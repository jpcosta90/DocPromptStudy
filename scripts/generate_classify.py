#!/usr/bin/env python3
"""
Classificação generativa zero-shot: o modelo gera texto e fazemos
keyword matching contra as 16 classes do RVL-CDIP.

Contraste com classify.py (modo embedding):
  classify.py         → mean pool hidden states → cosine similarity vs class embeddings
  generate_classify.py → gera texto → parse de classe por keyword matching

Uso:
  # Modo generativo com prompt explícito de classificação:
  python scripts/generate_classify.py \\
      --model internvl3-2b \\
      --prompt-ids generative_classify unrelated \\
      --split test \\
      --max-per-class 100 \\
      --output results/results.csv

  # Todos os prompts (inclui no_prompt como baseline generativo):
  python scripts/generate_classify.py --model qwen25vl-2b --all-prompts --split test
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_prompt_study import cache as C
from doc_prompt_study.extractor import build_extractor
from doc_prompt_study.rvl_cdip import load_rvl_cdip, RVL_CDIP_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword matching: texto gerado → índice de classe
# Mais robusto que exact match — cobre variações de resposta do modelo
# ---------------------------------------------------------------------------

_CLASS_KEYWORDS: dict[int, list[str]] = {
    0:  ["letter"],
    1:  ["form"],
    2:  ["email", "e-mail", "electronic mail"],
    3:  ["handwritten", "handwriting", "hand written", "hand-written"],
    4:  ["advertisement", "advert", "ad "],
    5:  ["scientific report", "technical report"],
    6:  ["scientific publication", "journal article", "journal paper", "publication"],
    7:  ["specification", "spec "],
    8:  ["file folder", "folder"],
    9:  ["news article", "newspaper", "news paper"],
    10: ["budget"],
    11: ["invoice", "billing"],
    12: ["presentation", "slide"],
    13: ["questionnaire", "survey"],
    14: ["resume", "curriculum vitae", "cv "],
    15: ["memo", "memorandum"],
}

# Prioridade: classes mais específicas (frases longas) devem ser verificadas
# antes de classes com keywords curtas que podem ser substrings de outras.
_PRIORITY_ORDER = sorted(
    _CLASS_KEYWORDS.items(),
    key=lambda kv: -max(len(k) for k in kv[1]),
)


def text_to_label(text: str) -> int:
    """
    Converte resposta gerada pelo modelo em índice de classe (0-15).
    Retorna -1 se nenhuma classe for identificada.
    """
    t = text.lower().strip()
    for label, keywords in _PRIORITY_ORDER:
        for kw in keywords:
            if kw in t:
                return label
    return -1  # não identificado


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--prompt-ids", nargs="*", default=[])
    p.add_argument("--all-prompts", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--hf-dataset", default="jbxai/rvl-cdip")
    p.add_argument("--output", default=str(ROOT / "results" / "results.csv"))
    p.add_argument("--force", action="store_true", help="Re-extrai mesmo que já cacheado")
    return p.parse_args()


def run_generation(extractor, ds, prompt, cache_dir, model_name, prompt_id, split, max_new_tokens, force):
    if not force and C.gen_exists(cache_dir, model_name, prompt_id, split):
        logger.info("  [%s] já cacheado — pulando (use --force para re-extrair).", prompt_id)
        data = C.gen_load(cache_dir, model_name, prompt_id, split)
        return data[0], data[1]  # labels, predicted_labels

    n = len(ds)
    display = repr(prompt[:60] + "...") if len(prompt) > 60 else repr(prompt)
    logger.info("  [%s] prompt=%s  (%d imagens)", prompt_id, display, n)

    all_labels:   list[int] = []
    all_preds:    list[int] = []
    all_texts:    list[str] = []
    no_match_count = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        image = ex["image"].convert("RGB")
        label = ex["label"]
        text  = extractor.generate_text(image, prompt, max_new_tokens=max_new_tokens)
        pred  = text_to_label(text)

        all_labels.append(label)
        all_preds.append(pred)
        all_texts.append(text)

        if pred == -1:
            no_match_count += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed   = (i + 1) / elapsed
            eta_min = (n - i - 1) / speed / 60
            running_acc = sum(p == l for p, l in zip(all_preds, all_labels) if p != -1) / max(1, i + 1)
            logger.info(
                "    %d/%d  |  %.1f img/s  |  ETA ~%.0f min  |  acc_so_far=%.3f  |  no_match=%d",
                i + 1, n, speed, eta_min, running_acc, no_match_count,
            )

    labels = np.array(all_labels)
    preds  = np.array(all_preds)

    C.gen_save(cache_dir, model_name, prompt_id, split, labels, preds, all_texts)

    if no_match_count > 0:
        logger.warning("  %d/%d respostas sem match de classe (pred=-1) — contadas como erros.", no_match_count, n)

    return labels, preds


def main():
    args = parse_args()

    models_cfg  = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    prompts_cfg = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text())["prompts"]

    prompt_ids = list(prompts_cfg) if args.all_prompts else args.prompt_ids
    if not prompt_ids:
        logger.error("Passe --prompt-ids ou --all-prompts.")
        sys.exit(1)

    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in args.model:
        if model_name not in models_cfg:
            logger.error("Modelo '%s' não encontrado.", model_name)
            continue

        logger.info("Carregando modelo: %s", model_name)
        extractor = build_extractor(models_cfg[model_name], load_in_4bit=not args.no_4bit, device=device)

        logger.info("=== Split: %s (max_per_class=%s) ===", args.split, args.max_per_class)
        ds = load_rvl_cdip(split=args.split, max_per_class=args.max_per_class, hf_dataset=args.hf_dataset)

        for pid in prompt_ids:
            if pid not in prompts_cfg:
                logger.warning("Prompt '%s' não encontrado — pulando.", pid)
                continue

            labels, preds = run_generation(
                extractor, ds, prompts_cfg[pid],
                args.cache_dir, model_name, pid, args.split,
                args.max_new_tokens, args.force,
            )

            # Acurácia: pred=-1 conta como erro
            acc = float((preds == labels).mean())
            match_rate = float((preds != -1).mean())

            logger.info("  %-20s | acc=%.4f | match_rate=%.3f", pid, acc, match_rate)
            rows.append({
                "model":      model_name,
                "prompt_id":  pid,
                "prompt":     prompts_cfg[pid][:80],
                "mode":       "generative",
                "layer":      "—",
                "split":      args.split,
                "accuracy":   f"{acc:.6f}",
                "match_rate": f"{match_rate:.4f}",
            })

        # Libera memória antes do próximo modelo
        del extractor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not rows:
        logger.warning("Nenhum resultado gerado.")
        return

    fieldnames = ["model", "prompt_id", "prompt", "mode", "layer", "split", "accuracy", "match_rate"]
    write_header = not Path(args.output).exists()
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    logger.info("Resultados salvos em %s", args.output)

    print("\n=== Ranking generativo (accuracy ↓) ===")
    for r in sorted(rows, key=lambda r: float(r["accuracy"]), reverse=True):
        print(f"  {r['model']:<20} {r['prompt_id']:<22} acc={float(r['accuracy']):.4f}  match={float(r['match_rate']):.3f}")


if __name__ == "__main__":
    main()
