#!/usr/bin/env python3
"""
Classificação zero-shot via cosine similarity entre embeddings de imagem e de classe.

Para cada imagem: predict = argmax_c cosine(embed_image, embed_class_c)
Nenhum treinamento — pure similarity matching.

Uso:
  python scripts/classify.py \\
      --model internvl3-2b \\
      --all-prompts \\
      --layer last \\
      --split test \\
      --output results/results.csv

  # Comparar camada 0 vs última:
  python scripts/classify.py --model internvl3-2b --all-prompts --layer 0 last

  # Múltiplos modelos de uma vez:
  python scripts/classify.py \\
      --model internvl3-2b qwen25vl-2b \\
      --prompt-ids no_prompt classify \\
      --layer last
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_prompt_study import cache as C
from doc_prompt_study.rvl_cdip import RVL_CDIP_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LAYER_ALIASES = {"0": 0, "last": -1, "-1": -1, "first": 0}


def cosine_classify(
    img_embs: np.ndarray,  # [N, dim]
    cls_embs: np.ndarray,  # [16, dim]
) -> np.ndarray:
    """Retorna predições [N] — índice da classe com maior cosine similarity."""
    # Normaliza L2
    img_n = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)
    cls_n = cls_embs / (np.linalg.norm(cls_embs, axis=1, keepdims=True) + 1e-8)
    # Similaridade: [N, 16]
    sims = img_n @ cls_n.T
    return sims.argmax(axis=1)


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def per_class_accuracy(preds, labels, n_classes=16) -> dict[str, float]:
    result = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        result[RVL_CDIP_CLASSES[c]] = float((preds[mask] == labels[mask]).mean())
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--prompt-ids", nargs="*", default=[])
    p.add_argument("--all-prompts", action="store_true")
    p.add_argument("--layer", nargs="+", default=["last"],
                   help="Índices de camada: '0' (embedding) ou 'last' / '-1' (última)")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--output", default=str(ROOT / "results" / "results.csv"))
    p.add_argument("--per-class", action="store_true", help="Imprime acurácia por classe")
    return p.parse_args()


def run_one(cache_dir, model, prompt_id, layer_idx, split, per_class=False):
    img_data = C.img_load(cache_dir, model, prompt_id, split, layer_idx)
    if img_data is None:
        logger.warning("Cache ausente: %s / %s / %s / layer%s — rode extract_embeddings.py.", model, prompt_id, split, layer_idx)
        return None

    cls_embs = C.cls_load_all(cache_dir, model, layer_idx)
    if cls_embs is None:
        logger.warning("Embeddings de classe ausentes para %s / layer%s — rode extract_embeddings.py --only-classes.", model, layer_idx)
        return None

    img_embs, labels = img_data
    preds = cosine_classify(img_embs, cls_embs)
    acc   = accuracy(preds, labels)

    if per_class:
        pc = per_class_accuracy(preds, labels)
        for cls_name, cls_acc in pc.items():
            logger.info("    %-28s %.3f", cls_name, cls_acc)

    return acc


def main():
    args = parse_args()

    prompts_cfg = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text())["prompts"]
    prompt_ids  = list(prompts_cfg) if args.all_prompts else args.prompt_ids
    if not prompt_ids:
        logger.error("Passe --prompt-ids ou --all-prompts.")
        sys.exit(1)

    layer_indices = []
    for l in args.layer:
        if l not in LAYER_ALIASES:
            logger.error("Camada '%s' inválida. Use: %s", l, list(LAYER_ALIASES))
            sys.exit(1)
        li = LAYER_ALIASES[l]
        if li not in layer_indices:
            layer_indices.append(li)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.model:
        for pid in prompt_ids:
            for li in layer_indices:
                layer_label = "layer0" if li == 0 else "layerlast"
                logger.info("%-18s | %-18s | %s | %s", model, pid, layer_label, args.split)
                acc = run_one(args.cache_dir, model, pid, li, args.split, args.per_class)
                if acc is not None:
                    logger.info("  Accuracy: %.4f", acc)
                    rows.append({
                        "model":       model,
                        "prompt_id":   pid,
                        "prompt":      prompts_cfg.get(pid, ""),
                        "layer":       layer_label,
                        "split":       args.split,
                        "accuracy":    f"{acc:.6f}",
                    })

    if not rows:
        logger.warning("Nenhum resultado gerado — verifique se os caches existem.")
        return

    fieldnames = ["model", "prompt_id", "prompt", "layer", "split", "accuracy"]
    write_header = not Path(args.output).exists()
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    logger.info("Resultados salvos em %s", args.output)

    # Ranking rápido no terminal
    rows_sorted = sorted(rows, key=lambda r: float(r["accuracy"]), reverse=True)
    print("\n=== Ranking (accuracy ↓) ===")
    for r in rows_sorted:
        print(f"  {r['model']:<20} {r['prompt_id']:<20} {r['layer']:<12} {float(r['accuracy']):.4f}")


if __name__ == "__main__":
    main()
