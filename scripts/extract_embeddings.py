#!/usr/bin/env python3
"""
Extrai e cacheia embeddings de imagens e de classes para um modelo.

Para cada imagem do RVL-CDIP: passa image + prompt pelo VLM e salva o
mean pool dos tokens visuais nas camadas 0 (embedding) e -1 (última).

Os embeddings das 16 classes (texto puro, sem imagem) também são extraídos
e salvos uma vez por modelo — são reutilizados para todos os prompts.

Uso:
  # Extrai imagens + classes para o internvl3-2b, todos os prompts:
  python scripts/extract_embeddings.py \\
      --model internvl3-2b \\
      --all-prompts \\
      --splits test \\
      --max-per-class 200 \\
      --gpu 0

  # Só extrai embeddings das classes (útil para novos modelos):
  python scripts/extract_embeddings.py --model qwen25vl-2b --only-classes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_prompt_study import cache as C
from doc_prompt_study.extractor import build_extractor
from doc_prompt_study.rvl_cdip import load_rvl_cdip, RVL_CDIP_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LAYERS = (0, -1)  # embedding layer e última camada


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-ids", nargs="*", default=[])
    p.add_argument("--all-prompts", action="store_true")
    p.add_argument("--only-classes", action="store_true",
                   help="Extrai apenas embeddings das classes (pula imagens)")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "validation", "test"])
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--hf-dataset", default="jbxai/rvl-cdip")
    return p.parse_args()


def extract_class_embeddings(extractor, classes_cfg, cache_dir, model_name):
    """Extrai e cacheia os embeddings das 16 classes (texto puro). Idempotente."""
    logger.info("=== Embeddings das classes ===")
    for cls in classes_cfg:
        key  = cls["key"]
        desc = cls["description"]
        missing = [li for li in LAYERS if not C.cls_exists(cache_dir, model_name, key, li)]
        if not missing:
            logger.info("  [%s] já cacheado — pulando.", key)
            continue
        logger.info("  [%s] '%s'", key, desc[:70])
        emb_dict = extractor.extract_text(desc, layers=tuple(missing))
        for li, emb in emb_dict.items():
            C.cls_save(cache_dir, model_name, key, li, emb)
    logger.info("Classes prontas.")


def extract_image_embeddings(extractor, ds, prompt_id, prompt, cache_dir, model_name, split):
    missing_layers = [li for li in LAYERS if not C.img_exists(cache_dir, model_name, prompt_id, split, li)]
    if not missing_layers:
        logger.info("  [%s/%s] já cacheado — pulando.", prompt_id, split)
        return

    n = len(ds)
    display = repr(prompt[:60] + "...") if len(prompt) > 60 else repr(prompt)
    logger.info("  [%s] prompt=%s  layers=%s  (%d imagens)", prompt_id, display, missing_layers, n)

    # Acumula embeddings por camada
    all_embs:   dict[int, list[np.ndarray]] = {li: [] for li in missing_layers}
    all_labels: list[int] = []
    t0 = time.time()

    for i, ex in enumerate(ds):
        image = ex["image"].convert("RGB")
        label = ex["label"]
        emb_dict = extractor.extract_image(image, prompt, layers=tuple(missing_layers))
        for li in missing_layers:
            all_embs[li].append(emb_dict[li])
        all_labels.append(label)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            speed   = (i + 1) / elapsed
            eta_min = (n - i - 1) / speed / 60
            logger.info("    %d/%d  |  %.1f img/s  |  ETA ~%.0f min", i + 1, n, speed, eta_min)

    labels = np.array(all_labels)
    for li in missing_layers:
        embs = np.stack(all_embs[li])
        path = C.img_save(cache_dir, model_name, prompt_id, split, li, embs, labels)
        logger.info("    layer %s → %s  salvo em %s", li, embs.shape, path)


def main():
    args = parse_args()

    models_cfg  = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    prompts_cfg = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text())["prompts"]
    classes_cfg = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text())["classes"]

    if args.model not in models_cfg:
        logger.error("Modelo '%s' não encontrado. Disponíveis: %s", args.model, list(models_cfg))
        sys.exit(1)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info("Carregando modelo: %s", args.model)
    extractor = build_extractor(models_cfg[args.model], load_in_4bit=not args.no_4bit, device=device)

    # 1. Embeddings das classes (sempre extraídos, idempotente)
    extract_class_embeddings(extractor, classes_cfg, args.cache_dir, args.model)

    if args.only_classes:
        return

    # 2. Embeddings das imagens
    prompt_ids = list(prompts_cfg) if args.all_prompts else args.prompt_ids
    if not prompt_ids:
        logger.error("Passe --prompt-ids ou --all-prompts (ou --only-classes).")
        sys.exit(1)

    for split in args.splits:
        logger.info("=== Split: %s (max_per_class=%s) ===", split, args.max_per_class)
        ds = load_rvl_cdip(split=split, max_per_class=args.max_per_class, hf_dataset=args.hf_dataset)

        for pid in prompt_ids:
            if pid not in prompts_cfg:
                logger.error("Prompt '%s' não encontrado.", pid)
                continue
            extract_image_embeddings(
                extractor, ds, pid, prompts_cfg[pid], args.cache_dir, args.model, split
            )


if __name__ == "__main__":
    main()
