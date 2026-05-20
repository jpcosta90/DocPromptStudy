#!/usr/bin/env python3
"""
Experiment: class-specific prompts vs generic prompts with multi-layer pooling.

Design:
  For each document image I, we evaluate two classification modes:

  Mode A – class-specific prompt (LOO centroid):
    For each candidate class C, embed image I using "Is this a [C]?" and
    compare to centroid of class-C training images embedded with the same prompt.
    Predicted class = argmax_C cosine(E(I, prompt_C), centroid_C).

  Mode B – generic prompt (LOO centroid):
    Single embedding per image ("What type of document is this?") compared to
    per-class centroids built with the same generic prompt.

  Mode C – no prompt (baseline).

  Layer strategies:
    last1  – mean pool of last transformer layer only
    last5  – mean pool of last 5 layers
    last12 – mean pool of last 12 layers
    all    – mean pool of all layers (including embedding layer 0)

Usage:
  TMPDIR=/tmp python scripts/experiment_class_prompts.py \\
      --local-data /mnt/data/zs_rvl_cdip/data \\
      --local-splits-csv /mnt/data/zs_rvl_cdip/splits.csv \\
      --max-per-class 3 \\
      --output results/report_class_prompts.html
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_prompt_study.extractor import build_extractor
from doc_prompt_study.rvl_cdip import load_local_rvl_cdip, RVL_CDIP_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES  = RVL_CDIP_CLASSES                              # display names (may have spaces)
CKEYS    = [c.replace(" ", "_") for c in CLASSES]        # dict keys / folder names

CLASS_PROMPTS: dict[str, str] = {
    key: f"Is this document a {name}?"
    for key, name in zip(CKEYS, CLASSES)
}
GENERIC_PROMPTS: dict[str, str] = {
    "generic":   "What type of document is this?",
    "no_prompt": "",
}
ALL_PROMPTS: dict[str, str] = {**CLASS_PROMPTS, **GENERIC_PROMPTS}
ALL_PKEYS  = list(ALL_PROMPTS)          # 16 class keys + 2 generic keys

LAYER_STRATEGIES = ["last1", "last5", "last12", "all"]
STRATEGY_LABELS  = {
    "last1":  "Last 1 layer",
    "last5":  "Last 5 layers",
    "last12": "Last 12 layers",
    "all":    "All layers",
}
MODES = {
    "class_specific": "Class-specific<br><em>Is this a [X]?</em>",
    "generic":        "Generic<br><em>What type of document?</em>",
    "no_prompt":      "No prompt",
}

# Families that cannot use 4-bit (e.g. Gemma-3n AltUp)
NO_4BIT_FAMILIES = {"gemma"}


# ---------------------------------------------------------------------------
# Multi-layer pooling
# ---------------------------------------------------------------------------

def pool_layers(layer_vecs: np.ndarray, strategy: str) -> np.ndarray:
    """
    layer_vecs: (L, dim) – L hidden states (index 0 = embedding layer, rest = transformer)
    Returns: (dim,) float32
    """
    n = len(layer_vecs)
    if strategy == "last1":
        return layer_vecs[-1]
    if strategy == "last5":
        return layer_vecs[max(0, n - 5):].mean(axis=0)
    if strategy == "last12":
        return layer_vecs[max(0, n - 12):].mean(axis=0)
    if strategy == "all":
        return layer_vecs.mean(axis=0)
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# LOO centroid classification
# ---------------------------------------------------------------------------

def loo_accuracy(
    embs: dict[str, np.ndarray],   # {pkey: (N, L, dim)}
    labels: np.ndarray,            # (N,) int
    mode: str,
    strategy: str,
) -> tuple[float, np.ndarray]:
    """
    LOO centroid classification.
    Returns (overall_acc, per_class_acc[16]).
    """
    N = len(labels)
    n_cls = len(CLASSES)
    correct = 0
    pc_correct = np.zeros(n_cls, dtype=int)
    pc_total   = np.zeros(n_cls, dtype=int)

    for i in range(N):
        true_cls = int(labels[i])
        pc_total[true_cls] += 1
        scores: list[float] = []

        for c_idx in range(n_cls):
            # Which prompt embedding to use for this candidate class
            if mode == "class_specific":
                pkey = CKEYS[c_idx]
            else:
                pkey = mode  # "generic" or "no_prompt"

            arr = embs[pkey]   # (N, L, dim)
            test_vec = pool_layers(arr[i], strategy)

            # Class centroid (LOO: exclude i if same class)
            mask = labels == c_idx
            cidx = [j for j in range(N) if mask[j] and j != i]
            if not cidx:
                scores.append(-2.0)
                continue

            vecs     = np.stack([pool_layers(arr[j], strategy) for j in cidx])
            centroid = vecs.mean(axis=0)

            tn = test_vec / (np.linalg.norm(test_vec) + 1e-10)
            cn = centroid  / (np.linalg.norm(centroid)  + 1e-10)
            scores.append(float(tn @ cn))

        pred = int(np.argmax(scores))
        if pred == true_cls:
            correct += 1
            pc_correct[true_cls] += 1

    acc = correct / N
    pc_acc = np.where(
        pc_total > 0,
        pc_correct / np.maximum(pc_total, 1),
        float("nan"),
    )
    return acc, pc_acc


# ---------------------------------------------------------------------------
# Embedding extraction (with per-model disk cache)
# ---------------------------------------------------------------------------

def _get_num_hidden_layers(extractor) -> int:
    cfg = extractor.model.config
    for sub in [cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)]:
        if sub is not None and hasattr(sub, "num_hidden_layers"):
            return sub.num_hidden_layers
    raise RuntimeError(f"Cannot detect num_hidden_layers for {extractor.hf_path}")


def extract_model_embeddings(
    model_name: str,
    model_cfg: dict,
    local_data: str,
    splits_csv: str | None,
    max_per_class: int,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Returns embs = {pkey: (N, L, dim)}, labels = (N,).
    Results are cached to disk so re-runs skip extraction.
    """
    cache_file = cache_dir / f"{model_name}_mpc{max_per_class}.npz"
    if cache_file.exists():
        logger.info("[%s] Loading from cache: %s", model_name, cache_file)
        data   = np.load(cache_file, allow_pickle=True)
        labels = data["__labels__"]
        embs   = {k: data[k] for k in ALL_PKEYS}
        return embs, labels

    # Load dataset
    ds     = load_local_rvl_cdip(data_dir=local_data, splits_csv=splits_csv, max_per_class=max_per_class)
    N      = len(ds)
    logger.info("[%s] Dataset: %d images", model_name, N)

    # Load model
    family   = model_cfg.get("family", "")
    use_4bit = family not in NO_4BIT_FAMILIES
    logger.info("[%s] Loading model (4bit=%s)...", model_name, use_4bit)
    extractor = build_extractor(model_cfg, load_in_4bit=use_4bit)

    n_layers   = _get_num_hidden_layers(extractor)
    all_layers = tuple(range(n_layers + 1))  # 0 = embedding, 1..N = transformer layers
    logger.info("[%s] num_hidden_layers=%d", model_name, n_layers)

    # Extraction loop: per image, run all prompts in one model load
    embs_raw: dict[str, list[np.ndarray]] = {k: [] for k in ALL_PKEYS}
    all_labels: list[int] = []

    for img_idx, ex in enumerate(ds):
        img = ex["image"]
        all_labels.append(int(ex["label"]))

        if img_idx % 8 == 0:
            logger.info("[%s]  %d/%d", model_name, img_idx, N)

        for pkey in ALL_PKEYS:
            layer_dict = extractor.extract_image(img, ALL_PROMPTS[pkey], layers=all_layers)
            mat = np.stack([layer_dict[i] for i in range(n_layers + 1)])  # (L, dim)
            embs_raw[pkey].append(mat)

    embs   = {k: np.stack(v) for k, v in embs_raw.items()}   # (N, L, dim)
    labels = np.array(all_labels, dtype=np.int32)

    # Cache to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, __labels__=labels, **embs)
    logger.info("[%s] Cached → %s", model_name, cache_file)

    # Free GPU
    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    return embs, labels


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _acc_cell(v: float, best: bool = False) -> str:
    if np.isnan(v):
        return '<td style="background:#ddd">–</td>'
    pct = v * 100
    # Red → yellow → green
    r = max(0, int(220 * (1 - v)))
    g = max(0, int(190 * v))
    bg = f"rgb({r},{g},60)"
    txt_color = "white" if v < 0.28 or v > 0.72 else "black"
    fw = "font-weight:bold;" if best else ""
    return f'<td style="background:{bg};color:{txt_color};{fw}">{pct:.1f}%</td>'


def generate_html(
    results: dict[str, dict[tuple, tuple]],
    max_per_class: int,
    date_str: str,
) -> str:
    model_names = list(results.keys())
    n_img = max_per_class * len(CLASSES)

    # ── Summary table ────────────────────────────────────────────────────────
    header = (
        "<tr><th>Model</th><th>Prompt mode</th>"
        + "".join(f"<th>{STRATEGY_LABELS[s]}</th>" for s in LAYER_STRATEGIES)
        + "</tr>"
    )
    rows = []
    for model in model_names:
        for mode_key, mode_label in MODES.items():
            accs = [results[model][(mode_key, s)][0] for s in LAYER_STRATEGIES]
            best_v = max(accs)
            cells = "".join(_acc_cell(a, a == best_v) for a in accs)
            rows.append(f"<tr><td>{model}</td><td>{mode_label}</td>{cells}</tr>")
    summary = f"""
<table>
  <thead>{header}</thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""

    # ── Per-class table (class-specific vs generic, best strategy) ───────────
    cls_header = (
        "<tr><th>Model</th>"
        + "".join(
            f'<th style="writing-mode:vertical-rl;transform:rotate(180deg);'
            f'white-space:nowrap;padding:4px 2px">{c}</th>'
            for c in CLASSES
        )
        + "</tr>"
    )
    cls_rows = []
    for model in model_names:
        best_strat = max(
            LAYER_STRATEGIES,
            key=lambda s: results[model][("class_specific", s)][0],
        )
        _, pc_cs  = results[model][("class_specific", best_strat)]
        _, pc_gen = results[model][("generic",        best_strat)]

        cells = []
        for c_idx in range(len(CLASSES)):
            v_cs  = pc_cs[c_idx]
            v_gen = pc_gen[c_idx]
            if np.isnan(v_cs):
                cells.append('<td style="background:#ddd">–</td>')
                continue
            delta = v_cs - v_gen
            if delta > 0.05:
                bg, sym = "#a8e6a3", "▲"
            elif delta < -0.05:
                bg, sym = "#f5a3a3", "▼"
            else:
                bg, sym = "#e8e8e8", "–"
            title = f"{CLASSES[c_idx]}: class-specific={v_cs:.0%} generic={v_gen:.0%}"
            cells.append(
                f'<td style="background:{bg}" title="{title}">'
                f'{sym} {v_cs:.0%}</td>'
            )

        cls_rows.append(
            f"<tr><td>{model}<br><small>(layer: {best_strat})</small></td>"
            + "".join(cells)
            + "</tr>"
        )
    per_class = f"""
<table>
  <thead>{cls_header}</thead>
  <tbody>{"".join(cls_rows)}</tbody>
</table>
<p style="font-size:.85em;color:#555">
  ▲ = class-specific prompt beats generic by &gt;5 pp &nbsp;|&nbsp;
  ▼ = generic beats class-specific by &gt;5 pp &nbsp;|&nbsp;
  – = within 5 pp. Cell shows class-specific accuracy.
  Hover for exact values.
</p>"""

    # ── Per-model best breakdown ─────────────────────────────────────────────
    details = []
    for model in model_names:
        rows_d = []
        for mode_key, mode_label in MODES.items():
            best_s = max(LAYER_STRATEGIES, key=lambda s: results[model][(mode_key, s)][0])
            acc, pc = results[model][(mode_key, best_s)]
            cell_row = "".join(
                f'<td title="{CLASSES[i]}">{pc[i]:.0%}</td>'
                for i in range(len(CLASSES))
            )
            rows_d.append(
                f"<tr><td>{mode_label}</td><td><strong>{acc:.1%}</strong> ({best_s})</td>{cell_row}</tr>"
            )
        details.append(f"""
<h3>{model}</h3>
<table style="font-size:.82em">
  <thead>
    <tr><th>Mode</th><th>Best overall</th>
    {"".join(f'<th style="writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap">{c}</th>' for c in CLASSES)}
    </tr>
  </thead>
  <tbody>{"".join(rows_d)}</tbody>
</table>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Class-Specific Prompt Experiment — RVL-CDIP</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2em 3em; background: #f8f8f8; color: #222; }}
    h1   {{ color: #1a1a2e; }}
    h2   {{ color: #444; border-bottom: 2px solid #ccc; padding-bottom: 6px; margin-top: 2em; }}
    h3   {{ color: #555; }}
    table         {{ border-collapse: collapse; margin-bottom: 1.5em; background: white;
                    box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    th            {{ background: #2c3e50; color: white; padding: 8px 12px; }}
    td            {{ padding: 6px 10px; text-align: center; border: 1px solid #ddd; }}
    td:first-child, td:nth-child(2) {{ text-align: left; }}
    .meta         {{ color: #666; font-size: .9em; margin-bottom: 1.5em; background: white;
                    padding: 1em; border-left: 4px solid #2c3e50; }}
    .note         {{ background: #fffbe6; border: 1px solid #f0c040; padding: 1em;
                    border-radius: 4px; margin-bottom: 1em; font-size: .9em; }}
  </style>
</head>
<body>
<h1>Class-Specific Prompt Experiment — RVL-CDIP</h1>
<div class="meta">
  <strong>Date:</strong> {date_str} &nbsp;|&nbsp;
  <strong>Sample:</strong> {max_per_class} images/class × {len(CLASSES)} classes = {n_img} images &nbsp;|&nbsp;
  <strong>Method:</strong> Leave-one-out centroid classification (cosine similarity)
</div>

<div class="note">
  <strong>Key question:</strong> Does embedding a document image under "Is this a letter?"
  (class-specific prompt) make it a better classifier of letters — compared to a generic prompt?<br>
  <strong>Layer strategy:</strong> Mean pool of last-1 / last-5 / last-12 / all hidden layers.
</div>

<h2>Top-1 Accuracy — Prompt Mode × Layer Strategy</h2>
<p>Bold = best across strategies for that model+mode combination.</p>
{summary}

<h2>Per-Class Accuracy — Class-Specific vs Generic (best layer strategy per model)</h2>
{per_class}

<h2>Per-Model Detail (best strategy per mode)</h2>
{"".join(details)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    cfg_models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    p.add_argument("--models", nargs="+", default=list(cfg_models))
    p.add_argument("--local-data",        default="/mnt/data/zs_rvl_cdip/data")
    p.add_argument("--local-splits-csv",  default="/mnt/data/zs_rvl_cdip/splits.csv")
    p.add_argument("--max-per-class",     type=int, default=3)
    p.add_argument("--output",            default=str(ROOT / "results" / "report_class_prompts.html"))
    p.add_argument("--embed-cache-dir",   default=str(ROOT / "cache" / "class_prompt_exp"))
    args = p.parse_args()

    cache_dir   = Path(args.embed_cache_dir)
    all_results: dict[str, dict] = {}

    for model_name in args.models:
        if model_name not in cfg_models:
            logger.error("Model '%s' not in models.yaml. Skipping.", model_name)
            continue
        logger.info("========== %s ==========", model_name)
        embs, labels = extract_model_embeddings(
            model_name, cfg_models[model_name],
            args.local_data, args.local_splits_csv,
            args.max_per_class, cache_dir,
        )
        model_res: dict[tuple, tuple] = {}
        for mode in ["class_specific", "generic", "no_prompt"]:
            for strat in LAYER_STRATEGIES:
                acc, pc = loo_accuracy(embs, labels, mode, strat)
                model_res[(mode, strat)] = (acc, pc)
                logger.info("  %-16s | %-6s → %.4f", mode, strat, acc)
        all_results[model_name] = model_res

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        generate_html(all_results, args.max_per_class, datetime.now().strftime("%Y-%m-%d %H:%M")),
        encoding="utf-8",
    )
    logger.info("Report → %s", out)


if __name__ == "__main__":
    main()
