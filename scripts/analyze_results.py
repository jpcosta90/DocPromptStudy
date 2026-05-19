#!/usr/bin/env python3
"""
Gera heatmaps comparando os três modos de classificação:
  • embedding layer0   — mean pool embedding puro (sem processamento do LM)
  • embedding layerlast — mean pool saída do LM (influenciada pelo prompt)
  • generativo         — modelo gera texto → keyword match nas classes

Um heatmap por modo: linhas = prompts, colunas = modelos.

Uso:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --input results/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

MODE_ORDER  = ["layer0", "layerlast", "generative"]
MODE_TITLES = {
    "layer0":     "Mean Pool — Embedding Puro (layer 0)",
    "layerlast":  "Mean Pool — Saída do LM (última camada)",
    "generative": "Classificação Generativa (keyword match)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default=str(ROOT / "results" / "results.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    return p.parse_args()


def _coerce_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica a coluna de modo: em classify.py o modo está em 'layer',
    em generate_classify.py está em 'mode'='generative'.
    Cria coluna 'mode_key' com: layer0, layerlast, generative.
    """
    rows = []
    for _, r in df.iterrows():
        mode_col = str(r.get("mode", "")).strip()
        layer    = str(r.get("layer", "")).strip()
        if mode_col == "generative":
            key = "generative"
        elif layer == "layer0":
            key = "layer0"
        else:
            key = "layerlast"
        r = r.copy()
        r["mode_key"] = key
        rows.append(r)
    return pd.DataFrame(rows)


def plot_heatmap(pivot: pd.DataFrame, title: str, output: str, vmax: float = 1.0):
    fig_h = max(4, len(pivot) * 0.65)
    fig_w = max(6, len(pivot.columns) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        vmin=0.0, vmax=vmax, linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Modelo", fontsize=10)
    ax.set_ylabel("Prompt", fontsize=10)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Figura salva: {output}")
    plt.close()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    df["accuracy"] = df["accuracy"].astype(float)
    df = _coerce_mode(df)

    present_modes = [m for m in MODE_ORDER if m in df["mode_key"].values]

    for mode in present_modes:
        sub = df[df["mode_key"] == mode]
        pivot = sub.pivot_table(index="prompt_id", columns="model", values="accuracy", aggfunc="mean")
        # Ordena prompts por média decrescente (melhores no topo)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        out = str(Path(args.output_dir) / f"heatmap_{mode}.png")
        plot_heatmap(pivot, title=f"RVL-CDIP — {MODE_TITLES[mode]}", output=out)

        print(f"\n=== {mode} — Ranking de prompts (média entre modelos) ===")
        print(pivot.mean(axis=1).sort_values(ascending=False).to_string(float_format="{:.4f}".format))
        print()

    # Gráfico de barras comparando os três modos por modelo (melhor prompt de cada)
    if len(present_modes) > 1:
        _plot_mode_comparison(df, present_modes, args.output_dir)


def _plot_mode_comparison(df: pd.DataFrame, modes: list[str], output_dir: str):
    """Barras: melhor accuracy por (modelo, modo) — mostra o ceiling de cada abordagem."""
    records = []
    for mode in modes:
        sub = df[df["mode_key"] == mode]
        for model, grp in sub.groupby("model"):
            best = grp["accuracy"].max()
            records.append({"model": model, "mode": mode, "best_accuracy": best})

    comp = pd.DataFrame(records)
    models = comp["model"].unique()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.5), 4))
    x = range(len(models))
    w = 0.25
    colors = {"layer0": "#AECBFA", "layerlast": "#1A73E8", "generative": "#E37400"}

    for i, mode in enumerate(modes):
        vals = [comp.loc[(comp.model == m) & (comp.mode_key == mode), "best_accuracy"].values for m in models]
        vals = [v[0] if len(v) else 0.0 for v in vals]
        ax.bar([xi + i * w for xi in x], vals, width=w, label=MODE_TITLES.get(mode, mode),
               color=colors.get(mode, "#888"), edgecolor="white")

    ax.set_xticks([xi + w for xi in x])
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Best Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("RVL-CDIP — Melhor accuracy por modo × modelo", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = str(Path(output_dir) / "comparison_modes.png")
    plt.savefig(out, dpi=150)
    print(f"Figura salva: {out}")
    plt.close()


if __name__ == "__main__":
    main()
