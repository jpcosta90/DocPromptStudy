#!/usr/bin/env python3
"""
Figura de paper: norma dos hidden states visuais por patch (Layer 0 vs Layer -1).

Layout 2×2:
  Linhas  : duas classes de documentos
  Colunas : Layer 0 (embeddings de entrada) | Layer -1 (embeddings de saída)

Cada painel mostra ||h_i^v||_2 por patch espacial — sem prompt, sem texto.
A diferença entre colunas evidencia o enriquecimento do LLM.
A diferença entre linhas evidencia distinção de classes.

Uso:
  python scripts/paper_figure_repr_change.py \\
      --output results/paper_figures/attention/fig_repr_change \\
      --gpu 0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   8,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_NUM  = 12
IMG_SIZE = 448
PROMPT   = "Analyze this document."   # necessário para o formato do InternVL; não afeta visuais

# Dois documentos por classe para mostrar consistência intra-classe
DOCUMENT_CLASSES = [
    {
        "label": "Cigarette Analysis Report",
        "dir":   "/mnt/data/la-cdip/data/cigarrete_analysis_report",
        "n_docs": 2,   # usa os 2 primeiros .tif da pasta
    },
    {
        "label": "Philip Morris Letter",
        "dir":   "/mnt/data/la-cdip/data/philip_morris_letter",
        "n_docs": 2,
    },
]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Carregando InternVL3-2B…")
    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False
    )
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model, tok


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def get_norm_maps(model, tokenizer, image: Image.Image):
    """
    Retorna (norm_map_l0, norm_map_lN, rows, cols):
      norm_map_lX: ndarray (rows, cols) — ||h_i^v||_2 médio por patch na camada X
    Sem hooks de atenção, sem dependência de prompt.
    """
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform, find_closest_aspect_ratio
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=MAX_NUM, image_size=IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16).to(
        next(model.parameters()).device
    )

    w, h   = image.size
    ratios = sorted(
        {(i, j) for n in range(1, MAX_NUM + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= MAX_NUM},
        key=lambda r: r[0] * r[1],
    )
    cols, rows = find_closest_aspect_ratio(w / h, ratios, w, h, IMG_SIZE)
    n_tok      = model.num_image_token
    n_spatial  = rows * cols

    inp        = prepare_inputs_for_multimodal_embedding(model, tokenizer, pv, PROMPT)
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    seq        = inp["input_ids"][0].cpu()
    img_pos    = (seq == img_ctx_id).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        out = model(
            input_ids=inp["input_ids"],
            pixel_values=inp["pixel_values"],
            image_flags=inp["image_flags"],
            output_hidden_states=True,
            return_dict=True,
        )

    def patch_norms(layer_idx):
        h = out.hidden_states[layer_idx][0].float().cpu()   # [seq, dim]
        norms = h[img_pos].norm(dim=-1)                      # [n_img_tokens]
        spatial = norms[:n_spatial * n_tok].reshape(n_spatial, n_tok).mean(dim=1)
        return spatial.reshape(rows, cols).numpy()

    l0 = patch_norms(0)
    lN = patch_norms(-1)

    def norm_pct(m):
        lo = np.percentile(m, 5)
        hi = np.percentile(m, 99)
        return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

    return norm_pct(l0), norm_pct(lN), rows, cols


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay(image: Image.Image, heat_map: np.ndarray,
            alpha: float, colormap: str) -> np.ndarray:
    gray    = np.array(image.convert("L").convert("RGB")).astype(np.float32)
    h, w    = gray.shape[:2]
    heat    = Image.fromarray((heat_map * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heat_np  = np.array(heat) / 255.0
    heat_rgb = plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255
    return np.clip(gray * (1 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

PANEL_LABELS = "abcdef"

# Layout 1×6:
# [cls_A L0] [cls_A doc1 L-1] [cls_A doc2 L-1] | [cls_B L0] [cls_B doc1 L-1] [cls_B doc2 L-1]
# panels = lista de 6 dicts: {image, heatmap, title, sublabel}


def make_figure_1x6(panels: list[dict], class_labels: list[str],
                    output_path: str, colormap: str, alpha: float, dpi: int):
    """
    6 painéis em linha com separador vertical entre as duas classes.
    panels[i]: {image, heatmap, col_title}
    """
    n_cols  = 6
    col_w   = 1.95
    row_h   = 3.2
    top_m   = 0.80   # headers
    bot_m   = 0.08
    left_m  = 0.15
    cbar_w  = 0.15
    right_p = 0.35

    fig_w = left_m + n_cols * col_w + right_p + cbar_w
    fig_h = top_m + row_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        1, n_cols + 1,
        left   = left_m / fig_w,
        right  = 1.0 - (right_p) / fig_w,
        top    = 1.0 - top_m / fig_h,
        bottom = bot_m / fig_h,
        wspace = 0.022,
        width_ratios=[1]*n_cols + [cbar_w / col_w],
    )
    cbar_ax = fig.add_subplot(gs[0, n_cols])

    # Títulos das colunas
    col_titles = [
        "Layer 0\n(input)",
        "Layer $-1$\ndoc.~1",
        "Layer $-1$\ndoc.~2",
        "Layer 0\n(input)",
        "Layer $-1$\ndoc.~1",
        "Layer $-1$\ndoc.~2",
    ]
    for c, title in enumerate(col_titles):
        pos = gs[0, c].get_position(fig)
        fig.text(pos.x0 + pos.width / 2,
                 1.0 - top_m * 0.22 / fig_h,
                 title, ha="center", va="top",
                 fontsize=7.5, fontweight="bold",
                 transform=fig.transFigure)

    # Labels de classe (acima dos grupos)
    for cls_idx, cls_label in enumerate(class_labels):
        c_start = cls_idx * 3
        pos_l = gs[0, c_start].get_position(fig)
        pos_r = gs[0, c_start + 2].get_position(fig)
        mid_x = (pos_l.x0 + pos_r.x1) / 2
        fig.text(mid_x, 1.0 - top_m * 0.68 / fig_h,
                 cls_label,
                 ha="center", va="top",
                 fontsize=8, fontweight="bold", style="italic",
                 transform=fig.transFigure)

        # Linha abaixo do label de classe
        fig.add_artist(matplotlib.lines.Line2D(
            [pos_l.x0, pos_r.x1],
            [1.0 - top_m * 0.82 / fig_h] * 2,
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=0.7,
        ))

    # Separador vertical entre as duas classes
    pos_mid_r = gs[0, 2].get_position(fig)
    pos_mid_l = gs[0, 3].get_position(fig)
    div_x = (pos_mid_r.x1 + pos_mid_l.x0) / 2
    fig.add_artist(matplotlib.lines.Line2D(
        [div_x, div_x],
        [bot_m / fig_h, 1.0 - top_m * 0.68 / fig_h],
        transform=fig.transFigure,
        color="#999999", linewidth=0.8, linestyle="--",
    ))

    for c, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(overlay(panel["image"], panel["heatmap"], alpha, colormap))
        ax.axis("off")
        ax.text(0.03, 0.97, f"({PANEL_LABELS[c]})",
                transform=ax.transAxes,
                fontsize=7, color="white", va="top", ha="left",
                fontweight="bold", fontfamily="serif",
                bbox=dict(boxstyle="square,pad=0.1",
                          fc="black", ec="none", alpha=0.45))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "", "High"])
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    cb.set_label("$\\|\\mathbf{h}_i^v\\|_2$", fontsize=7.5, labelpad=4)
    cb.outline.set_linewidth(0.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out}.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Salvo em {out}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default=str(ROOT / "results/paper_figures/attention/fig_repr_change"))
    p.add_argument("--colormap", default="inferno")
    p.add_argument("--alpha",    type=float, default=0.55)
    p.add_argument("--dpi",      type=int,   default=300)
    p.add_argument("--gpu",      type=int,   default=0)
    return p.parse_args()


def main():
    args  = parse_args()
    model, tokenizer = load_model()

    panels      = []
    class_labels = []

    for cls in DOCUMENT_CLASSES:
        tifs  = sorted(Path(cls["dir"]).glob("*.tif"))[: cls["n_docs"] + 1]
        label = cls["label"]
        class_labels.append(label)
        print(f"\n=== {label} ===")

        # Layer 0 e Layer -1 do primeiro documento
        img0  = Image.open(tifs[0]).convert("RGB")
        l0, lN, rows, cols = get_norm_maps(model, tokenizer, img0)
        print(f"  doc1  grid: {rows}×{cols}")
        panels.append({"image": img0, "heatmap": l0})   # L0
        panels.append({"image": img0, "heatmap": lN})   # L-1 doc1
        torch.cuda.empty_cache()

        # Apenas Layer -1 do segundo documento
        img1  = Image.open(tifs[1]).convert("RGB")
        _, lN2, _, _ = get_norm_maps(model, tokenizer, img1)
        print(f"  doc2  grid: {rows}×{cols}")
        panels.append({"image": img1, "heatmap": lN2})  # L-1 doc2
        torch.cuda.empty_cache()

    make_figure_1x6(panels, class_labels,
                    args.output, args.colormap, args.alpha, args.dpi)

    import shutil
    dest_dir = ROOT.parent / "CaVL-Doc" / "docs" / "assets"
    if dest_dir.exists():
        shutil.copy2(f"{args.output}.png", dest_dir / "fig_repr_change.png")
        print(f"Copiado para {dest_dir / 'fig_repr_change.png'}")


if __name__ == "__main__":
    main()
