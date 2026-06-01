#!/usr/bin/env python3
"""
Figura de paper: representação PCA-RGB dos hidden states visuais.

Cada patch recebe uma cor derivada dos seus 3 primeiros componentes principais
(ajustados em TODOS os patches das duas imagens e das duas camadas juntos).

  Cor  = direção no subespaço PCA  →  patches similares têm cores similares
  Brilho = norma ||h||_2           →  patches mais ativados ficam mais vívidos

Layout 2×2:
  Linhas  : documento 1 | documento 2
  Colunas : Layer 0 (raw ViT embeddings) | Layer -1 (output LLM)

Uso:
  python scripts/paper_figure_pca_rgb.py \\
      --output results/paper_figures/attention/fig_pca_rgb \\
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
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_NUM  = 12
IMG_SIZE = 448
PROMPT   = "Analyze this document."

DOCUMENTS = [
    ("/mnt/data/la-cdip/data/cigarrete_analysis_report",
     "Cigarette Analysis Report\n(dense table layout)"),
    ("/mnt/data/la-cdip/data/philip_morris_letter",
     "Philip Morris Letter\n(compact letter layout)"),
]

PANEL_LABELS = "abcd"
COL_HEADERS  = [
    "Layer 0\n(input embeddings)",
    "Layer $-1$\n(output embeddings)",
]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Carregando InternVL3-2B…")
    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False
    )
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model, tok


# ---------------------------------------------------------------------------
# Extração de hidden states por patch
# ---------------------------------------------------------------------------

TOK_SIDE = 16   # sqrt(256) — tokens por lado de cada tile após pixel shuffle

def get_token_hidden_states(model, tokenizer, image: Image.Image):
    """
    Retorna (tokens_l0, tokens_lN, h_full, w_full):
      tokens_lX: ndarray (h_full*w_full, dim) em ordem espacial row-major
      h_full = rows_tiles * TOK_SIDE   (ex: 4*16 = 64)
      w_full = cols_tiles * TOK_SIDE   (ex: 3*16 = 48)
    """
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform, find_closest_aspect_ratio
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(IMG_SIZE)
    blocks    = dynamic_preprocess(image, max_num=MAX_NUM, image_size=IMG_SIZE, use_thumbnail=True)
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
    cols_tiles, rows_tiles = find_closest_aspect_ratio(w / h, ratios, w, h, IMG_SIZE)
    n_tok     = model.num_image_token    # 256
    n_spatial = rows_tiles * cols_tiles

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

    # Dimensões do grid de tokens individuais
    h_full = rows_tiles * TOK_SIDE
    w_full = cols_tiles * TOK_SIDE

    def spatial_tokens(layer_idx) -> np.ndarray:
        hs = out.hidden_states[layer_idx][0].float().cpu()   # [seq, dim]
        vis = hs[img_pos][:n_spatial * n_tok].numpy()         # [n_spatial*n_tok, dim]

        # Reconstrói grid espacial: cada tile tem TOK_SIDE×TOK_SIDE tokens
        # Ordem na sequência: tile (r,c) → tokens row-major 16×16
        grid = np.zeros((h_full, w_full, vis.shape[-1]), dtype=np.float32)
        for tile_r in range(rows_tiles):
            for tile_c in range(cols_tiles):
                tile_idx = tile_r * cols_tiles + tile_c
                tile_tokens = vis[tile_idx * n_tok: (tile_idx + 1) * n_tok]  # [256, dim]
                tile_grid   = tile_tokens.reshape(TOK_SIDE, TOK_SIDE, -1)
                r0 = tile_r * TOK_SIDE
                c0 = tile_c * TOK_SIDE
                grid[r0:r0+TOK_SIDE, c0:c0+TOK_SIDE] = tile_grid

        return grid.reshape(h_full * w_full, -1)

    return spatial_tokens(0), spatial_tokens(-1), h_full, w_full


# ---------------------------------------------------------------------------
# PCA-RGB
# ---------------------------------------------------------------------------

def fit_pca1(all_tokens: list[np.ndarray]) -> tuple[PCA, float, float]:
    """Ajusta PCA com 1 componente em todos os tokens concatenados."""
    X   = np.vstack(all_tokens)
    pca = PCA(n_components=1, random_state=0)
    pca.fit(X)
    proj = pca.transform(X).flatten()
    lo   = float(np.percentile(proj, 2))
    hi   = float(np.percentile(proj, 98))
    return pca, lo, hi


def make_panel_image(image: Image.Image, tokens: np.ndarray,
                     h_full: int, w_full: int,
                     pca: PCA, lo: float, hi: float,
                     alpha: float, colormap: str) -> np.ndarray:
    """
    tokens  : (h_full*w_full, dim)
    Projeta no PC1, normaliza, aplica colormap divergente sobre escala de cinza.
    """
    gray  = np.array(image.convert("L").convert("RGB")).astype(np.float32)
    h_img, w_img = gray.shape[:2]

    proj  = pca.transform(tokens).flatten()             # (h_full*w_full,)
    # Normaliza para [-1, 1] respeitando bounds globais
    proj_n = np.clip((proj - lo) / (hi - lo + 1e-8) * 2 - 1, -1, 1)
    grid   = proj_n.reshape(h_full, w_full)             # (h_full, w_full)

    # Mapeia [-1,1] → [0,1] para colormap
    grid_01 = (grid + 1) / 2

    # Upsample para resolução da imagem
    heat_img = Image.fromarray((grid_01 * 255).astype(np.uint8))
    heat_img = heat_img.resize((w_img, h_img), resample=Image.BILINEAR)
    heat_np  = np.array(heat_img) / 255.0

    heat_rgb = plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255
    return np.clip(gray * (1 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def make_figure(rows_data: list[dict], output_path: str, alpha: float, dpi: int):
    n_rows, n_cols = len(rows_data), 2

    col_w, row_h  = 2.8, 3.6
    left_m        = 1.4
    top_m         = 0.72
    cbar_w        = 0.16
    bot_m         = 0.15   # espaço para nota de rodapé
    right_pad     = 0.4

    fig_w = left_m + n_cols * col_w + right_pad + cbar_w
    fig_h = top_m + n_rows * row_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        left   = left_m / fig_w,
        right  = 1.0 - (right_pad) / fig_w,
        top    = 1.0 - top_m / fig_h,
        bottom = bot_m / fig_h,
        wspace = 0.025,
        hspace = 0.04,
        width_ratios=[1, 1, cbar_w / col_w],
    )
    cbar_ax = fig.add_subplot(gs[:, n_cols])

    # Headers
    for c, header in enumerate(COL_HEADERS):
        pos = gs[0, c].get_position(fig)
        fig.text(
            pos.x0 + pos.width / 2,
            1.0 - top_m * 0.28 / fig_h,
            header, ha="center", va="top",
            fontsize=8.5, fontweight="bold",
            transform=fig.transFigure,
        )

    # Linha separadora do header
    fig.add_artist(matplotlib.lines.Line2D(
        [left_m / fig_w, 1.0 - right_pad / fig_w],
        [1.0 - top_m * 0.60 / fig_h] * 2,
        transform=fig.transFigure, color="#cccccc", linewidth=0.6,
    ))

    panel_idx = 0
    for r, row in enumerate(rows_data):
        # Label da linha
        pos0 = gs[r, 0].get_position(fig)
        fig.text(
            left_m * 0.38 / fig_w,
            (pos0.y0 + pos0.y1) / 2,
            row["label"],
            ha="center", va="center",
            fontsize=7.5, style="italic", rotation=90,
            transform=fig.transFigure,
        )

        if r > 0:
            y_sep = (gs[r-1, 0].get_position(fig).y0 +
                     gs[r, 0].get_position(fig).y1) / 2
            fig.add_artist(matplotlib.lines.Line2D(
                [left_m / fig_w, 1.0 - right_pad / fig_w],
                [y_sep] * 2,
                transform=fig.transFigure, color="#eeeeee", linewidth=0.5,
            ))

        for c, panel_img in enumerate([row["img_l0"], row["img_lN"]]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panel_img)
            ax.axis("off")
            ax.text(0.015, 0.975, f"({PANEL_LABELS[panel_idx]})",
                    transform=ax.transAxes,
                    fontsize=7.5, color="white", va="top", ha="left",
                    fontweight="bold", fontfamily="serif",
                    bbox=dict(boxstyle="square,pad=0.1",
                              fc="black", ec="none", alpha=0.45))
            panel_idx += 1

    # Colorbar com legenda explicativa
    sm = plt.cm.ScalarMappable(
        cmap=rows_data[0].get("colormap", "RdBu_r"),
        norm=plt.Normalize(-1, 1)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels([r"$-$PC$_1$", "0", r"$+$PC$_1$"])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
    cb.set_label("PC$_1$ projection", fontsize=7.5, labelpad=5)
    cb.outline.set_linewidth(0.5)

    fig.text(
        0.5, 0.01,
        r"Colour = projection onto PC$_1$ of $\mathbf{h}_i^v \in \mathbb{R}^{1536}$"
        " (fit on all tokens, both documents and layers). "
        "Tokens pointing in the same direction share the same colour.",
        ha="center", va="bottom", fontsize=6, color="#555555",
        transform=fig.transFigure,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out}.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Salvo em {out}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLORMAP = "RdBu_r"   # divergente: vermelho ↔ azul, branco no centro


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default=str(ROOT / "results/paper_figures/attention/fig_pca_rgb"))
    p.add_argument("--alpha",   type=float, default=0.70)
    p.add_argument("--dpi",     type=int,   default=300)
    p.add_argument("--gpu",     type=int,   default=0)
    return p.parse_args()


def main():
    args  = parse_args()
    model, tokenizer = load_model()

    # 1. Extrai hidden states individuais por token (alta resolução)
    extracted = []
    for doc_dir, label in DOCUMENTS:
        img_path = sorted(Path(doc_dir).glob("*.tif"))[0]
        image    = Image.open(img_path).convert("RGB")
        print(f"\nProcessando: {label.split(chr(10))[0]!r}")
        t_l0, t_lN, h_full, w_full = get_token_hidden_states(model, tokenizer, image)
        print(f"  grid tokens: {h_full}×{w_full}  dim: {t_l0.shape[1]}")
        extracted.append({
            "label": label, "image": image,
            "t_l0": t_l0, "t_lN": t_lN,
            "h": h_full, "w": w_full,
        })
        torch.cuda.empty_cache()

    # 2. PCA (1 componente) ajustado em TODOS os tokens
    all_tokens = []
    for e in extracted:
        all_tokens.extend([e["t_l0"], e["t_lN"]])
    pca, lo, hi = fit_pca1(all_tokens)
    print(f"\nPC1 variância explicada: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  bounds globais: [{lo:.3f}, {hi:.3f}]")

    # 3. Monta painéis
    rows_data = []
    for e in extracted:
        def to_panel(tokens, h=e["h"], w=e["w"]):
            return make_panel_image(
                e["image"], tokens, h, w, pca, lo, hi, args.alpha, COLORMAP
            )
        rows_data.append({
            "label":    e["label"],
            "img_l0":   to_panel(e["t_l0"]),
            "img_lN":   to_panel(e["t_lN"]),
            "colormap": COLORMAP,
        })

    make_figure(rows_data, args.output, args.alpha, args.dpi)

    import shutil
    dest = ROOT.parent / "CaVL-Doc" / "docs" / "assets"
    if dest.exists():
        shutil.copy2(f"{args.output}.png", dest / "fig_pca_rgb.png")
        print(f"Copiado para {dest / 'fig_pca_rgb.png'}")


if __name__ == "__main__":
    main()
