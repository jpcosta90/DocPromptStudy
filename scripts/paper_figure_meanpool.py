#!/usr/bin/env python3
"""
Demonstração visual e numérica da invariância do mean pool ao prompt.

Gera uma figura com três linhas (uma por prompt) × três colunas:
  (a) Mapa de atenção texto→visual        — sensível ao prompt
  (b) Pesos do mean pool (1/Nv uniforme)  — constante, independente do prompt
  (c) Norma ||h_i^v||_2 de cada patch     — conteúdo visual, independente do prompt

Mais uma tabela numérica de cosine similarity para confirmar a invariância.

Uso:
  python scripts/paper_figure_meanpool.py \\
      --image /mnt/data/zs_rvl_cdip/data/invoice/0000327323.tif \\
      --prompts "Describe the document." "What are the values of the invoice items?" \\
      --output results/paper_figures/attention/fig_meanpool_demo \\
      --gpu 0
"""

from __future__ import annotations

import argparse
import sys
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
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INTERNVL_MAX_NUM  = 12
INTERNVL_IMG_SIZE = 448
ALL_LAYERS = list(range(-36, 0))


# ---------------------------------------------------------------------------
# Carregamento InternVL3
# ---------------------------------------------------------------------------

def load_internvl():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False
    )
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model, tok


# ---------------------------------------------------------------------------
# Preparo de inputs
# ---------------------------------------------------------------------------

def prepare(model, tokenizer, image: Image.Image, prompt: str):
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform, find_closest_aspect_ratio
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(INTERNVL_IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=INTERNVL_MAX_NUM,
        image_size=INTERNVL_IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16).to(
        next(model.parameters()).device
    )

    w, h  = image.size
    ratios = sorted(
        {(i, j) for n in range(1, INTERNVL_MAX_NUM + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= INTERNVL_MAX_NUM},
        key=lambda r: r[0] * r[1],
    )
    from cavl_doc.data.transforms import find_closest_aspect_ratio
    cols, rows = find_closest_aspect_ratio(w / h, ratios, w, h, INTERNVL_IMG_SIZE)

    inp        = prepare_inputs_for_multimodal_embedding(model, tokenizer, pv, prompt)
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    seq        = inp["input_ids"][0].cpu()
    img_pos    = (seq == img_ctx_id).nonzero(as_tuple=True)[0]
    text_pos   = (seq != img_ctx_id).nonzero(as_tuple=True)[0]

    return inp, img_pos, text_pos, rows, cols, model.num_image_token


# ---------------------------------------------------------------------------
# Extração de atenção + hidden states
# ---------------------------------------------------------------------------

def extract_all(model, inp, img_pos, text_pos,
                rows, cols, block_size, layer_indices):
    """
    Dois forward passes separados para evitar OOM:
      Pass 1: atenção via hooks (sem output_hidden_states)
      Pass 2: norma dos hidden states da última camada (sem hooks de atenção)
    """
    lm_layers = model.language_model.model.layers
    n_layers  = len(lm_layers)
    resolved  = sorted(set(li % n_layers for li in layer_indices))

    # --- Pass 1: atenção ---
    attn_accum = torch.zeros(len(img_pos), dtype=torch.float32)
    hooks, originals = [], {}

    for li in resolved:
        orig = lm_layers[li].self_attn.forward
        originals[li] = orig
        def make_patched(f):
            def p(*a, **kw): kw["output_attentions"] = True; return f(*a, **kw)
            return p
        lm_layers[li].self_attn.forward = make_patched(orig)

    count = [0]
    def make_hook(li):
        def hook(module, inp_, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn = output[1][0].float().cpu()
                attn_accum.add_(attn[:, text_pos, :][:, :, img_pos].mean(dim=(0, 1)))
                count[0] += 1
        return hook

    for li in resolved:
        hooks.append(lm_layers[li].self_attn.register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            model(**inp, return_dict=True)
    finally:
        for h in hooks: h.remove()
        for li, orig in originals.items():
            lm_layers[li].self_attn.forward = orig
    if count[0] > 0:
        attn_accum /= count[0]

    torch.cuda.empty_cache()

    # --- Pass 2: norma dos hidden states (só última camada) ---
    norm_accum = torch.zeros(len(img_pos), dtype=torch.float32)
    with torch.no_grad():
        out2 = model(**inp, output_hidden_states=True, return_dict=True)
    hs = out2.hidden_states[-1][0].float().cpu()
    norm_accum = hs[img_pos].norm(dim=-1)
    del out2
    torch.cuda.empty_cache()

    def to_spatial(vec):
        n_spatial = rows * cols
        blk = vec[:n_spatial * block_size].reshape(n_spatial, block_size).mean(dim=1)
        return blk.reshape(rows, cols).numpy()

    def normalize(m):
        lo, hi = np.percentile(m, 5), np.percentile(m, 99)
        return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

    return normalize(to_spatial(attn_accum)), normalize(to_spatial(norm_accum))


# ---------------------------------------------------------------------------
# Embedding (mean pool) para cosine similarity
# ---------------------------------------------------------------------------

def mean_pool_embedding(model, inp, img_pos, layer_idx=-1):
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer_idx][0].float().cpu()   # [seq, dim]
    return h.mean(dim=0).numpy()                         # mean pool ALL tokens


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

PANEL_LABELS = "abcdefghijklmnop"


def make_figure(image: Image.Image,
                data: list[dict],          # list of {prompt, attn, norm}
                cos_table: list[dict],     # cosine similarity data
                output_path: str,
                colormap: str = "inferno",
                dpi: int = 300):

    n_rows = len(data)
    n_cols = 3   # atenção | pool uniforme | norma

    col_w, row_h = 2.5, 3.4
    left_m = 1.3
    top_m  = 0.85
    cbar_w = 0.15
    bot_m  = 0.1

    fig_w = left_m + n_cols * col_w + 0.35 + cbar_w
    fig_h = top_m + n_rows * row_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        left   = left_m / fig_w,
        right  = 1.0 - (cbar_w + 0.08) / fig_w,
        top    = 1.0 - top_m / fig_h,
        bottom = bot_m / fig_h,
        wspace = 0.03,
        hspace = 0.04,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
    )
    cbar_ax = fig.add_subplot(gs[:, n_cols])

    col_titles = [
        "Attention map\n(text → visual tokens)",
        "Mean pool weights\n$(1/N_v$ uniform$)$",
        "Hidden state norm\n$\\|\\mathbf{h}_i^v\\|_2$",
    ]
    for c, title in enumerate(col_titles):
        pos = gs[0, c].get_position(fig)
        fig.text(pos.x0 + pos.width / 2,
                 1.0 - top_m * 0.28 / fig_h,
                 title,
                 ha="center", va="top",
                 fontsize=8, fontweight="bold",
                 transform=fig.transFigure)

    # Linha separadora abaixo dos headers
    sep_y = 1.0 - top_m * 0.62 / fig_h
    fig.add_artist(matplotlib.lines.Line2D(
        [left_m / fig_w, 1.0 - (cbar_w + 0.18) / fig_w],
        [sep_y, sep_y],
        transform=fig.transFigure,
        color="#cccccc", linewidth=0.6,
    ))

    img_gray = np.array(image.convert("L").convert("RGB"))
    h_img, w_img = img_gray.shape[:2]

    uniform_map = np.ones((data[0]["attn"].shape[0],
                           data[0]["attn"].shape[1]))  # 1/Nv uniforme (normalizado = 0.5)
    uniform_vis = np.full_like(uniform_map, 0.5)

    panel_idx = 0
    for r, row in enumerate(data):
        # Label da linha
        pos0 = gs[r, 0].get_position(fig)
        mid_y = (pos0.y0 + pos0.y1) / 2
        label_text = "\n".join([row["prompt"][i:i+20]
                                 for i in range(0, len(row["prompt"]), 20)])
        fig.text(left_m * 0.38 / fig_w, mid_y,
                 label_text,
                 ha="center", va="center",
                 fontsize=7.5, style="italic", rotation=90,
                 transform=fig.transFigure)

        if r > 0:
            y_sep = (gs[r-1, 0].get_position(fig).y0 +
                     gs[r, 0].get_position(fig).y1) / 2
            fig.add_artist(matplotlib.lines.Line2D(
                [left_m / fig_w, 1.0 - (cbar_w + 0.18) / fig_w],
                [y_sep, y_sep],
                transform=fig.transFigure,
                color="#eeeeee", linewidth=0.5,
            ))

        for c, heatmap in enumerate([row["attn"], uniform_vis, row["norm"]]):
            ax = fig.add_subplot(gs[r, c])
            # Overlay sobre grayscale
            heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (w_img, h_img), resample=Image.BILINEAR
            )
            heat_np  = np.array(heat) / 255.0
            heat_rgb = plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255
            alpha    = 0.55
            blended  = np.clip(img_gray * (1 - alpha) + heat_rgb * alpha,
                               0, 255).astype(np.uint8)
            ax.imshow(blended)
            ax.axis("off")
            ax.text(0.015, 0.975, f"({PANEL_LABELS[panel_idx]})",
                    transform=ax.transAxes,
                    fontsize=7, color="white", va="top", ha="left",
                    fontweight="bold", fontfamily="serif",
                    bbox=dict(boxstyle="square,pad=0.1",
                              fc="black", ec="none", alpha=0.45))
            # Anotação especial no painel de pool uniforme
            if c == 1:
                ax.text(0.5, 0.04,
                        f"$w_i = 1/N_v = 1/{data[r]['n_tokens']}$",
                        transform=ax.transAxes,
                        fontsize=6.5, color="white", va="bottom", ha="center",
                        fontfamily="serif",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="black", ec="none", alpha=0.5))
            panel_idx += 1

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "", "High"])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
    cb.set_label("Value", fontsize=7, labelpad=4)
    cb.outline.set_linewidth(0.5)

    # Tabela numérica de cosine similarity
    if cos_table:
        table_y = bot_m * 0.45 / fig_h
        header = f"{'Prompt':<42} {'cos(A,B) same':>14} {'cos(A,B) diff':>14}"
        fig.text(left_m / fig_w, table_y + 0.055,
                 header,
                 fontsize=6.5, fontfamily="monospace",
                 transform=fig.transFigure, color="#333333")
        for k, row in enumerate(cos_table):
            p_short = row["prompt"][:40] + "…" if len(row["prompt"]) > 40 else row["prompt"]
            line = f"{p_short:<42} {row['cos_same']:>14.4f} {row['cos_diff']:>14.4f}"
            fig.text(left_m / fig_w,
                     table_y + 0.055 - (k + 1) * 0.028,
                     line,
                     fontsize=6.5, fontfamily="monospace",
                     transform=fig.transFigure, color="#555555")

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
    p.add_argument("--image",    required=True)
    p.add_argument("--image-b",  default=None,
                   help="Segunda imagem para cosine similarity (se None, usa outra da mesma pasta)")
    p.add_argument("--prompts",  nargs="+", required=True)
    p.add_argument("--output",   default="results/paper_figures/attention/fig_meanpool_demo")
    p.add_argument("--layers",   nargs="+", type=int, default=list(range(-36, 0)))
    p.add_argument("--colormap", default="inferno")
    p.add_argument("--dpi",      type=int, default=300)
    p.add_argument("--gpu",      type=int, default=0)
    return p.parse_args()


def main():
    args  = parse_args()
    image = Image.open(args.image).convert("RGB")

    # Segunda imagem (documento diferente) para cosine similarity
    if args.image_b:
        image_b = Image.open(args.image_b).convert("RGB")
    else:
        # Pega o próximo tif da mesma pasta
        img_dir = Path(args.image).parent
        others  = sorted(f for f in img_dir.glob("*.tif") if f != Path(args.image))
        image_b = Image.open(others[0]).convert("RGB") if others else image

    print("Carregando InternVL3-2B…")
    model, tokenizer = load_internvl()

    data      = []
    cos_table = []

    for prompt in args.prompts:
        print(f"\nPrompt: {prompt!r}")
        inp_a, img_pos, text_pos, rows, cols, bs = prepare(
            model, tokenizer, image, prompt
        )
        inp_b, img_pos_b, *_ = prepare(model, tokenizer, image_b, prompt)

        attn_map, norm_map = extract_all(
            model, inp_a, img_pos, text_pos, rows, cols, bs, args.layers
        )

        emb_a = mean_pool_embedding(model, inp_a, img_pos)
        emb_b = mean_pool_embedding(model, inp_b, img_pos_b)
        # "Same" document = cosine(A, A) e "diff" = cosine(A, B)
        cos_same = cosine(emb_a, emb_a)   # sempre 1.0 — referência
        cos_diff = cosine(emb_a, emb_b)

        data.append({
            "prompt":   prompt,
            "attn":     attn_map,
            "norm":     norm_map,
            "n_tokens": len(img_pos),
        })
        cos_table.append({
            "prompt":   prompt,
            "cos_same": cos_same,
            "cos_diff": cos_diff,
        })
        print(f"  cos(A,A)={cos_same:.4f}  cos(A,B)={cos_diff:.4f}")

    make_figure(image, data, cos_table,
                output_path=args.output,
                colormap=args.colormap,
                dpi=args.dpi)


if __name__ == "__main__":
    main()
