#!/usr/bin/env python3
"""
Gera figura de comparação de atenção para paper.

Layout: 2 linhas × 4 colunas
  Linhas  : prompts
  Colunas : InternVL3 Layer 0 | InternVL3 Layer -1 | Qwen Layer 0 | Qwen Layer -1

Uso:
  python scripts/paper_figure_attention.py \\
      --image /mnt/data/zs_rvl_cdip/data/invoice/0000327323.tif \\
      --prompts "Describe the document." "What are the values of the invoice items?" \\
      --output results/paper_figures/attention/fig_comparison.pdf \\
      --gpu 0
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    # Fonte serif próxima ao LaTeX — usada na maioria dos papers IEEE/ACM
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif"],
    "font.size":    8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    # Remove bordas de eixos desnecessárias
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left":   False,
    # PDF com fontes embutidas
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
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
ALL_LAYERS = list(range(-36, 0))   # todas as 36 camadas


# ---------------------------------------------------------------------------
# Carregamento
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


def load_qwen2vl():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, proc


# ---------------------------------------------------------------------------
# Preparação de inputs
# ---------------------------------------------------------------------------

def prepare_internvl(model, tokenizer, image: Image.Image, prompt: str):
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform, find_closest_aspect_ratio
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(INTERNVL_IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=INTERNVL_MAX_NUM, image_size=INTERNVL_IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16).to(
        next(model.parameters()).device
    )

    w, h = image.size
    ratios = sorted(
        {(i, j) for n in range(1, INTERNVL_MAX_NUM + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= INTERNVL_MAX_NUM},
        key=lambda r: r[0] * r[1],
    )
    cols, rows = find_closest_aspect_ratio(w / h, ratios, w, h, INTERNVL_IMG_SIZE)

    inp = prepare_inputs_for_multimodal_embedding(model, tokenizer, pv, prompt)
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    seq        = inp["input_ids"][0].cpu()
    img_pos    = (seq == img_ctx_id).nonzero(as_tuple=True)[0]
    text_pos   = (seq != img_ctx_id).nonzero(as_tuple=True)[0]

    return inp, img_pos, text_pos, rows, cols, model.num_image_token


def prepare_qwen2vl(model, processor, image: Image.Image, prompt: str):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    img_in, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, return_tensors="pt").to(
        next(model.parameters()).device
    )

    T, H, W = [int(x) for x in inputs["image_grid_thw"][0]]
    merge    = getattr(model.config.vision_config, "spatial_merge_size", 2)
    H_eff, W_eff = H // merge, W // merge

    img_token_id = model.config.image_token_id
    seq      = inputs.input_ids[0].cpu()
    img_pos  = (seq == img_token_id).nonzero(as_tuple=True)[0]

    tok = processor.tokenizer
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tok.convert_tokens_to_ids("<|im_end|>")
    starts = (seq == im_start_id).nonzero(as_tuple=True)[0]
    ends   = (seq == im_end_id).nonzero(as_tuple=True)[0]

    sys_pos  = torch.tensor([], dtype=torch.long)
    user_pos = (seq != img_token_id).nonzero(as_tuple=True)[0]
    if len(starts) >= 2 and len(ends) >= 2:
        s0, e0    = starts[0].item() + 1, ends[0].item()
        sys_pos   = torch.arange(s0, e0)
        s1, e1    = starts[1].item() + 1, ends[1].item()
        u_section = torch.arange(s1, e1)
        user_pos  = u_section[(seq[u_section] != img_token_id)]
    all_text = torch.cat([sys_pos, user_pos]) if len(sys_pos) > 0 else user_pos

    return inputs, img_pos, all_text, H_eff, W_eff


# ---------------------------------------------------------------------------
# Mapas de atenção
# ---------------------------------------------------------------------------

def _normalize(spatial: np.ndarray) -> np.ndarray:
    lo = np.percentile(spatial, 50)
    hi = np.percentile(spatial, 99)
    return np.clip((spatial - lo) / (hi - lo + 1e-8), 0, 1)


def attn_map_internvl_hooks(model, inp, img_pos, text_pos,
                             rows, cols, block_size, layer_indices) -> np.ndarray:
    lm_layers = model.language_model.model.layers
    n_layers  = len(lm_layers)
    resolved  = sorted(set(li % n_layers for li in layer_indices))

    accum = torch.zeros(len(img_pos), dtype=torch.float32)
    count = [0]
    hooks, originals = [], {}

    for li in resolved:
        orig = lm_layers[li].self_attn.forward
        originals[li] = orig
        def make_patched(f):
            def p(*a, **kw): kw["output_attentions"] = True; return f(*a, **kw)
            return p
        lm_layers[li].self_attn.forward = make_patched(orig)

    def make_hook(li):
        def hook(module, inp_, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn = output[1][0].float().cpu()
                accum.add_(attn[:, text_pos, :][:, :, img_pos].mean(dim=(0, 1)))
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
        accum /= count[0]
    n_spatial = rows * cols
    spatial = accum[:n_spatial * block_size].reshape(n_spatial, block_size).mean(dim=1)
    return _normalize(spatial.reshape(rows, cols).numpy())


def attn_map_qwen(model, inputs, img_pos, text_pos, H_eff, W_eff, layer_indices) -> np.ndarray:
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, return_dict=True)
    n_layers = len(out.attentions)
    resolved = [li % n_layers for li in layer_indices]
    accum = torch.zeros(len(img_pos), dtype=torch.float32)
    for li in resolved:
        attn = out.attentions[li][0].float().cpu()
        accum += attn[:, text_pos, :][:, :, img_pos].mean(dim=(0, 1))
    accum /= len(resolved)
    return _normalize(accum.reshape(H_eff, W_eff).numpy())


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay(image: Image.Image, attn_map: np.ndarray,
            alpha: float, colormap: str) -> np.ndarray:
    """
    Base em escala de cinza + colormap com alpha fixo.
    Grayscale dá contraste limpo; alpha fixo mantém o documento legível.
    """
    # Grayscale → RGB para manter 3 canais
    gray   = np.array(image.convert("L").convert("RGB")).astype(np.float32)
    h, w   = gray.shape[:2]
    heat   = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heat_np  = np.array(heat) / 255.0
    heat_rgb = plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255
    return np.clip(gray * (1 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figura de paper
# ---------------------------------------------------------------------------

PANEL_LABELS = "abcdefghijklmnop"

COL_HEADERS = [
    "InternVL3-2B\nLayer 0 (input)",
    "InternVL3-2B\nLayer −1 (output)",
    "Qwen2.5-VL-3B\nLayer 0 (input)",
    "Qwen2.5-VL-3B\nLayer −1 (output)",
]


def make_paper_figure(panels: list[list[np.ndarray]],
                      row_labels: list[str],
                      col_headers: list[str],
                      image: Image.Image,
                      output_path: str,
                      alpha: float,
                      colormap: str,
                      dpi: int = 300,
                      divider_after_col: int | None = None):
    """
    panels[row][col] = attention map (H×W float array)
    rows = prompts, cols = model×layer
    """
    n_rows = len(panels)
    n_cols = len(panels[0])

    # Dimensões calibradas para coluna dupla (7" max) em papers IEEE/ACM
    col_w       = 2.6      # polegadas por painel
    row_h       = 3.5      # polegadas por linha
    left_margin = 1.3      # espaço para labels de linha
    cbar_w      = 0.18     # colorbar estreita
    cbar_pad    = 0.12     # gap antes da colorbar
    top_margin  = 0.65     # espaço para headers
    bottom_pad  = 0.08

    fig_w = left_margin + n_cols * col_w + cbar_pad + cbar_w + 0.1
    fig_h = top_margin + n_rows * row_h + bottom_pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        left   = left_margin / fig_w,
        right  = 1.0 - (cbar_w + 0.08) / fig_w,
        top    = 1.0 - top_margin / fig_h,
        bottom = bottom_pad / fig_h,
        wspace = 0.025,
        hspace = 0.04,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
    )

    cbar_ax = fig.add_subplot(gs[:, n_cols])

    # Headers das colunas — negrito, fonte 8.5pt
    for c, header in enumerate(col_headers):
        pos = gs[0, c].get_position(fig)
        fig.text(
            pos.x0 + pos.width / 2,
            1.0 - top_margin * 0.28 / fig_h,
            header,
            ha="center", va="top",
            fontsize=8.5, fontweight="bold",
            transform=fig.transFigure,
        )

    # Linha horizontal separando header dos painéis
    header_y = 1.0 - top_margin * 0.60 / fig_h
    fig.add_artist(matplotlib.lines.Line2D(
        [left_margin / fig_w, 1.0 - (cbar_w + 0.18) / fig_w],
        [header_y, header_y],
        transform=fig.transFigure,
        color="#cccccc", linewidth=0.6,
    ))

    # Divisor vertical entre grupos de modelos (opcional)
    if divider_after_col is not None:
        pos_l = gs[0, divider_after_col].get_position(fig)
        pos_r = gs[0, divider_after_col + 1].get_position(fig)
        div_x = (pos_l.x1 + pos_r.x0) / 2
        fig.add_artist(matplotlib.lines.Line2D(
            [div_x, div_x],
            [bottom_pad / fig_h, 1.0 - top_margin * 0.60 / fig_h],
            transform=fig.transFigure,
            color="#cccccc", linewidth=0.6, linestyle="--",
        ))

    panel_idx = 0

    for r, row_maps in enumerate(panels):
        # Label da linha — itálico, rotacionado, centralizado
        pos0 = gs[r, 0].get_position(fig)
        pos1 = gs[r, n_cols - 1].get_position(fig)
        mid_y = (pos0.y0 + pos0.y1) / 2
        fig.text(
            left_margin * 0.38 / fig_w, mid_y,
            "\n".join(textwrap.wrap(row_labels[r], width=18)),
            ha="center", va="center",
            fontsize=8, style="italic",
            rotation=90,
            transform=fig.transFigure,
        )

        # Linha horizontal entre linhas (exceto após a última)
        if r > 0:
            sep_y = (gs[r - 1, 0].get_position(fig).y0 +
                     gs[r, 0].get_position(fig).y1) / 2
            fig.add_artist(matplotlib.lines.Line2D(
                [left_margin / fig_w, 1.0 - (cbar_w + 0.18) / fig_w],
                [sep_y, sep_y],
                transform=fig.transFigure,
                color="#eeeeee", linewidth=0.5,
            ))

        for c, attn_map in enumerate(row_maps):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(overlay(image, attn_map, alpha, colormap))
            ax.axis("off")

            # Label do painel — estilo paper: "(a)" branco, fundo preto semi-trans
            ax.text(0.015, 0.975, f"({PANEL_LABELS[panel_idx]})",
                    transform=ax.transAxes,
                    fontsize=7.5, color="white", va="top", ha="left",
                    fontweight="bold", fontfamily="serif",
                    bbox=dict(boxstyle="square,pad=0.1",
                              fc="black", ec="none", alpha=0.45))
            panel_idx += 1

    # Colorbar — fina, ticks mínimos, label vertical
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "", "High"])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
    cb.set_label("Attention", fontsize=7.5, labelpad=5)
    cb.outline.set_linewidth(0.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Salvo em {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--output", default="results/paper_figures/attention/fig_comparison.pdf")
    p.add_argument("--layers", nargs="+", type=int, default=ALL_LAYERS)
    p.add_argument("--alpha",    type=float, default=0.65)
    p.add_argument("--colormap", default="inferno")
    p.add_argument("--dpi",      type=int,   default=300)
    p.add_argument("--gpu",      type=int,   default=0)
    return p.parse_args()


def main():
    import matplotlib.lines   # noqa: ensure available for divider

    args  = parse_args()
    image = Image.open(args.image).convert("RGB")

    # ---- InternVL3 ----
    print("Carregando InternVL3-2B…")
    mdl_ivl, tok_ivl = load_internvl()

    ivl_maps: dict[tuple, np.ndarray] = {}
    for prompt in args.prompts:
        print(f"  InternVL3 · {prompt!r}")
        inp, img_pos, text_pos, rows, cols, block = prepare_internvl(
            mdl_ivl, tok_ivl, image, prompt
        )
        for layer_indices, key in [([0], "l0"), (args.layers, "ll")]:
            ivl_maps[(prompt, key)] = attn_map_internvl_hooks(
                mdl_ivl, inp, img_pos, text_pos, rows, cols, block, layer_indices
            )

    del mdl_ivl, tok_ivl
    torch.cuda.empty_cache()

    # ---- Qwen2.5-VL ----
    print("Carregando Qwen2.5-VL-3B…")
    mdl_qwn, proc_qwn = load_qwen2vl()

    qwn_maps: dict[tuple, np.ndarray] = {}
    for prompt in args.prompts:
        print(f"  Qwen2.5-VL · {prompt!r}")
        inp, img_pos, text_pos, H_eff, W_eff = prepare_qwen2vl(
            mdl_qwn, proc_qwn, image, prompt
        )
        for layer_indices, key in [([0], "l0"), (args.layers, "ll")]:
            qwn_maps[(prompt, key)] = attn_map_qwen(
                mdl_qwn, inp, img_pos, text_pos, H_eff, W_eff, layer_indices
            )

    del mdl_qwn, proc_qwn
    torch.cuda.empty_cache()

    stem = Path(args.output).with_suffix("")

    # ---- Figura 1: só InternVL3 (paper atual) ----
    ivl_out = f"{stem}_internvl3.png"
    panels_ivl = [[ivl_maps[(p, "l0")], ivl_maps[(p, "ll")]] for p in args.prompts]
    make_paper_figure(
        panels           = panels_ivl,
        row_labels       = args.prompts,
        col_headers      = ["Layer 0 (input)", "Layer −1 (output)"],
        image            = image,
        output_path      = ivl_out,
        alpha            = args.alpha,
        colormap         = args.colormap,
        dpi              = args.dpi,
        divider_after_col= None,
    )

    # Copia também para CaVL-Doc/docs/assets/
    cavl_assets = ROOT.parent / "CaVL-Doc" / "docs" / "assets"
    if cavl_assets.exists():
        import shutil
        dest = cavl_assets / "fig_attention_internvl3.png"
        shutil.copy2(ivl_out, dest)
        print(f"Copiado para {dest}")
    else:
        print(f"Aviso: {cavl_assets} não encontrado — pulando cópia.")

    # ---- Figura 2: ambos os modelos (paper futuro) ----
    panels_both = []
    for prompt in args.prompts:
        panels_both.append([
            ivl_maps[(prompt, "l0")],
            ivl_maps[(prompt, "ll")],
            qwn_maps[(prompt, "l0")],
            qwn_maps[(prompt, "ll")],
        ])
    make_paper_figure(
        panels           = panels_both,
        row_labels       = args.prompts,
        col_headers      = COL_HEADERS,
        image            = image,
        output_path      = f"{stem}_comparison.png",
        alpha            = args.alpha,
        colormap         = args.colormap,
        dpi              = args.dpi,
        divider_after_col= 1,
    )


if __name__ == "__main__":
    main()
