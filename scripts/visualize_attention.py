#!/usr/bin/env python3
"""
Visualiza quais patches visuais do documento são mais ativados por diferentes prompts.

Gera dois arquivos por execução:
  {output}_full.png  — atenção de todos os tokens de texto (system + user)
  {output}_user.png  — atenção apenas dos tokens do prompt do usuário

Cada painel mostra: heatmap de atenção + prompt (azul) + resposta gerada (verde).
O arquivo _full.png inclui também o texto do system prompt no rodapé.

Restrições:
  - attn_implementation="eager" obrigatório (flash_attention_2 não retorna atenção)
  - bf16 puro, sem 4-bit

Uso:
  python scripts/visualize_attention.py \\
      --image /mnt/data/zs_rvl_cdip/data/invoice/0000327323.tif \\
      --prompts "What is the title?" "What is the code in the bottom left corner?" \\
      --output results/attention_maps/invoice \\
      --family qwen2vl --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--family", default="qwen2vl", choices=["qwen2vl", "internvl"])
    p.add_argument("--image", required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts", nargs="+")
    src.add_argument("--prompt-ids", nargs="+")

    p.add_argument("--output", default="results/attention_maps/attn",
                   help="Prefixo de saída (sem extensão). Gera _full.png e _user.png")
    p.add_argument("--layers", nargs="+", type=int, default=[-4, -3, -2, -1])
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--colormap", default="inferno")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_qwen2vl(hf_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(hf_path)
    return model, processor


# ---------------------------------------------------------------------------
# Preparação dos inputs e separação de tokens
# ---------------------------------------------------------------------------

def prepare_qwen2vl(model, processor, image: Image.Image, prompt: str):
    """
    Processa imagem+prompt e retorna:
      inputs        — dict pronto para o modelo
      img_pos       — posições dos tokens visuais (CPU)
      all_text_pos  — posições de todos os tokens de texto (system + user)
      user_pos      — posições apenas dos tokens do prompt do usuário
      H_eff, W_eff  — grid espacial efetivo dos patches
      sys_text      — texto do system prompt decodificado
    """
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  prompt},
    ]}]
    text_tpl = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    img_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_tpl], images=img_inputs, return_tensors="pt"
    ).to(next(model.parameters()).device)

    T, H, W = [int(x) for x in inputs["image_grid_thw"][0]]
    merge   = getattr(model.config.vision_config, "spatial_merge_size", 2)
    H_eff, W_eff = H // merge, W // merge

    img_token_id = model.config.image_token_id
    seq = inputs.input_ids[0].cpu()
    img_pos = (seq == img_token_id).nonzero(as_tuple=True)[0]

    # Localiza seções via <|im_start|> / <|im_end|>
    tok = processor.tokenizer
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tok.convert_tokens_to_ids("<|im_end|>")
    starts = (seq == im_start_id).nonzero(as_tuple=True)[0]
    ends   = (seq == im_end_id).nonzero(as_tuple=True)[0]

    sys_text = ""
    sys_pos  = torch.tensor([], dtype=torch.long)
    user_pos = (seq != img_token_id).nonzero(as_tuple=True)[0]  # fallback

    if len(starts) >= 2 and len(ends) >= 2:
        # Seção system: entre starts[0]+1 e ends[0]
        s0, e0 = starts[0].item() + 1, ends[0].item()
        sys_pos  = torch.arange(s0, e0)
        sys_text = tok.decode(seq[sys_pos], skip_special_tokens=True).strip()

        # Seção user: entre starts[1]+1 e ends[1], excluindo tokens visuais
        s1, e1   = starts[1].item() + 1, ends[1].item()
        u_section = torch.arange(s1, e1)
        user_pos  = u_section[(seq[u_section] != img_token_id)]

    all_text_pos = torch.cat([sys_pos, user_pos]) if len(sys_pos) > 0 else user_pos

    return inputs, img_pos, all_text_pos, user_pos, H_eff, W_eff, sys_text


# ---------------------------------------------------------------------------
# Cálculo do mapa de atenção a partir do output já computado
# ---------------------------------------------------------------------------

def attn_map_from_output(
    out,
    img_pos: torch.Tensor,
    text_pos: torch.Tensor,
    H_eff: int,
    W_eff: int,
    layer_indices: list[int],
) -> np.ndarray:
    n_layers = len(out.attentions)
    resolved = [li % n_layers for li in layer_indices]

    accum = torch.zeros(len(img_pos), dtype=torch.float32)
    for li in resolved:
        attn    = out.attentions[li][0].float().cpu()   # [heads, seq, seq]
        attn_ti = attn[:, text_pos, :][:, :, img_pos]   # [heads, n_text, n_img]
        accum  += attn_ti.mean(dim=(0, 1))
    accum /= len(resolved)

    spatial = accum.reshape(H_eff, W_eff).numpy()
    lo, hi  = spatial.min(), spatial.max()
    if hi > lo:
        spatial = (spatial - lo) / (hi - lo)
    return spatial


# ---------------------------------------------------------------------------
# Geração de resposta
# ---------------------------------------------------------------------------

def generate_response(model, processor, image: Image.Image, prompt: str,
                      max_new_tokens: int = 50) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  prompt},
    ]}]
    text_tpl = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_tpl], images=img_inputs, return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def overlay_heatmap(image: Image.Image, attn_map: np.ndarray,
                    alpha: float, colormap: str) -> np.ndarray:
    img_np = np.array(image.convert("RGB"))
    h, w   = img_np.shape[:2]
    heat   = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heat_np  = np.array(heat) / 255.0
    heat_rgb = (plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255).astype(np.uint8)
    return (img_np * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)


def _wrap(text: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(text, width))


def make_figure(image: Image.Image, panels: list[dict], sys_text: str,
                show_sys_text: bool, colormap: str, alpha: float,
                output_path: str):
    """
    panels: lista de dicts com chaves 'prompt', 'response', 'attn_map'
    show_sys_text: se True, adiciona rodapé com o texto do system prompt
    """
    n   = len(panels)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.5 * (n + 1), 8), dpi=150)
    fig.patch.set_facecolor("#1a1a1a")

    # Painel original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10, color="white", pad=6)
    axes[0].axis("off")

    for i, panel in enumerate(panels):
        blended = overlay_heatmap(image, panel["attn_map"], alpha, colormap)
        ax = axes[i + 1]
        ax.set_facecolor("#1a1a1a")
        ax.imshow(blended)
        ax.axis("off")

        # Prompt (azul, cima)
        ax.set_title(_wrap(panel["prompt"], 32),
                     fontsize=8, color="#90caf9", fontweight="bold",
                     pad=6, loc="center")

        # Resposta (verde, baixo)
        if panel.get("response"):
            ax.text(0.5, -0.02, _wrap(f"→ {panel['response']}", 36),
                    transform=ax.transAxes,
                    fontsize=8, color="#a5d6a7", va="top", ha="center",
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#2a2a2a", ec="none"))

    # Rodapé com system prompt (só no _full)
    if show_sys_text and sys_text:
        fig.text(0.5, 0.01,
                 f"System prompt: \"{sys_text}\"",
                 ha="center", va="bottom", fontsize=7,
                 color="#9e9e9e", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04 if (show_sys_text and sys_text) else 0, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Salvo em {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve prompts
    if args.prompt_ids:
        cfg = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text())["prompts"]
        missing = [p for p in args.prompt_ids if p not in cfg]
        if missing:
            print(f"Prompt IDs não encontrados: {missing}", file=sys.stderr)
            sys.exit(1)
        prompts = [(pid, cfg[pid]) for pid in args.prompt_ids]
    else:
        prompts = [(p, p) for p in args.prompts]

    print(f"Carregando {args.model}…")
    model, processor = load_qwen2vl(args.model)
    image = Image.open(args.image).convert("RGB")

    n = len(prompts)
    panels_full = []
    panels_user = []
    sys_text_global = ""

    for i, (label, prompt) in enumerate(prompts):
        print(f"\n[{i+1}/{n}] {prompt!r}")

        # Prepara inputs e identifica posições de tokens
        inputs, img_pos, all_text_pos, user_pos, H_eff, W_eff, sys_text = \
            prepare_qwen2vl(model, processor, image, prompt)
        sys_text_global = sys_text  # igual para todos os prompts

        print(f"  tokens — system: {len(sys_text_global.split())}, "
              f"user: {len(user_pos)}, img: {len(img_pos)}")

        # Um único forward pass com atenção
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, return_dict=True)

        # Dois mapas a partir do mesmo output
        map_full = attn_map_from_output(out, img_pos, all_text_pos, H_eff, W_eff, args.layers)
        map_user = attn_map_from_output(out, img_pos, user_pos,     H_eff, W_eff, args.layers)

        # Resposta gerada
        response = generate_response(model, processor, image, prompt)
        print(f"  → {response}")

        panels_full.append({"prompt": prompt, "response": response, "attn_map": map_full})
        panels_user.append({"prompt": prompt, "response": response, "attn_map": map_user})

    stem = args.output.removesuffix(".png")
    make_figure(image, panels_full, sys_text_global,
                show_sys_text=True,
                colormap=args.colormap, alpha=args.alpha,
                output_path=f"{stem}_full.png")
    make_figure(image, panels_user, sys_text_global,
                show_sys_text=False,
                colormap=args.colormap, alpha=args.alpha,
                output_path=f"{stem}_user.png")


if __name__ == "__main__":
    main()
