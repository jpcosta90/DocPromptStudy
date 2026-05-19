#!/usr/bin/env python3
"""
Visualiza quais patches visuais do documento são mais ativados por diferentes prompts.

Extrai pesos de atenção do Qwen2.5-VL: média sobre as últimas N camadas e todas as
cabeças de atenção, dos tokens de texto → tokens visuais. O resultado é um heatmap
sobreposto à imagem original, um por prompt.

Restrições:
  - Usa attn_implementation="eager" (flash_attention_2 não expõe pesos de atenção)
  - Roda em bf16 puro, sem quantização 4-bit (pesos de atenção ficam corrompidos em 4-bit)
  - Modelos 2B em bf16 requerem ~4 GB VRAM

Uso:
  python scripts/visualize_attention.py \\
      --image path/para/documento.png \\
      --prompts "Qual o título?" "Quantas tabelas tem?" "Qual o tamanho da fonte?" \\
      --output atenção_prompts.png

  # Usar prompts do arquivo de configuração:
  python scripts/visualize_attention.py \\
      --image path/para/documento.png \\
      --prompt-ids minimal classify ocr_task \\
      --output atenção_prompts.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-2B-Instruct",
                   help="HF path do modelo (padrão: Qwen/Qwen2.5-VL-2B-Instruct)")
    p.add_argument("--image", required=True, help="Caminho para a imagem do documento")

    prompt_src = p.add_mutually_exclusive_group(required=True)
    prompt_src.add_argument("--prompts", nargs="+",
                            help="Prompts literais a comparar")
    prompt_src.add_argument("--prompt-ids", nargs="+",
                            help="IDs de prompts definidos em configs/prompts.yaml")

    p.add_argument("--output", default="attention_map.png",
                   help="Arquivo de saída (PNG)")
    p.add_argument("--layers", nargs="+", type=int, default=[-4, -3, -2, -1],
                   help="Índices das camadas (negativos = a partir do final; padrão: últimas 4)")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Opacidade do heatmap sobre a imagem (padrão: 0.55)")
    p.add_argument("--colormap", default="inferno",
                   help="Colormap matplotlib para o heatmap (padrão: inferno)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_model(hf_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Carregando {hf_path} em bf16 (eager attention)…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # necessário para output_attentions=True
    ).eval()
    processor = AutoProcessor.from_pretrained(hf_path)
    return model, processor


def compute_attention_map(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    layer_indices: list[int],
) -> np.ndarray:
    """
    Retorna mapa de atenção 2D (H_patches × W_patches) normalizado para [0, 1].

    Média dos pesos: tokens de texto → tokens visuais, sobre as camadas
    e cabeças especificadas.
    """
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  prompt},
    ]}]
    text_template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    img_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_template], images=img_inputs, return_tensors="pt"
    ).to(next(model.parameters()).device)

    # Grid espacial: image_grid_thw = (T, H, W) em número de patches
    T, H, W = [int(x) for x in inputs["image_grid_thw"][0]]
    # Fator de merge espacial (padrão=2 no Qwen2.5-VL)
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
    H_eff, W_eff = H // merge, W // merge

    img_token_id = model.config.image_token_id
    seq = inputs.input_ids[0]
    img_pos  = (seq == img_token_id).nonzero(as_tuple=True)[0]   # tokens visuais
    text_pos = (seq != img_token_id).nonzero(as_tuple=True)[0]   # tokens de texto

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, return_dict=True)

    # out.attentions: tupla de [1, heads, seq, seq] por camada
    n_layers = len(out.attentions)
    resolved = [li % n_layers for li in layer_indices]

    attn_accum = torch.zeros(len(img_pos), dtype=torch.float32)
    for li in resolved:
        attn = out.attentions[li][0].float().cpu()  # [heads, seq, seq]
        # Atenção de tokens de texto → tokens visuais: [heads, n_text, n_img]
        attn_ti = attn[:, text_pos, :][:, :, img_pos]
        attn_accum += attn_ti.mean(dim=(0, 1))       # média sobre heads e posições de texto

    attn_accum /= len(resolved)

    spatial = attn_accum.reshape(H_eff, W_eff).numpy()
    lo, hi = spatial.min(), spatial.max()
    if hi > lo:
        spatial = (spatial - lo) / (hi - lo)
    return spatial


def overlay_heatmap(
    image: Image.Image,
    attn_map: np.ndarray,
    alpha: float,
    colormap: str,
) -> np.ndarray:
    """Upsampla o mapa de atenção para o tamanho da imagem e combina com colormap."""
    import matplotlib.cm as cm

    img_np = np.array(image.convert("RGB"))
    h, w   = img_np.shape[:2]

    heat_img = Image.fromarray((attn_map * 255).astype(np.uint8))
    heat_img = heat_img.resize((w, h), resample=Image.BILINEAR)
    heat_np  = np.array(heat_img) / 255.0

    heat_rgb = (cm.get_cmap(colormap)(heat_np)[:, :, :3] * 255).astype(np.uint8)
    return (img_np * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)


def main():
    args = parse_args()

    # Resolve prompts
    if args.prompt_ids:
        prompts_cfg = yaml.safe_load(
            (ROOT / "configs" / "prompts.yaml").read_text()
        )["prompts"]
        missing = [pid for pid in args.prompt_ids if pid not in prompts_cfg]
        if missing:
            print(f"Prompt IDs não encontrados: {missing}", file=sys.stderr)
            sys.exit(1)
        prompts = [(pid, prompts_cfg[pid]) for pid in args.prompt_ids]
    else:
        prompts = [(p, p) for p in args.prompts]

    model, processor = load_model(args.model)
    image = Image.open(args.image).convert("RGB")

    n = len(prompts)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 6), dpi=150)

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for i, (label, prompt) in enumerate(prompts):
        display = repr(prompt[:50] + "…") if len(prompt) > 50 else repr(prompt)
        print(f"[{i+1}/{n}] {display}")

        attn_map = compute_attention_map(model, processor, image, prompt, args.layers)
        blended  = overlay_heatmap(image, attn_map, args.alpha, args.colormap)

        axes[i + 1].imshow(blended)
        title = label if len(label) <= 35 else label[:33] + "…"
        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"\nSalvo em {args.output}")


if __name__ == "__main__":
    main()
