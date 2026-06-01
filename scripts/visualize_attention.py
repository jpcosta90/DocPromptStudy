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

def load_internvl(hf_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        hf_path, trust_remote_code=True, use_fast=False
    )
    # Obrigatório para o forward pass do InternVL (substitui os IMG_CONTEXT no embedding)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # Garante eager no LM interno (Qwen2)
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config._attn_implementation = "eager"
    return model, tokenizer


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
# InternVL3 — preparação e geração
# ---------------------------------------------------------------------------

_INTERNVL_MAX_NUM  = 12
_INTERNVL_IMG_SIZE = 448


def _internvl_patch_grid(image: Image.Image):
    """Retorna (rows, cols) do grid de patches para a imagem."""
    from cavl_doc.data.transforms import find_closest_aspect_ratio
    w, h = image.size
    ratios = sorted(
        {(i, j) for n in range(1, _INTERNVL_MAX_NUM + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= _INTERNVL_MAX_NUM},
        key=lambda r: r[0] * r[1],
    )
    # find_closest_aspect_ratio retorna (cols, rows) — [0]=largura, [1]=altura
    cols, rows = find_closest_aspect_ratio(w / h, ratios, w, h, _INTERNVL_IMG_SIZE)
    return rows, cols


def prepare_internvl(model, tokenizer, image: Image.Image, prompt: str):
    """
    Retorna:
      inp           — dict com input_ids, pixel_values, image_flags
      img_pos       — posições dos tokens <IMG_CONTEXT> (CPU)
      all_text_pos  — posições de todos os tokens de texto (= user_pos, sem seção system)
      user_pos      — idem
      rows, cols    — grid espacial de patches (excluindo thumbnail)
      n_img_token   — tokens por patch (256)
      sys_text      — "" (InternVL não usa system section neste formato)
    """
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(_INTERNVL_IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=_INTERNVL_MAX_NUM,
        image_size=_INTERNVL_IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16)

    rows, cols  = _internvl_patch_grid(image)
    n_img_token = model.num_image_token   # 256

    inp = prepare_inputs_for_multimodal_embedding(model, tokenizer, pv, prompt)

    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    seq        = inp["input_ids"][0].cpu()
    img_pos    = (seq == img_ctx_id).nonzero(as_tuple=True)[0]
    text_pos   = (seq != img_ctx_id).nonzero(as_tuple=True)[0]

    return inp, img_pos, text_pos, text_pos, rows, cols, n_img_token, ""


def generate_response_internvl(model, tokenizer, image: Image.Image,
                                prompt: str, max_new_tokens: int = 50) -> str:
    """Geração com formato de chat correto (internvl2_5 + Qwen2 backbone)."""
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform

    transform = build_transform(_INTERNVL_IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=_INTERNVL_MAX_NUM,
        image_size=_INTERNVL_IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16).to(
        next(model.parameters()).device
    )
    n_patches   = pv.shape[0]
    n_img_token = model.num_image_token
    img_tokens  = "<img>" + "<IMG_CONTEXT>" * n_img_token * n_patches + "</img>"

    # Formato de chat internvl2_5 com backbone Qwen2
    conv = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{img_tokens}\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    enc = tokenizer(conv, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            pixel_values=pv,
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    # O InternVL usa inputs_embeds internamente: o output contém só os tokens gerados
    generated = out[0]
    if generated.shape[0] > enc["input_ids"].shape[1]:
        generated = generated[enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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

# aliases para interface uniforme com InternVL3 (rows=H_eff, cols=W_eff, block_size=1)
_prepare_qwen2vl_orig = prepare_qwen2vl
def prepare_qwen2vl(model, processor, image, prompt):
    inp, img_pos, all_tp, user_pos, H, W, sys = _prepare_qwen2vl_orig(
        model, processor, image, prompt)
    return inp, img_pos, all_tp, user_pos, H, W, sys


# ---------------------------------------------------------------------------
# Cálculo do mapa de atenção
# ---------------------------------------------------------------------------

def _normalize_spatial(spatial: np.ndarray) -> np.ndarray:
    """Percentile clip para realçar os patches com maior atenção."""
    lo = np.percentile(spatial, 50)
    hi = np.percentile(spatial, 99)
    if hi > lo:
        return np.clip((spatial - lo) / (hi - lo), 0, 1)
    return np.zeros_like(spatial)


def _accum_to_spatial(accum: torch.Tensor, rows: int, cols: int,
                       block_size: int) -> np.ndarray:
    if block_size > 1:
        n_spatial = rows * cols
        accum = accum[: n_spatial * block_size].reshape(n_spatial, block_size).mean(dim=1)
    return _normalize_spatial(accum.reshape(rows, cols).numpy())


def attn_map_from_output(
    out,
    img_pos: torch.Tensor,
    text_pos: torch.Tensor,
    rows: int,
    cols: int,
    layer_indices: list[int],
    block_size: int = 1,
) -> np.ndarray:
    """Qwen2.5-VL: usa out.attentions (sequências curtas, sem OOM)."""
    if out.attentions is None:
        raise RuntimeError("out.attentions é None — use attn_implementation='eager'.")

    n_layers = len(out.attentions)
    resolved = [li % n_layers for li in layer_indices]
    accum = torch.zeros(len(img_pos), dtype=torch.float32)
    for li in resolved:
        attn    = out.attentions[li][0].float().cpu()
        attn_ti = attn[:, text_pos, :][:, :, img_pos]
        accum  += attn_ti.mean(dim=(0, 1))
    accum /= len(resolved)
    return _accum_to_spatial(accum, rows, cols, block_size)


def attn_map_with_hooks(
    model,
    inputs: dict,
    img_pos: torch.Tensor,
    text_pos: torch.Tensor,
    rows: int,
    cols: int,
    layer_indices: list[int],
    block_size: int = 1,
) -> np.ndarray:
    """
    InternVL3: extrai atenção via hooks, acumulando na CPU camada a camada.
    Evita OOM causado por armazenar todas as matrizes de atenção na GPU.

    Patcha temporariamente as camadas alvo para forçar output_attentions=True
    apenas nelas, sem ativar o flag globalmente (que guardaria tudo na GPU).
    """
    lm_layers = model.language_model.model.layers
    n_layers   = len(lm_layers)
    resolved   = sorted(set(li % n_layers for li in layer_indices))

    accum      = torch.zeros(len(img_pos), dtype=torch.float32)
    count      = [0]
    hooks      = []
    originals  = {}

    # Patcha cada camada alvo para sempre retornar attn_weights
    for li in resolved:
        orig = lm_layers[li].self_attn.forward
        originals[li] = orig

        def make_patched(orig_fwd):
            def patched(*args, **kwargs):
                kwargs["output_attentions"] = True
                return orig_fwd(*args, **kwargs)
            return patched

        lm_layers[li].self_attn.forward = make_patched(orig)

    # Hook que captura e acumula na CPU imediatamente
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn    = output[1][0].float().cpu()          # [heads, seq, seq]
                attn_ti = attn[:, text_pos, :][:, :, img_pos] # [heads, n_text, n_img]
                accum.add_(attn_ti.mean(dim=(0, 1)))
                count[0] += 1
        return hook

    for li in resolved:
        hooks.append(lm_layers[li].self_attn.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs, return_dict=True)   # sem output_attentions global
    finally:
        for h in hooks:
            h.remove()
        for li, orig in originals.items():
            lm_layers[li].self_attn.forward = orig

    if count[0] > 0:
        accum /= count[0]
    return _accum_to_spatial(accum, rows, cols, block_size)


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
    """Alpha por pixel proporcional à atenção — baixa atenção mostra documento original."""
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    h, w   = img_np.shape[:2]
    heat   = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heat_np  = np.array(heat) / 255.0                                   # [h, w] ∈ [0,1]
    heat_rgb = (plt.get_cmap(colormap)(heat_np)[:, :, :3] * 255)        # [h, w, 3]
    px_alpha = (heat_np * alpha)[:, :, np.newaxis]                       # por pixel
    return np.clip(img_np * (1 - px_alpha) + heat_rgb * px_alpha, 0, 255).astype(np.uint8)


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

    print(f"Carregando {args.model} ({args.family})…")
    if args.family == "internvl":
        model, processor = load_internvl(args.model)
    else:
        model, processor = load_qwen2vl(args.model)

    image = Image.open(args.image).convert("RGB")

    n = len(prompts)
    panels = []
    sys_text_global = ""

    for i, (label, prompt) in enumerate(prompts):
        print(f"\n[{i+1}/{n}] {prompt!r}")

        if args.family == "internvl":
            inp, img_pos, all_text_pos, _, rows, cols, block_size, sys_text = \
                prepare_internvl(model, processor, image, prompt)
        else:
            inp, img_pos, all_text_pos, _, rows, cols, sys_text = \
                prepare_qwen2vl(model, processor, image, prompt)
            block_size = 1
            sys_text_global = sys_text

        print(f"  tokens — system: {len(sys_text.split()) if sys_text else 0}, "
              f"text: {len(all_text_pos)}, img: {len(img_pos)}, "
              f"grid: {rows}×{cols}" + (f", bloco: {block_size}" if block_size > 1 else ""))

        if args.family == "internvl":
            attn_map = attn_map_with_hooks(
                model, inp, img_pos, all_text_pos, rows, cols, args.layers, block_size)
        else:
            with torch.no_grad():
                out = model(**dict(**inp, output_attentions=True, return_dict=True))
            attn_map = attn_map_from_output(
                out, img_pos, all_text_pos, rows, cols, args.layers, block_size)

        if args.family == "internvl":
            response = generate_response_internvl(model, processor, image, prompt)
        else:
            response = generate_response(model, processor, image, prompt)
        print(f"  → {response}")

        panels.append({"prompt": prompt, "response": response, "attn_map": attn_map})

    stem = args.output.removesuffix(".png")
    make_figure(image, panels, sys_text_global,
                show_sys_text=bool(sys_text_global),
                colormap=args.colormap, alpha=args.alpha,
                output_path=f"{stem}.png")


if __name__ == "__main__":
    main()
