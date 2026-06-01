#!/usr/bin/env python3
"""
Avalia embeddings de documentos via similaridade de cosseno num protocolo zero-shot.

Replica o eval_embeddings.py do CaVL-Doc para comparar InternVL3 e Qwen2.5-VL
na primeira e última camada de hidden states.

Métrica: EER (Equal Error Rate) — quanto menor, melhor.

Uso:
  python scripts/eval_embeddings.py \\
      --val-csv /path/para/split0/validation_pairs.csv \\
      --image-dir /mnt/data/la-cdip/data \\
      --output results/eval_embeddings_split0.html \\
      --gpu 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EMBEDDING_PROMPT_DEFAULT = "Analyze this document."


# ---------------------------------------------------------------------------
# Extratores
# ---------------------------------------------------------------------------

class InternVL3Embedder:
    MAX_NUM    = 12   # igual ao paper
    IMAGE_SIZE = 448

    def __init__(self, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cavl_doc.data.transforms import build_transform
        print("Carregando InternVL3-2B…")
        self.model = AutoModelForCausalLM.from_pretrained(
            "OpenGVLab/InternVL3-2B",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False
        )
        self._transform = build_transform(self.IMAGE_SIZE)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._token_counts: list[int] = []

    @torch.no_grad()
    def embed(self, img: Image.Image, layer_idx: int,
              prompt: str = EMBEDDING_PROMPT_DEFAULT) -> np.ndarray:
        from cavl_doc.data.transforms import dynamic_preprocess
        from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

        blocks = dynamic_preprocess(
            img, max_num=self.MAX_NUM, image_size=self.IMAGE_SIZE, use_thumbnail=True
        )
        pv  = torch.stack([self._transform(b) for b in blocks]).to(torch.bfloat16).to(
            next(self.model.parameters()).device
        )
        inp = prepare_inputs_for_multimodal_embedding(
            self.model, self.tokenizer, pv, prompt
        )
        n_vis = (inp["input_ids"][0].cpu() == self.model.img_context_token_id).sum().item()
        self._token_counts.append(n_vis)

        out = self.model(
            input_ids=inp["input_ids"],
            pixel_values=inp["pixel_values"],
            image_flags=inp["image_flags"],
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[layer_idx]   # [1, seq, dim]
        return h.mean(dim=1).squeeze(0).float().cpu().numpy()

    def avg_visual_tokens(self) -> float:
        return float(np.mean(self._token_counts)) if self._token_counts else 0.0


class Qwen2VLEmbedder:
    # Equaliza com InternVL3 max_num=12: ~13 tiles × 256 = 3328 tokens
    # 3328 tokens × merge_size² × patch_size² = 3328 × 4 × 196 = 2_609_152 pixels
    TARGET_PIXELS = 2_609_152

    def __init__(self, device: str, equalize_tokens: bool = True):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Carregando Qwen2.5-VL-3B… (equalize_tokens={equalize_tokens})")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        if equalize_tokens:
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                min_pixels=self.TARGET_PIXELS,
                max_pixels=self.TARGET_PIXELS,
            )
        else:
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.equalize_tokens = equalize_tokens
        self._token_counts: list[int] = []

    @torch.no_grad()
    def embed(self, img: Image.Image, layer_idx: int) -> np.ndarray:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  EMBEDDING_PROMPT},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        img_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_inputs, return_tensors="pt"
        ).to(next(self.model.parameters()).device)

        img_token_id = self.model.config.image_token_id
        n_vis = (inputs.input_ids[0].cpu() == img_token_id).sum().item()
        self._token_counts.append(n_vis)

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[layer_idx]   # [1, seq, dim]
        return h.mean(dim=1).squeeze(0).float().cpu().numpy()

    def avg_visual_tokens(self) -> float:
        return float(np.mean(self._token_counts)) if self._token_counts else 0.0


# ---------------------------------------------------------------------------
# Cosine + EER
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    auc = float(roc_auc_score(labels, scores))
    return eer, float(thr[idx]), auc


# ---------------------------------------------------------------------------
# Avaliação de um par (modelo × camada)
# ---------------------------------------------------------------------------

def run_eval(embedder, layer_idx: int, val_csv: Path,
             image_dir: Path, label: str,
             prompt: str = EMBEDDING_PROMPT_DEFAULT) -> dict:
    import pandas as pd

    df    = pd.read_csv(val_csv)
    cache: dict[str, np.ndarray] = {}
    rows  = []
    t0    = time.time()
    n     = len(df)

    for i, row in df.iterrows():
        for col in ("file_a_path", "file_b_path"):
            p = row[col]
            if p not in cache:
                full = image_dir / p
                # Alguns CSVs têm um subdiretório extra no prefixo (ex: separacao-rvl_cdip/)
                if not full.exists():
                    full = image_dir / Path(p).relative_to(Path(p).parts[0])
                img = Image.open(full).convert("RGB")
                cache[p] = embedder.embed(img, layer_idx, prompt=prompt)

        score = cosine(cache[row["file_a_path"]], cache[row["file_b_path"]])
        rows.append({"is_equal": int(row["is_equal"]), "score": score})

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}]  {time.time()-t0:.0f}s")

    df_res = pd.DataFrame(rows)
    eer, thr, auc = compute_eer(df_res["score"].values, df_res["is_equal"].values)
    elapsed = time.time() - t0

    avg_tok = embedder.avg_visual_tokens() if hasattr(embedder, "avg_visual_tokens") else 0
    print(f"  {label}: EER={eer*100:.2f}%  AUC={auc:.4f}  avg_vis_tokens={avg_tok:.0f}  ({elapsed/60:.1f} min)")
    return {"label": label, "eer": eer, "eer_pct": round(eer*100, 2),
            "auc": round(auc, 4), "threshold": round(thr, 4),
            "avg_visual_tokens": round(avg_tok),
            "n_pairs": n, "elapsed_min": round(elapsed/60, 1)}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def make_html(rows: list[dict], output_path: Path, split: str):
    import pandas as pd
    df = pd.DataFrame(rows)

    rows_html = ""
    for _, r in df.iterrows():
        best = r["eer_pct"] == df["eer_pct"].min()
        style = ' style="font-weight:bold;background:#e8f5e9;"' if best else ""
        rows_html += (
            f'<tr{style}><td>{r["label"]}</td>'
            f'<td>{r["eer_pct"]:.2f}%</td>'
            f'<td>{r["auc"]:.4f}</td>'
            f'<td>{r["threshold"]:.4f}</td>'
            f'<td>{r.get("avg_visual_tokens", "—")}</td>'
            f'<td>{r["n_pairs"]}</td>'
            f'<td>{r["elapsed_min"]} min</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>DocPromptStudy — Eval Embeddings Split {split}</title>
<style>
body {{font-family:Georgia,serif;background:#f7f7f7;color:#222;padding:40px;max-width:900px;margin:auto;}}
h1 {{font-size:1.5rem;color:#1a1a2e;margin-bottom:4px;}}
.subtitle {{color:#666;font-size:0.88rem;margin-bottom:30px;}}
.section {{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:22px 26px;margin-bottom:26px;}}
h2 {{font-size:1.05rem;color:#1a1a2e;border-bottom:2px solid #4C78A8;padding-bottom:4px;margin-bottom:14px;}}
p.desc {{font-size:0.85rem;color:#555;margin-bottom:14px;line-height:1.55;}}
table {{border-collapse:collapse;width:100%;font-size:0.82rem;}}
th {{background:#1a1a2e;color:#fff;padding:7px 12px;text-align:left;}}
td {{padding:6px 12px;border-bottom:1px solid #e0e0e0;}}
tr:nth-child(even) td {{background:#f2f2f8;}}
footer {{font-size:0.72rem;color:#aaa;margin-top:36px;text-align:center;}}
</style>
</head>
<body>
<h1>DocPromptStudy — Avaliação de Embeddings</h1>
<p class="subtitle">Split {split} · {df["n_pairs"].iloc[0]} pares · Prompt: "{EMBEDDING_PROMPT}"</p>

<div class="section">
<h2>Resultados — EER por Modelo e Camada</h2>
<p class="desc">
  EER (Equal Error Rate): menor é melhor.<br>
  Camada 0 = embedding layer (entrada do LM) · Camada -1 = última camada do LM.<br>
  InternVL3: max_num=12 (~3328 tokens visuais). Qwen2.5-VL: equalizado via min_pixels=max_pixels=2_609_152.<br>
  Embeddings: mean pool sobre todos os tokens da sequência.
</p>
<table>
  <tr>
    <th>Modelo / Camada</th><th>EER ↓</th><th>AUC ↑</th>
    <th>Threshold</th><th>Avg tokens visuais</th><th>Pares</th><th>Tempo</th>
  </tr>
  {rows_html}
</table>
</div>
<footer>Gerado por scripts/eval_embeddings.py · DocPromptStudy</footer>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"\nHTML salvo em {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-csv",   default="/home/joaopaulo/Projects/CaVL-Doc/data/generated_splits/split0/validation_pairs.csv")
    p.add_argument("--image-dir", default="/mnt/data/la-cdip/data")
    p.add_argument("--output",    default=str(ROOT / "results" / "eval_embeddings_split0.html"))
    p.add_argument("--models",    nargs="+", default=["internvl3", "qwen2vl"],
                   choices=["internvl3", "qwen2vl"])
    p.add_argument("--prompts",   nargs="+", default=[EMBEDDING_PROMPT_DEFAULT],
                   help="Prompts a avaliar (um por run; padrão: 'Analyze this document.')")
    p.add_argument("--gpu",       type=int, default=0)
    return p.parse_args()


def main():
    args       = parse_args()
    val_csv    = Path(args.val_csv)
    image_dir  = Path(args.image_dir)
    output_path = Path(args.output)
    device     = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    model_configs = [
        ("internvl3", 0,  "InternVL3-2B · Layer 0  (input)"),
        ("internvl3", -1, "InternVL3-2B · Layer -1 (last)"),
        ("qwen2vl",   0,  "Qwen2.5-VL-3B · Layer 0  (input)"),
        ("qwen2vl",   -1, "Qwen2.5-VL-3B · Layer -1 (last)"),
    ]
    model_configs = [(m, l, lbl) for m, l, lbl in model_configs if m in args.models]

    results            = []
    current_model_key  = None
    embedder           = None

    for prompt in args.prompts:
        prompt_short = prompt[:40] + "…" if len(prompt) > 40 else prompt
        print(f"\n{'#'*60}")
        print(f"PROMPT: {prompt_short!r}")
        print(f"{'#'*60}")

        for model_key, layer_idx, base_label in model_configs:
            if model_key != current_model_key:
                if embedder is not None:
                    del embedder
                    torch.cuda.empty_cache()
                embedder = InternVL3Embedder(device) if model_key == "internvl3" \
                           else Qwen2VLEmbedder(device)
                current_model_key = model_key

            label = f"{base_label} | {prompt_short!r}"
            print(f"\n{'='*60}\n{label}\n{'='*60}")
            row = run_eval(embedder, layer_idx, val_csv, image_dir, label, prompt=prompt)
            row["prompt"] = prompt
            results.append(row)

    make_html(results, output_path, split="0")

    print(f"\n{'='*70}")
    print(f"{'Modelo/Camada':<45} {'EER':>8} {'AUC':>8}")
    print(f"{'='*70}")
    for r in results:
        print(f"{r['label'][:45]:<45} {r['eer_pct']:>7.2f}% {r['auc']:>8.4f}")


if __name__ == "__main__":
    main()
