"""
Cache de embeddings em disco.

Estrutura:
  cache/{model}/{prompt_id}/{split}_layer{idx}.npz   — embeddings de imagens
    → embeddings: float32 [N, dim]
    → labels:     int64   [N]

  cache/{model}/_classes/{class_key}_layer{idx}.npy  — embeddings das classes
    → float32 [dim]

layer_idx: 0 = embedding layer, -1 = última camada (salvo como 0 e "last")
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

# Converte layer_idx numérico para nome de arquivo legível
def _layer_tag(layer_idx: int) -> str:
    return "layer0" if layer_idx == 0 else f"layer{layer_idx}" if layer_idx > 0 else "layerlast"


# ---------------------------------------------------------------------------
# Embeddings de imagens
# ---------------------------------------------------------------------------

def _img_path(cache_dir: str | Path, model: str, prompt_id: str, split: str, layer_idx: int) -> Path:
    p = Path(cache_dir) / model / prompt_id
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{split}_{_layer_tag(layer_idx)}.npz"


def img_exists(cache_dir, model, prompt_id, split, layer_idx) -> bool:
    return _img_path(cache_dir, model, prompt_id, split, layer_idx).exists()


def img_save(cache_dir, model, prompt_id, split, layer_idx,
             embeddings: np.ndarray, labels: np.ndarray) -> Path:
    p = _img_path(cache_dir, model, prompt_id, split, layer_idx)
    np.savez_compressed(p, embeddings=embeddings.astype(np.float32), labels=labels.astype(np.int64))
    return p


def img_load(cache_dir, model, prompt_id, split, layer_idx) -> tuple[np.ndarray, np.ndarray] | None:
    p = _img_path(cache_dir, model, prompt_id, split, layer_idx)
    if not p.exists():
        return None
    data = np.load(p)
    return data["embeddings"], data["labels"]


# ---------------------------------------------------------------------------
# Embeddings das classes (texto puro, sem imagem)
# ---------------------------------------------------------------------------

def _cls_path(cache_dir: str | Path, model: str, class_key: str, layer_idx: int) -> Path:
    p = Path(cache_dir) / model / "_classes"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{class_key}_{_layer_tag(layer_idx)}.npy"


def cls_exists(cache_dir, model, class_key, layer_idx) -> bool:
    return _cls_path(cache_dir, model, class_key, layer_idx).exists()


def cls_save(cache_dir, model, class_key, layer_idx, embedding: np.ndarray) -> Path:
    p = _cls_path(cache_dir, model, class_key, layer_idx)
    np.save(p, embedding.astype(np.float32))
    return p


def cls_load(cache_dir, model, class_key, layer_idx) -> np.ndarray | None:
    p = _cls_path(cache_dir, model, class_key, layer_idx)
    return np.load(p) if p.exists() else None


def cls_load_all(cache_dir, model, layer_idx, n_classes: int = 16) -> np.ndarray | None:
    """Carrega matriz [n_classes, dim] com todos os embeddings de classe, em ordem de label."""
    from doc_prompt_study.rvl_cdip import RVL_CDIP_CLASSES
    embs = []
    for name in RVL_CDIP_CLASSES:
        key = name.replace(" ", "_")  # "scientific report" → "scientific_report"
        e = cls_load(cache_dir, model, key, layer_idx)
        if e is None:
            return None
        embs.append(e)
    return np.stack(embs)  # [16, dim]


# ---------------------------------------------------------------------------
# Resultados generativos: texto gerado + classe predita
# cache/{model}/_generative/{prompt_id}/{split}.npz  → labels, predicted_labels
# cache/{model}/_generative/{prompt_id}/{split}.txt  → textos gerados (1 por linha)
# ---------------------------------------------------------------------------

def _gen_path(cache_dir: str | Path, model: str, prompt_id: str, split: str, ext: str) -> Path:
    p = Path(cache_dir) / model / "_generative" / prompt_id
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{split}{ext}"


def gen_exists(cache_dir, model, prompt_id, split) -> bool:
    return _gen_path(cache_dir, model, prompt_id, split, ".npz").exists()


def gen_save(
    cache_dir, model, prompt_id, split,
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    generated_texts: list[str],
) -> Path:
    p_npz = _gen_path(cache_dir, model, prompt_id, split, ".npz")
    p_txt = _gen_path(cache_dir, model, prompt_id, split, ".txt")
    np.savez_compressed(
        p_npz,
        labels=labels.astype(np.int64),
        predicted_labels=predicted_labels.astype(np.int64),
    )
    p_txt.write_text("\n".join(generated_texts), encoding="utf-8")
    return p_npz


def gen_load(cache_dir, model, prompt_id, split):
    """Retorna (labels, predicted_labels, generated_texts) ou None."""
    p_npz = _gen_path(cache_dir, model, prompt_id, split, ".npz")
    p_txt = _gen_path(cache_dir, model, prompt_id, split, ".txt")
    if not p_npz.exists():
        return None
    data  = np.load(p_npz)
    texts = p_txt.read_text(encoding="utf-8").splitlines() if p_txt.exists() else []
    return data["labels"], data["predicted_labels"], texts


# ---------------------------------------------------------------------------
# Listagem
# ---------------------------------------------------------------------------

def list_cached(cache_dir: str | Path) -> list[dict]:
    entries = []
    for npz in Path(cache_dir).rglob("*.npz"):
        parts = npz.relative_to(cache_dir).parts
        if len(parts) == 3 and not parts[1].startswith("_"):
            model, prompt_id, fname = parts
            stem = fname.replace(".npz", "")
            # stem format: {split}_{layer_tag}
            split_part, _, layer_part = stem.rpartition("_")
            entries.append({
                "model": model, "prompt_id": prompt_id,
                "split": split_part, "layer": layer_part, "path": str(npz),
            })
    return entries
