"""
Carregamento do RVL-CDIP via HuggingFace datasets.

RVL-CDIP: 400k imagens, 16 classes de documentos.
  Train: 320k | Validation: 40k | Test: 40k
  25k imagens por classe em treino, 2.5k por classe em val/test.

Para experimentos rápidos usar max_per_class (e.g. 500 treino / 200 test).
"""

from __future__ import annotations

from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

RVL_CDIP_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

NUM_CLASSES = len(RVL_CDIP_CLASSES)   # 16


def load_rvl_cdip(
    split: str = "train",
    max_per_class: int | None = None,
    hf_dataset: str = "jbxai/rvl-cdip",
    streaming: bool = False,
):
    """
    Retorna um HuggingFace Dataset (ou IterableDataset se streaming=True).

    Args:
        split: "train", "validation" ou "test"
        max_per_class: limita amostras por classe (None = todas)
        hf_dataset: nome do dataset no HF Hub
        streaming: não baixa tudo de uma vez (útil para o split de treino completo)
    """
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=split, streaming=streaming, trust_remote_code=True)

    if max_per_class is not None and not streaming:
        counts: dict[int, int] = defaultdict(int)
        indices = []
        for i, ex in enumerate(ds):
            lbl = ex["label"]
            if counts[lbl] < max_per_class:
                indices.append(i)
                counts[lbl] += 1
            if sum(counts.values()) == max_per_class * NUM_CLASSES:
                break
        ds = ds.select(indices)

    return ds


class RVLCDIPDataset(Dataset):
    """Wrapper torch Dataset em cima do HF Dataset carregado."""

    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image: Image.Image = ex["image"].convert("RGB")
        label: int = ex["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
