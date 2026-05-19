"""
Carregamento do RVL-CDIP — HuggingFace datasets ou dataset local.

Dataset local esperado (/mnt/data/zs_rvl_cdip/data):
  data/{class_name}/{filename}.tif   (200 imagens por classe)
  splits.csv  → class_name, class_number, doc_path, subset, ...

RVL-CDIP HuggingFace: 400k imagens, 16 classes.
  Para experimentos rápidos: max_per_class (e.g. 10 por classe = 160 total)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
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

# Mapeamento: nome da pasta local → índice padrão RVL-CDIP (mesma ordem de classes.yaml)
# O dataset local usa ordenação alfabética internamente; remapeamos para a convenção padrão.
_FOLDER_TO_LABEL: dict[str, int] = {
    "letter":                  0,
    "form":                    1,
    "email":                   2,
    "handwritten":             3,
    "advertisement":           4,
    "scientific_report":       5,
    "scientific_publication":  6,
    "specification":           7,
    "file_folder":             8,
    "news_article":            9,
    "budget":                  10,
    "invoice":                 11,
    "presentation":            12,
    "questionnaire":           13,
    "resume":                  14,
    "memo":                    15,
}


class LocalRVLCDIPEntry:
    """Representa uma entrada do dataset local (lazy — carrega imagem sob demanda)."""
    __slots__ = ("path", "label")

    def __init__(self, path: Path, label: int):
        self.path = path
        self.label = label

    @property
    def image(self) -> Image.Image:
        img = Image.open(self.path)
        # TIF multi-página: usa só a primeira página
        img.seek(0)
        return img.convert("RGB")


class LocalRVLCDIPList:
    """Lista de entradas do dataset local compatível com o loop de extração."""

    def __init__(self, entries: list[LocalRVLCDIPEntry]):
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for e in self._entries:
            yield {"image": e.image, "label": e.label}

    def __getitem__(self, idx):
        e = self._entries[idx]
        return {"image": e.image, "label": e.label}


def load_local_rvl_cdip(
    data_dir: str | Path,
    splits_csv: str | Path | None = None,
    max_per_class: int | None = None,
    seed: int = 42,
) -> LocalRVLCDIPList:
    """
    Carrega o dataset local em /mnt/data/zs_rvl_cdip/data (ou similar).

    Args:
        data_dir:      Pasta raiz com subpastas por classe (advertisement/, budget/, ...)
        splits_csv:    Opcional — usa splits.csv para filtrar subset. Se None,
                       varre todas as imagens nas subpastas.
        max_per_class: Limita amostras por classe (amostragem aleatória com seed fixo).
        seed:          Semente para reprodutibilidade da amostragem.
    """
    import pandas as pd

    data_dir = Path(data_dir)

    if splits_csv is not None:
        df = pd.read_csv(splits_csv)
        entries = []
        for _, row in df.iterrows():
            folder = row["doc_path"].split("/")[0]
            label  = _FOLDER_TO_LABEL.get(folder)
            if label is None:
                continue
            img_path = data_dir / row["doc_path"]
            if img_path.exists():
                entries.append(LocalRVLCDIPEntry(img_path, label))
    else:
        entries = []
        for folder, label in _FOLDER_TO_LABEL.items():
            folder_path = data_dir / folder
            if not folder_path.exists():
                continue
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
                for img_path in sorted(folder_path.glob(ext)):
                    entries.append(LocalRVLCDIPEntry(img_path, label))

    if max_per_class is not None:
        rng = random.Random(seed)
        by_class: dict[int, list] = defaultdict(list)
        for e in entries:
            by_class[e.label].append(e)
        sampled = []
        for lbl, lst in by_class.items():
            rng.shuffle(lst)
            sampled.extend(lst[:max_per_class])
        entries = sampled

    return LocalRVLCDIPList(entries)


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
