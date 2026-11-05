"""MVTec AD loading (file listing only; images are read lazily)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def list_train(root: str, cat: str) -> list[Path]:
    return sorted((Path(root) / cat / "train" / "good").glob("*.png"))


def list_test(root: str, cat: str) -> list[tuple[Path, int, str]]:
    """(path, label 0=good/1=defect, defect_type) for the official test split."""
    out = []
    for d in sorted((Path(root) / cat / "test").iterdir()):
        out += [(p, int(d.name != "good"), d.name) for p in sorted(d.glob("*.png"))]
    return out


def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img
