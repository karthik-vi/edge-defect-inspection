"""Classical machine-vision baseline: moment-based alignment + golden-image rule checks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GRID = 8


def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """BGR/gray uint8 -> gray uint8 of shape (size, size)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def part_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground mask via Otsu; picks the polarity whose largest blob is not the border."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if border.mean() > 127:  # background is white -> invert
        m = 255 - m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def align(gray: np.ndarray, ecc_min: float = 1.3) -> np.ndarray:
    """Translate part centroid to image centre; rotate to principal axis if the part is elongated."""
    h, w = gray.shape
    m = part_mask(gray)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    c = max(cnts, key=cv2.contourArea)
    mo = cv2.moments(c)
    if mo["m00"] < 50:
        return gray
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    a, b, d = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    common = np.sqrt(((a - b) / 2) ** 2 + d**2)
    l1, l2 = (a + b) / 2 + common, (a + b) / 2 - common
    ecc = np.sqrt(l1 / max(l2, 1e-6))
    angle = 0.5 * np.degrees(np.arctan2(2 * d, a - b)) if ecc > ecc_min else 0.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _cell_stats(gray: np.ndarray) -> np.ndarray:
    """Per-grid-cell edge density and Laplacian variance -> (2, GRID, GRID)."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150).astype(np.float32) / 255
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    ch, cw = h // GRID, w // GRID
    e = edges[: ch * GRID, : cw * GRID].reshape(GRID, ch, GRID, cw).mean((1, 3))
    t = lap[: ch * GRID, : cw * GRID].reshape(GRID, ch, GRID, cw).var((1, 3))
    return np.stack([e, np.log1p(t)])


def _raw_scores(gray: np.ndarray, mu: np.ndarray, sd: np.ndarray, cmu: np.ndarray,
                csd: np.ndarray) -> np.ndarray:
    z = cv2.GaussianBlur(np.abs(gray.astype(np.float32) - mu) / sd, (7, 7), 0)
    diff = np.percentile(z, 99.5)
    cz = np.abs(_cell_stats(gray) - cmu) / csd
    return np.array([diff, cz[0].max(), cz[1].max()], dtype=np.float32)


@dataclass
class GoldenModel:
    mu: np.ndarray
    sd: np.ndarray
    cmu: np.ndarray
    csd: np.ndarray
    rule_mean: np.ndarray = None  # per-rule normalisation from val-good scores
    rule_std: np.ndarray = None

    def raw(self, aligned: np.ndarray) -> np.ndarray:
        return _raw_scores(aligned, self.mu, self.sd, self.cmu, self.csd)

    def score(self, aligned: np.ndarray) -> float:
        """Image anomaly score: max over normalised rule scores (higher = more anomalous)."""
        return float(((self.raw(aligned) - self.rule_mean) / self.rule_std).max())


def fit_golden(golden: list[np.ndarray], val_good: list[np.ndarray]) -> GoldenModel:
    """golden / val_good: aligned gray images of good parts."""
    stack = np.stack([g.astype(np.float32) for g in golden])
    mu = stack.mean(0)
    sd = np.maximum(cv2.GaussianBlur(stack.std(0), (9, 9), 0), 8.0)
    cs = np.stack([_cell_stats(g) for g in golden])
    cmu, csd = cs.mean(0), np.maximum(cs.std(0), 1e-3 + 0.1 * cs.std())
    gm = GoldenModel(mu, sd, cmu, csd)
    raws = np.stack([gm.raw(v) for v in val_good])
    gm.rule_mean, gm.rule_std = raws.mean(0), np.maximum(raws.std(0), 1e-6)
    return gm
