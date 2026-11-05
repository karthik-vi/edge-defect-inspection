"""Drift monitor: image statistics + feature-embedding distribution shift (PCA + KS test)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from edgeinspect.classical import part_mask

STAT_NAMES = ("brightness", "log_laplacian_var", "centroid_x", "centroid_y")


def image_stats(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """[mean brightness, log blur metric (Laplacian variance), part centroid x, y in [0,1]]."""
    g = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    lap = float(np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()))
    mo = cv2.moments(part_mask(g), binaryImage=True)
    cx, cy = (mo["m10"] / mo["m00"] / size, mo["m01"] / mo["m00"] / size) if mo["m00"] > 0 else (0.5, 0.5)
    return np.array([g.mean(), lap, cx, cy], np.float32)


@dataclass
class DriftReport:
    score: float          # >= 1.0 means alert
    alert: bool
    stat_z: dict[str, float]
    ks_p: float           # smallest KS p-value over PCA components (uncorrected)


class DriftMonitor:
    """Compares a window of production images against a reference set of good parts.

    score = max(max_i d_i / z_thr, log(p_bonf)/log(alpha)); alert when score >= 1.
    d_i: effect size = |window mean - ref mean| / ref per-image sd of statistic i. (A significance
    z-test on window means false-alarmed on clean held-out parts: they sit ~1 sd from the train
    reference, so an effect-size tolerance is used instead.)
    p_bonf: Bonferroni-corrected min KS p-value over the PCA components of the mean-pooled embedding.
    """

    def __init__(self, ref_stats: np.ndarray, ref_emb: np.ndarray, z_thr: float = 3.0,
                 alpha: float = 1e-3, n_comp: int = 5) -> None:
        self.mu, self.sd = ref_stats.mean(0), np.maximum(ref_stats.std(0), 1e-6)
        self.pca = PCA(n_components=n_comp, random_state=0).fit(ref_emb)
        self.ref_proj = self.pca.transform(ref_emb)
        self.z_thr, self.alpha, self.n_comp = z_thr, alpha, n_comp

    def check(self, stats: np.ndarray, emb: np.ndarray) -> DriftReport:
        z = np.abs(stats.mean(0) - self.mu) / self.sd
        proj = self.pca.transform(emb)
        p = min(ks_2samp(self.ref_proj[:, k], proj[:, k]).pvalue for k in range(self.n_comp))
        p_bonf = min(1.0, p * self.n_comp)
        s_img = float(z.max() / self.z_thr)
        s_emb = float(np.log(max(p_bonf, 1e-300)) / np.log(self.alpha))
        score = max(s_img, s_emb)
        return DriftReport(score, score >= 1.0, dict(zip(STAT_NAMES, map(float, z), strict=True)),
                           float(p))


class StreamDriftMonitor:
    """Online wrapper: buffers per-image observations, emits a report every `window` images."""

    def __init__(self, monitor: DriftMonitor, window: int = 20) -> None:
        self.m, self.window = monitor, window
        self._s: list[np.ndarray] = []
        self._e: list[np.ndarray] = []
        self.last: DriftReport | None = None

    def update(self, stats: np.ndarray, emb: np.ndarray) -> DriftReport | None:
        self._s.append(stats)
        self._e.append(emb)
        if len(self._s) < self.window:
            return None
        self.last = self.m.check(np.stack(self._s), np.stack(self._e))
        self._s, self._e = [], []
        return self.last
