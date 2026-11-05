"""Eval scorers: AUROC, recall at a fixed false-reject rate, latency percentiles."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def threshold_from_good(val_good_scores: np.ndarray, frr: float) -> float:
    """Threshold so that ~frr of good validation parts are rejected (uses good parts only)."""
    return float(np.quantile(val_good_scores, 1.0 - frr))


def recall_at_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> tuple[float, float]:
    """Returns (defect recall, false-reject rate on test goods) at score > thr."""
    pred = scores > thr
    return float(pred[labels == 1].mean()), float(pred[labels == 0].mean())


def recall_at_frr_oracle(labels: np.ndarray, scores: np.ndarray, frr: float) -> float:
    """Upper bound: threshold taken from the test goods themselves (needs labels; reference only)."""
    thr = np.quantile(scores[labels == 0], 1.0 - frr)
    return float((scores[labels == 1] > thr).mean())


def latency_percentiles(ms: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(ms, p)) for p in (50, 95, 99)}
