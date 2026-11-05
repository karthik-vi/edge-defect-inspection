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


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """masks (N,H,W) in {0,1}; maps same shape."""
    return float(roc_auc_score(masks.reshape(-1).astype(np.uint8), maps.reshape(-1)))


def pro_auc(masks: np.ndarray, maps: np.ndarray, fpr_max: float = 0.3, n_thr: int = 100) -> float:
    """Area under per-region-overlap vs pixel-FPR curve for FPR<=fpr_max, normalised to [0,1].

    Thresholds are quantiles of the anomaly-map values of normal pixels (n_thr steps).
    """
    import cv2

    regions = []
    for i, m in enumerate(masks):
        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        regions += [(i, lab == k) for k in range(1, n)]
    if not regions:
        return float("nan")
    neg = maps[masks == 0]
    thrs = np.quantile(neg, np.linspace(0.0, 1.0, n_thr, endpoint=False))[::-1]  # high -> low
    fprs, pros = [0.0], [0.0]
    for t in thrs:
        pred = maps > t
        fpr = pred[masks == 0].mean()
        if fpr > fpr_max:
            break
        fprs.append(float(fpr))
        pros.append(float(np.mean([pred[i][r].mean() for i, r in regions])))
    if len(fprs) < 2:
        return 0.0
    return float(np.trapezoid(pros, fprs) / fpr_max)
