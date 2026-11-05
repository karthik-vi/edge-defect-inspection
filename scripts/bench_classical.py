"""Phase 1 benchmark: classical golden-image baseline on MVTec AD (per category)."""
from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime

import numpy as np

from edgeinspect.classical import align, fit_golden, preprocess
from edgeinspect.config import load_settings
from edgeinspect.data import list_test, list_train, read_gray
from edgeinspect.metrics import (
    image_auroc,
    latency_percentiles,
    recall_at_frr_oracle,
    recall_at_threshold,
    threshold_from_good,
)


def run(cat: str, cfg) -> dict:
    rng = np.random.default_rng(cfg.seed)
    train = list_train(cfg.data_root, cat)
    idx = rng.permutation(len(train))
    gold = [train[i] for i in idx[: cfg.n_golden]]
    val = [train[i] for i in idx[cfg.n_golden: cfg.n_golden + cfg.n_val_good]]
    aln = {"none": lambda g: g, "translate": lambda g: align(g, ecc_min=1e9), "rotate": align}[
        cfg.align_mode]

    def prep(p):
        return aln(preprocess(read_gray(p), cfg.image_size))

    model = fit_golden([prep(p) for p in gold], [prep(p) for p in val])
    val_scores = np.array([model.score(prep(p)) for p in val])
    thr = threshold_from_good(val_scores, cfg.false_reject_rate)

    test = list_test(cfg.data_root, cat)
    labels = np.array([t[1] for t in test])
    scores, ms = [], []
    for p, _, _ in test:
        g = preprocess(read_gray(p), cfg.image_size)  # image read excluded from latency
        t0 = time.perf_counter()
        scores.append(model.score(aln(g)))
        ms.append((time.perf_counter() - t0) * 1e3)
    scores, ms = np.array(scores), np.array(ms)
    rec, frr = recall_at_threshold(labels, scores, thr)
    return {
        "category": cat, "n_test": len(test), "n_defect": int(labels.sum()),
        "image_auroc": round(image_auroc(labels, scores), 4),
        "recall_at_val_thr": round(rec, 4), "test_frr_at_val_thr": round(frr, 4),
        "recall_at_1pct_frr_oracle": round(recall_at_frr_oracle(labels, scores, 0.01), 4),
        "latency_ms": {k: round(v, 2) for k, v in latency_percentiles(ms).items()},
    }


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(1)
    cfg = load_settings()
    rows = [run(c, cfg) for c in cfg.categories]
    out = {"date": datetime.now(UTC).date().isoformat(), "cmd": "make bench-classical", "cpu": platform.processor(),
           "opencv_threads": 1, "n_golden": cfg.n_golden, "n_val_good": cfg.n_val_good,
           "image_size": cfg.image_size, "align_mode": cfg.align_mode, "rows": rows}
    with open(f"{cfg.results_dir}/classical.json", "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(r["category"], r["image_auroc"], r["recall_at_val_thr"], r["test_frr_at_val_thr"],
              r["recall_at_1pct_frr_oracle"], r["latency_ms"])
