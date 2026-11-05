"""Phase 2 benchmark: PatchCore on MVTec AD. One run = one config over the categories."""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from edgeinspect.classical import align
from edgeinspect.config import load_settings
from edgeinspect.data import list_test, list_train
from edgeinspect.metrics import (
    image_auroc,
    latency_percentiles,
    pixel_auroc,
    pro_auc,
    recall_at_frr_oracle,
    recall_at_threshold,
    threshold_from_good,
)
from edgeinspect.patchcore import Extractor, PatchCore

ALIGN = {"none": lambda g: g, "translate": lambda g: align(g, ecc_min=1e9), "rotate": align}


def load_mask(root: str, cat: str, path: Path, label: int, size: int) -> np.ndarray:
    if not label:
        return np.zeros((size, size), np.uint8)
    mp = Path(root) / cat / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    return (cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def run(cat: str, a: argparse.Namespace, cfg, ext: Extractor) -> dict:
    aln = ALIGN[a.align]

    def rd(p):
        return aln(cv2.imread(str(p), cv2.IMREAD_COLOR))

    rng = np.random.default_rng(cfg.seed)
    train = list_train(cfg.data_root, cat)
    idx = rng.permutation(len(train))
    val = [train[i] for i in idx[: cfg.pc_val_good]]
    fit = [train[i] for i in idx[cfg.pc_val_good: cfg.pc_val_good + cfg.pc_train_max]]
    pc = PatchCore(ext, cfg.pc_size, a.ratio, seed=cfg.seed)
    t0 = time.perf_counter()
    pc.fit([rd(p) for p in fit])
    fit_s = time.perf_counter() - t0
    thr = threshold_from_good(pc.score([rd(p) for p in val])[0], cfg.false_reject_rate)

    test = list_test(cfg.data_root, cat)
    labels = np.array([t[1] for t in test])
    imgs = [rd(p) for p, _, _ in test]  # alignment is part of preprocessing, excluded from latency
    masks = np.stack([load_mask(cfg.data_root, cat, p, l, cfg.pc_size) for p, l, _ in test])
    scores, maps, ms = [], [], []
    for im in imgs:  # batch size 1, as on a line
        t0 = time.perf_counter()
        s, m = pc.score([im])
        ms.append((time.perf_counter() - t0) * 1e3)
        scores.append(s[0])
        maps.append(m[0])
    scores, maps = np.array(scores), np.stack(maps)
    rec, frr = recall_at_threshold(labels, scores, thr)
    return {
        "category": cat, "n_test": len(test), "n_bank_imgs": len(fit), "bank_patches": len(pc.bank),
        "bank_mb": round(pc.bank_mb, 2), "fit_s": round(fit_s, 1),
        "image_auroc": round(image_auroc(labels, scores), 4),
        "pixel_auroc": round(pixel_auroc(masks, maps), 4),
        "pro": round(pro_auc(masks, maps), 4),
        "recall_at_val_thr": round(rec, 4), "test_frr_at_val_thr": round(frr, 4),
        "recall_at_1pct_frr_oracle": round(recall_at_frr_oracle(labels, scores, 0.01), 4),
        "latency_ms": {k: round(v, 1) for k, v in latency_percentiles(np.array(ms)).items()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--align", default="none", choices=list(ALIGN))
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cats", default="")
    a = ap.parse_args()
    cfg = load_settings()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    ext = Extractor(a.backbone, tuple(int(x) for x in a.layers.split(",")), a.dim)
    rows = [run(c, a, cfg, ext) for c in (a.cats.split(",") if a.cats else cfg.categories)]
    out = {"tag": a.tag, "args": vars(a), "date": datetime.now(UTC).date().isoformat(),
           "cpu": platform.processor() or "Intel Xeon (virtualised), 4 cores", "torch": torch.__version__,
           "n_val_good": cfg.pc_val_good, "pc_train_max": cfg.pc_train_max, "rows": rows}
    Path(cfg.results_dir, "patchcore").mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir, "patchcore", f"{a.tag}.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(a.tag, r["category"], r["image_auroc"], r["pixel_auroc"], r["pro"],
              r["recall_at_val_thr"], r["test_frr_at_val_thr"], r["latency_ms"], r["bank_mb"], flush=True)
