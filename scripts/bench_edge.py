"""Phase 3: PyTorch vs ONNX Runtime vs OpenVINO, FP32 vs INT8 (QDQ PTQ). One backend per process.

Accuracy at 4 threads over the full test split; latency at 1/2/4 threads (thread-limited
emulation of a small edge CPU, NOT real Jetson/RPi hardware).
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
import torch

from bench_patchcore import run
from edgeinspect.config import load_settings
from edgeinspect.data import list_test, list_train
from edgeinspect.metrics import latency_percentiles
from edgeinspect.patchcore import Extractor, PatchCore


class OrtExt:
    dim = 256

    def __init__(self, path: str, threads: int) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # avoids contention with FAISS OpenMP
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.s.run(None, {"image": x.numpy()})[0])


class OvExt:
    dim = 256

    def __init__(self, path: str, threads: int) -> None:
        import openvino as ov
        self.m = ov.Core().compile_model(path, "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                       "PERFORMANCE_HINT": "LATENCY",
                                                       "INFERENCE_PRECISION_HINT": "f32"})

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.array(self.m(x.numpy())[0]))


def make_ext(backend: str, threads: int):
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    faiss.omp_set_num_threads(threads)
    if backend == "torch":
        return Extractor("resnet18", (2, 3), 256)
    kind, prec = backend.split("_")
    path = f"models/r18_{'fp32' if prec == 'fp32' else 'int8'}.onnx"
    return (OrtExt if kind == "ort" else OvExt)(path, threads)


def latency_sweep(backend: str, cfg, ratio: float, n_img: int) -> dict:
    out: dict[str, dict] = {}
    for cat in cfg.categories:
        rng = np.random.default_rng(cfg.seed)
        tr = list_train(cfg.data_root, cat)
        idx = rng.permutation(len(tr))[cfg.pc_val_good: cfg.pc_val_good + cfg.pc_train_max]
        pc = PatchCore(make_ext(backend, 4), cfg.pc_size, ratio, seed=cfg.seed)
        pc.fit([cv2.imread(str(tr[i])) for i in idx])
        test = list_test(cfg.data_root, cat)
        sel = np.random.default_rng(1).permutation(len(test))[:n_img]
        imgs = [cv2.imread(str(test[i][0])) for i in sel]
        for t in (1, 2, 4):
            pc.ext = make_ext(backend, t)
            pc.score(imgs[:3])  # warm-up
            ms = []
            for im in imgs:
                t0 = time.perf_counter()
                pc.score([im])
                ms.append((time.perf_counter() - t0) * 1e3)
            out.setdefault(str(t), {})[cat] = latency_percentiles(np.array(ms))
    return {t: {k: round(float(np.mean([v[k] for v in d.values()])), 1) for k in ("p50", "p95", "p99")}
            for t, d in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)  # torch | ort_fp32 | ort_int8 | ov_fp32 | ov_int8
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--n-lat", type=int, default=40)
    a = ap.parse_args()
    cfg = load_settings()
    rows = [run(c, Namespace(align="none", ratio=a.ratio), cfg, make_ext(a.backend, 4))
            for c in cfg.categories]
    lat = latency_sweep(a.backend, cfg, a.ratio, a.n_lat)
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 4)
    model_mb = 0.0
    if a.backend != "torch":
        model_mb = Path(f"models/r18_{a.backend.split('_')[1]}.onnx").stat().st_size / 2**20
    res = {"backend": a.backend, "date": datetime.now(UTC).date().isoformat(), "ratio": a.ratio,
           "cpu": platform.processor() or "Intel Xeon (virtualised), 4 cores",
           "mean_image_auroc": mean("image_auroc"), "mean_pixel_auroc": mean("pixel_auroc"),
           "mean_pro": mean("pro"), "mean_recall_at_val_thr": mean("recall_at_val_thr"),
           "mean_test_frr": mean("test_frr_at_val_thr"), "bank_mb": rows[0]["bank_mb"],
           "model_mb": round(model_mb, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "latency_ms_by_threads": lat, "rows": rows}
    Path("results/edge").mkdir(parents=True, exist_ok=True)
    Path(f"results/edge/{a.backend}.json").write_text(json.dumps(res, indent=1))
    print(a.backend, res["mean_image_auroc"], res["mean_pixel_auroc"], res["mean_recall_at_val_thr"],
          lat, res["peak_rss_mb"], flush=True)
