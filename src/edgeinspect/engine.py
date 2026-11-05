"""Inspection engine: one trained category = memory bank + threshold + drift reference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edgeinspect.drift import DriftMonitor, image_stats
from edgeinspect.patchcore import PatchCore
from edgeinspect.runtime import OrtExtractor


@dataclass
class Verdict:
    score: float
    threshold: float
    reject: bool
    anomaly_map: np.ndarray
    stats: np.ndarray
    embedding: np.ndarray


def overlay_png(bgr: np.ndarray, amap: np.ndarray, thr: float, alpha: float = 0.5) -> bytes:
    """Heatmap overlay for the operator: colour scale is relative to the reject threshold
    (full red = 2x threshold)."""
    h, w = bgr.shape[:2]
    norm = np.clip(cv2.resize(amap, (w, h)) / (2.0 * thr), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


class Engine:
    def __init__(self, category: str, pc: PatchCore, thr: float, version: str,
                 monitor: DriftMonitor) -> None:
        self.category, self.pc, self.thr, self.version, self.monitor = (
            category, pc, thr, version, monitor)

    def inspect(self, bgr: np.ndarray) -> Verdict:
        s, m, e = self.pc.infer([bgr])
        return Verdict(float(s[0]), self.thr, bool(s[0] > self.thr), m[0], image_stats(bgr), e[0])

    @classmethod
    def load(cls, category: str, cfg) -> Engine:
        d = Path(cfg.deploy_dir)
        z = np.load(d / f"{category}.npz")
        pc = PatchCore(OrtExtractor(cfg.onnx_model, cfg.threads), cfg.pc_size, cfg.pc_ratio)
        pc.set_bank(z["bank"])
        ver = json.loads((d / "version.json").read_text())["model_version"]
        mon = DriftMonitor(z["ref_stats"], z["ref_emb"], cfg.drift_z, cfg.drift_alpha)
        return cls(category, pc, float(z["thr"]), ver, mon)
