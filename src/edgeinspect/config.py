"""Single config source: config.yaml (override path with EDGE_CONFIG)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    data_root: str = "data/mvtec_ad"
    categories: list[str] = ["bottle", "metal_nut", "screw"]
    results_dir: str = "results"
    image_size: int = 256
    false_reject_rate: float = 0.01
    seed: int = 0
    n_golden: int = 20
    n_val_good: int = 20
    pc_val_good: int = 40
    pc_train_max: int = 100
    pc_size: int = 224
    pc_ratio: float = 0.05
    threads: int = 4
    deploy_dir: str = "models/deploy"
    onnx_model: str = "models/r18_int8.onnx"
    mlflow_uri: str = "sqlite:///mlflow.db"
    drift_window: int = 20
    drift_z: float = 5.0         # alert if |z| of a window-mean image statistic exceeds this
    drift_alpha: float = 0.001   # alert if Bonferroni-corrected KS p-value on PCA comps is below this
    service_port: int = 8000
    align_mode: str = "none"  # none | translate | rotate


def load_settings(path: str | None = None) -> Settings:
    p = Path(path or os.environ.get("EDGE_CONFIG", "config.yaml"))
    return Settings(**(yaml.safe_load(p.read_text()) if p.exists() else {}))
