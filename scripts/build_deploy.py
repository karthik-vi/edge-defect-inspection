"""Build per-category deployment artifacts (bank, threshold, drift reference) and log to MLflow.

The MLflow run id is the model version; every service verdict reports it, so a verdict can be
traced back to the run (params, metrics, artifacts).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import mlflow
import numpy as np

from edgeinspect.config import load_settings
from edgeinspect.data import list_test, list_train
from edgeinspect.drift import image_stats
from edgeinspect.metrics import image_auroc, recall_at_threshold, threshold_from_good
from edgeinspect.patchcore import PatchCore
from edgeinspect.runtime import OrtExtractor

if __name__ == "__main__":
    cfg = load_settings()
    out = Path(cfg.deploy_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("edge-inspection")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    ext = OrtExtractor(cfg.onnx_model, cfg.threads)
    with mlflow.start_run(run_name="deploy-ort-int8") as run:
        mlflow.log_params({"backbone": "resnet18", "layers": "2,3", "runtime": "onnxruntime-int8",
                           "coreset_ratio": cfg.pc_ratio, "pc_train_max": cfg.pc_train_max,
                           "pc_val_good": cfg.pc_val_good, "git": git})
        for cat in cfg.categories:
            rng = np.random.default_rng(cfg.seed)
            tr = list_train(cfg.data_root, cat)
            idx = rng.permutation(len(tr))
            val = [cv2.imread(str(tr[i])) for i in idx[: cfg.pc_val_good]]
            fit = [cv2.imread(str(tr[i])) for i in idx[cfg.pc_val_good: cfg.pc_val_good + cfg.pc_train_max]]
            pc = PatchCore(ext, cfg.pc_size, cfg.pc_ratio, seed=cfg.seed)
            pc.fit(fit)
            sv, _, ev = pc.infer(val)
            thr = threshold_from_good(sv, cfg.false_reject_rate)
            ref = fit + val
            ref_emb = np.concatenate([pc.infer(ref[k:k + 16])[2] for k in range(0, len(ref), 16)])
            ref_stats = np.stack([image_stats(i) for i in ref])
            np.savez(out / f"{cat}.npz", bank=pc.bank, thr=thr, ref_stats=ref_stats, ref_emb=ref_emb)
            test = list_test(cfg.data_root, cat)
            sc = np.array([pc.infer([cv2.imread(str(p))])[0][0] for p, _, _ in test])
            lab = np.array([t[1] for t in test])
            rec, frr = recall_at_threshold(lab, sc, thr)
            mlflow.log_metrics({f"{cat}_image_auroc": image_auroc(lab, sc), f"{cat}_threshold": thr,
                                f"{cat}_recall": rec, f"{cat}_test_frr": frr, f"{cat}_bank_mb": pc.bank_mb})
            print(cat, round(image_auroc(lab, sc), 4), round(rec, 3), round(frr, 3), flush=True)
            mlflow.log_artifact(str(out / f"{cat}.npz"))
        mlflow.set_tag("deployed", "true")
        (out / "version.json").write_text(json.dumps({"model_version": run.info.run_id[:12],
                                                      "mlflow_run_id": run.info.run_id, "git": git}))
