"""Phase 4: inject synthetic drift at increasing severity; compare the drift monitor with the
operational metrics (recall and false-reject rate at the fixed deployed threshold).

Per category: shuffled official test split (good + defect), windows of `drift_window` images.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from edgeinspect.config import load_settings
from edgeinspect.data import list_test
from edgeinspect.engine import Engine
from edgeinspect.metrics import image_auroc, recall_at_threshold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIFTS = {  # name -> (label, 6 severity levels, fn(img, level, rng))
    "lighting": ("dimming gain", [1.0, 0.94, 0.88, 0.82, 0.7, 0.55],
                 lambda im, v, r: np.clip(im.astype(np.float32) * v, 0, 255).astype(np.uint8)),
    "defocus": ("blur sigma (px @1024)", [0, 0.5, 1, 2, 3, 5],
                lambda im, v, r: im if v == 0 else cv2.GaussianBlur(im, (0, 0), v)),
    "camera_shift": ("shift (px @1024)", [0, 4, 8, 16, 32, 64],
                     lambda im, v, r: cv2.warpAffine(im, np.float32([[1, 0, v], [0, 1, v / 2]]),
                                                     im.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)),
    "sensor_noise": ("noise sigma (0-255)", [0, 2, 4, 8, 16, 32],
                     lambda im, v, r: np.clip(im + r.normal(0, v, im.shape), 0, 255).astype(np.uint8)
                     if v else im),
}


def evaluate(eng: Engine, imgs, labels, fn, level, cfg) -> dict:
    rng = np.random.default_rng(cfg.seed)
    sc, stats, emb = [], [], []
    for im in imgs:
        v = eng.inspect(fn(im, level, rng))
        sc.append(v.score)
        stats.append(v.stats)
        emb.append(v.embedding)
    sc, stats, emb = np.array(sc), np.stack(stats), np.stack(emb)
    rec, frr = recall_at_threshold(labels, sc, eng.thr)
    w = cfg.drift_window
    reps = [eng.monitor.check(stats[i:i + w], emb[i:i + w]) for i in range(0, len(sc) - w + 1, w)]
    return {"recall": rec, "frr": frr, "auroc": image_auroc(labels, sc),
            "drift_score": float(np.mean([r.score for r in reps])),
            "alert_frac": float(np.mean([r.alert for r in reps])), "n_windows": len(reps)}


if __name__ == "__main__":
    cfg = load_settings()
    cats = ["bottle", "metal_nut"]  # screw baseline recall is 0.45: no room to show degradation
    res: dict = {"date": datetime.now(UTC).date().isoformat(), "window": cfg.drift_window,
                 "z_thr": cfg.drift_z, "alpha": cfg.drift_alpha, "cats": {}}
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    for cat in cats:
        eng = Engine.load(cat, cfg)
        test = list_test(cfg.data_root, cat)
        order = np.random.default_rng(0).permutation(len(test))
        imgs = [cv2.imread(str(test[i][0])) for i in order]
        labels = np.array([test[i][1] for i in order])
        res["cats"][cat] = {}
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=False)
        for ax, (name, (lab, levels, fn)) in zip(axes, DRIFTS.items(), strict=True):
            rows = [evaluate(eng, imgs, labels, fn, lv, cfg) for lv in levels]
            base = rows[0]
            first_alert = next((i for i, r in enumerate(rows) if r["alert_frac"] >= 0.5), None)
            first_deg = next((i for i, r in enumerate(rows)
                              if r["recall"] <= base["recall"] - 0.10 or r["frr"] >= base["frr"] + 0.05), None)
            res["cats"][cat][name] = {"levels": levels, "rows": rows, "first_alert_level": first_alert,
                                      "first_degraded_level": first_deg}
            x = np.arange(len(levels))
            ax.plot(x, [r["recall"] for r in rows], "o-", c="tab:blue", label="recall @ fixed thr")
            ax.plot(x, [r["frr"] for r in rows], "s-", c="tab:red", label="false-reject rate")
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks(x, [str(v) for v in levels], fontsize=7)
            ax.set_xlabel(lab, fontsize=8)
            ax2 = ax.twinx()
            ax2.plot(x, [r["drift_score"] for r in rows], "^--", c="tab:green", label="drift score")
            ax2.axhline(1.0, c="tab:green", lw=0.8, ls=":")
            ax2.set_yscale("log")
            for lvl, col in ((first_alert, "tab:green"), (first_deg, "k")):
                if lvl is not None:
                    ax.axvline(lvl, c=col, lw=1, alpha=0.5)
            ax.set_title(f"{cat}: {name}", fontsize=9)
            if name == "lighting":
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="center left")
            print(cat, name, "alert@", first_alert, "degraded@", first_deg, flush=True)
        fig.suptitle("Green line = first level where the monitor alerts (>=50% windows); black = first "
                     "level where recall drops >=0.10 or FRR rises >=0.05", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"results/plots/drift_{cat}.png", dpi=90)
    Path("results/drift.json").write_text(json.dumps(res, indent=1))
