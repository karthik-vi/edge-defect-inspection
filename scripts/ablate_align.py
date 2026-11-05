"""Phase 1 ablation: alignment on/off/rotation-threshold (writes results/classical_align_ablation.json)."""
import functools
import json

import cv2

import bench_classical as b
from edgeinspect import classical
from edgeinspect.config import load_settings

cv2.setNumThreads(1)
cfg = load_settings()
orig = classical.align
res = {}
for name, fn in {
    "none": lambda g: g,
    "translate_only": functools.partial(orig, ecc_min=1e9),
    "translate+rotate(ecc>1.3)": orig,
}.items():
    b.align = fn
    res[name] = {c: {k: v for k, v in b.run(c, cfg).items() if k in ("image_auroc", "recall_at_val_thr")}
                 for c in cfg.categories}
    print(name, res[name])
with open(f"{cfg.results_dir}/classical_align_ablation.json", "w") as f:
    json.dump(res, f, indent=1)
