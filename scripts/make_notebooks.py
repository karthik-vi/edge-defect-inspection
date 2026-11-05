"""Generates notebooks/*.ipynb (kept as code so they are reviewable). Notebooks are NOT executed here."""
import json

BANNER = ("> **Not executed in this repo's CPU container.** Written to run on Colab (Runtime -> GPU). "
          "No numbers in this notebook were produced by the author; fill the results table from your run "
          "and record hardware/date in `results/RESUME_METRICS.md`.")


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "accelerator": "GPU"}}


SETUP = """
!git clone -q https://github.com/karthik-vi/edge-defect-inspection.git repo
%cd repo
!pip -q install torch torchvision faiss-cpu opencv-python-headless scikit-learn pyyaml pydantic-settings
!pip -q install -e . --no-deps
!bash scripts/download_data.sh   # 3 MVTec AD categories only (CC BY-NC-SA 4.0, non-commercial)
"""

SUP = [
    ("markdown", "# Supervised segmentation vs. PatchCore: when is it worth labelling defects?\n\n" + BANNER + """

**Question:** PatchCore needs 0 defect labels. A supervised segmenter needs masks. At 0 / 5 / 20 labelled defects per category, which wins on image AUROC and on recall at a 1%-false-reject threshold chosen from good parts only?

**Protocol (same for both):** k defect images (with masks) are taken *from the official test split* for supervised training; all metrics are computed on the **remaining** test images (so the evaluation set is identical for every k and both methods). Threshold = 99% quantile of scores of 40 held-out good *train* images. Supervised training also uses 100 good train images as negatives."""),
    ("code", SETUP),
    ("code", """
import numpy as np, cv2, torch, json, random
from pathlib import Path
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from edgeinspect.config import load_settings
from edgeinspect.data import list_train, list_test
from edgeinspect.metrics import image_auroc, recall_at_threshold, threshold_from_good
from edgeinspect.patchcore import Extractor, PatchCore

cfg = load_settings(); dev = "cuda" if torch.cuda.is_available() else "cpu"; S = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def load(p):  # image -> normalised CHW tensor
    x = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())

def mask(cat, p, lab):
    if not lab: return np.zeros((S, S), np.float32)
    m = cv2.imread(str(Path(cfg.data_root) / cat / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"), 0)
    return (cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
"""),
    ("code", """
def run_category(cat, ks=(0, 5, 20), epochs=30, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = list_train(cfg.data_root, cat); idx = rng.permutation(len(tr))
    val = [tr[i] for i in idx[:40]]; fit = [tr[i] for i in idx[40:140]]
    test = list_test(cfg.data_root, cat)
    defects = [t for t in test if t[1] == 1]; order = rng.permutation(len(defects))
    res = {}
    for k in ks:
        lab_idx = set(order[:k].tolist())
        lab = [defects[i] for i in lab_idx]
        held = [t for t in test if t[1] == 0] + [d for i, d in enumerate(defects) if i not in lab_idx]
        y = np.array([t[1] for t in held])
        # --- PatchCore (0 labels; independent of k, evaluated on the same held-out set) ---
        pc = PatchCore(Extractor("resnet18", (2, 3), 256), S, cfg.pc_ratio, seed=seed)
        pc.fit([cv2.imread(str(p)) for p in fit])
        sv = pc.score([cv2.imread(str(p)) for p in val])[0]
        ss = np.concatenate([pc.score([cv2.imread(str(t[0])) for t in held[i:i+16]])[0] for i in range(0, len(held), 16)])
        thr = threshold_from_good(sv, 0.01); rec, frr = recall_at_threshold(y, ss, thr)
        res[f"patchcore_k{k}"] = dict(auroc=image_auroc(y, ss), recall=rec, frr=frr)
        if k == 0: continue   # a supervised model cannot be trained with 0 defects
        # --- supervised LRASPP-MobileNetV3 (torchvision, BSD) ---
        X = [load(t[0]) for t in lab] * 5 + [load(p) for p in fit]            # oversample the few defects
        M = [torch.from_numpy(mask(cat, t[0], 1)) for t in lab] * 5 + [torch.zeros(S, S)] * len(fit)
        net = lraspp_mobilenet_v3_large(weights_backbone="IMAGENET1K_V1", num_classes=1).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3); net.train()
        for ep in range(epochs):
            for b in torch.randperm(len(X)).split(8):
                x = torch.stack([X[i] for i in b]).to(dev); m = torch.stack([M[i] for i in b]).to(dev)[:, None]
                if random.random() < .5: x, m = x.flip(-1), m.flip(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)["out"], m, pos_weight=torch.tensor(10., device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        def score(paths):
            out = []
            with torch.no_grad():
                for i in range(0, len(paths), 16):
                    x = torch.stack([load(p) for p in paths[i:i+16]]).to(dev)
                    out += torch.sigmoid(net(x)["out"]).amax((1, 2, 3)).cpu().tolist()
            return np.array(out)
        sv = score(val); ss = score([t[0] for t in held])
        thr = threshold_from_good(sv, 0.01); rec, frr = recall_at_threshold(y, ss, thr)
        res[f"supervised_k{k}"] = dict(auroc=image_auroc(y, ss), recall=rec, frr=frr)
    return res

results = {c: run_category(c) for c in cfg.categories}
Path("results").mkdir(exist_ok=True); Path("results/supervised_vs_anomaly.json").write_text(json.dumps(results, indent=1))
"""),
    ("code", """
import pandas as pd
rows = [dict(category=c, method=k, **v) for c, r in results.items() for k, v in r.items()]
pd.DataFrame(rows).round(3).pivot(index="category", columns="method", values=["auroc", "recall"])
"""),
    ("markdown", """## Results (fill in after running)
| category | labelled defects k | PatchCore AUROC / recall | supervised AUROC / recall |
|---|---|---|---|
| | 0 | | n/a |
| | 5 | | |
| | 20 | | |

**How to read it:** the crossover k (if any) is the number of labelled defects where supervised beats PatchCore on recall at a fixed false-reject rate. Caveats: one seed, a small segmenter, 30 epochs, k defects chosen at random; repeat over seeds before quoting."""),
]

TRT = [
    ("markdown", "# ONNX -> TensorRT (FP16 / INT8) on a T4 vs the CPU numbers\n\n" + BANNER + """

Compares per-part latency/throughput and cost per 1,000 parts of the PatchCore feature extractor on a Colab T4 (ONNX Runtime CUDA and TensorRT execution providers) with the CPU results in `results/edge_summary.md`. FAISS kNN search stays on CPU here (as in the CPU benchmark); say so when you quote numbers."""),
    ("code", SETUP.replace("!pip -q install torch", "!pip -q install onnxruntime-gpu onnx tensorrt torch") ),
    ("code", """
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python scripts/export_onnx.py 32            # models/r18_fp32.onnx and models/r18_int8.onnx (QDQ)
"""),
    ("code", """
import time, numpy as np, onnxruntime as ort
print(ort.get_available_providers())
def session(path, kind):
    opts = {"fp32_cuda": [("CUDAExecutionProvider", {})],
            "trt_fp16": [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                        "trt_engine_cache_path": "trt_cache_fp16"})],
            "trt_int8": [("TensorrtExecutionProvider", {"trt_int8_enable": True, "trt_fp16_enable": True,
                                                        "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache_int8"})]}[kind]
    return ort.InferenceSession(path, providers=opts + [("CPUExecutionProvider", {})])

def bench(sess, bs, n=200, warm=20):
    x = np.random.rand(bs, 3, 224, 224).astype(np.float32); ms = []
    for i in range(n + warm):
        t = time.perf_counter(); sess.run(None, {"image": x}); ms.append((time.perf_counter() - t) * 1e3)
    ms = np.array(ms[warm:]); return {"bs": bs, "p50": np.percentile(ms, 50), "p95": np.percentile(ms, 95), "p99": np.percentile(ms, 99),
                                     "parts_per_s": bs * 1e3 / np.percentile(ms, 50)}
rows = []
for name, path in [("fp32_cuda", "models/r18_fp32.onnx"), ("trt_fp16", "models/r18_fp32.onnx"), ("trt_int8", "models/r18_int8.onnx")]:
    s = session(path, name)
    for bs in (1, 8): rows.append({"backend": name, **bench(s, bs)})
import pandas as pd; df = pd.DataFrame(rows).round(2); df
"""),
    ("code", """
# Cost per 1,000 parts. PRICE IS YOUR INPUT (instance $/hour for the T4 you used); it is not a measurement.
T4_USD_PER_HOUR = None   # e.g. your cloud's on-demand T4 price
if T4_USD_PER_HOUR:
    df["usd_per_1k_parts"] = 1000 / df.parts_per_s / 3600 * T4_USD_PER_HOUR
df
"""),
    ("code", """
# Accuracy check for the INT8 engine: rerun the repo's evaluation with this session as the extractor
# (see scripts/bench_edge.py, OrtExt). Record image/pixel AUROC next to the CPU INT8 numbers.
"""),
    ("markdown", """## Results (fill in after running)
| backend | batch | p50 / p99 ms | parts/s | $/1k parts | image AUROC |
|---|---|---|---|---|---|
| CPU ORT INT8, 4 threads (from `results/edge_summary.md`) | 1 | 25 / 37 | 40 | | 0.968 |
| T4 CUDA FP32 | 1 | | | | |
| T4 TensorRT FP16 | 1 | | | | |
| T4 TensorRT INT8 | 1 | | | | |

Decision to record in `docs/DECISIONS.md`: does a GPU pay off at the line's actual rate (parts/s)? A T4 is idle most of the time at one part every few seconds; the CPU result may win on cost."""),
]

for name, cells in (("supervised_vs_anomaly", SUP), ("tensorrt_benchmark", TRT)):
    with open(f"notebooks/{name}.ipynb", "w") as f:
        json.dump(nb(cells), f, indent=1)
