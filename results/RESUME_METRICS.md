# Resume metrics (every number from a run in this repo)
## Phase 1: classical golden-image baseline (no learning beyond mean/std images)
- Command: `make bench-classical` (raw: `results/classical.json`), 2026-11-05
- Hardware: Intel Xeon (virtualised), 4 cores, 15 GB RAM, no GPU; OpenCV 5.0.0 single-thread, Python 3.13
- Data: MVTec AD bottle/metal_nut/screw, official test split (83 / 115 / 160 images), 20 golden + 20 val-good train images, 256x256, align_mode=none
- Threshold from 20 val-good scores (99% quantile, i.e. ~max of 20 -> noisy); oracle = threshold from test goods (reference only)

| category | image AUROC | recall @ val thr | test FRR @ val thr | recall @ 1% FRR (oracle) | latency ms p50/p95/p99 |
|---|---|---|---|---|---|
| bottle | 0.985 | 0.952 | 0.000 | 0.952 | 1.18 / 1.27 / 1.31 |
| metal_nut | 0.637 | 0.355 | 0.045 | 0.290 | 1.22 / 1.35 / 1.39 |
| screw | 0.526 | 0.034 | 0.171 | 0.008 | 1.05 / 1.17 / 1.26 |

Mean AUROC 0.716. Screw is ~chance: global golden-image rules cannot handle randomly rotated parts.

## Phase 2: PatchCore (ResNet18 layers 2+3, FAISS, 10% coreset)
- Command: `make bench-patchcore` (raw: `results/patchcore/*.json`, table: `results/patchcore_summary.md`; `python3 scripts/summarize_phase2.py`), 2026-11-05
- Hardware: Intel Xeon (virtualised), 4 cores, 15 GB RAM, no GPU; PyTorch 2.15.0+cpu, 4 threads, batch size 1, 224x224
- Data: same 3 MVTec AD categories, official test split; bank from 100 good train images (7.7 MB), threshold from 40 other good train images (99% quantile); 1 seed
- Latency = feature extraction + FAISS search + map blur, per part, excludes image decode

| category | img AUROC | pixel AUROC | PRO | recall @ val thr | test FRR | oracle recall @1% FRR |
|---|---|---|---|---|---|---|
| bottle | 1.000 | 0.981 | 0.926 | 1.00 | 0.00 | 1.00 |
| metal_nut | 0.992 | 0.981 | 0.929 | 0.99 | 0.09 | 0.98 |
| screw | 0.923 | 0.987 | 0.940 | 0.35 | 0.00 | 0.54 |
| **mean** | **0.972** (classical 0.716) | 0.983 | 0.932 | 0.78 | 0.03 | |

Latency p50/p95/p99 (mean over categories): 34 / 41 / 54 ms (~29 parts/s on 4 threads) vs classical 1.2 ms. Best-latency config: 5% coreset, 24 / 32 / 39 ms, AUROC 0.970. WideResNet50: 0.954 AUROC at 70 / 83 / 91 ms (not better here).

## Phase 3: edge optimisation (ONNX Runtime INT8, thread-limited emulation, not real Jetson/RPi hardware)
- Command: `make bench-edge` (table: `results/edge_summary.md`, raw: `results/edge/*.json`), 2026-11-05; same 4-vCPU Xeon VM, ResNet18 layers 2+3, 5% coreset, batch 1, 224x224
- INT8 PTQ (ORT static QDQ, 32 good-part calibration images): model 11.0 → 3.1 MB; mean image AUROC 0.9696 (FP32) → 0.9678 (INT8), delta −0.0018; pixel AUROC 0.9829 → 0.9827
- Latency p50 (1 / 2 / 4 threads): PyTorch FP32 71 / 40 / 28 ms; ORT INT8 50 / 29 / 25 ms (p99 at 4 threads 37 ms; ~40 parts/s); OpenVINO INT8 45 / 25 / 24 ms
- Cost per 1k parts uses an *assumed* $0.04/vCPU-hour: ~$0.001 (assumption, not a measurement)
