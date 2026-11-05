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
