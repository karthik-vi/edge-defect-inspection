# edge-defect-inspection
Inspection station that learns from **good parts only**, runs on CPU, explains decisions with heatmaps and detects camera drift. Spec: `docs/PROJECT_BRIEF.md`; status: `PROGRESS.md`.

## Results so far (MVTec AD: bottle, metal_nut, screw; CPU, 4 threads, batch 1)
| method | bottle AUROC | metal_nut AUROC | screw AUROC | mean pixel AUROC / PRO | recall @ val-good threshold (bottle/nut/screw) | latency p50 / p99 |
|---|---|---|---|---|---|---|
| Classical golden-image baseline (Phase 1) | 0.985 | 0.637 | 0.526 | n/a | 0.95 / 0.35 / 0.03 | 1.2 / 1.3 ms |
| PatchCore ResNet18, 10% coreset (Phase 2) | 1.000 | 0.992 | 0.923 | 0.983 / 0.932 | 1.00 / 0.99 / 0.35 | 34 / 54 ms |
| PatchCore, 5% coreset | 1.000 | 0.992 | 0.917 | 0.983 / 0.932 | see `results/patchcore_summary.md` | 24 / 39 ms |
| PatchCore ONNX Runtime INT8, 5% coreset (Phase 3) | see `docs/DECISIONS.md` ADR-004 (mean 0.9678 vs 0.9696 FP32) | | | 0.983 | 0.81 mean | 25 / 37 ms (4 thr) |

Findings (details and caveats in `docs/DECISIONS.md`): the coreset gives the same accuracy as the full bank at 6.5x lower latency and 10x less memory; WideResNet50 and Phase 1 alignment did not improve results; screw defects are the weak point (recall 0.35 at a threshold that rejects 0% of good parts). Single seed, 3 categories, so small differences are within noise.

Reproduce: `make setup download bench-classical bench-patchcore`. Hardware and commands: `results/RESUME_METRICS.md`. Docker/GPU: not verified in this environment.
