# edge-defect-inspection
Inspection station that learns from **good parts only**, runs on CPU, explains decisions with heatmaps and detects camera drift. Spec: `docs/PROJECT_BRIEF.md`; status: `PROGRESS.md`.

## Results so far (MVTec AD: bottle, metal_nut, screw; CPU, 1 thread)
| method | bottle AUROC | metal_nut AUROC | screw AUROC | latency p50 |
|---|---|---|---|---|
| Classical golden-image baseline (Phase 1) | 0.985 | 0.637 | 0.526 | ~1.2 ms |
| PatchCore / INT8-ONNX | not yet run | | | |

Reproduce: `make setup download bench-classical`. Details, hardware and caveats: `results/RESUME_METRICS.md`; decisions: `docs/DECISIONS.md`. Docker/GPU: not verified in this environment.
