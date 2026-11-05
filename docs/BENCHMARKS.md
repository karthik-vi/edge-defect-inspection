# Benchmark methodology
- Threshold chosen from held-out *good train* images only (no defect labels at setup): score threshold = (1-FRR) quantile of val-good scores.
- Reported: image AUROC, defect recall at the 1% false-reject threshold (applied to test), latency p50/p95/p99 per part, single process.
- Test set = official MVTec test split (good + defect).
