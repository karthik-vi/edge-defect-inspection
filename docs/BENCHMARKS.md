# Benchmark methodology
- Threshold chosen from held-out *good train* images only (no defect labels at setup): score threshold = (1-FRR) quantile of val-good scores.
- Reported: image AUROC, defect recall at the 1% false-reject threshold (applied to test), latency p50/p95/p99 per part, single process.
- Test set = official MVTec test split (good + defect).

## PatchCore / edge / drift (Phases 2-4)
- PatchCore: bank from 100 random good train images (seed 0), threshold from 40 other good train images; image latency = feature extraction + FAISS search + map blur, batch 1, image decode excluded. Pixel AUROC at 224x224 against nearest-resized masks; PRO = area under region-overlap vs FPR curve for FPR<=0.3 (100 thresholds).
- Edge: accuracy over the full test split at 4 threads; latency at 1/2/4 threads on 40 random test images per category after 3 warm-up calls; FAISS threads = backend threads. Thread-limited emulation on a 4-vCPU cloud VM, not real edge hardware.
- Drift: stream = 60 held-out good parts (train parts unused for bank/threshold/reference + test goods), windows of 20; recall on 40 test defects; synthetic perturbations at 6 severities applied at original resolution. Operational degradation = recall -0.10 or FRR +0.05 vs severity 0.
- All timing runs are sequential (no concurrent jobs) to avoid contaminating latency.
