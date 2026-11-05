# PROGRESS
## Done
- Phase 0: skeleton, Makefile, CI (ruff+pytest, CPU torch), config.yaml, docs, `make download` (3 MVTec categories; MVTec not blocked).
- Phase 1: classical baseline; mean AUROC 0.716 (ADR-001).
- Phase 2: PatchCore (`src/edgeinspect/patchcore.py`, FAISS, greedy coreset), pixel AUROC/PRO, 9 runs (`make bench-patchcore`, ~15 min): backbone, layers, coreset, alignment ablations (ADR-002/003). Mean img AUROC 0.972, pixel 0.983, PRO 0.932, 34 ms p50.
## In progress
- Nothing. Phase 2 acceptance met (metrics per category, ablations, threshold from val-good documented).
## Next step
- Phase 3 (edge optimisation): export ResNet18 extractor (layers 2+3) to ONNX; benchmark PyTorch vs ONNX Runtime (already installed) vs OpenVINO (needs `pip install openvino`, record in DECISIONS.md); INT8 PTQ with good-part calibration; 1/2/4 thread latency; accuracy change from INT8; deployment table. Use coreset 5%. Do not add a heatmap-example plot yet (README still lacks heatmap images and plots/).
## Blockers
- None. Docker/GPU/TensorRT unavailable: label "not verified in this environment".
## How to resume
`make setup && make download && make test`; `make bench-classical bench-patchcore`; read docs/PROJECT_BRIEF.md.
Caveats: val threshold from 40 good images is noisy (metal_nut test FRR 9%); alignment-ablation pixel metrics invalid (masks not warped); 1 seed; layer2-only+5% coreset combination untested.
