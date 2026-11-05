# PROGRESS
## Done
- Phase 0: skeleton, Makefile, CI (ruff+pytest, CPU torch), config.yaml, docs, `make download` (3 MVTec categories; MVTec not blocked).
- Phase 1: classical baseline; mean AUROC 0.716 (ADR-001).
- Phase 2: PatchCore (`src/edgeinspect/patchcore.py`, FAISS, greedy coreset), pixel AUROC/PRO, 9 runs (`make bench-patchcore`, ~15 min): backbone, layers, coreset, alignment ablations (ADR-002/003). Mean img AUROC 0.972, pixel 0.983, PRO 0.932, 34 ms p50.
- Phase 3: ONNX export, INT8 PTQ, PyTorch/ORT/OpenVINO benchmark at 1/2/4 threads (ADR-004). ORT INT8: AUROC −0.0018, model 3.1 MB, 25 ms p50 @4 threads.
- Phase 4: FastAPI service, MLflow-tracked deploy build, drift monitor + synthetic-drift experiment (ADR-005/006), Prometheus alerts, Dockerfile/compose (unverified: no daemon), README rewritten.
## In progress
- Phase 5 (Colab notebooks, Could).
## Next step
- Phase 3 (edge optimisation): export ResNet18 extractor (layers 2+3) to ONNX; benchmark PyTorch vs ONNX Runtime (already installed) vs OpenVINO (needs `pip install openvino`, record in DECISIONS.md); INT8 PTQ with good-part calibration; 1/2/4 thread latency; accuracy change from INT8; deployment table. Use coreset 5%. Do not add a heatmap-example plot yet (README still lacks heatmap images and plots/).
## Blockers
- None. Docker/GPU/TensorRT unavailable: label "not verified in this environment".
## How to resume
`make setup && make download && make test`; `make bench-classical bench-patchcore`; read docs/PROJECT_BRIEF.md.
Caveats: val threshold from 40 good images is noisy (metal_nut test FRR 9%); alignment-ablation pixel metrics invalid (masks not warped); 1 seed; layer2-only+5% coreset combination untested.
