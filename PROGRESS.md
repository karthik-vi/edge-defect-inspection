# PROGRESS
## Done
- Phase 0: skeleton, Makefile, CI (ruff+pytest, CPU torch), config.yaml, docs, `make download` (3 MVTec categories; MVTec not blocked).
- Phase 1: classical baseline; mean AUROC 0.716 (ADR-001).
- Phase 2: PatchCore (`src/edgeinspect/patchcore.py`, FAISS, greedy coreset), pixel AUROC/PRO, 9 runs (`make bench-patchcore`, ~15 min): backbone, layers, coreset, alignment ablations (ADR-002/003). Mean img AUROC 0.972, pixel 0.983, PRO 0.932, 34 ms p50.
- Phase 3: ONNX export, INT8 PTQ, PyTorch/ORT/OpenVINO benchmark at 1/2/4 threads (ADR-004). ORT INT8: AUROC −0.0018, model 3.1 MB, 25 ms p50 @4 threads.
- Phase 4: FastAPI service, MLflow-tracked deploy build, drift monitor + synthetic-drift experiment (ADR-005/006), Prometheus alerts, Dockerfile/compose (unverified: no daemon), README rewritten.
- Phase 5: two Colab notebooks written (ADR-007) but NOT executed (no GPU here): `notebooks/*.ipynb`, no numbers claimed.
## In progress
- Nothing. All phases (0-5) are done within what can be verified on CPU.
## Next step (for the user)
- Run the two notebooks on Colab (T4) and fill their result tables + `results/RESUME_METRICS.md`; verify `docker compose up` and Prometheus rules on a machine with a Docker daemon.
- Improve screw (recall 0.45): higher resolution / more good images / per-category thresholds with more validation parts; re-check drift thresholds on an independent clean set.
## Blockers
- None. Not verified here: Docker build/run, Prometheus scraping, GPU/TensorRT, real edge hardware.
## How to resume
`make setup download test`; `make bench-classical bench-patchcore bench-edge build-deploy bench-drift`; `make serve`. Read docs/DECISIONS.md (ADR-001..007) and results/RESUME_METRICS.md.
Caveats: 1 seed, 3 categories; thresholds from 40 good parts are noisy; drift thresholds tuned once on the same data (ADR-006); alignment-ablation pixel metrics invalid (masks not warped).
