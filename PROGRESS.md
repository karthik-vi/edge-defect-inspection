# PROGRESS
## Done
- Phase 0: skeleton, Makefile, CI (ruff+pytest), config.yaml, docs, `make download` (bottle/metal_nut/screw from mydrive.ch via mvtec.com links; MVTec was NOT blocked; ~480 MB, gitignored).
- Phase 1: classical baseline (`src/edgeinspect/classical.py`), `make bench-classical`, alignment ablation, ADR-001, results/RESUME_METRICS.md. Mean AUROC 0.716 (bottle 0.985, metal_nut 0.637, screw 0.526); ~1.2 ms/part p50.
## In progress
- Nothing. Phase 1 acceptance (metrics + latency + tests) met.
## Next step
- Phase 2: PatchCore (ResNet18 first) — needs torch, torchvision, faiss-cpu (not installed; record each in DECISIONS.md). Reuse `data.py`, `metrics.py`, threshold-from-val-good. Start with bottle only to size runtime (<15 min per run).
## Blockers
- None. Docker/GPU/TensorRT unavailable: label "not verified in this environment".
## How to resume
`make setup && make download && make test && make bench-classical`; read docs/PROJECT_BRIEF.md.
Notes: use `python3 -m pytest`; val threshold from 20 good images is noisy (consider more val-good in Phase 2).
