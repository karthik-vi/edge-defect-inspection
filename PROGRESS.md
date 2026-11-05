# PROGRESS
## Done
- Phase 0: skeleton, Makefile, CI, config.yaml, docs, data download script (bottle/metal_nut/screw from mydrive.ch via mvtec.com links; ~480MB, gitignored).
## In progress
- Phase 1 classical baseline.
## Next step
- Implement src/edgeinspect/classical.py + scripts/bench_classical.py, tests, record results.
## Blockers
- None. (Docker/GPU not available: not verified here.)
## How to resume
`make setup && make download && make test`; read docs/PROJECT_BRIEF.md.
