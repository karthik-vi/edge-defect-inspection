.PHONY: setup download test lint bench-classical bench-patchcore
setup:
	pip install -e . pytest ruff
download:
	scripts/download_data.sh
lint:
	python3 -m ruff check .
test:
	python3 -m pytest -q
bench-patchcore:
	scripts/run_phase2.sh
bench-classical:
	python3 scripts/bench_classical.py
