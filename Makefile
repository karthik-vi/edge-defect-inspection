.PHONY: setup download test lint bench-classical
setup:
	pip install -e . pytest ruff
download:
	scripts/download_data.sh
lint:
	python3 -m ruff check .
test:
	python3 -m pytest -q
bench-classical:
	python3 scripts/bench_classical.py
