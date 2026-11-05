.PHONY: setup download test lint bench-classical
setup:
	pip install -e . pytest ruff
download:
	scripts/download_data.sh
lint:
	ruff check .
test:
	pytest -q
bench-classical:
	python scripts/bench_classical.py
