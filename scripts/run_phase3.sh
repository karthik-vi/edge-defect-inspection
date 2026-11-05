#!/usr/bin/env bash
# Phase 3: export ONNX (+INT8 PTQ with good-part calibration) and benchmark all backends sequentially.
set -u
[ -f models/r18_int8.onnx ] || python3 scripts/export_onnx.py 32
for b in torch ort_fp32 ort_int8 ov_fp32 ov_int8; do python3 scripts/bench_edge.py --backend $b; done
