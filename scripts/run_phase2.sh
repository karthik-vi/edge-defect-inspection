#!/usr/bin/env bash
# Phase 2 main run + ablations (sequential so latency numbers are not contaminated). ~15-25 min on 4 cores.
set -u
B="python3 scripts/bench_patchcore.py"
$B --tag r18_main
$B --tag r18_layer2 --layers 2
$B --tag r18_layer3 --layers 3
$B --tag r18_coreset001 --ratio 0.01
$B --tag r18_coreset005 --ratio 0.05
$B --tag r18_nocoreset --ratio 1.0
$B --tag r18_align_translate --align translate
$B --tag r18_align_rotate --align rotate
$B --tag wrn50_main --backbone wide_resnet50_2
