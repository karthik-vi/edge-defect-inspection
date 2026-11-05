# Decisions (ADR: Context -> Options -> Decision -> Evidence)
Phase entries are added as each component is measured.

## ADR-001: Classical baseline alignment (Phase 1)
- **Context:** Phase 1 baseline = golden-image rules (pixel z-diff vs mean/std golden image, per-cell edge density, per-cell Laplacian variance; max of val-normalised rule scores). Alignment was meant to reduce pose variation.
- **Options:** (a) none, (b) centroid translation, (c) centroid translation + principal-axis rotation (only if eccentricity > 1.3).
- **Decision:** `align_mode: none` as the fixed baseline config.
- **Evidence** (image AUROC, 20 golden + 20 val-good, MVTec test split; `results/classical_align_ablation.json`):

| align | bottle | metal_nut | screw |
|---|---|---|---|
| none | 0.985 | 0.637 | 0.526 |
| translate | 0.960 | 0.613 | 0.569 |
| translate+rotate | 0.960 | 0.613 | **0.103** |

  Moment-based rotation is unstable (180° ambiguity, near-symmetric nut) and inverts the screw result; centroid translation did not help on 2/3 categories. Alignment is therefore *not* earning its place in the classical pipeline; it is re-tested in Phase 2 on learned features.
  Caveat: the mode was compared on the test split with 3 configs; "none" is the best average (0.716), not tuned beyond that.

## ADR-002: PatchCore (ResNet18, FAISS coreset) vs the classical baseline (Phase 2)
- **Problem / metric:** the classical baseline cannot handle pose variation or subtle texture defects (mean image AUROC 0.716, screw ~chance). Metric: image AUROC, pixel AUROC, PRO, recall at the val-good-derived threshold.
- **Decision:** PatchCore-style memory bank (ResNet18 layers 2+3, 3x3 local aggregation, 256-d, kNN distance via FAISS `IndexFlatL2`), fit on good images only (100 per category; 40 other good train images held out for the threshold).
- **Evidence** (`results/patchcore_summary.md`, 3 categories, 1 seed, test split): mean image AUROC **0.972** (bottle 1.000, metal_nut 0.992, screw 0.923) vs 0.716 classical; pixel AUROC 0.983, PRO 0.932; ~34 ms/part p50 on 4 threads vs ~1.2 ms classical. The accuracy costs ~30x latency; the classical rules are still the cheaper check where they work (bottle 0.985).
- **Dependencies justified:** torch/torchvision (pretrained ResNet, BSD), faiss-cpu (kNN search, MIT) — each is used by the core model.

## ADR-003: Ablations — what earned its place
All numbers are means over bottle/metal_nut/screw, one seed; differences < ~0.01 AUROC are within noise (screw has 160 test images, 1 seed).
| ablation | finding | verdict |
|---|---|---|
| Coreset ratio 1% / 5% / 10% / 100% | AUROC 0.932 / 0.970 / 0.972 / 0.973; latency p50 15 / 24 / 34 / 220 ms; bank 0.8 / 3.8 / 7.7 / 76.6 MB | **Coreset earns its place:** 10% gives the same accuracy as no coreset at 6.5x lower latency and 10x less memory. 5% loses 0.002 AUROC for -30% latency. 1% breaks screw (0.81). **Choose 5%.** |
| Layers 2 / 3 / 2+3 | 0.978 / 0.922 / 0.972 | Layer 3 alone is bad (screw 0.78) but 2x faster. Layer 2 alone was at least as good as 2+3 on every category (differences ≤0.02, not significant on 1 seed) — **candidate, not yet verified in combination with 5% coreset.** Keep 2+3 as default. |
| Backbone ResNet18 vs WideResNet50 | 0.972 vs 0.954 AUROC; 34 vs 70 ms | **WRN50 does not pay off here:** slower and not better on these 3 categories (worse on screw: 0.874). Use ResNet18. (Both reduced to 256-d, so bank size is equal.) |
| Phase 1 alignment (none / translate / translate+rotate) | image AUROC 0.972 / 0.966 / 0.975 | **Alignment does not earn its place:** +0.003 for rotate is noise and only helps screw (0.923 → 0.934); translate is worse. Note: ground-truth masks were *not* warped with the images, so pixel AUROC/PRO of the aligned variants (0.876/0.975 …) are invalid and ignored. |

**Threshold selection (documented):** the threshold is the 99% quantile of anomaly scores of 40 held-out *good train* images (no defect labels needed at setup). It gave mean test false-reject 3% (target 1%): bottle 0%, metal_nut 9%, screw 0%. With 40 samples the 99% quantile is essentially the max, so it is noisy; more validation parts would tighten it.
**Known weakness:** recall at this threshold is 1.00 / 0.99 / **0.35** (bottle / nut / screw). Screw AUROC is 0.92 but its defects are small and score only slightly above the worst good parts, so a threshold that rejects 0% of good parts misses most of them. Screw is the honest limit of this model at 224 px.

## ADR-004: Edge optimisation — runtime and INT8 (Phase 3)
- **Problem / metric:** run on a small CPU/edge box; metrics = latency p50/p99 at 1/2/4 threads (*thread-limited emulation, not real Jetson/RPi hardware*), image/pixel AUROC change, model/bank size, cost per 1,000 parts.
- **Options:** PyTorch FP32; ONNX Runtime FP32/INT8; OpenVINO FP32/INT8. INT8 = static post-training quantisation (QDQ, per-channel weights, 32 good train images across the 3 categories as calibration; no test images). Config: ResNet18 layers 2+3, 5% coreset (ADR-003).
- **Evidence** (`results/edge_summary.md`, mean of 3 categories, full test split for accuracy, 40 random test images/category for latency, batch 1; FAISS threads = backend threads):

| backend | img AUROC | pix AUROC | recall@val-thr | p50 ms 1 / 2 / 4 thr | model MB |
|---|---|---|---|---|---|
| PyTorch FP32 (baseline) | 0.9696 | 0.9829 | 0.78 | 71 / 40 / 28 | 11.0 |
| ONNX Runtime FP32 | 0.9696 | 0.9829 | 0.78 | 59 / 33 / 30 | 11.0 |
| **ONNX Runtime INT8** | 0.9678 | 0.9827 | 0.81 | 50 / 29 / 25 | 3.1 |
| OpenVINO FP32 | 0.9696 | 0.9829 | 0.78 | 59 / 33 / 26 | 11.0 |
| OpenVINO INT8 | 0.9643 | 0.9825 | 0.77 | 45 / 25 / 24 | 3.1 |

- **Findings:** (1) ONNX FP32 reproduces PyTorch exactly. (2) INT8 shrinks the model 3.5x (11.0 → 3.1 MB); AUROC change is −0.0018 (ORT) / −0.0053 (OpenVINO), recall at the val threshold is within noise (+0.03 / −0.01). (3) The speed-up is modest: at 4 threads INT8 is ~15% faster than ORT FP32 and about the same as PyTorch FP32 (25 vs 28 ms) because FAISS search and map post-processing are not quantised; at 1 thread, OpenVINO INT8 is 1.6x faster than PyTorch (45 vs 71 ms). (4) At 4 threads on this 4-vCPU box all backends land within 24–30 ms, so the runtime choice matters mostly for *small* (1–2 thread) edge boxes.
- **Decision (deployment config):** ONNX Runtime INT8, ResNet18 layers 2+3, 5% coreset: smallest accuracy change among the INT8 options, 3.1 MB model, single light runtime dependency (OpenVINO is only 2–4 ms faster and costs 0.005 AUROC). Not verified on real edge hardware.
- **Bug found and fixed on the way (worth knowing for interviews):** the first ONNX export traced the pooling-matrix construction element by element (thousands of nodes) and ONNX Runtime output differed 31% from PyTorch; fixing it to be a constant gives an exact match. Second: ORT at 4 threads was 2x *slower* than at 2 threads (65 vs 33 ms) due to thread spinning contention with FAISS' OpenMP threads; `session.intra_op.allow_spinning=0` fixed it (65 → 30 ms).
- **Cost:** 1,000 parts take ~25 s of a 4-vCPU box at 4 threads (40 parts/s); at an *assumed* $0.04/vCPU-hour (placeholder, not measured) that is ~$0.001 per 1,000 parts. Peak RSS in `results/edge/*.json` is for the whole benchmark process (it holds all test images), so it is an upper bound, not the service footprint.
- **Dependencies added:** onnx + onnxruntime (export + inference, MIT), openvino (comparison, Apache-2.0).
