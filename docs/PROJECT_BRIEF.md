# Project brief: Edge visual-inspection system for factory defect detection

## Pitch
Manual visual inspection is slow and inconsistent. Supervised defect models need thousands of labeled defects that new production lines don't have yet. This project is an **inspection station that learns from good parts only** (anomaly detection), runs in real time on a **CPU or edge box**, explains its decisions to operators with **defect heatmaps**, and **detects when the camera or lighting drifts** before accuracy silently collapses.

Results are reported in factory terms, not just AUROC:
- **defect recall at a fixed false-reject rate** (e.g. 1% of good parts wrongly scrapped)
- **parts per second per CPU core**
- **cost per 1,000 parts inspected**

## Hard constraints
- **CPU-only in this container.** GPU and TensorRT work goes in Colab notebooks that the user runs.
- **Data:** the MVTec AD dataset (CC BY-NC-SA 4.0, fine for a non-commercial portfolio). Start with 3 categories that have different defect types (e.g. `bottle`, `metal_nut`, `screw`). If the download is blocked or too large, use VisA (CC BY 4.0). Record the license in `docs/DATA.md`. Never commit images.
- **Licenses:** don't use AGPL code (e.g. Ultralytics); use torchvision or permissively licensed libraries. If a download fails because of network access, stop, write it in `PROGRESS.md`, push, and tell the user which domain to allow.

## Priorities
- **Must:** Phases 0–3.
- **Should:** Phase 4.
- **Could:** Phase 5.

## Phase 0: Environment check and skeleton (Must)
Do the same as the standing rules require: `ENVIRONMENT.md`, layout, Makefile, CI, `PROGRESS.md`, push.

## Phase 1: Classical machine-vision baseline (Must)
**What to build:** an OpenCV pipeline:
- part alignment and ROI extraction (template matching or edges plus contour moments)
- rule-based checks (edge-density and texture statistics against golden-image references)

**Why:** this is what many factories run today (HALCON-style). It sets the bar a learned model has to beat, and alignment gets reused as pre-processing.

**Metrics:**
- image-level AUROC
- recall at 1% false-reject rate
- latency per part

## Phase 2: Anomaly detection learned from good parts only (Must)
**What to build:** PatchCore-style anomaly detection:
- pretrained CNN features (start with ResNet18 for CPU, then WideResNet50 as an accuracy reference)
- a coreset-subsampled memory bank in FAISS
- pixel-level anomaly maps

**Ablations:**
- backbone
- feature layers
- coreset size (memory and latency vs. accuracy)
- with vs. without Phase 1 alignment

**Metrics (per category):**
- image AUROC and pixel AUROC / PRO
- recall at 1% false-reject rate
- latency
- memory

**Threshold selection:** choose the decision threshold from validation good parts only. That's realistic, because there are no defect labels at setup time. Document it.

## Phase 3: Edge optimization (Must)
- Export the feature extractor to ONNX.
- Benchmark PyTorch vs. ONNX Runtime vs. OpenVINO.
- Apply INT8 post-training quantization with a calibration set of good parts.
- Measure latency and throughput at 1, 2 and 4 threads to emulate small edge CPUs. Label these "thread-limited emulation, not real Jetson/RPi hardware".
- Report the accuracy change from INT8.
- Decide the deployment configuration with evidence: a latency vs. accuracy vs. memory table.

## Phase 4: Production service and monitoring (Should)
**Service:** a FastAPI inspection service. It takes an image and returns the verdict, the score, and a heatmap overlay for the operator. Batch endpoint included.

**Model registry:** use MLflow for runs, metrics and the deployed-model version, so any verdict can be traced to a model version.

**Drift monitor:**
- **Signals:** image statistics (brightness, blur/Laplacian variance, alignment offset) plus the distribution shift of the feature embeddings (MMD or a KS test on PCA components).
- **Test:** inject synthetic drift (lighting change, defocus blur, camera shift, sensor noise) at increasing severity.
- **Show:** the monitor alerts *before* recall at the fixed false-reject rate degrades. Plot both curves together.

**Observability and deployment:**
- Prometheus metrics (throughput, latency, reject rate, drift score) and alert rules.
- A Dockerfile and docker-compose. Verify what you can here and label the rest.

## Phase 5: Colab notebooks (Could)
- **`supervised_vs_anomaly.ipynb`:** train a supervised segmentation model (torchvision or segmentation_models_pytorch) using the few available defect masks. Compare it with PatchCore at 0, 5 and 20 labeled defects per category. This answers when it's worth labeling defects.
- **`tensorrt_benchmark.ipynb`:** benchmark ONNX → TensorRT FP16/INT8 on a T4 and compare with the CPU numbers on throughput and cost per 1k parts.

## Definition of done
- README with: problem framing in factory terms, an architecture diagram, results tables (classical vs. PatchCore vs. INT8-ONNX), heatmap examples, the drift plot, decisions, and limitations.
- `results/RESUME_METRICS.md` with every headline number, its reproduction command, and the hardware.
