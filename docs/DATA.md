# Data and models
| Item | Source | License | Use |
|---|---|---|---|
| MVTec AD: bottle, metal_nut, screw only | https://www.mvtec.com/research-teaching/datasets/mvtec-ad (files on mydrive.ch) | CC BY-NC-SA 4.0 (non-commercial) | train on good images only; test on good+defect |
Images are never committed (`data/` is gitignored). Fetch with `make download`.

## Pretrained models
| Model | Source | License | Use |
|---|---|---|---|
| ResNet18 / WideResNet50-2 ImageNet-1k weights | torchvision (`IMAGENET1K_V1`, downloaded on first use) | torchvision code BSD-3-Clause; weights trained on ImageNet (research-use terms of ImageNet apply) | frozen feature extractor |
| MobileNetV3-Large backbone (Phase 5 notebook only) | torchvision | same as above | supervised segmentation baseline |
Runtime/libs: PyTorch (BSD), ONNX Runtime (MIT), OpenVINO (Apache-2.0), FAISS (MIT), OpenCV (Apache-2.0), FastAPI (MIT), MLflow (Apache-2.0). No AGPL code (e.g. Ultralytics) is used.
