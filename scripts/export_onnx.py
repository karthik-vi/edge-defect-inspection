"""Export the PatchCore feature extractor to ONNX (FP32) and make an INT8 (QDQ) copy using good-part calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from edgeinspect.config import load_settings
from edgeinspect.data import list_train
from edgeinspect.patchcore import Extractor, to_tensor


class Reader(CalibrationDataReader):
    def __init__(self, arrs: list[np.ndarray]) -> None:
        self.it = iter(arrs)

    def get_next(self):
        a = next(self.it, None)
        return None if a is None else {"image": a}


if __name__ == "__main__":
    cfg = load_settings()
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    out = Path("models")
    out.mkdir(exist_ok=True)
    ext = Extractor("resnet18", (2, 3), 256)
    dummy = torch.zeros(1, 3, cfg.pc_size, cfg.pc_size)
    ext(dummy)  # build the cached pooling matrix eagerly so it is exported as a constant
    torch.onnx.export(ext, dummy, str(out / "r18_fp32.onnx"), input_names=["image"],
                      output_names=["feat"], dynamic_axes={"image": {0: "b"}, "feat": {0: "b"}},
                      opset_version=17, dynamo=False)
    # calibration: good train images pooled across categories (never test images)
    rng = np.random.default_rng(cfg.seed)
    cal = []
    for cat in cfg.categories:
        tr = list_train(cfg.data_root, cat)
        for i in rng.permutation(len(tr))[: n_calib // len(cfg.categories) + 1]:
            cal.append(to_tensor(cv2.imread(str(tr[i])), cfg.pc_size).numpy()[None])
    quantize_static(str(out / "r18_fp32.onnx"), str(out / "r18_int8.onnx"), Reader(cal[:n_calib]),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    for f in ("r18_fp32.onnx", "r18_int8.onnx"):
        print(f, round((out / f).stat().st_size / 2**20, 1), "MB")
