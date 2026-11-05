"""ONNX Runtime feature extractor (drop-in for patchcore.Extractor)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch


class OrtExtractor:
    def __init__(self, path: str, threads: int = 4, dim: int = 256) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        # spinning ORT threads contend with FAISS' OpenMP threads (2x slower at 4 threads, ADR-004)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.s.run(None, {"image": x.numpy()})[0]))
