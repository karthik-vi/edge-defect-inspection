"""PatchCore-style anomaly detection: pretrained CNN patch features + coreset memory bank (FAISS)."""
from __future__ import annotations

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Extractor(torch.nn.Module):
    """Frozen backbone returning locally-aggregated, channel-reduced patch features (B,H,W,D)."""

    def __init__(self, backbone: str = "resnet18", layers: tuple[int, ...] = (2, 3),
                 dim: int = 256) -> None:
        super().__init__()
        ctor = {"resnet18": torchvision.models.resnet18,
                "wide_resnet50_2": torchvision.models.wide_resnet50_2}[backbone]
        m = ctor(weights="IMAGENET1K_V1").eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.blocks = torch.nn.ModuleList([m.layer2, m.layer3, m.layer4])
        self.layers, self.dim = layers, dim
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feats = []
        for i, blk in enumerate(self.blocks, start=2):
            x = blk(x)
            if i in self.layers:
                feats.append(F.avg_pool2d(x, 3, 1, 1))  # 3x3 neighbourhood aggregation
            if i >= max(self.layers):
                break
        size = feats[0].shape[-2:]
        feats = [f if f.shape[-2:] == size else F.interpolate(f, size=size, mode="bilinear",
                                                              align_corners=False) for f in feats]
        f = torch.cat(feats, 1)                                  # B,C,H,W
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, c)                  # B*H*W, C
        f = F.adaptive_avg_pool1d(f.unsqueeze(1), self.dim).squeeze(1)
        return f.reshape(b, h, w, self.dim)


def to_tensor(bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def greedy_coreset(x: np.ndarray, n: int, seed: int = 0, proj: int = 32) -> np.ndarray:
    """Greedy k-center selection (indices) on a random projection of x (N,D)."""
    if n >= len(x):
        return np.arange(len(x))
    g = torch.Generator().manual_seed(seed)
    z = torch.from_numpy(x) @ torch.randn(x.shape[1], proj, generator=g) / proj**0.5
    idx = [int(torch.randint(len(z), (1,), generator=g))]
    d = torch.cdist(z, z[idx[0]][None]).squeeze(1)
    for _ in range(n - 1):
        i = int(d.argmax())
        idx.append(i)
        d = torch.minimum(d, torch.cdist(z, z[i][None]).squeeze(1))
    return np.array(idx)


class PatchCore:
    def __init__(self, extractor: Extractor, size: int = 224, coreset_ratio: float = 0.1,
                 sigma: float = 4.0, seed: int = 0) -> None:
        self.ext, self.size, self.ratio, self.sigma, self.seed = (
            extractor, size, coreset_ratio, sigma, seed)
        self.index: faiss.Index | None = None
        self.bank: np.ndarray | None = None

    def embed(self, imgs: list[np.ndarray], bs: int = 16) -> np.ndarray:
        out = [self.ext(torch.stack([to_tensor(i, self.size) for i in imgs[k:k + bs]]))
               for k in range(0, len(imgs), bs)]
        return torch.cat(out).numpy()  # N,H,W,D

    def fit(self, good: list[np.ndarray]) -> None:
        f = self.embed(good).reshape(-1, self.ext.dim)
        sel = greedy_coreset(f, max(1, int(len(f) * self.ratio)), self.seed)
        self.bank = np.ascontiguousarray(f[sel])
        self.index = faiss.IndexFlatL2(self.bank.shape[1])
        self.index.add(self.bank)

    def maps(self, imgs: list[np.ndarray]) -> np.ndarray:
        """Pixel anomaly maps (N,size,size) from nearest-neighbour patch distance."""
        f = self.embed(imgs)
        n, h, w, d = f.shape
        dist, _ = self.index.search(np.ascontiguousarray(f.reshape(-1, d)), 1)
        m = np.sqrt(np.maximum(dist, 0)).reshape(n, h, w)
        out = np.stack([cv2.GaussianBlur(cv2.resize(x, (self.size, self.size)), (0, 0), self.sigma)
                        for x in m])
        return out

    def score(self, imgs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        m = self.maps(imgs)
        return m.reshape(len(m), -1).max(1), m

    @property
    def bank_mb(self) -> float:
        return self.bank.nbytes / 2**20
