import numpy as np
import torch

from edgeinspect.metrics import pixel_auroc, pro_auc
from edgeinspect.patchcore import PatchCore, greedy_coreset


class StubExt:
    """Patch feature = mean colour of the 8x8 cell; no pretrained weights needed."""
    dim = 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, 8).permute(0, 2, 3, 1)


def test_coreset_covers_clusters():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.01, (200, 4)) for c in (0, 5, 10)]).astype(np.float32)
    sel = greedy_coreset(x, 3)
    assert len(set(sel // 200)) == 3  # one point from each cluster
    assert len(greedy_coreset(x, 10_000)) == len(x)


def test_patchcore_flags_defect_and_localises():
    rng = np.random.default_rng(0)

    def img(defect=False):
        a = np.full((64, 64, 3), 100, np.uint8) + rng.integers(0, 3, (64, 64, 3), dtype=np.uint8)
        if defect:
            a[8:24, 8:24] = 250
        return a

    pc = PatchCore(StubExt(), size=64, coreset_ratio=0.5, sigma=1.0)
    pc.fit([img() for _ in range(8)])
    s_good, _ = pc.score([img()])
    s_bad, m = pc.score([img(True)])
    assert s_bad[0] > 5 * s_good[0]
    assert m[0, 8:24, 8:24].mean() > 3 * m[0, 40:, 40:].mean()
    assert pc.bank_mb > 0


def test_pixel_metrics():
    masks = np.zeros((2, 32, 32), np.uint8)
    masks[0, 10:20, 10:20] = 1
    perfect = masks.astype(np.float32) + np.random.default_rng(0).random(masks.shape) * 0.1
    assert pixel_auroc(masks, perfect) > 0.99
    assert pro_auc(masks, perfect) > 0.9
    assert pro_auc(masks, np.random.default_rng(1).random(masks.shape)) < 0.3


def test_pool_matrix_equals_adaptive_avg_pool():
    import torch.nn.functional as F

    from edgeinspect.patchcore import Extractor

    ext = Extractor.__new__(Extractor)  # skip loading weights
    torch.nn.Module.__init__(ext)
    ext.dim = 256
    x = torch.randn(10, 384)
    ref = F.adaptive_avg_pool1d(x.unsqueeze(1), 256).squeeze(1)
    assert torch.allclose(x @ ext._pool_matrix(384), ref, atol=1e-5)
