import numpy as np

from edgeinspect.classical import align, fit_golden, part_mask
from edgeinspect.metrics import (
    image_auroc,
    latency_percentiles,
    recall_at_frr_oracle,
    recall_at_threshold,
    threshold_from_good,
)


def _part(dx=0, dy=0, defect=False, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256), 200, np.uint8)
    img[88 + dy:168 + dy, 108 + dx:148 + dx] = 60
    if defect:
        img[120 + dy:135 + dy, 115 + dx:140 + dx] = 230
    return np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def test_mask_finds_dark_part_on_light_bg():
    assert part_mask(_part()).sum() > 0 and part_mask(_part())[128, 128] == 255


def test_align_removes_translation():
    a, b = align(_part()), align(_part(dx=30, dy=-20))
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 5


def test_golden_scores_defect_higher_than_good():
    g = [align(_part(seed=i)) for i in range(10)]
    v = [align(_part(dx=i, seed=20 + i)) for i in range(10)]
    m = fit_golden(g, v)
    assert m.score(align(_part(defect=True, seed=99))) > m.score(align(_part(seed=98))) + 1


def test_metrics():
    y, s = np.array([0, 0, 0, 1, 1]), np.array([.1, .2, .3, .9, .25])
    assert image_auroc(y, s) == 5 / 6
    assert threshold_from_good(np.arange(100.0), 0.01) > 98
    assert recall_at_threshold(y, s, 0.35) == (0.5, 0.0)
    assert recall_at_frr_oracle(y, s, 0.01) >= 0.5
    assert latency_percentiles(np.arange(1.0, 101.0))["p50"] == 50.5
