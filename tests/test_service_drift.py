import cv2
import numpy as np
from fastapi.testclient import TestClient

from edgeinspect.drift import DriftMonitor, StreamDriftMonitor, image_stats
from edgeinspect.engine import Verdict
from edgeinspect.service import create_app


def _img(rng, gain=1.0, noise=2.0):
    a = np.full((128, 128, 3), 200, np.float32)
    a[40:90, 40:90] = 60
    return np.clip(a * gain + rng.normal(0, noise, a.shape), 0, 255).astype(np.uint8)


def _monitor(rng):
    imgs = [_img(rng) for _ in range(80)]
    emb = np.stack([cv2.resize(i, (4, 4)).reshape(-1).astype(np.float32) for i in imgs])
    return DriftMonitor(np.stack([image_stats(i) for i in imgs]), emb), emb


def test_no_false_alarm_then_alert_on_drift():
    rng = np.random.default_rng(0)
    mon, _ = _monitor(rng)

    def window(gain):
        im = [_img(rng, gain) for _ in range(20)]
        e = np.stack([cv2.resize(i, (4, 4)).reshape(-1).astype(np.float32) for i in im])
        return mon.check(np.stack([image_stats(i) for i in im]), e)

    assert not window(1.0).alert
    assert window(0.7).alert


def test_stream_emits_every_window():
    rng = np.random.default_rng(1)
    mon, _ = _monitor(rng)
    st = StreamDriftMonitor(mon, window=5)
    outs = [st.update(image_stats(_img(rng)), np.zeros(48, np.float32)) for _ in range(10)]
    assert [o is not None for o in outs] == [False] * 4 + [True] + [False] * 4 + [True]


class FakeEngine:
    category, version, thr = "bottle", "test123", 1.0

    def __init__(self):
        rng = np.random.default_rng(0)
        self.monitor, _ = _monitor(rng)

    def inspect(self, bgr):
        s = float(bgr.mean() / 100)
        return Verdict(s, self.thr, s > self.thr, np.full((32, 32), s, np.float32),
                       image_stats(bgr), np.zeros(48, np.float32))


def _png(v):
    return cv2.imencode(".png", np.full((64, 64, 3), v, np.uint8))[1].tobytes()


def test_service_endpoints():
    c = TestClient(create_app({"bottle": FakeEngine()}))
    r = c.post("/inspect/bottle", files={"image": ("a.png", _png(50), "image/png")}).json()
    assert r["verdict"] == "accept" and r["model_version"] == "test123" and r["heatmap_png_base64"]
    r = c.post("/inspect/bottle?overlay=false", files={"image": ("a.png", _png(250), "image/png")}).json()
    assert r["verdict"] == "reject" and "heatmap_png_base64" not in r
    b = c.post("/inspect_batch/bottle", files=[("images", ("a.png", _png(50), "image/png")),
                                                ("images", ("b.png", _png(250), "image/png"))]).json()
    assert b["n"] == 2 and b["n_reject"] == 1
    assert c.post("/inspect/bottle", files={"image": ("x", b"junk", "image/png")}).status_code == 400
    assert c.post("/inspect/nope", files={"image": ("a.png", _png(5), "image/png")}).status_code == 404
    m = c.get("/metrics").text
    assert 'inspect_parts_total{category="bottle",verdict="reject"} 2.0' in m
    assert c.get("/model").json()["bottle"]["model_version"] == "test123"
