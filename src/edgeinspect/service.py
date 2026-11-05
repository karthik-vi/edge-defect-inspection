"""FastAPI inspection service: verdict + score + heatmap overlay, batch endpoint, Prometheus metrics."""
from __future__ import annotations

import base64
import time
from collections import deque
from typing import Annotated, Protocol

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from edgeinspect.config import Settings, load_settings
from edgeinspect.drift import StreamDriftMonitor
from edgeinspect.engine import Engine, Verdict, overlay_png


class EngineLike(Protocol):
    category: str
    version: str
    thr: float
    monitor: object

    def inspect(self, bgr: np.ndarray) -> Verdict: ...


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a decodable image")
    return img


def create_app(engines: dict[str, EngineLike] | None = None, cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or load_settings()
    app = FastAPI(title="edge-defect-inspection", version="0.1.0")
    engines = engines if engines is not None else {}
    reg = CollectorRegistry()
    n_parts = Counter("inspect_parts_total", "Inspected parts", ["category", "verdict"], registry=reg)
    lat = Histogram("inspect_latency_seconds", "Per-part inference latency", ["category"], registry=reg,
                    buckets=(.01, .02, .03, .05, .075, .1, .2, .5, 1, 2))
    reject_rate = Gauge("inspect_reject_rate", "Reject rate over last 100 parts", ["category"], registry=reg)
    drift_score = Gauge("inspect_drift_score", "Latest drift score (>=1 means alert)", ["category"], registry=reg)
    drift_alert = Gauge("inspect_drift_alert", "1 if the latest drift window alerted", ["category"], registry=reg)
    recent: dict[str, deque] = {}
    streams: dict[str, StreamDriftMonitor] = {}

    def engine(cat: str) -> EngineLike:
        if cat not in engines:
            if cat not in cfg.categories:
                raise HTTPException(404, f"unknown category {cat}")
            engines[cat] = Engine.load(cat, cfg)
        return engines[cat]

    def run(cat: str, img: np.ndarray, with_overlay: bool) -> dict:
        eng = engine(cat)
        t0 = time.perf_counter()
        v = eng.inspect(img)
        dt = time.perf_counter() - t0
        verdict = "reject" if v.reject else "accept"
        n_parts.labels(cat, verdict).inc()
        lat.labels(cat).observe(dt)
        recent.setdefault(cat, deque(maxlen=100)).append(v.reject)
        reject_rate.labels(cat).set(float(np.mean(recent[cat])))
        if cat not in streams:
            streams[cat] = StreamDriftMonitor(eng.monitor, cfg.drift_window)  # type: ignore[arg-type]
        rep = streams[cat].update(v.stats, v.embedding)
        if rep is not None:
            drift_score.labels(cat).set(rep.score)
            drift_alert.labels(cat).set(float(rep.alert))
        last = streams[cat].last
        out = {"category": cat, "verdict": verdict, "score": v.score, "threshold": v.threshold,
               "model_version": eng.version, "latency_ms": round(dt * 1e3, 2),
               "drift": {"score": last.score, "alert": last.alert} if last else None}
        if with_overlay:
            out["heatmap_png_base64"] = base64.b64encode(overlay_png(img, v.anomaly_map, v.threshold)).decode()
        return out

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/model")
    def model() -> dict:
        return {c: {"model_version": e.version, "threshold": e.thr} for c, e in engines.items()}

    @app.post("/inspect/{category}")
    async def inspect(category: str, image: Annotated[UploadFile, File()], overlay: bool = True) -> dict:
        return run(category, decode(await image.read()), overlay)

    @app.post("/inspect_batch/{category}")
    async def inspect_batch(category: str, images: Annotated[list[UploadFile], File()],
                            overlay: bool = False) -> dict:
        res = [run(category, decode(await f.read()), overlay) for f in images]
        return {"n": len(res), "n_reject": sum(r["verdict"] == "reject" for r in res), "results": res}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory edgeinspect.service:app_factory
    return create_app()
