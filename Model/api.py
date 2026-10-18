"""
api.py - FastAPI wrapper around service.py. Run on the VM:

    pip install fastapi uvicorn
    uvicorn api:app --host 127.0.0.1 --port 8000

Files it reads (same folder, override with env vars): river_model.joblib (MODEL_PATH),
hourly.csv (HOURLY_CSV, for replay/benchmark), metrics.json (METRICS_JSON, from bench + export_metrics).

Endpoints
  GET  /health                          model + machine info, replay range
  GET  /predict?mode=balanced|sensitive live EA data -> status
  POST /predict                         {"readings":[{"time","level","rain"}], "mode", "now"} -> status
  GET  /replay?start&end&mode           hour-by-hour scores for a past period + true outcome
  GET  /replay/stream?start&end&speed   Server-Sent Events, one realistic prediction per hour
  GET  /metrics                         cross-validation table + reliability curves (+ SYNTHETIC stress tests)
  GET  /benchmark?n=200                 latency + throughput on this machine (also saved to benchmark.json)
"""
import json
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import service as svc

MODEL_PATH = os.environ.get("MODEL_PATH", "river_model.joblib")
HOURLY_CSV = os.environ.get("HOURLY_CSV", "hourly.csv")
METRICS_JSON = os.environ.get("METRICS_JSON", "metrics.json")
BENCH_JSON = os.environ.get("BENCH_JSON", "benchmark.json")
STRESS_JSON = os.environ.get("STRESS_JSON", "stress.json")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE["bundle"] = svc.load_bundle(MODEL_PATH)
    STATE["hourly"] = svc.load_hourly_csv(HOURLY_CSV) if os.path.exists(HOURLY_CSV) else None
    yield


app = FastAPI(title="River level alert service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _hourly():
    if STATE.get("hourly") is None:
        raise HTTPException(503, f"{HOURLY_CSV} not found; save it from the notebook with hourly.to_csv(...)")
    return STATE["hourly"]


def _mode(mode):
    if mode not in ("balanced", "sensitive"):
        raise HTTPException(400, "mode must be 'balanced' or 'sensitive'")
    return mode


@app.get("/health")
def health():
    b = STATE["bundle"]
    h = STATE.get("hourly")
    return svc._clean({
        "ok": True, "station": b["meta"].get("station"), "model": b["model_name"],
        "level_threshold_m": b["level_threshold"], "horizon_h": b["horizon_h"],
        "operating_points": b["operating_points"], "trained_until": b["trained_until"],
        "holdout_start": b["holdout_start"],
        "replay_range": None if h is None else [str(h.index[0]), str(h.index[-1])],
        "suggested_replay_range": None if h is None else list(svc.holdout_range(b, h)),
        "environment": svc.environment(), "disclaimer": svc.DISCLAIMER,
    })


@app.get("/predict")
def predict_live(mode: str = "balanced"):
    try:
        return svc._clean(svc.predict_live(STATE["bundle"], _mode(mode)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Live data unavailable: {e}")


class Reading(BaseModel):
    time: str
    level: Optional[float] = None
    rain: Optional[float] = None


class PredictBody(BaseModel):
    readings: List[Reading]
    mode: str = "balanced"
    now: Optional[str] = None


@app.post("/predict")
def predict_posted(body: PredictBody):
    if not body.readings:
        raise HTTPException(400, "no readings")
    df = pd.DataFrame([r.model_dump() for r in body.readings])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    level = df["level"].dropna()
    rain = df["rain"].dropna()
    if level.empty:
        raise HTTPException(400, "no level values")
    hourly = svc.prepare_hourly(level, [rain] if len(rain) else [], end=body.now or df.index[-1])
    r = svc.predict_window(STATE["bundle"], hourly.iloc[-svc.WINDOW_H:], _mode(body.mode), now=body.now)
    r["source"] = "posted"
    return svc._clean(r)


@app.get("/replay")
def replay(start: str, end: str, mode: str = "balanced"):
    try:
        out, events, warnings = svc.replay(STATE["bundle"], _hourly(), start, end, _mode(mode))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(out) > 24 * 120:
        raise HTTPException(400, "max 120 days per request")
    return {"rows": svc.jsonable(out), "events": events, "warnings": warnings,
            "level_threshold_m": STATE["bundle"]["level_threshold"], "disclaimer": svc.DISCLAIMER}


@app.get("/replay/stream")
def replay_stream(start: str, end: str, mode: str = "balanced",
                  speed: float = Query(4.0, gt=0, le=1000, description="simulated hours per second")):
    b, h, m = STATE["bundle"], _hourly(), _mode(mode)

    def gen():
        for r in svc.replay_steps(b, h, start, end, m):
            yield f"data: {json.dumps(svc._clean(r))}\n\n"
            time.sleep(1.0 / speed)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_JSON):
        raise HTTPException(503, f"{METRICS_JSON} not found; run svc.export_metrics(summary, scores) in the notebook")
    with open(METRICS_JSON) as f:
        out = json.load(f)
    if os.path.exists(BENCH_JSON):
        with open(BENCH_JSON) as f:
            out["last_benchmark"] = json.load(f)
    if os.path.exists(STRESS_JSON):
        with open(STRESS_JSON) as f:
            out["stress_tests"] = json.load(f)        # labelled SYNTHETIC, keep separate from real metrics
    return out


@app.get("/benchmark")
def benchmark(n: int = Query(200, ge=10, le=5000), mode: str = "balanced"):
    return svc._clean(svc.benchmark(STATE["bundle"], _hourly(), n=n, mode=_mode(mode), save_path=BENCH_JSON))
