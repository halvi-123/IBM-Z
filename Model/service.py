"""
service.py - serving core for the river-level alert model (decision support, not an official warning).

Pipeline: readings -> hourly frame -> features (bench.build_features) -> score -> data-quality flags -> status.
The same score_frame() is used for live prediction, replay and the benchmark, so there is no train/serve skew.

Notebook usage (hourly = the notebook's DataFrame with columns level, rain):

    import service as svc
    bundle = svc.train(hourly, meta={"station": str(station.label),
                                     "level_measure": level_measure, "rain_measures": rain_measures})
    svc.save_bundle(bundle)                       # -> river_model.joblib
    hourly.to_csv("hourly.csv")                   # replay data for the API
    svc.export_metrics(summary, scores)           # -> metrics.json (from bench.run_cv)
    print(svc.benchmark(bundle, hourly))          # latency + throughput on this machine
    print(svc.predict_live(bundle))               # live EA data

By default the model is trained on everything BEFORE the last full water year, which is kept as an
untouched holdout so replay shows the model on data it never saw. train(..., holdout_start=None) uses all data.
"""
import io
import json
import os
import platform
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import build_features, future_max, make_dataset, water_year, pick_thresholds, find_events, \
    FEATURE_WINDOW_H, reliability_table

H = pd.Timedelta(hours=1)
WINDOW_H = FEATURE_WINDOW_H + 72      # 240 h of history per prediction (longest feature is 168 h)
STALE_H = 3                           # no observed level for longer than this -> DATA STALE
MAX_GAP_H = 6                         # more missing level hours than this in the last 24 h -> DATA ISSUE
STUCK_H = 24                          # level flat to the millimetre for this long -> sensor stuck
DISCLAIMER = "Decision support, not an official warning. Check Environment Agency flood warnings."
REASON_FEATURES = {
    "level": ("River level", "m"),
    "level_chg_3h": ("Level rise over 3 h", "m"),
    "level_chg_12h": ("Level rise over 12 h", "m"),
    "rain_6h": ("Rain in last 6 h", "mm"),
    "rain_24h": ("Rain in last 24 h", "mm"),
    "rain_72h": ("Rain in last 72 h", "mm"),
}


def new_model(seed=42):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------- data preparation
def finish(df):
    """Regular hourly grid + observation masks + the same gap handling as training
    (interpolate level gaps up to 6 h, missing rain counts as 0). Idempotent."""
    df = df.copy().sort_index()
    df = df[~df.index.duplicated()]
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="1h"))
    df["level_obs"] = df["level_obs"].fillna(False).astype(bool) if "level_obs" in df else df.level.notna()
    df["rain_obs"] = df["rain_obs"].fillna(False).astype(bool) if "rain_obs" in df else df.rain.notna()
    df["level"] = df.level.interpolate(limit=6, limit_area="inside")   # never extrapolate past the last reading
    df["rain"] = df.rain.fillna(0.0)
    return df


def prepare_hourly(level, rains=(), end=None):
    """15-min level series + list of 15-min rain series -> finished hourly frame up to `end`."""
    lv = level.resample("1h").max()
    rs = [r.resample("1h").sum(min_count=1) for r in rains if len(r)]
    rain = pd.concat(rs, axis=1).mean(axis=1) if rs else pd.Series(np.nan, index=lv.index)
    stop = _ts(end).floor("h") if end is not None else max(lv.index.max(), rain.index.max())
    idx = pd.date_range(lv.index.min().floor("h"), stop, freq="1h")
    return finish(pd.DataFrame({"level": lv.reindex(idx), "rain": rain.reindex(idx)}))


def load_hourly_csv(path="hourly.csv"):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return finish(df[[c for c in df.columns if c in ("level", "rain", "level_obs", "rain_obs")]])


# ---------------------------------------------------------------- training
def train(hourly, horizon=24, pct=0.95, holdout_start="last_year", min_recall=0.8, min_precision=0.5,
          seed=42, meta=None, min_year_rows=2000):
    hourly = finish(hourly)
    gap = pd.Timedelta(hours=horizon)
    wy = water_year(hourly.index)
    if holdout_start == "last_year":
        full = sorted(y for y, c in Counter(wy).items() if c >= min_year_rows)
        hs = hourly.index[wy == full[-1]][0]
    else:
        hs = None if holdout_start is None else _ts(holdout_start)
    fit_end = hourly.index[-1] + H if hs is None else hs - gap   # targets of training rows stay before holdout

    thr = float(hourly.level[hourly.index < fit_end].quantile(pct))
    data = make_dataset(hourly[["level", "rain"]], thr, horizon)
    tr = data[data.index < fit_end]
    feats = [c for c in data.columns if c != "target"]

    # operating points: fit on all but the last training water year, choose thresholds on that year
    tr_wy = water_year(tr.index)
    val_y = max(y for y, c in Counter(tr_wy).items() if c >= min_year_rows)
    val = tr[tr_wy == val_y]
    inner = tr[tr.index < val.index[0] - gap]
    if inner.target.sum() >= 5 and val.target.sum() > 0:
        m_in = new_model(seed).fit(inner[feats], inner.target)
        ops = pick_thresholds(val.target, m_in.predict_proba(val[feats])[:, 1], min_recall, min_precision)
    else:
        ops = {"high_recall": 0.1, "balanced": 0.3, "high_precision": 0.5}
        print("WARNING: too few positives for a validation year; using default operating points")

    model = new_model(seed).fit(tr[feats], tr.target)
    d1 = hourly.level[(hourly.index < fit_end) & hourly.level_obs].diff().abs()
    bundle = {
        "model": model, "model_name": "HistGradientBoostingClassifier", "features": feats,
        "level_threshold": thr, "level_threshold_pct": pct, "horizon_h": horizon,
        "operating_points": ops, "validation_year": f"WY{val_y}",
        "trained_from": str(tr.index[0]), "trained_until": str(fit_end),
        "holdout_start": None if hs is None else str(hs), "data_end": str(hourly.index[-1]),
        "train_rows": int(len(tr)), "train_positive_rate": float(tr.target.mean()),
        "feature_quantiles": {f: np.quantile(tr[f], np.linspace(0, 1, 101)) for f in REASON_FEATURES},
        "spike_m": float(max(d1.quantile(0.9995) * 1.5, 0.05)),
        "sklearn_version": sklearn.__version__, "trained_on": platform.machine(),
        "meta": meta or {},
    }
    print(f"Trained {bundle['model_name']} on {len(tr)} rows ({tr.target.mean():.1%} positive) "
          f"until {fit_end:%Y-%m-%d} | threshold {thr:.3f} m | operating points "
          + ", ".join(f"{k}={v:.3f}" for k, v in ops.items()))
    return bundle


def save_bundle(bundle, path="river_model.joblib"):
    joblib.dump(bundle, path, compress=3)
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def load_bundle(path="river_model.joblib"):
    b = joblib.load(path)
    if b.get("sklearn_version") != sklearn.__version__:
        print(f"WARNING: model saved with scikit-learn {b.get('sklearn_version')}, running {sklearn.__version__}")
    return b


# ---------------------------------------------------------------- scoring
def mode_thresholds(bundle, mode="balanced"):
    """(alert, watch) probability thresholds. balanced: alert at the F1 point, watch at the high-recall point.
    sensitive: alert at the high-recall point."""
    ops = bundle["operating_points"]
    lo, hi = sorted((ops["high_recall"], ops["balanced"]))
    alert, watch = (lo, lo / 2) if mode == "sensitive" else (hi, max(lo, hi / 2))
    if watch >= alert:
        watch = alert / 2
    return float(alert), float(watch)


def score_frame(bundle, df, mode="balanced", timings=None):
    """Score every hour of df. Each hour uses only data at or before it. Returns (scores, features)."""
    t0 = time.perf_counter()
    df = finish(df)
    feats = build_features(df[["level", "rain"]])[bundle["features"]]
    t1 = time.perf_counter()
    has = df.level.notna().to_numpy()
    p = np.full(len(df), np.nan)
    if has.any():
        p[has] = bundle["model"].predict_proba(feats[has])[:, 1]
    t2 = time.perf_counter()

    pos = np.arange(len(df), dtype=float)
    last_obs = pd.Series(np.where(df.level_obs, pos, np.nan)).ffill().to_numpy()
    age_h = pos - last_obs                               # NaN if never observed in this frame
    lv_obs = df.level.where(df.level_obs)
    flat = (lv_obs.rolling(STUCK_H, min_periods=STUCK_H // 2).max()
            - lv_obs.rolling(STUCK_H, min_periods=STUCK_H // 2).min())
    out = pd.DataFrame(index=df.index)
    out["level"] = df.level.ffill()
    out["prob"] = pd.Series(p, index=df.index).ffill()
    out["age_h"] = age_h
    out["gap_h_24h"] = (~df.level_obs).rolling(24, min_periods=1).sum()
    out["stuck"] = (flat < 0.0005).fillna(False).astype(bool)
    out["spike"] = (lv_obs.diff().abs() > bundle["spike_m"]).astype(bool)
    out["rain_offline"] = ((~df.rain_obs).rolling(24, min_periods=24).sum() >= 24).astype(bool)

    alert, watch = mode_thresholds(bundle, mode)
    stale = ~(out.age_h <= STALE_H)                      # NaN age counts as stale
    issue = out.stuck | (out.gap_h_24h > MAX_GAP_H) | out.rain_offline
    out["status"] = np.select(
        [out.level >= bundle["level_threshold"], stale, out.prob >= alert, issue, out.prob >= watch],
        ["ABOVE THRESHOLD", "DATA STALE", "ALERT", "DATA ISSUE", "WATCH"], default="LOW")
    t3 = time.perf_counter()
    if timings is not None:
        timings.update(features=(t1 - t0) * 1e3, model=(t2 - t1) * 1e3, flags=(t3 - t2) * 1e3,
                       total=(t3 - t0) * 1e3)
    return out, feats


def reasons(bundle, feat_row, top=3, min_pct=80):
    """Inputs that are unusual compared with training hours (descriptive, not model attributions)."""
    out = []
    for f, (label, unit) in REASON_FEATURES.items():
        v = feat_row.get(f, np.nan)
        if pd.isna(v):
            continue
        q = bundle["feature_quantiles"][f]
        pct = int(np.clip(np.searchsorted(q, v, side="right") - 1, 0, 100))
        if pct >= min_pct and v > 0:
            out.append({"feature": f, "label": label, "value": round(float(v), 3), "unit": unit,
                        "percentile": pct, "text": f"{label}: {v:.2f} {unit} (higher than {pct}% of training hours)"})
    return sorted(out, key=lambda r: -r["percentile"])[:top]


def predict_window(bundle, df, mode="balanced", now=None):
    """Prediction for the latest hour of df (raw or finished hourly frame). If `now` is given the grid is
    extended to now, so missing recent data shows up as DATA STALE instead of a confident LOW."""
    if now is not None:
        df = df.reindex(pd.date_range(df.index[0], max(df.index[-1], _ts(now).floor("h")), freq="1h"))
    tm = {}
    sf, feats = score_frame(bundle, df, mode, tm)
    row = sf.iloc[-1]
    scored = feats.index[finish(df).level.notna().to_numpy()]
    feat_row = feats.loc[scored[-1]] if len(scored) else pd.Series(dtype=float)
    alert, watch = mode_thresholds(bundle, mode)
    flags = {"stale": bool(not (row.age_h <= STALE_H)), "stuck_sensor": bool(row.stuck),
             "sudden_spike": bool(row.spike), "rain_offline": bool(row.rain_offline),
             "missing_level_hours_24h": int(row.gap_h_24h)}
    return {
        "station": bundle["meta"].get("station"),
        "as_of": str(sf.index[-1]),
        "level_m": None if pd.isna(row.level) else round(float(row.level), 3),
        "level_threshold_m": round(bundle["level_threshold"], 3),
        "horizon_h": bundle["horizon_h"],
        "probability": None if pd.isna(row.prob) else round(float(row.prob), 4),
        "status": str(row.status), "mode": mode,
        "alert_threshold": round(alert, 4), "watch_threshold": round(watch, 4),
        "data_age_h": None if pd.isna(row.age_h) else float(row.age_h),
        "flags": flags,
        "reasons": reasons(bundle, feat_row) if row.status in ("ALERT", "WATCH", "ABOVE THRESHOLD") else [],
        "latency_ms": {k: round(v, 3) for k, v in tm.items()},
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------- replay
def replay(bundle, hourly, start, end, mode="balanced"):
    """Score a past period hour by hour (vectorised; identical to predict_window on each window)."""
    s, e = _ts(start), _ts(end)
    hz = pd.Timedelta(hours=bundle["horizon_h"])
    df = hourly[(hourly.index >= s - WINDOW_H * H) & (hourly.index <= e + hz)]
    if df.empty:
        raise ValueError("No data in that period")
    sf, _ = score_frame(bundle, df, mode)
    fut = future_max(finish(df).level, bundle["horizon_h"])
    sf["crosses_within_h"] = (fut >= bundle["level_threshold"]).where(fut.notna())
    out = sf.loc[s:e]
    events = find_events(finish(df).level.loc[s:e + hz], bundle["level_threshold"])
    warnings = []
    if s < _ts(bundle["trained_until"]):
        warnings.append(f"Period overlaps training data (trained until {bundle['trained_until']}); "
                        "results are not a fair test. Replay from the holdout instead.")
    return out, [str(t) for t in events], warnings


def replay_steps(bundle, hourly, start, end, mode="balanced"):
    """Generator: one realistic predict_window call per hour (used by the streaming API)."""
    hourly = finish(hourly)
    fut = future_max(hourly.level, bundle["horizon_h"])
    for t in hourly.loc[_ts(start):_ts(end)].index:
        win = hourly.loc[t - (WINDOW_H - 1) * H: t]
        r = predict_window(bundle, win, mode)
        f = fut.get(t)
        r["truth_crosses_within_h"] = None if pd.isna(f) else bool(f >= bundle["level_threshold"])
        yield r


def holdout_range(bundle, hourly):
    s = _ts(bundle["holdout_start"]) if bundle.get("holdout_start") else None
    return (str(s) if s is not None else None), str(hourly.index[-1])


# ---------------------------------------------------------------- benchmark
def environment():
    return {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
            "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cpu_count": os.cpu_count(), "node": platform.node()}


def benchmark(bundle, hourly, n=200, seed=0, mode="balanced", save_path=None):
    """Latency of one prediction (data in hand -> status) and throughput, on this machine."""
    hourly = finish(hourly)
    rng = np.random.default_rng(seed)
    start = _ts(bundle["holdout_start"]) if bundle.get("holdout_start") else hourly.index[WINDOW_H]
    cand = np.where(hourly.index >= start)[0]
    cand = cand[cand >= WINDOW_H - 1]
    picks = rng.choice(cand, size=min(n, len(cand)), replace=False)
    predict_window(bundle, hourly.iloc[picks[0] - WINDOW_H + 1: picks[0] + 1], mode)    # warm-up
    lat = {"features": [], "model": [], "flags": [], "total": [], "wall": []}
    for i in picks:
        win = hourly.iloc[i - WINDOW_H + 1: i + 1]
        t0 = time.perf_counter()
        r = predict_window(bundle, win, mode)
        lat["wall"].append((time.perf_counter() - t0) * 1e3)
        for k in ("features", "model", "flags", "total"):
            lat[k].append(r["latency_ms"][k])

    span = hourly.loc[start:]
    t0 = time.perf_counter()
    sf, feats = score_frame(bundle, span, mode)
    t_vec = time.perf_counter() - t0
    X = feats.dropna(how="all")
    t0 = time.perf_counter()
    bundle["model"].predict_proba(X)
    t_model = time.perf_counter() - t0

    pctl = lambda v: {"p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                      "p99": round(float(np.percentile(v, 99)), 3), "mean": round(float(np.mean(v)), 3)}
    res = {
        "environment": environment(),
        "single_prediction_ms": {k: pctl(v) for k, v in lat.items()},
        "single_prediction_n": int(len(picks)), "window_h": WINDOW_H,
        "replay_hours": int(len(span)), "replay_seconds": round(t_vec, 4),
        "replay_hours_per_second": round(len(span) / t_vec, 1),
        "model_batch_rows": int(len(X)), "model_predictions_per_second": round(len(X) / t_model, 1),
        "note": "Latency covers features + model + data-quality flags with data already in memory; "
                "network fetch from the EA API is measured separately in predict_live.",
        "ran_at": str(pd.Timestamp.now(tz="UTC")),
    }
    if save_path:
        with open(save_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------- live data (EA Hydrology API)
BASE = "https://environment.data.gov.uk/hydrology"
HEADERS = {"User-Agent": "ibmz-datathon-river-alert"}
BAD_QUALITY = {"Suspect", "Missing"}


def _get(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (attempt + 1)); continue
        raise RuntimeError(f"EA API error {r.status_code} for {r.url}")
    raise RuntimeError(f"EA API gave up after {retries} attempts: {url}")


def fetch_measure(measure, start, end):
    r = _get(f"{BASE}/id/measures/{measure}/readings.csv", {"mineq-date": start, "max-date": end, "_limit": 20000})
    df = pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
    if df.empty or "value" not in df:
        return pd.Series(dtype=float)
    if "quality" in df:
        df = df[~df["quality"].isin(BAD_QUALITY)]
    df = df.dropna(subset=["value"])
    df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True)
    return df.drop_duplicates("dateTime").set_index("dateTime")["value"].astype(float).sort_index()


def fetch_recent(bundle, days=11, now=None):
    m = bundle["meta"]
    if not m.get("level_measure"):
        raise ValueError("bundle meta has no level_measure; pass meta=... to train()")
    now = _ts(now) if now is not None else pd.Timestamp.now(tz="UTC")
    a, b = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t0 = time.perf_counter()
    level = fetch_measure(m["level_measure"], a, b)
    rains = [fetch_measure(x, a, b) for x in m.get("rain_measures", [])]
    fetch_ms = (time.perf_counter() - t0) * 1e3
    if level.empty:
        raise RuntimeError("No recent level readings returned")
    return prepare_hourly(level, rains, end=now), fetch_ms


def predict_live(bundle, mode="balanced"):
    now = pd.Timestamp.now(tz="UTC")
    df, fetch_ms = fetch_recent(bundle, now=now)
    r = predict_window(bundle, df.iloc[-WINDOW_H:], mode, now=now)
    r["latency_ms"]["fetch"] = round(fetch_ms, 1)
    r["source"] = "live"
    return r


# ---------------------------------------------------------------- metrics export
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def export_metrics(summary, scores, path="metrics.json", note=""):
    """Write bench.run_cv results (+ reliability curves) for the API's /metrics endpoint."""
    rel = {}
    for name in ("hist_gb", "rf", "rf_platt", "logreg"):
        if name in scores:
            rel[name] = reliability_table(scores, name).round(4).to_dict(orient="records")
    out = {"summary": summary.to_dict(orient="records"), "reliability": rel, "note": note,
           "environment": environment(), "exported_at": str(pd.Timestamp.now(tz="UTC"))}
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print("Saved", path)


def jsonable(df):
    d = df.reset_index().rename(columns={"index": "time"})
    d["time"] = d["time"].astype(str)
    return _clean(d.to_dict(orient="records"))


# ---------------------------------------------------------------- self-test on synthetic data
if __name__ == "__main__":
    from bench import synthetic_hourly
    hourly = synthetic_hourly()
    b = train(hourly, meta={"station": "SYNTHETIC"})
    print(json.dumps(predict_window(b, hourly.iloc[-WINDOW_H:]), indent=2, default=str))
    print(json.dumps(benchmark(b, hourly, n=100), indent=2))
