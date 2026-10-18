"""
stress.py - SYNTHETIC stress tests for the served river-level model (decision support, not an official warning).

Takes each real high-level event in the holdout year, damages the inputs in one way (heavier rain, a sensor
fault, missing or late readings) and replays the PRE_H hours before the event hour by hour with
service.predict_window, as the live service would see it. Event times always come from the undamaged data.
Nothing here is used for training; these are stress tests, not real-world performance.

Run on the VM next to river_model.joblib and hourly.csv:

    python stress.py                 # -> stress.json (served by api.py /metrics as "stress_tests")
    python stress.py --synthetic     # self-test on synthetic data, no export

Columns
  events_caught       ALERT at least once in the horizon_h before the event
  flagged_events      DATA STALE or DATA ISSUE shown in that window
  silent_misses       neither caught nor flagged (the case responders must never see)
  median_lead_h       event start minus first ALERT, caught events only
  status_changed_pct  share of replayed hours whose status differs from the undamaged baseline
  low_while_fault_h   hours showing a confident LOW while the fault is active
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

import service as svc
from bench import find_events

H = svc.H
LABEL = "SYNTHETIC"
PRE_H = 48            # hours replayed before each event
SPIKE_M = 0.5         # size of the one-reading spike
SPIKE_AT_H = 12       # hours before the event
STUCK_FROM_H = 36     # sensor freezes this many hours before the event
OFFLINE_FROM_H = 12   # level gauge stops reporting this many hours before the event
GAP_FRAC = 0.4        # share of level readings dropped in the gaps scenario
LATE_H = 4            # readings arrive this many hours late (STALE_H is 3)
FLAG_STATUSES = ("DATA STALE", "DATA ISSUE")
NOTE = ("SYNTHETIC stress tests: real holdout events with artificially damaged inputs. "
        "Not real-world performance; see the cross-validation summary for that.")


# ---------------------------------------------------------------- scenarios
# each takes (segment, event start, rng) and returns (damaged segment, (fault_from, fault_to) or None)
def _drop_level(seg, mask):
    seg.loc[mask, "level"] = np.nan
    seg.loc[mask, "level_obs"] = False


def baseline(seg, e, rng):
    return seg, None


def doubled_rain(seg, e, rng):
    seg["rain"] = seg.rain * 2
    return seg, None


def sudden_spike(seg, e, rng):
    t = e - SPIKE_AT_H * H
    seg.loc[t, "level"] = seg.level.loc[:t].ffill().iloc[-1] + SPIKE_M
    seg.loc[t, "level_obs"] = True
    return seg, (t, t + H)


def stuck_sensor(seg, e, rng):
    t = e - STUCK_FROM_H * H
    m = seg.index >= t
    seg.loc[m, "level"] = seg.level.loc[:t].ffill().iloc[-1]
    seg.loc[m, "level_obs"] = True
    return seg, (t, e - H)


def gauge_offline(seg, e, rng):
    t = e - OFFLINE_FROM_H * H
    _drop_level(seg, seg.index >= t)
    return seg, (t, e - H)


def gaps(seg, e, rng):
    t = e - PRE_H * H
    _drop_level(seg, (seg.index >= t) & (rng.random(len(seg)) < GAP_FRAC))
    return seg, (t, e - H)


def late_data(seg, e, rng):
    return seg, (e - PRE_H * H, e - H)


# (name, description, function, delay_h); baseline must come first
SCENARIOS = [
    ("baseline", "Undamaged holdout data", baseline, 0),
    ("doubled_rain", "Rain doubled over the whole input window", doubled_rain, 0),
    ("sudden_spike", f"One level reading +{SPIKE_M} m, {SPIKE_AT_H} h before the event", sudden_spike, 0),
    ("stuck_sensor", f"Level frozen from {STUCK_FROM_H} h before the event", stuck_sensor, 0),
    ("gauge_offline", f"No level readings from {OFFLINE_FROM_H} h before the event", gauge_offline, 0),
    ("gaps", f"{GAP_FRAC:.0%} of level readings missing in the {PRE_H} h before the event", gaps, 0),
    ("late_data", f"All readings arrive {LATE_H} h late", late_data, LATE_H),
]


# ---------------------------------------------------------------- replay
def holdout_events(bundle, hourly):
    hs = svc._ts(bundle["holdout_start"]) if bundle.get("holdout_start") else None
    if hs is None:
        print("WARNING: bundle has no holdout; stress events overlap training data")
    ho = hourly if hs is None else hourly[hourly.index >= hs]
    events = find_events(ho.level, bundle["level_threshold"])
    first = hourly.index[0] + (PRE_H + svc.WINDOW_H) * H
    keep = [e for e in events if e >= first and (hs is None or e - PRE_H * H >= hs)]
    if len(keep) < len(events):
        print(f"Skipped {len(events) - len(keep)} event(s) too close to the start of the holdout")
    return keep


def replay_event(bundle, seg, e, mode, delay_h=0):
    """Status for each hour before e, using only data received by then."""
    rows = []
    for t in pd.date_range(e - PRE_H * H, e - H, freq="1h"):
        win = seg.loc[t - (svc.WINDOW_H - 1) * H: t - delay_h * H]
        r = svc.predict_window(bundle, win, mode, now=t)
        rows.append({"time": t, "status": r["status"], "prob": r["probability"]})
    return pd.DataFrame(rows).set_index("time")


def score_event(st, base, e, horizon, fault):
    last = st[st.index >= e - horizon * H]
    alerts = last.index[last.status == "ALERT"]
    caught = len(alerts) > 0
    flagged = bool(last.status.isin(FLAG_STATUSES).any())
    low = None
    if fault is not None:
        active = (st.index >= fault[0]) & (st.index <= fault[1])
        low = int((active & (st.status == "LOW").to_numpy()).sum())
    return {"event": str(e), "caught": caught, "lead_h": (e - alerts[0]) / H if caught else None,
            "flagged": flagged, "silent_miss": not caught and not flagged,
            "changed_pct": 100 * float((st.status != base.status).mean()),
            "low_while_fault_h": low, "status_hours": st.status.value_counts().to_dict()}


def run(bundle, hourly, modes=("balanced", "sensitive"), seed=0):
    hourly = svc.finish(hourly)
    events = holdout_events(bundle, hourly)
    hz = bundle["horizon_h"]
    per_event = []
    for mode in modes:
        print(f"[{mode}] {len(events)} events x {len(SCENARIOS)} scenarios x {PRE_H} h")
        for i, e in enumerate(events):
            src = hourly.loc[e - (PRE_H + svc.WINDOW_H) * H: e - H]
            base = None
            for name, _, fn, delay in SCENARIOS:
                seg, fault = fn(src.copy(), e, np.random.default_rng([seed, i]))
                st = replay_event(bundle, seg, e, mode, delay)
                base = st if base is None else base
                per_event.append({"mode": mode, "scenario": name, **score_event(st, base, e, hz, fault)})
    return summarise(per_event), per_event


def summarise(per_event):
    df = pd.DataFrame(per_event)
    out = []
    for (mode, name), g in df.groupby(["mode", "scenario"], sort=False):
        leads = g.lead_h.dropna()
        low = g.low_while_fault_h.dropna()
        out.append({"mode": mode, "scenario": name,
                    "events_caught": f"{int(g.caught.sum())}/{len(g)}",
                    "flagged_events": int(g.flagged.sum()),
                    "silent_misses": int(g.silent_miss.sum()),
                    "median_lead_h": float(leads.median()) if len(leads) else np.nan,
                    "status_changed_pct": round(float(g.changed_pct.mean()), 1),
                    "low_while_fault_h": int(low.sum()) if len(low) else None,
                    "status_hours": dict(sum((Counter(s) for s in g.status_hours), Counter()))})
    return pd.DataFrame(out)


def export(summary, per_event, bundle, path="stress.json"):
    out = {"label": LABEL, "note": NOTE, "station": bundle["meta"].get("station"),
           "level_threshold_m": bundle["level_threshold"], "horizon_h": bundle["horizon_h"],
           "pre_event_h": PRE_H, "holdout_start": bundle.get("holdout_start"),
           "scenarios": [{"name": n, "description": d} for n, d, _, _ in SCENARIOS],
           "summary": summary.to_dict(orient="records"), "per_event": per_event,
           "environment": svc.environment(), "ran_at": str(pd.Timestamp.now(tz="UTC"))}
    with open(path, "w") as f:
        json.dump(svc._clean(out), f, indent=2)
    print("Saved", path)


def show(summary, note=NOTE):
    print(f"\n{note}\n")
    print(summary.drop(columns="status_hours").to_string(index=False))


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        from bench import synthetic_hourly
        hourly = synthetic_hourly()
        bundle = svc.train(hourly, meta={"station": "SYNTHETIC"})
        summary, _ = run(bundle, hourly, modes=("balanced",))
        show(summary, "SYNTHETIC self-test on bench.synthetic_hourly data (checks the code, not the model).")
    else:
        bundle = svc.load_bundle()
        hourly = svc.load_hourly_csv()
        summary, per_event = run(bundle, hourly)
        show(summary)
        export(summary, per_event, bundle)
