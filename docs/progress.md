# Progress so far

Status as of Sunday 18 October 2026 (submission deadline 17:30 UK the same day).

## Summary

| Area | State |
|---|---|
| Data pipeline (EA Hydrology API, 3 water years, hourly) | Done |
| Evaluation suite (`Model/bench.py`) | Done, run on the VM, results in `Model/metrics.json` |
| Serving core (`Model/service.py`) and API (`Model/api.py`) | Done, running on the VM, reached through an SSH tunnel on port 8000 |
| Trained model bundle (`Model/river_model.joblib`) | Trained on the VM on WY2023 and WY2024, WY2025 kept back as holdout |
| Speed benchmark on LinuxONE | Done, results in `Model/benchmark.json` |
| Holdout replay check | Done (numbers below) |
| Alert threshold review | Analysed, decision pending (see "Open decisions") |
| Stress tests (`Model/stress.py`) | Done, run on the VM on the 9 holdout events, results in `Model/stress.json` |
| Django backend (`Backend/`) | Done: models, admin, JSON API, model-service client |
| Frontend (`Frontend/`) | Not built yet, API contract written in `Frontend/README.md` |
| README, flowchart, screenshots | README drafted, flowchart in [how-it-works.md](how-it-works.md), screenshots still to take |

## What we did, step by step

### 1. Picked the problem and the gauge

We chose one hazard (river level) and one gauge (River Exe at Thorverton) so we could be honest about accuracy
instead of spreading thin. The pipeline is shaped `readings -> features -> score -> data-quality flags -> status`
so other hazards or gauges can be added later without rewriting it.

### 2. Got the data

`Model/prediction.ipynb` searches the Environment Agency (EA) Hydrology API for the station, finds its
15-minute level series and nearby rain gauges, downloads three water years (October 2022 to September 2025),
caches the raw files in `hydro_cache/`, and turns them into one hourly table:

- level: the highest reading in each hour
- rain: total rainfall in each hour, averaged across the nearby rain gauges

The hourly table is saved as `hourly.csv` on the VM. It is not committed because it can be rebuilt from the API.

### 3. Built an honest evaluation before choosing a model

The first notebook version trained a random forest on a simple time split and printed accuracy. We replaced
that with `bench.py`, because accuracy is misleading here: the river is above the high level only about 5% of
the time, so a model that always says "no" is about 95% accurate and useless.

`bench.run_cv` does leave-one-water-year-out testing: train on two water years, test on the third, rotate.
Inside each fold the high-level threshold and the alert cut-offs are chosen from training years only, so the
test year is never peeked at. Results are counted in events (did we alert before the river crossed the line?)
and false alerts per month, which is what a responder actually experiences.

### 4. Compared models against our own rules

Cross-validation across all three years (45 test events: 11, 25 and 9 per year), from `Model/metrics.json`:

| Model | Operating point | Events caught | False alerts / month | Median lead (h) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| HistGradientBoosting | high recall | 45/45 | 2.56 | 24.0 | 0.906 | 0.352 |
| HistGradientBoosting | balanced | 45/45 | 2.28 | 17.0 | 0.906 | 0.352 |
| HistGradientBoosting | high precision | 30/45 | 0.94 | 11.5 | 0.906 | 0.352 |
| Random forest | balanced | 44/45 | 1.92 | 16.5 | 0.881 | 0.343 |
| Random forest + Platt | balanced | 42/45 | 1.44 | 13.0 | 0.880 | 0.318 |
| Logistic regression | balanced | 44/45 | 2.44 | 23.5 | 0.884 | 0.339 |
| Rule: 24 h rain (ours) | balanced | 33/45 | 2.67 | 14.0 | 0.799 | 0.143 |
| Rule: 24 h rain (ours) | high recall | 45/45 | 3.94 | 24.0 | 0.799 | 0.143 |
| Rule: 3 h level rise (ours) | balanced | 45/45 | 3.67 | 7.0 | 0.513 | 0.155 |

How to read this:

- The learned models clearly beat the rain rule: PR-AUC about 0.35 against about 0.14, and at the balanced
  point boosting catches all 45 events where the rain rule catches 33.
- Random forest, boosting and logistic regression are within noise of each other. One gauge and 45 events is
  not enough to rank them confidently, and we say so.
- The 3-hour rise rule has ROC-AUC 0.51, which is chance. It "catches" every event only because it fires very
  often. We do not present it as a fair baseline.
- The 24 h median lead at the high-recall point is capped by the 24 h horizon. The correct wording is
  "alerts as soon as the 24 h window opens", not "24 hours of warning on average".

### 5. Checked calibration and chose the served model

Calibration asks: when the model says 30%, does the river cross about 30% of the time?

| Model | Top bin: predicted | Top bin: observed | Brier score |
|---|---|---|---|
| HistGradientBoosting | 0.30 | 0.30 | 0.036 |
| Random forest | 0.44 | 0.30 | 0.038 |
| Logistic regression | not reported | not reported | 0.117 |

Boosting is well calibrated out of the box, random forest is overconfident, and Platt scaling on the forest
barely helped. So the served model is HistGradientBoosting with no calibration step.

### 6. Built the serving core

`Model/service.py` holds everything needed at prediction time. One function, `score_frame`, is used for live
prediction, replay and the benchmark, so the numbers we report come from the same code that serves alerts.
We checked that hour-by-hour scoring and the fast vectorised replay give identical probabilities.

It also adds the data-quality layer: stale data, missing hours, stuck sensor, sudden spike and rain gauge
offline are detected and turned into DATA STALE or DATA ISSUE statuses, so bad data never produces a
confident LOW. Details in [how-it-works.md](how-it-works.md).

### 7. Trained the serving bundle on the VM

`service.train` trains on WY2023 and WY2024 and keeps WY2025 (from 2024-10-01) completely untouched as a
holdout. The bundle on the VM reports:

| Field | Value |
|---|---|
| Model | HistGradientBoostingClassifier |
| High-level threshold | 1.375 m (95th percentile of training levels) |
| Horizon | 24 h |
| Trained until | 2024-09-30 |
| Holdout | 2024-10-01 to 2025-09-30 |
| Operating points (probability) | high recall 0.0047, balanced 0.0218, high precision 0.93 |
| Built with | Python 3.12.3, scikit-learn 1.4.1.post1, on s390x |

### 8. Wrapped it in an API and exposed it to the laptop

`Model/api.py` (FastAPI, port 8000) serves health, live prediction, replay, a live-style replay stream,
metrics (now including stress tests) and the benchmark. It runs on the VM and is reached from the laptop with
`ssh -L 8000:localhost:8000 <vm>`.

### 9. Measured speed on LinuxONE

From `Model/benchmark.json` (machine `s390x`, 3 CPUs, host `surrey`, 200 single predictions from the holdout):

| Step | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| Build features | 3.19 | 3.40 | 3.46 |
| Model | 2.80 | 3.11 | 4.32 |
| Data-quality flags | 1.88 | 2.11 | 2.32 |
| Total, data in memory to status | 7.92 | 8.40 | 9.41 |
| Wall clock incl. overhead | 8.82 | 9.36 | 10.33 |

Throughput: a full year of hourly replay (8,760 hours) scored in 0.038 s, about 232,000 hours per second.
The model alone runs about 306,000 predictions per second. Network fetch from the EA API is not included; the
live endpoint reports it separately. These are measured numbers. We do not claim LinuxONE beats a laptop.

### 10. Checked the holdout year

We replayed the whole holdout year (8,760 hours, 9 events) through the live service in balanced mode:

| Status | Share of hours |
|---|---|
| LOW | 83.0% |
| ALERT | 11.1% (971 h) |
| WATCH | 3.9% (340 h) |
| ABOVE THRESHOLD | 2.0% (174 h) |

- All 9 holdout events were preceded by an ALERT, with about 2.25 false alert episodes per month. That matches
  cross-validation (2.28).
- WATCH covers only 3.9% of hours, so it does not need raising.
- On clean history the stuck-sensor flag fired 0% of the time, the spike flag 0.07%, rain offline 0%.
- ALERT is on for 11% of the year because the balanced cut-off is very low (probability 0.022).
  See "Open decisions".

### 11. Built the Django backend

`Backend/` stores stations, scenarios, predictions, alerts and benchmark runs, provides an admin site that
doubles as a simple performance UI, and exposes a JSON API under `/api/` for the frontend. It never imports
model code; it calls the model service over HTTP. Details in [repository.md](repository.md).

### 12. Wrote and self-tested the stress tests

`Model/stress.py` takes each real holdout event, damages the input data in one way, and replays the 48 hours
before the event hour by hour exactly as the live service would. Scenarios: doubled rain, a +0.5 m spike, a
stuck sensor, the gauge going offline, 40% of readings missing, readings arriving 4 hours late. Results go to
`stress.json` and are served under `/metrics` as `stress_tests`, labelled SYNTHETIC.

The self-test (`python stress.py --synthetic`) ran on the VM. It uses fake data, so it checks that the code
works, not how good the model is:

| Scenario (SYNTHETIC self-test, 15 fake events) | Caught | Flagged | Silent misses | LOW hours during fault |
|---|---|---|---|---|
| baseline | 15/15 | 0 | 0 | n/a |
| doubled_rain | 14/15 | 0 | 1 | n/a |
| sudden_spike | 15/15 | 0 | 0 | 9 |
| stuck_sensor | 4/15 | 13 | 1 | 293 |
| gauge_offline | 3/15 | 15 | 0 | 33 |
| gaps | 14/15 | 14 | 0 | 225 |
| late_data | 0/15 | 15 | 0 | 0 |

What it already tells us about the design:

- Late or missing data is handled as intended: the status switches to DATA STALE or DATA ISSUE instead of a
  confident LOW, so "0 caught" there comes with "0 silent misses".
- A stuck sensor is the weak spot. It is only detected after 24 flat hours, so LOW can show during the first day.
- A spike only raises a flag. It does not change the status to DATA ISSUE.
- More rain lowered the alert for one fake event. The model is not forced to treat more rain as more risk.
  This did not happen on the real holdout events (below).

### 13. Ran the stress tests on the real holdout (SYNTHETIC)

`python stress.py` on the VM, 9 real holdout events, 48 hours replayed before each, both modes. The events are
real; the damage to the data is artificial, so these are SYNTHETIC results, not real-world performance.

**Balanced mode (default)**

| Scenario | Caught (of 9) | Flagged | Silent misses | Median lead (h) | Hours with changed status | LOW hours during fault |
|---|---|---|---|---|---|---|
| baseline | 9/9 | 0 | 0 | 11.0 | 0.0% | n/a |
| doubled_rain | 9/9 | 0 | 0 | 24.0 | 21.8% | n/a |
| sudden_spike | 9/9 | 0 | 0 | 12.0 | 8.1% | 3 |
| stuck_sensor | 3/9 | 5 | **1** | 24.0 | 24.3% | 122 |
| gauge_offline | 4/9 | 9 | 0 | 24.0 | 19.7% | 15 |
| gaps | 9/9 | 5 | 0 | 9.0 | 35.6% | 87 |
| late_data | 0/9 | 9 | 0 | n/a | 95.8% | 0 |

**Sensitive mode**

| Scenario | Caught (of 9) | Flagged | Silent misses | Median lead (h) | Hours with changed status | LOW hours during fault |
|---|---|---|---|---|---|---|
| baseline | 9/9 | 0 | 0 | 24.0 | 0.0% | n/a |
| doubled_rain | 9/9 | 0 | 0 | 24.0 | 26.4% | n/a |
| sudden_spike | 9/9 | 0 | 0 | 24.0 | 3.7% | 0 |
| stuck_sensor | 7/9 | 5 | **1** | 23.0 | 22.9% | 35 |
| gauge_offline | 8/9 | 9 | 0 | 24.0 | 19.2% | 3 |
| gaps | 9/9 | 5 | 0 | 24.0 | 27.1% | 5 |
| late_data | 0/9 | 9 | 0 | n/a | 95.8% | 0 |

What this shows:

- **The baseline matches the holdout check.** 9/9 caught with an 11 h median lead in balanced mode, the same
  as the full-year replay. Sensitive mode alerts as soon as the 24 h window opens.
- **Across 6 damage types and 2 modes, there is 1 silent miss, and it is the stuck sensor.** A frozen
  sensor reads as calm water, and the check needs 24 flat hours before it fires. This is our main known weakness.
- **Missing or late data is handled safely.** gauge_offline and late_data are flagged on every event (9/9), so
  even when no ALERT shows, the responder is told the data cannot be trusted. late_data catches 0 because 4 h
  late is past the 3 h stale limit, so every hour shows DATA STALE instead.
- **Gaps barely hurt.** 40% of readings missing still gives 9/9 caught in both modes.
- **Heavier rain makes the model alert earlier, never later** (median lead 11 h to 24 h in balanced mode).
- **A 0.5 m spike did not cause trouble here**: no missed events, and only a few LOW hours while it was in the data.
- **Sensitive mode is much more robust to faults** (stuck sensor 7/9 against 3/9, gauge offline 8/9 against
  4/9), at the cost of more alerts in normal times.
- "Flagged" counts events where DATA STALE or DATA ISSUE was the shown status. ALERT outranks DATA ISSUE, so
  when the model alerts anyway the flag is still in the result but not the headline status. That is why gaps
  shows 9 caught but only 5 flagged.

One thing to check live: if EA readings routinely arrive more than 3 hours late, the live status would sit on
DATA STALE. The live `/predict` result reports `data_age_h`, which tells us the real delay.

## Open decisions

### Alert cut-off is low

The balanced cut-off is chosen by maximising F1 over individual hours in the validation year. Because positive
hours are rare, that picks a very low probability (0.022), so ALERT is on for 11% of the holdout year.

On the holdout, a higher cut-off would reduce false alerts (for example 0.05 still caught 9/9 at 1.83 false
alerts per month). But we must not pick a number from that table: it is the untouched holdout, and tuning on it
would break the claim that thresholds are chosen inside training data only.

Two honest options:

1. Change the selection rule in `bench.pick_thresholds` (for example: the highest cut-off that still catches
   every validation event), re-run cross-validation and retraining on the VM, then report the holdout once.
2. Keep the current cut-off and show the ALERT share openly. The false alert rate (about 2.3 per month) is
   consistent between cross-validation and holdout, so this is defensible.

## Still to do (priority order)

1. Restart the API on the VM so `/metrics` serves `stress.json`, and copy `stress.json` back to the repo.
2. Decide the alert cut-off question above.
3. Screenshots in `docs/screenshots/`: `uname -m`, benchmark output, the running API, stress test output.
4. Frontend: responder view and judge view.
5. Fill the results section of the root README from `metrics.json`, `benchmark.json` and `stress.json`.
6. Record the video (3 minutes max).
7. If time: a second gauge on a faster-responding river.
