# Repository guide

What is in the repo, what each file does, and how the parts connect.

## Top level

```
IBM-Z/
  Model/       ML: evaluation, serving, API, stress tests. Runs on the LinuxONE VM (s390x)
  Backend/     Django 6.1 app: stores predictions and alerts, JSON API for the frontend, admin site
  Frontend/    Web UI (not built yet; API contract in Frontend/README.md)
  docs/        These docs and platform screenshots
  README.md    Project overview, run steps, open-source library links
  .gitignore   Python template plus project rules (data, model files, secrets)
```

`CLAUDE.md` (project notes for the coding assistant) exists locally but is git-ignored.

## How the parts talk to each other

```
 LinuxONE VM                              Laptop
+---------------------------+            +-------------------------------+
| Model/api.py  :8000       | <-- SSH -- | Backend (Django) :8001        |
|   service.py, bench.py    |   tunnel   |   alerts/model_client.py      |
|   river_model.joblib      |            |   stores Prediction, Alert    |
|   hourly.csv, *.json      |            |   /api/ for the frontend      |
+---------------------------+            +-------------------------------+
              ^                                         ^
              |  EventSource /replay/stream             |  /api/...
              +------------------- Frontend ------------+
```

- The tunnel: `ssh -L 8000:localhost:8000 <user>@<vm>` makes the VM's port 8000 appear on the laptop.
- The backend never imports model code. It calls the model API over HTTP at `MODEL_API_URL`
  (default `http://localhost:8000`).
- The frontend talks only to the backend's `/api/`, except for the live-style replay stream, which it opens
  directly on the model API.

## Model/

| File | What it does |
|---|---|
| `bench.py` | The evaluation suite. Feature building, target creation, leave-one-water-year-out cross-validation, operating point selection, event metrics, reliability tables, and a synthetic data generator for self-tests. |
| `service.py` | The serving core. Data cleaning (`finish`), training the served bundle (`train`), scoring with data-quality flags and statuses (`score_frame`), one prediction with reasons (`predict_window`), replay, benchmark, live EA data fetch, metrics export. |
| `api.py` | FastAPI app wrapping `service.py`. Loads the bundle and `hourly.csv` at start-up. |
| `stress.py` | SYNTHETIC stress tests over real holdout events. Writes `stress.json`. |
| `prediction.ipynb` | The notebook: finds the station and rain gauges on the EA API, downloads and caches data, builds `hourly`, and an early random forest experiment. Used on the VM to produce the hourly table, the bundle and the metrics. |
| `requirements.txt` | numpy, pandas, scikit-learn, joblib, requests, matplotlib, fastapi, uvicorn. |
| `metrics.json` | Real cross-validation results from the VM (summary table and reliability curves). |
| `benchmark.json` | Real latency and throughput results from the VM. |
| `river_model.joblib` | The trained bundle (model, features, threshold, operating points, training info). Built on the VM. |

Generated on the VM and git-ignored: `hourly.csv`, `hydro_cache/`, `jupyter-env/`, `requirements.lock`.

### Key functions in `service.py`

| Function | Purpose |
|---|---|
| `finish(df)` | Puts data on a clean hourly grid, records which hours were really observed, fills short level gaps, sets missing rain to 0 |
| `prepare_hourly(level, rains, end)` | 15-minute readings to the finished hourly table |
| `train(hourly, ...)` | Trains the served model on everything before the holdout year, picks operating points on the last training year, stores what the service needs |
| `mode_thresholds(bundle, mode)` | ALERT and WATCH cut-offs for `balanced` or `sensitive` |
| `score_frame(bundle, df, mode)` | The single scoring path: features, probabilities, flags and status for every hour |
| `reasons(bundle, row)` | Inputs that are unusually high compared with training hours |
| `predict_window(bundle, df, mode, now)` | Full result for the latest hour (status, probability, flags, reasons, latency) |
| `replay(...)`, `replay_steps(...)` | Fast whole-period replay, and hour-by-hour replay for the stream |
| `benchmark(...)` | Latency and throughput on the current machine |
| `predict_live(bundle, mode)` | Fetches the last 11 days from the EA API and predicts now |
| `export_metrics(summary, scores)` | Writes `metrics.json` |

### Model API endpoints (`api.py`, port 8000)

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Model and machine info, threshold, operating points, holdout range |
| GET | `/predict?mode=` | Live prediction from current EA data |
| POST | `/predict` | Prediction from readings you send `{"readings": [{"time","level","rain"}], "mode", "now"}` |
| GET | `/replay?start&end&mode` | Hour-by-hour scores for a past period (max 120 days) plus the true outcome |
| GET | `/replay/stream?start&end&mode&speed` | Server-Sent Events, one realistic prediction per simulated hour |
| GET | `/metrics` | Cross-validation table, reliability curves, last benchmark, stress tests (SYNTHETIC) |
| GET | `/benchmark?n=` | Runs the benchmark on the VM and saves `benchmark.json` |

Interactive docs at `http://localhost:8000/docs` once the tunnel is open.

## Backend/

Django 6.1 project `config` with one app, `alerts`. Plain Django views, no Django REST Framework.

| File | What it does |
|---|---|
| `config/settings.py` | Adds `alerts` and `corsheaders`; reads `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `MODEL_API_URL`, `MODEL_API_TIMEOUT`, `CORS_ALLOWED_ORIGINS` from the environment |
| `config/urls.py` | `/admin/` and `/api/` |
| `alerts/models.py` | The database tables (below) |
| `alerts/model_client.py` | HTTP client for the model API; raises `ModelServiceError` with a clear message if the tunnel is down |
| `alerts/services.py` | `record_prediction` stores a result and raises an Alert (12 h cooldown per station, same as the evaluation); `record_benchmark` stores a latency run |
| `alerts/views.py` | JSON endpoints for the frontend |
| `alerts/urls.py` | Routes under `/api/` |
| `alerts/admin.py` | Admin pages; the admin doubles as a simple performance UI |
| `alerts/management/commands/seed_station.py` | Creates the Thorverton station and copies the threshold from the model API if it is reachable |
| `alerts/migrations/0001_initial.py` | Creates the tables |
| `requirements.txt` | django 6.1.2, django-cors-headers, requests |

### Database tables

| Model | Holds |
|---|---|
| `Station` | Gauge name, river, location, EA measure ids, threshold, horizon, optional per-station model URL |
| `Scenario` | A replay period, live run or synthetic stress test (shown as SYNTHETIC in admin) |
| `Prediction` | One model result: time, source (live, replay, posted), mode, status, probability, level, data age, flags, reasons, latency, and the true outcome for replays |
| `Alert` | Raised from an ALERT or ABOVE THRESHOLD prediction; can be acknowledged by a named person |
| `BenchmarkRun` | Stored latency, cross-validation or stress results, with the machine type (for example s390x) |

### Backend API (port 8001, under `/api/`)

| Method | Path | Returns |
|---|---|---|
| GET | `health/` | Backend status plus model API health |
| GET | `stations/` | Active stations |
| GET | `stations/<id>/status/?mode=` | Live prediction, stored, may raise an alert |
| GET | `stations/<id>/replay/?start&end&mode&speed` | Replay rows, events, warnings and a `stream_url` for EventSource |
| GET | `predictions/?station&source&limit` | Stored predictions |
| GET | `alerts/?station` | Alert feed |
| POST | `alerts/<id>/ack/` | Acknowledge an alert, body `{"by": "name"}` |
| GET | `metrics/` | Passes through the model API's `/metrics` |
| GET, POST | `benchmarks/` | Stored runs, or run a new benchmark on the VM and store it |

## Frontend/

Plain HTML and JavaScript, Chart.js from a CDN, no build step. Serve with `python -m http.server 5173` inside
`Frontend/`, with the tunnel to the model API on port 8000 open. Both pages call the model API directly.

| File | What it does |
|---|---|
| `index.html` | Product page: what it is, the problem, how it works, live evidence figures from `/metrics`, strengths and known limits. "Run the Dec 2024 simulation" opens the console. |
| `console.html` | Responder console. Status banner with a plain-language line, chance of crossing, level against the high level, alert lead time, data age and data checks. Live chart, reasons, per-step latency. Buttons for the Dec 2024 replay (speed 1, 4 or 20), Live now, Break the sensor (SYNTHETIC) and balanced or sensitive mode. Below that, the judge evidence: stat tiles, model vs rule table, reliability curve, SYNTHETIC stress table. `console.html#replay` starts the replay on load. |
| `README.md` | How to serve the pages and the backend API contract. |

## docs/

These pages and `screenshots/`, the platform evidence for the submission:

| File | Shows |
|---|---|
| `01-linuxone-s390x.png` | `uname -m` = `s390x` on the VM |
| `02-benchmark.png` | Benchmark results measured on the VM |
| `03-service-running.png` | Model API running on the VM, `/health` output |
| `04-api-docs.png` | FastAPI docs running a December 2024 replay |
| `05-stress-tests.png` | `stress.json` on the VM, labelled SYNTHETIC |

## How to run

### Model service (on the VM)

```bash
cd ~/IBM-Z/Model
pip install -r requirements.txt
# in prediction.ipynb: build `hourly`, then
#   from bench import run_cv; summary, per_fold, scores = run_cv(hourly)
#   import service as svc
#   bundle = svc.train(hourly, meta={...}); svc.save_bundle(bundle)
#   hourly.to_csv("hourly.csv"); svc.export_metrics(summary, scores)
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Tunnel (on the laptop)

```bash
ssh -L 8000:localhost:8000 <user>@<vm>
```

### Backend (on the laptop)

```bash
cd Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_station
python manage.py createsuperuser     # for /admin
python manage.py runserver 8001      # 8000 is taken by the tunnel
```

### Self-tests (synthetic data, any machine with the Model requirements)

```bash
python Model/bench.py
python Model/service.py
python Model/stress.py --synthetic   # run from inside Model/
```

### Stress tests on real data (on the VM)

```bash
cd ~/IBM-Z/Model
python stress.py     # writes stress.json; restart uvicorn so /metrics serves it
```

## What is never committed

- `.env` files, SSH private keys (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`), Jupyter token files.
- Data and generated files: `hourly.csv`, `hydro_cache/`, `jupyter-env/`.
- `CLAUDE.md`.
