# Qubit: River Level Alert

**IBM Z Datathon 2026 · Theme: Real-time AI for critical decisions · Team Qubit**

> Decision support for first responders, not an official warning. Always check Environment Agency flood warnings.

Know up to 24 hours ahead when a river will cross its high level, and know when the sensor data can't be trusted.
Trained, served and benchmarked on IBM LinuxONE (s390x).

## Description

Rivers are one of the few hazards that warn you before they arrive. Rain falls on the hills first and reaches a river
gauge hours later, so there is a window in which responders can close fords and low roads, move vehicles and
equipment, put out barriers and pumps, and check on vulnerable residents. That window is only useful if someone reads
the signs early, and if the readings they are looking at are real.

Today responders mostly see two things: live gauge readings, which show where the river is now but not where it is
heading, and official flood warnings, which remain the authority. Simple rules on top of the gauge do not close the
gap. A level alarm fires only once the river is already high. A rain alarm fires too often, and false alarms wear out
trust until people stop responding. And sensors fail quietly: a frozen gauge or a dropped feed looks exactly like a
calm river.

We built **Qubit: River Level Alert** to address all three. Every hour it reads the last 10 days of river level and
rainfall for one gauge (River Exe at Thorverton, Devon), estimates the chance that the river will cross its high
level within the next 24 hours, and turns that into a single status a responder can read in a second. Before trusting
the model it checks the data, and when the data looks broken it says so (DATA STALE or DATA ISSUE) instead of reporting
a confident LOW. Every claim about accuracy comes with the number of events behind it and a comparison with simple
rules we built ourselves.

We predict "crossing the gauge's high level", not "flooding": whether water reaches homes depends on local defences,
so we only claim what we can measure and check.

## What it does

- **Hourly risk status**: LOW, WATCH, ALERT, ABOVE THRESHOLD, DATA STALE, DATA ISSUE, with a plain-language line for each.
- **Chance of crossing in 24 h** from a calibrated gradient boosting model, in balanced or sensitive mode.
- **Data-quality checks** on every prediction: stale feed, missing hours, stuck sensor, sudden spike, rain gauge offline.
- **Reasons**: inputs that are unusually high compared with training hours (descriptive, not model attributions).
- **Replay** of real past events hour by hour, exactly as the live service would have seen them.
- **SYNTHETIC wildcards and stress tests**: the same events with damaged data (gauge offline, frozen sensor, late or
  missing readings, spikes, doubled rain) to show how the system behaves when data goes wrong.
- **Benchmark suite**: honest accuracy (events caught, false alerts per month, lead time, calibration) and latency
  measured on LinuxONE.
- **Responder console and product page** in the browser, plus a Django backend that stores predictions and alerts.

## Results

River Exe at Thorverton, EA Hydrology API, water years 2023 to 2025, hourly data.

**Cross-validation** (leave one water year out, 3 water years, **45 high-level events**, one gauge; thresholds chosen
inside the training years only), balanced operating point:

| Model | Events caught | False alerts / month | PR-AUC |
|---|---|---|---|
| **Gradient boosting (served)** | **45/45** | **2.28** | **0.35** |
| Random forest | 44/45 | 1.92 | 0.34 |
| Logistic regression | 44/45 | 2.44 | 0.34 |
| Rule: 24 h rain (ours) | 33/45 | 2.67 | 0.14 |
| Rule: 3 h level rise (ours) | 45/45 | 3.67 | 0.16 |

The 3 h rise rule is at chance (ROC-AUC 0.51): it catches everything only by alerting constantly. The three models are
within noise of each other on 45 events.

**Untouched holdout year** (WY2025, never used for training or picking thresholds, **9 events**): 9/9 caught,
2.25 false alerts per month, 11 h median lead. On the December 2024 event the first ALERT came 27 h before the river
crossed its high level.

**Calibration**: in testing, hours given about 30% crossed about 30% of the time (random forest said 44% for the same
30%, so we serve gradient boosting with no extra calibration step).

**Stress tests (SYNTHETIC)**, 9 real holdout events with damaged inputs, balanced mode: one silent miss (no ALERT and
no data warning), caused by a frozen sensor, which takes 24 h to detect. Gauge offline and late data were flagged on
every event. Full table in [docs/progress.md](docs/progress.md).

**Speed on IBM LinuxONE (s390x, 3 CPUs)**, reported as measured: 7.9 ms median from data in memory to status
(features, model and data checks, 200 predictions); about 306,000 model predictions per second in batch; a full year
of hourly history (8,760 h) scored in 0.038 s.

## How it works

```
EA Hydrology API -> hourly level + rain -> features (last 10 days) -> gradient boosting on LinuxONE
                 -> data-quality checks -> status + reasons -> FastAPI (:8000) -> console / Django backend
```

1. **Readings**: 15-minute level and rainfall from the Environment Agency, cleaned to an hourly grid.
2. **Features**: level now, rises over 1 to 24 h, recent peaks, rain totals from 1 h to 7 days.
3. **Model**: scikit-learn HistGradientBoostingClassifier, trained and served on the LinuxONE VM.
4. **Data checks**: stale (> 3 h), missing hours (> 6 in 24 h), stuck sensor (flat for 24 h), spike, rain offline.
5. **Status**: ABOVE THRESHOLD > DATA STALE > ALERT > DATA ISSUE > WATCH > LOW. Bad data never yields a confident LOW.

One scoring function (`service.score_frame`) serves live prediction, replay and the benchmark, so what we measure is
exactly what we serve. Diagrams of every part are in [docs/architecture.md](docs/architecture.md), and a plain-language
explanation is in [docs/how-it-works.md](docs/how-it-works.md).

## Why IBM LinuxONE

The whole model side runs on an IBM LinuxONE virtual machine (s390x, Ubuntu): data download, training, the model API
and the benchmark. Critical alerting needs a platform built for reliability and always-on service, and we measured
what the model costs to run there instead of assuming it. Evidence screenshots are in
[docs/screenshots/](docs/screenshots/).

## Tech stack

**Model service (LinuxONE VM)**
- Python 3.12
- scikit-learn (HistGradientBoosting, random forest, logistic regression), pandas, NumPy, joblib
- FastAPI + Uvicorn + Pydantic (JSON API and Server-Sent Events replay stream)
- Jupyter (data download and exploration)

**Backend**
- Django 6.1 (plain views, no DRF), SQLite
- django-cors-headers, Requests (HTTP client to the model service)

**Frontend**
- Plain HTML, CSS and JavaScript, no build step
- Chart.js (live chart, reliability curve)

**Platform**
- IBM LinuxONE Community Cloud VM (s390x, Ubuntu 24.04), reached from the laptop through an SSH tunnel

## Repository structure

```
IBM-Z/
├── Model/               runs on the LinuxONE VM
│   ├── bench.py         evaluation suite: cross-validation, rule baselines, event metrics, calibration
│   ├── service.py       serving core: features, scoring, data checks, statuses, replay, benchmark, live EA fetch
│   ├── api.py           FastAPI app on :8000
│   ├── stress.py        SYNTHETIC stress tests over real holdout events
│   ├── prediction.ipynb data download and exploration
│   ├── metrics.json     cross-validation results (from the VM)
│   ├── benchmark.json   latency and throughput (from the VM)
│   └── river_model.joblib  served model bundle (trained on the VM)
├── Backend/             Django project `config`, app `alerts` (stations, predictions, alerts, benchmark runs)
├── Frontend/            index.html (product page), console.html (responder console + evidence)
├── docs/                project docs and LinuxONE screenshots
└── README.md
```

## Local setup

### 1. Model service (on the LinuxONE VM)

```bash
git clone https://github.com/halvi-123/IBM-Z.git
cd IBM-Z/Model
python3 -m venv ~/jupyter-env && source ~/jupyter-env/bin/activate
pip install -r requirements.txt
```

Build the data and model in `prediction.ipynb` (downloads from the EA API into `hydro_cache/`, then):

```python
from bench import run_cv
import service as svc
summary, per_fold, scores = run_cv(hourly)           # cross-validation
bundle = svc.train(hourly, meta={...})               # trains on WY2023-24, keeps WY2025 as holdout
svc.save_bundle(bundle); hourly.to_csv("hourly.csv"); svc.export_metrics(summary, scores)
```

Then, from `~/IBM-Z/Model`:

```bash
python stress.py                                     # SYNTHETIC stress tests -> stress.json
nohup uvicorn api:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &
```

### 2. SSH tunnel (on the laptop)

```bash
ssh -L 8000:localhost:8000 <user>@<vm>
```

The model API is now at http://localhost:8000 (interactive docs at http://localhost:8000/docs).

### 3. Frontend (on the laptop)

```bash
cd Frontend
python -m http.server 5173
```

Open http://localhost:5173. "Run the Dec 2024 simulation" opens the console and starts the replay;
`console.html#wildcard-gauge_offline` runs the SYNTHETIC gauge-offline wildcard.

### 4. Backend (optional, on the laptop)

```bash
cd Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_station
python manage.py createsuperuser     # for /admin
python manage.py runserver 8001      # 8000 is taken by the tunnel
```

## Environment variables

All optional, with working defaults.

| Variable | Where | Default |
|---|---|---|
| `MODEL_PATH`, `HOURLY_CSV`, `METRICS_JSON`, `BENCH_JSON`, `STRESS_JSON` | Model API | files in `Model/` |
| `MODEL_API_URL` | Backend | `http://localhost:8000` |
| `MODEL_API_TIMEOUT` | Backend | `30` |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG` | Backend | dev key, debug on |
| `CORS_ALLOWED_ORIGINS` | Backend | `http://localhost:5173,http://localhost:3000` |

## API examples

Model service (port 8000):

```
GET  /health                                         model, threshold, holdout range, machine, wildcards
GET  /predict?mode=balanced                          live prediction from current EA data
POST /predict                                        prediction from readings you send
GET  /replay?start=2024-12-01&end=2024-12-07         hour-by-hour scores and the true outcome
GET  /replay/stream?start=...&end=...&speed=4        live-style replay (Server-Sent Events)
GET  /replay/stream?...&scenario=gauge_offline       SYNTHETIC wildcard replay
GET  /metrics                                        cross-validation, reliability, benchmark, stress tests
GET  /benchmark?n=200                                run the latency benchmark on this machine
```

Backend (port 8001): `GET /api/stations/`, `GET /api/stations/<id>/status/`, `GET /api/alerts/`,
`POST /api/alerts/<id>/ack/`, `GET /api/metrics/`. Full list in [docs/repository.md](docs/repository.md).

## Running tests

Self-tests on synthetic data:

```bash
cd Model
python bench.py
python service.py
python stress.py --synthetic
```

## Honesty notes and limits

- We say "crossing a high river level", not "flood", and label everything "decision support, not an official warning".
- Baselines are rules we built ourselves; every metric is shown with its event count.
- Synthetic stress tests and wildcards are reported separately, labelled SYNTHETIC, and never used for training.
- One gauge and 45 events: results are noisy. A frozen sensor takes 24 h to detect, and a single spike can show a
  false ABOVE THRESHOLD for one hour.
- Benchmark numbers are reported as measured on the VM, with no comparison claimed against other hardware.

## Team

Team Qubit:

- Hassaan Alvi
- Sai Raghavan Commandur
- Adarsh Ram Sivakumar
- Adit Sharma

## Docs

- [docs/architecture.md](docs/architecture.md): architecture and solution flowcharts (16 diagrams)
- [docs/README.md](docs/README.md): overview and ground rules
- [docs/progress.md](docs/progress.md): what we did, step by step, with all results
- [docs/how-it-works.md](docs/how-it-works.md): the system explained from scratch
- [docs/repository.md](docs/repository.md): every file and endpoint
- [docs/glossary.md](docs/glossary.md): terms

## Data

- Environment Agency Hydrology API: https://environment.data.gov.uk/hydrology/doc/reference
- Licensed under the Open Government Licence v3.0: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Open-source libraries

- Python: https://www.python.org
- NumPy: https://numpy.org
- pandas: https://pandas.pydata.org
- scikit-learn: https://scikit-learn.org
- joblib: https://joblib.readthedocs.io
- Requests: https://requests.readthedocs.io
- Matplotlib: https://matplotlib.org
- Jupyter: https://jupyter.org
- FastAPI: https://fastapi.tiangolo.com
- Uvicorn: https://www.uvicorn.org
- Pydantic: https://docs.pydantic.dev
- Django: https://www.djangoproject.com
- django-cors-headers: https://github.com/adamchainz/django-cors-headers
- Chart.js: https://www.chartjs.org
- IBM Plex (slide deck fonts): https://github.com/IBM/plex
