# River-level alert system – IBM Z Datathon 2026

> Decision support for first responders, not an official warning. Always check Environment Agency flood warnings.

Predicts whether a river will cross a high level within the next 24 hours, flags bad or missing sensor
data instead of reporting a confident "low risk", and measures its own speed and accuracy on IBM LinuxONE.

## How it fits together
```
EA Hydrology API ──> Model service (LinuxONE VM, FastAPI :8000) ──> Django backend (/api/, admin) ──> Frontend
                     features -> HistGradientBoosting -> data-quality flags -> status
```

| Folder | What | Runs on |
|---|---|---|
| `Model/` | Evaluation suite (`bench.py`), serving core (`service.py`), API (`api.py`), training notebook | LinuxONE VM (s390x) |
| `Backend/` | Django: stations, scenarios, predictions, alerts, benchmark runs; admin as performance UI | laptop / any |
| `Frontend/` | Responder and judge views | browser |
| `docs/` | Flowchart and platform screenshots | – |

## Results (River Exe at Thorverton, 3 water years, 45 high-level events)
_To be filled from `Model/metrics.json` and `Model/benchmark.json` on the VM._

## Run it

### 1. Model service (on the VM)
```bash
cd Model
pip install -r requirements.txt
# in prediction.ipynb after loading `hourly`:
#   from bench import run_cv; summary, per_fold, scores = run_cv(hourly)
#   import service as svc
#   bundle = svc.train(hourly, meta={"station": str(station.label), "level_measure": level_measure, "rain_measures": rain_measures})
#   svc.save_bundle(bundle); hourly.to_csv("hourly.csv"); svc.export_metrics(summary, scores)
uvicorn api:app --host 127.0.0.1 --port 8000
```
From your laptop: `ssh -L 8000:localhost:8000 <user>@<vm>`, then open http://localhost:8000/docs.

### 2. Backend (laptop)
```bash
cd Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_station
python manage.py createsuperuser     # for /admin
python manage.py runserver 8001      # port 8000 is taken by the model tunnel
```

### 3. Frontend
See `Frontend/README.md` for the API contract.

## Data
- Environment Agency Hydrology API (Open Government Licence): https://environment.data.gov.uk/hydrology/doc/reference

## Open-source libraries
- scikit-learn – https://scikit-learn.org
- pandas – https://pandas.pydata.org
- NumPy – https://numpy.org
- joblib – https://joblib.readthedocs.io
- Requests – https://requests.readthedocs.io
- Matplotlib – https://matplotlib.org
- Jupyter – https://jupyter.org
- FastAPI – https://fastapi.tiangolo.com
- Uvicorn – https://www.uvicorn.org
- Django – https://www.djangoproject.com
- django-cors-headers – https://github.com/adamchainz/django-cors-headers
