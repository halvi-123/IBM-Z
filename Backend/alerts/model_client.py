"""HTTP client for the model service (Model/api.py on the LinuxONE VM, reached through the SSH tunnel)."""
import requests
from django.conf import settings


class ModelServiceError(Exception):
    pass


def _base(station=None):
    return ((station and station.model_api_url) or settings.MODEL_API_URL).rstrip("/")


def _call(method, path, station=None, timeout=None, **kw):
    url = f"{_base(station)}{path}"
    try:
        r = requests.request(method, url, timeout=timeout or settings.MODEL_API_TIMEOUT, **kw)
    except requests.RequestException as e:
        raise ModelServiceError(f"Model service unreachable at {url}: {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ModelServiceError(f"Model service {r.status_code}: {detail}")
    return r.json()


def health(station=None):
    return _call("GET", "/health", station)


def predict_live(station=None, mode="balanced"):
    return _call("GET", "/predict", station, params={"mode": mode})


def replay(start, end, station=None, mode="balanced"):
    return _call("GET", "/replay", station, params={"start": start, "end": end, "mode": mode}, timeout=120)


def metrics(station=None):
    return _call("GET", "/metrics", station)


def benchmark(station=None, n=200):
    return _call("GET", "/benchmark", station, params={"n": n}, timeout=300)


def stream_url(start, end, station=None, mode="balanced", speed=4):
    """URL the frontend can open directly with EventSource for live-style replay."""
    return f"{_base(station)}/replay/stream?start={start}&end={end}&mode={mode}&speed={speed}"
