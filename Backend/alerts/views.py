"""JSON endpoints for the frontend. Plain Django views (no DRF) to keep dependencies small."""
import json

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import model_client as mc
from .models import Alert, BenchmarkRun, Prediction, Station
from .services import record_benchmark, record_prediction


def _err(e, code=502):
    return JsonResponse({"error": str(e)}, status=code)


def _station_dict(s):
    return {"id": s.id, "name": s.name, "river": s.river, "lat": s.lat, "lon": s.lon,
            "level_threshold_m": s.level_threshold_m, "horizon_h": s.horizon_h}


def _prediction_dict(p):
    return {"id": p.id, "station": p.station_id, "as_of": p.as_of.isoformat(), "source": p.source, "mode": p.mode,
            "status": p.status, "probability": p.probability, "level_m": p.level_m, "data_age_h": p.data_age_h,
            "flags": p.flags, "reasons": p.reasons, "latency_ms": p.latency_ms,
            "truth_crosses_within_h": p.truth_crosses_within_h}


@require_GET
def health(request):
    try:
        return JsonResponse({"backend": "ok", "model_service": mc.health()})
    except mc.ModelServiceError as e:
        return JsonResponse({"backend": "ok", "model_service": None, "error": str(e)}, status=503)


@require_GET
def stations(request):
    return JsonResponse({"stations": [_station_dict(s) for s in Station.objects.filter(active=True)]})


@require_GET
def station_status(request, pk):
    """Live prediction from the model service; stored as a Prediction (and Alert if needed)."""
    s = get_object_or_404(Station, pk=pk)
    try:
        result = mc.predict_live(s, request.GET.get("mode", "balanced"))
    except mc.ModelServiceError as e:
        return _err(e)
    p = record_prediction(s, result, source="live")
    return JsonResponse({**result, "prediction_id": p.id, "station": _station_dict(s)})


@require_GET
def station_replay(request, pk):
    """Proxy for the model service's vectorised replay (not stored)."""
    s = get_object_or_404(Station, pk=pk)
    start, end = request.GET.get("start"), request.GET.get("end")
    if not (start and end):
        return _err("start and end are required", 400)
    mode = request.GET.get("mode", "balanced")
    try:
        data = mc.replay(start, end, s, mode)
    except mc.ModelServiceError as e:
        return _err(e)
    data["stream_url"] = mc.stream_url(start, end, s, mode, request.GET.get("speed", 4))
    return JsonResponse(data)


@require_GET
def predictions(request):
    qs = Prediction.objects.select_related("station")
    if request.GET.get("station"):
        qs = qs.filter(station_id=request.GET["station"])
    if request.GET.get("source"):
        qs = qs.filter(source=request.GET["source"])
    return JsonResponse({"predictions": [_prediction_dict(p) for p in qs[: int(request.GET.get("limit", 100))]]})


@require_GET
def alerts(request):
    qs = Alert.objects.select_related("station", "prediction")
    if request.GET.get("station"):
        qs = qs.filter(station_id=request.GET["station"])
    return JsonResponse({"alerts": [
        {"id": a.id, "station": a.station.name, "raised_at": a.raised_at.isoformat(), "status": a.status,
         "probability": a.probability, "acknowledged": a.acknowledged, "reasons": a.prediction.reasons,
         "source": a.prediction.source}
        for a in qs[: int(request.GET.get("limit", 50))]]})


@csrf_exempt
@require_POST
def acknowledge_alert(request, pk):
    a = get_object_or_404(Alert, pk=pk)
    body = json.loads(request.body or "{}")
    a.acknowledged, a.acknowledged_by = True, body.get("by", "")
    a.save(update_fields=["acknowledged", "acknowledged_by"])
    return JsonResponse({"ok": True})


@require_GET
def metrics(request):
    """Cross-validation table + reliability curves for the judge view."""
    try:
        return JsonResponse(mc.metrics())
    except mc.ModelServiceError as e:
        return _err(e)


@csrf_exempt
def benchmarks(request):
    """GET: stored benchmark runs. POST: run a new latency benchmark on the model service and store it."""
    s = Station.objects.filter(active=True).first()
    if request.method == "POST":
        if s is None:
            return _err("no station; run manage.py seed_station", 400)
        try:
            run = record_benchmark(s, mc.benchmark(s, int(request.GET.get("n", 200))))
        except mc.ModelServiceError as e:
            return _err(e)
        return JsonResponse({"id": run.id, "results": run.results})
    runs = BenchmarkRun.objects.all()[:20]
    return JsonResponse({"benchmarks": [
        {"id": r.id, "kind": r.kind, "machine": r.machine, "p50_ms": r.p50_ms, "p95_ms": r.p95_ms,
         "predictions_per_second": r.predictions_per_second, "created": r.created.isoformat()} for r in runs]})
