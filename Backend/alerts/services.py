from datetime import timedelta

from django.utils.dateparse import parse_datetime

from .models import Alert, Prediction

ALERTING = {"ALERT", "ABOVE THRESHOLD"}
ALERT_COOLDOWN = timedelta(hours=12)   # same cooldown as Model/bench.py episodes


def record_prediction(station, result, source="live", scenario=None, truth=None):
    """Store one model-service result; raise an Alert unless one was raised for this station recently."""
    as_of = parse_datetime(result["as_of"])
    p = Prediction.objects.create(
        station=station, scenario=scenario, source=source, as_of=as_of,
        mode=result.get("mode", "balanced"), status=result["status"],
        probability=result.get("probability"), level_m=result.get("level_m"),
        data_age_h=result.get("data_age_h"), flags=result.get("flags", {}),
        reasons=result.get("reasons", []), latency_ms=result.get("latency_ms", {}),
        truth_crosses_within_h=truth if truth is not None else result.get("truth_crosses_within_h"),
    )
    if p.status in ALERTING:
        recent = Alert.objects.filter(station=station, raised_at__gte=as_of - ALERT_COOLDOWN,
                                      prediction__source=source).exists()
        if not recent:
            Alert.objects.create(station=station, prediction=p, raised_at=as_of,
                                 status=p.status, probability=p.probability)
    return p


def record_benchmark(station, result):
    from .models import BenchmarkRun
    sp = result.get("single_prediction_ms", {}).get("total", {})
    return BenchmarkRun.objects.create(
        station=station, kind="latency", machine=result.get("environment", {}).get("machine", ""),
        p50_ms=sp.get("p50"), p95_ms=sp.get("p95"),
        predictions_per_second=result.get("model_predictions_per_second"), results=result,
    )
