from django.db import models


class Station(models.Model):
    name = models.CharField(max_length=100, unique=True)
    river = models.CharField(max_length=100, blank=True)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    ea_level_measure = models.CharField(max_length=200, blank=True)
    ea_rain_measures = models.JSONField(default=list, blank=True)
    level_threshold_m = models.FloatField(null=True, blank=True, help_text="High-level threshold used by the model")
    horizon_h = models.PositiveIntegerField(default=24)
    model_api_url = models.CharField(max_length=200, blank=True,
                                     help_text="Overrides settings.MODEL_API_URL for this station (multi-station later)")
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.river})" if self.river else self.name


class Scenario(models.Model):
    """A period or synthetic case the model is run on. Real replays and synthetic stress tests are kept apart."""
    KIND_CHOICES = [("replay", "Real replay"), ("stress", "Synthetic stress test"), ("live", "Live")]
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="scenarios")
    name = models.CharField(max_length=120)
    kind = models.CharField(max_length=10, choices=KIND_CHOICES, default="replay")
    start = models.DateTimeField(null=True, blank=True)
    end = models.DateTimeField(null=True, blank=True)
    params = models.JSONField(default=dict, blank=True, help_text="e.g. {'perturbation': 'rain_x2'}")
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        label = "SYNTHETIC" if self.kind == "stress" else self.get_kind_display()
        return f"{self.name} [{label}]"


class BenchmarkRun(models.Model):
    KIND_CHOICES = [("cv", "Cross-validation"), ("latency", "Latency/throughput"), ("stress", "Stress tests")]
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="benchmarks")
    kind = models.CharField(max_length=10, choices=KIND_CHOICES)
    machine = models.CharField(max_length=50, blank=True, help_text="e.g. s390x for LinuxONE")
    p50_ms = models.FloatField(null=True, blank=True)
    p95_ms = models.FloatField(null=True, blank=True)
    predictions_per_second = models.FloatField(null=True, blank=True)
    results = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

    def __str__(self):
        return f"{self.get_kind_display()} on {self.machine or '?'} @ {self.created:%Y-%m-%d %H:%M}"


class Prediction(models.Model):
    SOURCE_CHOICES = [("live", "Live"), ("replay", "Replay"), ("posted", "Posted readings")]
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="predictions")
    scenario = models.ForeignKey(Scenario, null=True, blank=True, on_delete=models.SET_NULL, related_name="predictions")
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default="live")
    as_of = models.DateTimeField()
    mode = models.CharField(max_length=12, default="balanced")
    status = models.CharField(max_length=20)
    probability = models.FloatField(null=True, blank=True)
    level_m = models.FloatField(null=True, blank=True)
    data_age_h = models.FloatField(null=True, blank=True)
    flags = models.JSONField(default=dict, blank=True)
    reasons = models.JSONField(default=list, blank=True)
    latency_ms = models.JSONField(default=dict, blank=True)
    truth_crosses_within_h = models.BooleanField(null=True, blank=True, help_text="Known only for replays")
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-as_of"]
        indexes = [models.Index(fields=["station", "as_of"])]

    def __str__(self):
        return f"{self.station.name} {self.as_of:%Y-%m-%d %H:%M} {self.status}"


class Alert(models.Model):
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="alerts")
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name="alert")
    raised_at = models.DateTimeField()
    status = models.CharField(max_length=20)
    probability = models.FloatField(null=True, blank=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.CharField(max_length=100, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-raised_at"]

    def __str__(self):
        return f"{self.station.name} {self.status} @ {self.raised_at:%Y-%m-%d %H:%M}"
