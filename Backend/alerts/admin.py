from django.contrib import admin

from .models import Alert, BenchmarkRun, Prediction, Scenario, Station

admin.site.site_header = "River alert - model performance admin"


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ("name", "river", "level_threshold_m", "horizon_h", "active")


@admin.register(Scenario)
class ScenarioAdmin(admin.ModelAdmin):
    list_display = ("name", "station", "kind", "start", "end", "created")
    list_filter = ("kind", "station")


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    list_display = ("created", "station", "kind", "machine", "p50_ms", "p95_ms", "predictions_per_second")
    list_filter = ("kind", "machine")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("as_of", "station", "source", "mode", "status", "probability", "level_m", "truth_crosses_within_h")
    list_filter = ("status", "source", "mode", "station")
    date_hierarchy = "as_of"


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("raised_at", "station", "status", "probability", "acknowledged")
    list_filter = ("status", "acknowledged", "station")
