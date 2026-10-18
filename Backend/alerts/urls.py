from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health),
    path("stations/", views.stations),
    path("stations/<int:pk>/status/", views.station_status),
    path("stations/<int:pk>/replay/", views.station_replay),
    path("predictions/", views.predictions),
    path("alerts/", views.alerts),
    path("alerts/<int:pk>/ack/", views.acknowledge_alert),
    path("metrics/", views.metrics),
    path("benchmarks/", views.benchmarks),
]
