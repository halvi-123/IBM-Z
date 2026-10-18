from django.core.management.base import BaseCommand

from alerts import model_client as mc
from alerts.models import Station


class Command(BaseCommand):
    help = "Create the Thorverton station; fills the threshold from the model service if it is reachable."

    def handle(self, *args, **opts):
        s, created = Station.objects.get_or_create(
            name="Thorverton", defaults={"river": "River Exe", "horizon_h": 24})
        try:
            h = mc.health()
            s.level_threshold_m, s.horizon_h = h["level_threshold_m"], h["horizon_h"]
            s.save()
            self.stdout.write(f"Model service reachable; threshold {s.level_threshold_m:.3f} m")
        except mc.ModelServiceError as e:
            self.stdout.write(self.style.WARNING(f"Model service not reachable ({e}); threshold left blank"))
        self.stdout.write(self.style.SUCCESS(f"{'Created' if created else 'Updated'} station {s}"))
