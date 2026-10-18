# Frontend

Plain HTML + JS, Chart.js from a CDN, no build step. Serve with `python -m http.server 5173` in this folder
(model API tunnel on :8000 must be open).

- `index.html`: product page (what it is, how it works, live evidence figures). "Run the Dec 2024 simulation" opens the console.
- `console.html`: responder console plus judge evidence. `console.html#replay` starts the Dec 2024 replay on load,
  `#replay-20` also sets the speed.

Planned views (both now in `console.html`):

- **Responder view**: current status (LOW / WATCH / ALERT / ABOVE THRESHOLD / DATA STALE / DATA ISSUE),
  probability of crossing within 24 h, reasons, data-quality flags, alert feed, live/replay toggle with speed control.
- **Judge view**: model vs rule baseline table (events caught, false alerts per month, lead time),
  reliability curve, synthetic stress-test results (labelled SYNTHETIC), latency on s390x.

Every screen shows: "Decision support, not an official warning."

## API (Django backend, default http://localhost:8001/api/)
| Method | Path | Returns |
|---|---|---|
| GET | `health/` | backend + model-service info (`model_service.suggested_replay_range`) |
| GET | `stations/` | stations |
| GET | `stations/<id>/status/?mode=balanced\|sensitive` | live prediction (stored, may raise an alert) |
| GET | `stations/<id>/replay/?start=&end=&mode=` | hourly rows `{time, level, prob, status, age_h, stuck, spike, rain_offline, crosses_within_h}`, `events`, `warnings`, `stream_url` |
| GET | `predictions/?station=&source=&limit=` | stored predictions |
| GET | `alerts/?station=` | alert feed |
| POST | `alerts/<id>/ack/` `{"by": "name"}` | acknowledge |
| GET | `metrics/` | `summary` (cross-validation table), `reliability` curves, `last_benchmark` |
| GET / POST | `benchmarks/` | stored latency runs / run a new one on the VM |

Live-style replay: open `stream_url` with `new EventSource(url)`. Each message is one prediction
(same shape as `status/`) plus `truth_crosses_within_h`; a final `done` event ends the stream.
Replay the holdout year (from 2024-10-01); earlier periods overlap training and come with a warning.
