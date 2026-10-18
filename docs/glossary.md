# Glossary

| Term | Meaning |
|---|---|
| ABOVE THRESHOLD | Status when the river is already at or above the high level. |
| ALERT | Status when the probability of crossing the high level within 24 h is at or above the alert cut-off. |
| Balanced (mode or operating point) | The cut-off with the best F1 on the validation year. The default mode. |
| Brier score | Average squared gap between predicted probability and what happened. Lower is better. |
| Bundle | The saved file `river_model.joblib`: the model plus everything the service needs (features, threshold, cut-offs, training info). |
| Calibration | Whether probabilities mean what they say. If the model says 30%, the river should cross about 30% of the time. |
| Cooldown | ALERT hours less than 12 h apart count as one alert episode, and the backend will not raise a new Alert within 12 h. |
| Cross-validation (leave-one-water-year-out) | Train on two water years, test on the third, rotate so each year is tested once. |
| DATA ISSUE | Status when data looks unreliable: too many missing hours, stuck sensor, or rain gauges offline. |
| DATA STALE | Status when no real level reading has arrived for more than 3 hours. |
| EA Hydrology API | Environment Agency open data service for river level and rainfall readings. |
| Event | A crossing of the high level. Crossings less than 12 h apart count as one event. |
| Events caught | Events with at least one ALERT in the 24 h before they started. |
| F1 | A single score balancing catching positives and avoiding false alarms. |
| False alerts per month | Alert episodes not followed by an event, per 30.44 days. |
| Feature | A number describing the recent past, for example "rain in the last 24 h". |
| Flag | A data-quality check result shown with each prediction (stale, missing hours, stuck sensor, spike, rain offline). |
| High-level threshold | 95th percentile of training river levels (1.375 m for the served model). |
| HistGradientBoosting | The served model: many small decision trees, each correcting the last. From scikit-learn. |
| Holdout | WY2025 (from 2024-10-01). Never used for training or picking cut-offs. Demo replays use it. |
| Horizon | How far ahead we predict: 24 hours. |
| Lead time | Time between the first ALERT and the start of the event. Capped at 24 h. |
| LinuxONE | IBM's Linux mainframe platform. Our VM runs Ubuntu on s390x. |
| LOW | Status when the probability is low and the data looks fine. |
| Operating point | A probability cut-off picked for a goal: high recall, balanced or high precision. |
| Platt scaling | A way to recalibrate probabilities by fitting a small logistic model. Tried on random forest; barely helped. |
| PR-AUC | Area under the precision-recall curve. Useful when positives are rare. Ours: about 0.35 for models, 0.14 for the rain rule. |
| Reasons | Inputs that are unusually high compared with training hours. Descriptive, not model attributions. |
| Reliability curve | Chart of predicted probability against observed frequency, used to judge calibration. |
| Replay | Running the model over a past period hour by hour, as if it were live. |
| ROC-AUC | Chance the model ranks a random positive hour above a random negative one. 0.5 is chance. |
| Rule baseline | A simple rule we built ourselves (24 h rain, or 3 h level rise) scored the same way as the models. |
| s390x | The CPU architecture of IBM Z and LinuxONE. |
| Sensitive (mode) | Alerts at the high-recall cut-off: more alerts, fewer misses. |
| Silent miss | In stress tests: an event with neither an ALERT nor a data warning beforehand. The worst outcome. |
| SSH tunnel | `ssh -L 8000:localhost:8000 <vm>` makes the VM's model API reachable on the laptop. |
| Stress test | A SYNTHETIC test where real input data is deliberately damaged. Never used for training. |
| SYNTHETIC | Label for anything built from artificial or damaged data. Always reported apart from real results. |
| Water year (WY) | October to September, named by the year it ends. WY2025 = 1 Oct 2024 to 30 Sep 2025. |
| WATCH | Status when the probability is elevated but below the alert cut-off. |
