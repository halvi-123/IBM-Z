# Project docs

Qubit: River Level Alert, team Qubit's entry for the IBM Z Datathon 2026 (theme: "Real-time AI for critical decisions").

> Decision support, not an official warning. Always check Environment Agency flood warnings.

## The project in one paragraph

We watch one river gauge (River Exe at Thorverton, Devon). Every hour the system looks at the last 10 days
of river level and rainfall and estimates the chance that the river will cross a high level within the next
24 hours. It turns that chance into a simple status a first responder can act on (LOW, WATCH, ALERT, ABOVE
THRESHOLD), and it refuses to say LOW when the sensor data looks broken (DATA STALE, DATA ISSUE). The model is
trained and served on an IBM LinuxONE virtual machine (s390x). The project includes a benchmark suite that
measures how fast it runs there and how accurate it really is, compared with a simple rule we built ourselves.

## Where to read next

| Doc | What it covers | Read it if you want to... |
|---|---|---|
| [progress.md](progress.md) | What we have done so far, real results, decisions made, what is still open | know where the project stands today |
| [architecture.md](architecture.md) | Full architecture: 16 diagrams covering system, deployment, data pipeline, training and evaluation, serving, status logic, API, replay, stress tests, benchmark, backend data model, frontend, plus design decisions | see exactly how every part fits together |
| [how-it-works.md](how-it-works.md) | Data, features, model, evaluation, statuses, data-quality flags and stress tests, explained from scratch | understand or explain the system (for example in the video) |
| [repository.md](repository.md) | Every folder and file, how the parts talk to each other, how to run them | find your way around the code |
| [glossary.md](glossary.md) | Short definitions of every term we use | look up a word quickly |
| `screenshots/` | 5 screenshots from the LinuxONE VM: `uname -m`, benchmark, running API, API docs replay, stress tests | prove it ran on LinuxONE |

## Ground rules we keep everywhere

These apply to the UI, the README, these docs and the video.

- Say "crossing a high river level", not "flood".
- Label everything "decision support, not an official warning".
- Our baseline is a rule we built ourselves. We say so, and we do not quote outside accuracy numbers.
- Every metric sits next to the number of events behind it (45 events across 3 years, or 9 in the holdout year).
- Synthetic stress tests are reported separately and labelled SYNTHETIC. They are never used as training data.
- Benchmark numbers from the VM are reported as measured, even if a laptop would be as fast.
