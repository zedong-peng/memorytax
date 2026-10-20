# Code

```text
data/locomo-conv26.json ─► reproduce.py ─► results/<method>/<model>/
```

| Path | Role |
| --- | --- |
| `reproduce.py` | entry point: each cell (method × model), its command, where it writes, and whether it finished |
| `run.py` | the per-question runner for full context, BM25 and Mem0 |
| `check_api.py` | checks the keys, endpoints and usage fields in `.env` with a few real requests |
| `methods/<method>/` | each method's retrieval or store, and its own runners |
| `methods/agent/` | the per-question runner for Pi, Codex and Claude Code: one Harbor container per question, open only to the model endpoint |
| `common/` | model client (four roles; every request and response saved), checkpoints and locks, data loading, the shared answer step, judge, official F1 (`locomo/`), prices |

## A run

Model calls have four roles: `WORK` builds the memory, `ANSWER` answers, `JUDGE` grades and `EMBEDDING` embeds. Each model has its own key and endpoint in `.env` (see `.env.example`), whichever role calls it.

Each run is a directory:

- `manifest.json`: data, code, models, settings and prices. A resume is refused if any of them differs; a new configuration needs a new directory.
- `source/`: a snapshot of the source at the start, without `.env`.
- `build/`: the build's checkpoints (Mem0's database after each turn), so an interrupted build resumes from the last commit.
- `questions/<id>/{retrieval,answer,judge}/`: each question's three stages, saved apart. Every request and response is saved before it is parsed; a resume replays them without calling or paying again.
- `summary.json`: per-question results and the cost ledger. Failed or uncertain requests are in the ledger too; an unknown cost is null, never zero.

Running the same command again resumes.
