# Mem0 OSS 2.1.0

SDK [`v2.1.0`](https://github.com/mem0ai/mem0/tree/v2.1.0) (`19f7134`). The answer and judge prompts of the official [LoCoMo benchmark](https://github.com/mem0ai/memory-benchmarks/tree/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/locomo) are copied unchanged into `prompts.py`; their hashes are in `source.json`.

- **Build**: one store per conversation, written one turn at a time in order. `speaker_a` is the user and the other speaker the assistant; only the text and the session dates go in, not the photo captions. The OSS SDK takes no timestamp, so the date goes into the message prefix and `created_at`.
- **Extraction**: the SDK's ADD-only prompt and default settings (temperature 0.1, max_tokens 2000, top_p 0.1).
- **Retrieval**: `search` with top_k 200 and threshold 0.1, no reranking; the SDK's fusion of dense, BM25 and entity search is unchanged.
- **Shared protocol** (the paper's Mem0 row): the Mem0 2025 answer template and judge, as for every other method.
- **Native protocol** (`reanswer.py --protocol native`): the benchmark's answer prompt, with the last session's date as the reference date, and its 2026 judge; category 3 reference answers are cut at the first semicolon, as upstream does.

| File | Role |
| --- | --- |
| `method.py` | build (each turn's API responses and a database snapshot are saved; an interrupted build resumes from the last committed turn), retrieval, and the native answer and judge |
| `reanswer.py` | answers again from a built store's saved retrievals, with another answer model (the writer–reader cells) or the native protocol |

Before the first run: `python -m spacy download en_core_web_sm`.
