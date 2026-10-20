# Zep: Graphiti OSS 0.30.2

Zep's open-source engine is [Graphiti](https://github.com/getzep/graphiti) (`16cdf70`), run here on Kuzu 0.11.3. Its dependencies are pinned in `requirements.txt` and installed in a separate `.venv-zep`, since Graphiti pins pydantic 2.13.5.

- **Graph**: one Kuzu database per conversation. Each session is one message episode, added in date order, with Graphiti's own extraction, deduplication and temporal edge updates; no communities and no custom ontology. Only the speakers, the text and the session dates go in. Extraction uses Graphiti's `OpenAIGenericClient` JSON schema with `max_tokens` 16384. The same model and effort build the graph and answer.
- **Retrieval**: `EDGE_HYBRID_SEARCH_RRF` (BM25 and cosine) filtered to the conversation, top 20 edges; the context is these facts with their `valid_at` and `invalid_at`. Answer and judge as for the other methods (Mem0 2025 template, GPT-4o-mini judge).
- **Patches**, in `method.py`, with the upstream source untouched:
  - the Kuzu driver does not set `_database`; here it is the conversation id;
  - async query results are released before the database closes, avoiding a use-after-free in Kuzu.

After each session a database snapshot is committed with its hash; an interrupted build resumes from the last committed graph.

```sh
.venv-zep/bin/python src/methods/zep/run.py --model claude-opus-5-5 --reasoning-effort high \
  --output results/zep/opus-opus --run
```

A writer–reader cell (a graph one model built, answered by another) uses `reanswer.py`, in the main venv: it reuses the retrievals saved per question in the store, matched by question text, and changes only the answer model.

```sh
.venv/bin/python src/methods/zep/reanswer.py --store results/zep/mini-mini --answer-model claude-opus-5-5 \
  --reasoning-effort high --output results/zep/mini-opus --run
```
