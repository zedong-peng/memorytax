"""Pinned Graphiti OSS with durable API recording and local graph snapshots."""

import asyncio
import functools
import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"

from common.artifacts import checkpoint, digest, save, verified
from common.llm import Client
from methods.mem0.method import timestamp

VERSION = "0.30.2"
COMMIT = "16cdf7045378c8d53ae01f94e2fa60d238cb0f68"
MODELS = dict(
    work="gpt-4o-mini",
    answer="gpt-4o-mini",
    judge="gpt-4o-mini",
    embedding="text-embedding-3-small",
)
# run.py sets the build/answer model and its reasoning effort before anything is called.
SETTINGS = dict(reasoning_effort="none")


def client():
    return Client(models=MODELS, reasoning_effort=SETTINGS["reasoning_effort"], timeout=180)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(sample, turn_limit=None):
    groups = {}
    turns = sorted(sample.turns, key=lambda t: timestamp(t.date))
    for turn in turns[:turn_limit]:
        groups.setdefault((turn.session, timestamp(turn.date)), []).append(turn)
    return [
        dict(
            name=f"{sample.id}/session_{sid}",
            date=date,
            body="\n".join(f"{t.speaker}: {t.text}" for t in rows),
            turn_ids=[t.id for t in rows],
        )
        for (sid, date), rows in groups.items()
    ]


class Recorder:
    """Separate request scopes keep Graphiti's async API calls replayable."""

    def __init__(self, directory, semaphore, phase):
        self.directory, self.semaphore, self.phase = directory, semaphore, phase
        self.counts = {}
        self.calls = []

    async def request(self, endpoint, role, payload):
        api = client()
        if role == "work":
            # Same model settings as every other path: the run's effort, no temperature.
            payload = api.generation_payload(payload, role)
            schema = (payload.get("response_format") or {}).get("json_schema")
            if schema is not None and "strict" not in schema and "luna" in payload["model"]:
                # Graphiti leaves strict unset on purpose (its schemas are not strict-mode);
                # some OpenAI-compatible routes treat unset as strict and reject them.
                payload = {
                    **payload,
                    "response_format": {
                        **payload["response_format"],
                        "json_schema": {**schema, "strict": False},
                    },
                }
        key = digest({"endpoint": endpoint, "role": role, "payload": payload})
        occurrence = self.counts.get(key, 0)
        self.counts[key] = occurrence + 1
        directory = self.directory / "api" / f"{key}-{occurrence}"

        def invoke():
            api.reset(directory)
            try:
                body = api.post(endpoint, payload, role=role)
                api.record(
                    body,
                    "embedding" if role == "embedding" else "construction",
                    payload["model"],
                    phase=self.phase,
                    request=payload,
                    role=role,
                )
                save(directory / "usage.json", api.calls)
                return body, api.calls
            finally:
                api.close()

        async with self.semaphore:
            body, calls = await asyncio.to_thread(invoke)
        self.calls.extend(calls)
        if role == "work":
            from openai.types.chat import ChatCompletion

            if body["choices"][0].get("finish_reason") == "length":
                raise ValueError(
                    "Graphiti extraction truncated; retain response and fail"
                )
            json.loads(body["choices"][0]["message"]["content"])
            return ChatCompletion.model_validate(body)
        from openai.types import CreateEmbeddingResponse

        vectors = sorted(body["data"], key=lambda x: x["index"])
        inputs = payload["input"]
        n = len(inputs) if isinstance(inputs, list) else 1
        if [x["index"] for x in vectors] != list(range(n)):
            raise ValueError("Embedding indices/count differ")
        if any(len(x["embedding"]) != 1536 for x in vectors):
            raise ValueError("Expected 1536-dimensional embeddings")
        return CreateEmbeddingResponse.model_validate(body)

    def sdk(self):
        async def chat(**payload):
            return await self.request("/chat/completions", "work", payload)

        async def embedding(**payload):
            return await self.request("/embeddings", "embedding", payload)

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
            embeddings=SimpleNamespace(create=embedding),
        )


def graph(database, recorder):
    import kuzu
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    # Resource settings only; upstream driver, schema and search stay intact.
    if not isinstance(kuzu.Database, functools.partial):
        kuzu.Database = functools.partial(
            kuzu.Database, buffer_pool_size=256 * 1024**2, max_num_threads=2
        )

    class UnusedReranker(CrossEncoderClient):
        async def rank(self, query, passages):
            raise RuntimeError("This protocol uses RRF, never a cross encoder")

    driver = KuzuDriver(str(database))
    sdk = recorder.sdk()
    return Graphiti(
        graph_driver=driver,
        llm_client=OpenAIGenericClient(
            LLMConfig(model=MODELS["work"], small_model=MODELS["work"], temperature=0),
            client=sdk,
            max_tokens=16384,
            structured_output_mode="json_schema",
        ),
        embedder=OpenAIEmbedder(
            OpenAIEmbedderConfig(
                embedding_model=MODELS["embedding"], embedding_dim=1536
            ),
            client=sdk,
        ),
        cross_encoder=UnusedReranker(),
        max_coroutines=8,
    )


def set_recorder(g, recorder):
    sdk = recorder.sdk()
    g.llm_client.client = sdk
    g.embedder.client = sdk


async def close_graph(g):
    # AsyncConnection futures retain native query results until the next loop tick.
    # Closing the database first causes a use-after-free in their destructors.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    gc.collect()
    g.driver.client.executor.shutdown(wait=True)
    for connection in g.driver.client.connections:
        connection.close()
    g.driver.db.close()


async def build(sample, directory, semaphore, turn_limit=None):
    from graphiti_core.nodes import EpisodeType

    directory.mkdir(parents=True, exist_ok=True)
    items = episodes(sample, turn_limit)
    identity = dict(conversation=sample.id, episodes=items, graphiti=COMMIT)
    state_path = directory / "state.json"
    state = verified(state_path, identity)
    database = directory / "runtime.kuzu"
    # Discard only this run's scratch graph; immutable committed snapshots survive.
    for path in directory.glob("runtime.kuzu*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    start = 0
    if state:
        snapshot = directory / state["snapshot"]
        if sha(snapshot) != state["snapshot_sha256"]:
            raise ValueError("Graph snapshot checksum mismatch")
        shutil.copy2(snapshot, database)
        start = state["completed"]
    recorder = Recorder(directory / "setup", semaphore, "build")
    g = graph(database, recorder)
    # 0.30.2's Kuzu constructor omits the base driver's namespace field.
    # Each conversation already has its own physical database.
    g.driver._database = sample.id
    try:
        if start == 0:
            await g.driver.graph_ops.build_indices_and_constraints(g.driver)
        for i, episode in enumerate(items[start:], start):
            scope = directory / "episodes" / f"{i:04d}"
            recorder = Recorder(scope, semaphore, "build")
            set_recorder(g, recorder)
            began = time.perf_counter()
            value = await g.add_episode(
                name=episode["name"],
                episode_body=episode["body"],
                source_description="LoCoMo text-only conversation session",
                reference_time=datetime.fromisoformat(episode["date"]),
                source=EpisodeType.message,
                group_id=sample.id,
                update_communities=False,
            )
            await g.driver.execute_query("CHECKPOINT")
            snapshot = directory / "snapshots" / f"{i:04d}.kuzu"
            snapshot.parent.mkdir(exist_ok=True)
            temporary = snapshot.with_suffix(".pending")
            shutil.copy2(database, temporary)
            os.replace(temporary, snapshot)
            checkpoint(
                scope / "result.json",
                episode,
                dict(
                    graph=value.model_dump(mode="json"),
                    calls=recorder.calls,
                    seconds=time.perf_counter() - began,
                ),
            )
            state = dict(
                completed=i + 1,
                snapshot=str(snapshot.relative_to(directory)),
                snapshot_sha256=sha(snapshot),
                turns=sum(len(e["turn_ids"]) for e in items[: i + 1]),
            )
            checkpoint(state_path, identity, state)
            print(
                f"build {sample.id} {i + 1}/{len(items)} ({state['turns']} turns)",
                flush=True,
            )
    except BaseException:
        await close_graph(g)
        raise
    return g, state


async def retrieve(g, qa, directory, semaphore, graph_hash):
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    identity = dict(
        question=qa.get("original_question", qa["question"]),
        group=qa["id"].split(":")[0],
        graph_sha256=graph_hash,
        recipe="EDGE_HYBRID_SEARCH_RRF",
        k=20,
    )
    path = directory / "result.json"
    previous = verified(path, identity)
    if previous is not None:
        return previous
    recorder = Recorder(directory, semaphore, "query")
    set_recorder(g, recorder)
    began = time.perf_counter()
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 20
    result = await g.search_(
        identity["question"], config=config, group_ids=[identity["group"]]
    )
    edges = [
        e.model_dump(mode="json", exclude={"fact_embedding"}) for e in result.edges
    ]
    if any(e["group_id"] != identity["group"] for e in edges):
        raise ValueError("Cross-conversation retrieval")
    context = "\n".join(
        f"[valid_at={e['valid_at']}; invalid_at={e['invalid_at']}] {e['fact']}"
        for e in edges
    )
    value = dict(
        value=dict(context=context, edges=edges),
        calls=recorder.calls,
        active_seconds=time.perf_counter() - began,
        resumed=False,
    )
    checkpoint(path, identity, value)
    return value
