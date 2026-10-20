"""Mem0 OSS 2.1.0 with the public 2026 LoCoMo answer/judge protocol."""

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from common.artifacts import checkpoint, digest, now, read, save, verified

from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    preprocess_answer,
)

VERSION = "2.1.0"
PROTOCOL = "mem0_benchmarks_4b61c5d_locomo"
SOURCE = Path(__file__).with_name("source.json")


def check_environment():
    os.environ["MEM0_TELEMETRY"] = "false"
    if importlib.metadata.version("mem0ai") != VERSION:
        raise ValueError(f"Install mem0ai[nlp]=={VERSION}")
    from mem0.utils.spacy_models import get_nlp_full, get_nlp_lemma

    if get_nlp_full() is None or get_nlp_lemma() is None:
        raise ValueError(
            "Install en_core_web_sm: python -m spacy download en_core_web_sm"
        )


@lru_cache(maxsize=1024)
def timestamp(date):
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    raise ValueError(f"Invalid LoCoMo date: {date}")


def messages(sample):
    if not sample.speaker_a:
        raise ValueError("Mem0 requires the dataset's explicit speaker_a")
    return [
        {
            "turn_id": t.id,
            "created_at": timestamp(t.date),
            "message": {
                "role": "user" if t.speaker == sample.speaker_a else "assistant",
                "content": f"[{t.date}] {t.speaker}: {t.text}",
            },
        }
        for t in sorted(sample.turns, key=lambda t: timestamp(t.date))
    ]


def sdk_client(client):
    """Keep SDK prompts/parameters intact; meter both SDK transports by role."""
    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion

    def request(endpoint, payload, role):
        try:
            if role == "work":
                payload = client.generation_payload(payload, role)
            body = client.post(endpoint, payload, role=role)
            phase = getattr(client.local, "memory_phase", "query")
            client.record(
                body,
                "embedding" if role == "embedding" else "construction",
                payload["model"],
                phase=phase,
                request=payload,
                role=role,
            )
            if role == "work":
                client.validate_generation_response(body, payload["model"])
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Mem0 extraction truncated")
                # Mem0 strips a Markdown code fence before parsing; check what it will parse.
                from mem0.memory.utils import remove_code_blocks

                facts = json.loads(remove_code_blocks(choice["message"]["content"]))
                if not isinstance(facts, dict) or not isinstance(
                    facts.get("memory"), list
                ):
                    raise ValueError("Mem0 extraction requires a memory list")
                if any(
                    not isinstance(m, dict)
                    or not isinstance(m.get("text"), str)
                    or not m["text"].strip()
                    for m in facts["memory"]
                ):
                    raise ValueError("Invalid extracted memory")
            else:
                data = sorted(body["data"], key=lambda x: x["index"])
                if [x["index"] for x in data] != list(range(len(payload["input"]))):
                    raise ValueError("Embedding count/index mismatch")
                if any(
                    len(x["embedding"]) != 1536
                    or not all(math.isfinite(v) for v in x["embedding"])
                    for x in data
                ):
                    raise ValueError("Invalid embedding vectors")
            schema = ChatCompletion if role == "work" else CreateEmbeddingResponse
            return schema.model_validate(body)
        except Exception as error:
            if isinstance(error, (ValueError, KeyError, TypeError)):
                client.local.invalid_output = True
            # SDK catches some failures internally; never commit a partial build/search.
            client.local.sdk_failed = True
            raise

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **p: request("/chat/completions", p, "work")
            )
        ),
        embeddings=SimpleNamespace(
            create=lambda **p: request("/embeddings", p, "embedding")
        ),
        close=lambda: None,
    )


def open_memory(directory, client):
    from mem0 import Memory

    directory.mkdir(parents=True, exist_ok=True)
    work, embedding = client.connections["work"], client.connections["embedding"]
    memory = Memory.from_config(
        {
            "history_db_path": str(directory / "history.db"),
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(directory / "qdrant"),
                    "collection_name": "locomo",
                    "embedding_model_dims": 1536,
                    "on_disk": True,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": work["model"],
                    "api_key": work["key"],
                    "openai_base_url": work["base_url"],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "top_p": 0.1,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding["model"],
                    "api_key": embedding["key"],
                    "openai_base_url": embedding["base_url"],
                    "embedding_dims": 1536,
                },
            },
        }
    )
    memory.llm.client.close()
    memory.embedding_model.client.close()
    memory.llm.client = memory.embedding_model.client = sdk_client(client)
    if memory.vector_store._get_bm25_encoder() is None:
        close(memory)
        raise ValueError("Mem0 requires working BM25 (fastembed)")
    return memory


def close(memory):
    memory.close()
    memory.vector_store.client.close()


def file_hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def snapshot(runtime, destination):
    """Back up committed SQLite pages while the owning conversation is idle."""
    if destination.exists():
        shutil.rmtree(
            destination
        )  # Only an uncommitted snapshot with this turn number.
    destination.mkdir(parents=True)
    for source in runtime.rglob("*"):
        if not source.is_file() or source.name.endswith(
            ("-wal", "-shm", "-journal", ".lock")
        ):
            continue
        target = destination / source.relative_to(runtime)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream:
            is_sqlite = stream.read(16) == b"SQLite format 3\x00"
        if is_sqlite:
            src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
            dst = sqlite3.connect(target)
            try:
                src.backup(dst)
            finally:
                src.close()
                dst.close()
        else:
            shutil.copy2(source, target)
    return file_hashes(destination)


def build(sample, client, directory, stop, turn_limit=None):
    content = messages(sample)
    if turn_limit is not None:
        content = content[:turn_limit]
    identity = {
        "version": VERSION,
        "client": client.identity,
        "content": digest(content),
        "method": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    current = verified(directory / "current.json", identity)
    runtime = directory / "runtime"
    if runtime.exists():
        shutil.rmtree(
            runtime
        )  # Derived state; the committed snapshot is authoritative.
    start = 0
    if current:
        saved = directory / "snapshots" / str(current["next_turn"])
        if file_hashes(saved) != current["files"]:
            raise ValueError("Mem0 database snapshot checksum mismatch")
        shutil.copytree(saved, runtime)
        start = current["next_turn"]
    memory = open_memory(runtime, client)
    try:
        for index in range(start, len(content)):
            if stop.is_set():
                raise InterruptedError("Stopping Mem0 before next turn")
            item = content[index]
            scope = directory / "turns" / f"{index:05d}"
            client.reset(scope)
            client.local.memory_phase = "construction"
            client.local.sdk_failed = False
            client.local.invalid_output = False
            started_at, started = now(), time.perf_counter()
            value = memory.add(
                [item["message"]],
                user_id=sample.id,
                metadata={"created_at": item["created_at"], "turn_id": item["turn_id"]},
            )
            if client.local.sdk_failed:
                raise ValueError(
                    "Mem0 swallowed an API failure; database not committed"
                )
            for record in value["results"]:
                stored = memory.vector_store.get(record["id"])
                if stored is None or stored.payload.get("data") != record["memory"]:
                    raise ValueError("Mem0 did not persist an extracted memory")
            state = directory / "snapshots" / str(index + 1)
            hashes = snapshot(runtime, state)
            save(
                scope / "result.json",
                {
                    "value": value,
                    "calls": client.calls,
                    "started_at": started_at,
                    "active_seconds": time.perf_counter() - started,
                },
            )
            checkpoint(
                directory / "current.json",
                identity,
                {"next_turn": index + 1, "files": hashes},
            )
            for old in (directory / "snapshots").iterdir():
                if old != state:
                    shutil.rmtree(old)
            if (index + 1) % 25 == 0 or index + 1 == len(content):
                print(f"Mem0 {sample.id}: {index + 1}/{len(content)} turns", flush=True)
        calls = [
            c
            for p in sorted((directory / "turns").glob("*/result.json"))
            for c in read(p)["calls"]
        ]
        save(directory / "index.json", {"turns": len(content), "calls": calls})
        return memory
    except BaseException:
        close(memory)
        if getattr(client.local, "invalid_output", False):
            scope = client.local.scope
            if (scope / "calls").exists():
                number = len(list(scope.glob("invalid-calls-*"))) + 1
                shutil.move(
                    str(scope / "calls"), str(scope / f"invalid-calls-{number:04d}")
                )
        raise


def retrieve(memory, sample, question, client, top_k):
    client.local.sdk_failed = False
    client.local.memory_phase = "query"
    result = memory.search(
        question, top_k=top_k, filters={"user_id": sample.id}, threshold=0.1,
        rerank=False,
    )
    if client.local.sdk_failed:
        raise ValueError("Mem0 swallowed an API failure during search")
    memories = result["results"]
    reference_date = (
        max(sample.turns, key=lambda t: timestamp(t.date)).date
        if sample.turns else None
    )
    return {
        "type": "mem0_oss",
        "memories": memories,
        "reference_date": reference_date,
        "context": "\n".join(
            f"{m.get('created_at', '')}: {m['memory']}"
            for m in sorted(memories, key=lambda m: m.get("created_at", ""))
        ),
        "answer_prompt": get_answer_generation_prompt(
            question, memories, reference_date=reference_date
        ),
    }


def answer(client, context):
    result = client.chat(
        None, context, role="answer", max_tokens=4096, seed=None,
        token_limit_parameter="max_tokens",
    )
    return result.rsplit("ANSWER:", 1)[-1].strip()


def judge(client, qa, prediction):
    reference = preprocess_answer(qa["category"], qa["reference"])
    prompt = get_judge_prompt(qa["category"], qa["question"], reference, prediction)
    result = json.loads(
        client.chat(
            JUDGE_SYSTEM_PROMPT,
            prompt,
            role="judge",
            kind="judge",
            max_tokens=4096,
            seed=None,
            token_limit_parameter="max_tokens",
            response_format={"type": "json_object"},
        )
    )
    label = result.get("label", "").upper()
    if label not in ("CORRECT", "WRONG"):
        raise ValueError("Invalid Mem0 judge label")
    return int(label == "CORRECT")
