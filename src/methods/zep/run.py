"""Zep (Graphiti OSS) on LoCoMo conv-26: build the graph, then retrieve, answer and judge each question.

Runs in its own environment (.venv-zep, requirements.txt here). Running the same command again
resumes from the committed graph snapshot and the per-question stages.
"""

import argparse
import asyncio
import importlib.metadata
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.artifacts import digest, read, run_lock, save
from common.data import load
from common.eval import judge, summarize, token_f1
from common.llm import Client, connection_settings, context_metrics, public_connections, token_total
from common.usage import PRICES, billed_cost, standard_cost, summarize_resources
from common.answer import generate_answer
from methods.zep.method import COMMIT, MODELS, SETTINGS, VERSION, build, close_graph, episodes, retrieve, sha
from run import ledger, stage

SOURCE = Path(__file__).resolve().parents[2]
DATA = SOURCE.parent / "data/locomo-conv26.json"


def answer_question(qa, retrieval, directory):
    context = retrieval["value"]["context"]
    client = Client(models=MODELS, reasoning_effort=SETTINGS["reasoning_effort"], timeout=180)
    try:
        answer = stage(
            directory / "answer",
            dict(question=qa["question"], context_sha256=digest(context), template="mem0_2025"),
            client,
            lambda: generate_answer(client, qa["question"], context),
        )
        judgment = stage(
            directory / "judge",
            dict(question=qa["question"], reference=qa["reference"], prediction=answer["value"], judge="mem0_2025"),
            client,
            lambda: judge(client, qa["question"], qa["reference"], answer["value"], "mem0"),
        )
        physical = ledger(directory)
        calls = [c for c in physical["calls"] if c["role"] != "judge"]
        unknown = [c for c in physical["unresolved_or_failed_requests"] if c["role"] != "judge"]
        row = {
            **qa,
            "prediction": answer["value"],
            "correct": judgment["value"],
            "f1": token_f1(answer["value"], qa["reference"]),
            "calls": calls + judgment["calls"],
            "generation_tokens": token_total(calls, "generation"),
            "judge_tokens": token_total(judgment["calls"], "judge"),
            **context_metrics(context),
            "query_cost_usd": None if unknown else standard_cost(calls),
            "query_known_cost_usd": standard_cost(calls),
            "unresolved_or_failed_requests": unknown,
            "query_billed_cost_usd": billed_cost(calls),
            "query_latency_s": None,
            "retrieval_latency_s": retrieval["active_seconds"],
            "answer_latency_s": answer["active_seconds"],
            "composed_query_latency_s": retrieval["active_seconds"] + answer["active_seconds"],
            "latency_basis": "separately checkpointed retrieval and answer; not end-to-end",
        }
        save(directory / "row.json", row)
        return row
    finally:
        client.close()


def collect(output, config, failures):
    order = {q: i for i, q in enumerate(config["question_ids"])}
    rows = sorted((read(p) for p in (output / "questions").glob("*/row.json")), key=lambda r: order[r["id"]])
    physical = ledger(output)
    build_calls = [c for c in physical["calls"] if c["phase"] == "construction"]
    # Missing/failed attempts cannot be priced as zero.
    uncertain = [c for c in physical["unresolved_or_failed_requests"] if c["path"].startswith("build/")]
    cost = None if uncertain else standard_cost(build_calls)
    complete = [r["id"] for r in rows] == config["question_ids"] and not failures
    doc = dict(
        method="zep_oss",
        model=config["roles"]["answer"]["model"],
        status="complete" if complete else "incomplete",
        settings=config,
        rows=rows,
        failures=failures,
        summary=summarize(rows) if rows else {},
        resources=summarize_resources(rows) if rows else {},
        construction=[dict(cost_usd=cost)],
        logical_construction_cost_usd=cost,
        construction_known_cost_usd=standard_cost(build_calls),
        uncertain_construction_attempts=uncertain,
        build_calls=build_calls,
    )
    save(output / "summary.json", doc)
    return doc


async def execute(args, config, sample, questions):
    output = args.output
    api_limit = asyncio.Semaphore(16)
    answering = asyncio.Semaphore(args.workers)
    failures, tasks, g = [], [], None

    async def answer(qa, retrieval, directory):
        async with answering:
            try:
                await asyncio.to_thread(answer_question, qa, retrieval, directory)
                print(f"answered {qa['id']}", flush=True)
            except Exception as error:
                traceback.print_exc()
                failures.append(dict(id=qa["id"], type=type(error).__name__, error=str(error)))

    try:
        g, state = await build(sample, output / "build" / sample.id, api_limit, args.turn_limit)
        # Graph searches share one Kuzu connection, so retrieval is sequential; answers overlap.
        for qa in questions:
            directory = output / "questions" / qa["id"].replace(":", "_")
            if (directory / "row.json").exists():
                continue
            retrieval = await retrieve(g, qa, directory / "retrieval", api_limit, state["snapshot_sha256"])
            tasks.append(asyncio.create_task(answer(qa, retrieval, directory)))
        await asyncio.gather(*tasks)
    except Exception as error:
        traceback.print_exc()
        failures.append(dict(conversation=sample.id, type=type(error).__name__, error=str(error)))
    finally:
        if g is not None:
            await close_graph(g)
    return collect(output, config, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="graph construction and answering; judge and embedding stay fixed")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="none")
    parser.add_argument("--workers", type=int, default=8, help="questions answered concurrently")
    parser.add_argument("--limit", type=int, help="smoke run: only the first N questions")
    parser.add_argument("--turn-limit", type=int, help="smoke run: only the first N turns of the history")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    MODELS.update(work=args.model, answer=args.model)
    SETTINGS.update(reasoning_effort=args.reasoning_effort)
    args.output = args.output.resolve()
    if args.turn_limit and not args.limit:
        raise ValueError("Partial history is only allowed in a smoke run")
    if importlib.metadata.version("graphiti-core") != VERSION:
        raise ValueError("Graphiti version mismatch")
    pairs = load(DATA)
    sample = pairs[0][0]
    questions = [q for _, q in pairs][: args.limit]
    config = dict(
        graphiti_commit=COMMIT,
        graphiti_version=VERSION,
        data_sha256=sha(DATA),
        population="smoke" if args.limit else "full",
        question_ids=[q["id"] for q in questions],
        turn_limit=args.turn_limit,
        ingestion="one complete text-only session per episode, sequential chronological order",
        retrieval="EDGE_HYBRID_SEARCH_RRF, k=20, conversation filter, timestamped facts",
        backend="Kuzu 0.11.3; upstream driver; 256 MiB buffer per conversation",
        extraction="upstream OpenAIGenericClient json_schema, temperature=0, max_tokens=16384",
        answer_protocol="mem0_2025",
        judge_protocol="mem0_2025",
        reasoning_effort=args.reasoning_effort,
        prices=PRICES,
        roles=public_connections(connection_settings(models=MODELS)),
        api_concurrency=16,
        answer_workers=args.workers,
        source_sha256={str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob("*.py"))},
        dependencies={k: importlib.metadata.version(k)
                      for k in ["graphiti-core", "kuzu", "openai", "anthropic", "pydantic", "neo4j", "numpy", "httpx"]},
    )
    with run_lock(args.output):
        manifest = args.output / "manifest.json"
        if manifest.exists():
            if read(manifest) != config:
                raise ValueError("Frozen run configuration/source differs; use the saved source or a new output")
        else:
            save(manifest, config)
            shutil.copytree(SOURCE, args.output / "source/src", ignore=shutil.ignore_patterns("__pycache__"))
        print(json.dumps(dict(population=config["population"], questions=len(questions),
                              episodes=len(episodes(sample, args.turn_limit)))), flush=True)
        if args.run:
            result = asyncio.run(execute(args, config, sample, questions))
            if result["status"] != "complete":
                raise SystemExit(1)


if __name__ == "__main__":
    main()
