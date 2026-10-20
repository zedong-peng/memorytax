"""Checkpointed LoCoMo experiments. Dry run by default; --run enables API calls."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common.answer import generate_answer
from common.artifacts import checkpoint, digest, now, read, run_lock, save, verified
from common.data import load
from common.eval import judge, summarize, token_f1
from common.llm import (
    DEFAULT_ENV,
    ROLES,
    Client,
    connection_settings,
    context_metrics,
    public_connections,
    token_total,
    usage_record,
)
from common.usage import (
    PRICES,
    billed_cost,
    standard_cost,
    summarize_resources,
    summarize_roles,
)
from methods.bm25 import method as bm25
from methods.full_context import method as full_context
from methods.mem0 import method as mem0_oss

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"
METHODS = ("full_context", "bm25", "mem0_oss")
STOP = threading.Event()


def native_mem0(args):
    return args.method == "mem0_oss" and getattr(args, "mem0_protocol", "native") == "native"


def fixed_files():
    return sorted(
        p
        for p in SOURCE.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and (p.suffix in (".py", ".json", ".md") or p.name == "LICENSE")
    )


def code_digest():
    return digest(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in fixed_files()
        }
    )


def selection(args):
    examples = load(args.data.read_bytes())
    if (args.population == "full") == bool(args.limit):
        raise ValueError("A full run takes every question; a smoke run needs --limit")
    return examples[: args.limit] if args.limit else examples


def stage(directory, identity, client, operation):
    """Resume completed stages or recover durable API responses in an unfinished one."""
    identity = {**identity, "client": client.identity}
    result_path = directory / "result.json"
    result = verified(result_path, identity)
    if result is not None:
        return result
    if STOP.is_set():
        raise InterruptedError("Stopping before next stage")
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted((directory / "attempts").glob("*.json"))
    attempt_path = directory / "attempts" / f"{len(attempts) + 1:04d}.json"
    started = time.perf_counter()
    started_at = now()
    save(attempt_path, {"status": "running", "started_at": started_at})
    client.reset(directory)
    try:
        value = operation()
        result = {
            "value": value,
            "calls": client.calls,
            "started_at": started_at,
            "active_seconds": time.perf_counter() - started,
            "ended_at": now(),
            "resumed": bool(attempts),
        }
        checkpoint(result_path, identity, result)
        save(
            attempt_path,
            {
                "status": "succeeded",
                "started_at": started_at,
                "ended_at": now(),
                "active_seconds": result["active_seconds"],
            },
        )
        return result
    except BaseException as error:
        save(
            attempt_path,
            {
                "status": "interrupted"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed",
                "error_type": type(error).__name__,
                "started_at": started_at,
                "ended_at": now(),
                "active_seconds": time.perf_counter() - started,
            },
        )
        # Invalid model output may be retried without losing the response/paid usage.
        # Transport failures keep successful earlier calls available for replay.
        if (
            isinstance(error, (ValueError, KeyError, TypeError))
            and (directory / "calls").exists()
        ):
            shutil.move(
                str(directory / "calls"),
                str(directory / f"invalid-calls-{len(attempts) + 1:04d}"),
            )
        raise


def retrieve(sample, qa, client, args, indexes):
    """The context a method gives the answer model for one question."""
    if args.method == "full_context":
        return full_context.retrieve(sample, qa)
    if args.method == "bm25":
        return bm25.retrieve(sample, qa, args.top_k, args.window)
    if args.method == "mem0_oss":
        return mem0_oss.retrieve(indexes[sample.id], sample, qa["question"], client, args.top_k)
    raise ValueError(f"Unsupported method: {args.method}")


def evaluate(sample, qa, client, args, directory, indexes):
    identity = {"qa_id": qa["id"], "question": qa["question"], "method": args.method}
    retrieval = stage(
        directory / "retrieval",
        identity,
        client,
        lambda: retrieve(sample, qa, client, args, indexes),
    )
    context = retrieval["value"]["context"]
    answer = stage(
        directory / "answer",
        {
            "question": qa["question"],
            "context_sha256": digest(context),
            "answer_prompt_sha256": digest(retrieval["value"].get("answer_prompt")),
            "template": mem0_oss.PROTOCOL
            if native_mem0(args)
            else "mem0_2025",
        },
        client,
        lambda: (
            mem0_oss.answer(client, retrieval["value"]["answer_prompt"])
            if native_mem0(args)
            else generate_answer(client, qa["question"], context)
        ),
    )
    prediction = answer["value"]
    judgment = stage(
        directory / "judge",
        {
            "prediction": prediction,
            "reference": qa["reference"],
            "question": qa["question"],
            "judge": mem0_oss.PROTOCOL if native_mem0(args) else "mem0_2025",
        },
        client,
        lambda: (
            mem0_oss.judge(client, qa, prediction)
            if native_mem0(args)
            else judge(client, qa["question"], qa["reference"], prediction, "mem0")
        ),
    )
    calls = retrieval["calls"] + answer["calls"] + judgment["calls"]
    query_calls = retrieval["calls"] + answer["calls"]
    # Resumed question latency is not mixed into uninterrupted latency percentiles.
    resumed = retrieval["resumed"] or answer["resumed"]
    row = {
        **qa,
        "prediction": prediction,
        "correct": judgment["value"],
        "f1": token_f1(prediction, qa["reference"]),
        "calls": calls,
        "input_tokens": token_total(calls, "generation", "input_tokens"),
        "output_tokens": token_total(calls, "generation", "output_tokens"),
        "generation_tokens": token_total(calls, "generation"),
        "embedding_tokens": token_total(calls, "embedding"),
        "judge_tokens": token_total(calls, "judge"),
        **context_metrics(context),
        "query_started_at": retrieval["started_at"],
        "query_latency_s": None
        if resumed
        else retrieval["active_seconds"] + answer["active_seconds"],
        "retrieval_latency_s": None
        if retrieval["resumed"]
        else retrieval["active_seconds"],
        "resumed": resumed,
        "query_cost_usd": standard_cost(query_calls),
        "query_billed_cost_usd": billed_cost(query_calls),
    }
    save(directory / "row.json", row)
    return row


def ledger(directory):
    """Recompute actual successful calls once, including failed/invalid attempts."""
    calls, uncertain = [], []
    for path in sorted(directory.rglob("response.json")):
        saved = read(path)
        response = verified(path, saved["identity"])
        body, request = response["body"], saved["identity"]
        relative = str(path.relative_to(directory))
        phase = (
            "construction"
            if relative.startswith("build/")
            else ("judge" if "/judge/" in relative else "query")
        )
        kind = (
            "embedding"
            if request["endpoint"] == "/embeddings"
            else ("judge" if phase == "judge" else "generation")
        )
        calls.append(
            {
                "path": relative,
                "phase": phase,
                **usage_record(
                    body, kind, request["payload"]["model"], request["role"]
                ),
                "base_url": request["base_url"],
                **response["timing"],
            }
        )
    for path in sorted(directory.rglob("attempt-*.json")):
        value = read(path)
        if value["status"] != "succeeded":
            # A crash after response commit but before attempt status update is resolved locally.
            if (
                value["status"] == "running"
                and (path.parent / "response.json").exists()
            ):
                timing = read(path.parent / "response.json")["result"]["timing"]
                if timing["started_at"] == value["started_at"]:
                    continue
            request = read(path.parent / "request.json")
            uncertain.append(
                {
                    "path": str(path.relative_to(directory)),
                    "role": request["role"],
                    **value,
                }
            )
    costs = {}
    for phase in ("construction", "query", "judge"):
        entries = [c for c in calls if c["phase"] == phase]
        costs[phase] = {
            "successful_calls": len(entries),
            "standard_cost_usd": standard_cost(entries),
            "provider_reported_cost_usd": billed_cost(entries),
        }
    construction = [
        c
        for p in sorted((directory / "build").glob("*/index.json"))
        for c in read(p)["calls"]
    ]
    return {
        "calls": calls,
        "phases": costs,
        "roles": summarize_roles(calls, uncertain),
        "logical_construction_cost_usd": standard_cost(construction),
        "construction_cache_hits": sum(bool(c.get("cache_hit")) for c in construction),
        "unresolved_or_failed_requests": uncertain,
        "actual_bill_complete": not uncertain
        and all(c["cost_usd"] is not None for c in calls),
    }


def role_options(args):
    return {
        "models": {r: getattr(args, r + "_model") for r in ROLES},
    }


def manifest(args, examples):
    connections = connection_settings(args.env_file, **role_options(args))
    public = public_connections(connections)
    return {
        "schema_version": 3,
        "dataset": "locomo",
        "method": args.method,
        "model": connections["answer"]["model"],
        "answer_protocol": mem0_oss.PROTOCOL
        if native_mem0(args)
        else "mem0_2025",
        "judge_protocol": mem0_oss.PROTOCOL
        if native_mem0(args)
        else "mem0_2025",
        "population": args.population,
        "question_ids": [q["id"] for _, q in examples],
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "code_sha256": code_digest(),
        "roles": public,
        "reasoning_effort": args.reasoning_effort,
        "configuration": {
            "top_k": args.top_k,
            "window": args.window,
            "workers": args.workers,
            "build_workers": args.build_workers,
            "build_turn_limit": args.build_turn_limit,
            "seed": args.seed,
            "mem0_protocol": getattr(args, "mem0_protocol", "native")
            if args.method == "mem0_oss" else None,
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "httpx", "numpy", "tiktoken", "openai", "jinja2",
                *(["mem0ai", "qdrant-client", "fastembed", "spacy", "en-core-web-sm"]
                  if args.method == "mem0_oss" else []),
            )
        },
        "method_source": read(mem0_oss.SOURCE) if args.method == "mem0_oss" else None,
        "mem0_version": importlib.metadata.version("mem0ai")
        if args.method == "mem0_oss"
        else None,
        "prices": PRICES,
        "price_basis": "Published list rates, uncached estimate; provider bill recorded separately",
    }


def run(args, client_factory=Client):
    examples = selection(args)
    if args.build_turn_limit and (
        args.population != "smoke" or args.method != "mem0_oss"
    ):
        raise ValueError("--build-turn-limit is only for mem0_oss smoke tests")
    config = manifest(args, examples)
    directory = args.output.resolve()
    if directory.suffix == ".json":
        raise ValueError(
            "New runs use --output results/<run-id>/ (a directory); old JSON results stay unchanged"
        )
    print(f"{args.method}: {len(examples)} questions -> {directory}", flush=True)
    if not args.run:
        print(json.dumps(config, indent=2))
        print("Dry run; no API calls or writes.")
        return config
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError(
                    "Run exists. Use --resume for identical configuration or choose a new output."
                )
            previous = read(manifest_path)
            differences = [
                k
                for k in set(config) | set(previous)
                if previous.get(k) != config.get(k)
            ]
            if differences:
                raise ValueError(
                    "Resume configuration differs: " + ", ".join(sorted(differences))
                )
        else:
            save(manifest_path, config)
            save(directory / "question_ids.json", config["question_ids"])
            for path in fixed_files():
                destination = directory / "source" / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            shutil.copy2(
                ROOT / "requirements.txt", directory / "source" / "requirements.txt"
            )
        counts = {
            s: sum(
                (
                    directory
                    / "questions"
                    / q["id"].replace(":", "_")
                    / s
                    / "result.json"
                ).exists()
                for _, q in examples
            )
            for s in ("retrieval", "answer", "judge")
        }
        print(f"Checkpoints: {counts}; total {len(examples)}", flush=True)
        client = client_factory(
            args.env_file, **role_options(args), reasoning_effort=args.reasoning_effort
        )
        rows, failures, indexes = [], [], {}
        save(directory / "status.json", {"status": "running", "updated_at": now()})
        try:
            # Build only conversations with an unfinished retrieval; judge-only resume needs no index.
            if args.method == "mem0_oss":
                mem0_oss.check_environment()
                samples = {
                    s.id: s
                    for s, q in examples
                    if not (
                        directory
                        / "questions"
                        / q["id"].replace(":", "_")
                        / "retrieval"
                        / "result.json"
                    ).exists()
                }
                with ThreadPoolExecutor(max_workers=args.build_workers) as pool:
                    futures = {
                        pool.submit(
                            mem0_oss.build,
                            sample,
                            client,
                            directory / "build" / sample.id,
                            STOP,
                            args.build_turn_limit,
                        ): sample.id
                        for sample in samples.values()
                    }
                    errors = []
                    for future in as_completed(futures):
                        try:
                            indexes[futures[future]] = future.result()
                        except Exception as error:
                            errors.append(error)
                    if errors:
                        raise errors[0]

            def work(pair):
                sample, qa = pair
                return evaluate(
                    sample,
                    qa,
                    client,
                    args,
                    directory / "questions" / qa["id"].replace(":", "_"),
                    indexes,
                )

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(work, pair): pair[1]["id"] for pair in examples}
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        rows.append(row)
                        if len(rows) % 10 == 0 or len(rows) == len(examples):
                            print(f"{len(rows)}/{len(examples)} complete", flush=True)
                    except Exception as error:
                        failures.append(
                            {
                                "id": futures[future],
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        print(f"Failed {futures[future]}: {error}", flush=True)
            positions = {q["id"]: i for i, (_, q) in enumerate(examples)}
            rows.sort(key=lambda row: positions[row["id"]])
            result = {
                "dataset": "locomo",
                "method": args.method,
                "model": config["model"],
                "settings": config,
                "rows": rows,
                "failures": failures,
                "expected_questions": len(examples),
                "status": "complete" if len(rows) == len(examples) else "incomplete",
                "summary": summarize(rows) if rows else {},
                "resources": summarize_resources(rows) if rows else {},
                "ledger": ledger(directory),
            }
            result["summary"]["coverage"] = len(rows) / len(examples)
            result["summary"]["fixed_denominator_accuracy"] = sum(
                r["correct"] for r in rows
            ) / len(examples)
            save(directory / "summary.json", result)
            save(
                directory / "status.json",
                {"status": result["status"], "updated_at": now()},
            )
            if failures:
                raise RuntimeError(
                    f"{len(failures)} unfinished questions; resume this run to continue"
                )
            return result
        except BaseException:
            save(
                directory / "status.json",
                {
                    "status": "interrupted" if STOP.is_set() else "incomplete",
                    "updated_at": now(),
                },
            )
            raise
        finally:
            if args.method == "mem0_oss":
                for memory in indexes.values():
                    mem0_oss.close(memory)
            client.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/locomo-conv26.json")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    for role in ROLES:
        p.add_argument(f"--{role}-model", help="Override ROLE_MODEL")
    p.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    p.add_argument("--mem0-protocol", choices=["native", "shared"], default="native")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--build-workers", type=int, default=10)
    p.add_argument(
        "--build-turn-limit", type=int, help="Mem0 smoke only: limit ingestion turns"
    )
    p.add_argument("--limit", type=int, help="smoke run: only the first N questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population", choices=["smoke", "full"], default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if (
        min(args.top_k, args.workers, args.build_workers) < 1
        or args.window < 0
        or any(
            x is not None and x < 1
            for x in (args.limit, args.build_turn_limit)
        )
    ):
        raise SystemExit("Counts must be positive and window nonnegative")

    def stop(signum, frame):
        STOP.set()
        print(
            "Stop requested: saving in-flight work; no new stages will start.",
            flush=True,
        )

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
