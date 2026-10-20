"""Answer a Mem0 store's saved retrievals again, with another model or Mem0's own protocol.

--store is a finished Mem0 run (results/mem0/<build>-<build>). Its per-question
retrievals are reused, never searched again, so a cell differs from the store's
run only in the model that answers and the protocol that answers and judges:

    shared  Mem0's 2025 answer template and judge, as every other row
    native  Mem0's own answer prompt (saved with each retrieval) and 2026 judge

Dry run by default; --run calls the API, and the same command resumes.
"""

import argparse
import hashlib
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run as experiment  # noqa: E402
from common.artifacts import digest, now, read, run_lock, save  # noqa: E402
from common.data import load  # noqa: E402
from common.eval import judge, summarize, token_f1  # noqa: E402
from common.llm import DEFAULT_ENV, Client, connection_settings, context_metrics, public_connections, token_total  # noqa: E402
from common.usage import PRICES, billed_cost, standard_cost, summarize_resources  # noqa: E402
from methods.mem0 import method as mem0  # noqa: E402
from common.answer import generate_answer  # noqa: E402

ROOT = experiment.ROOT
DATA = ROOT / "data/locomo-conv26.json"


def saved_retrieval(store, qa):
    """A store run's checkpointed retrieval of one question, checked against its digest."""
    path = store / "questions" / qa["id"].replace(":", "_") / "retrieval/result.json"
    record = read(path)
    if digest(record["result"]) != record["result_sha256"] or record["identity"]["question"] != qa["question"]:
        raise ValueError(f"Saved retrieval does not match: {path}")
    return record["result"]


def evaluate(qa, store, client, protocol, directory):
    retrieval = saved_retrieval(store, qa)
    value = retrieval["value"]
    native = protocol == "native"
    answer = experiment.stage(
        directory / "answer",
        {
            "question": qa["question"],
            "context_sha256": digest(value["context"]),
            "answer_prompt_sha256": digest(value.get("answer_prompt")),
            "template": mem0.PROTOCOL if native else "mem0_2025",
        },
        client,
        lambda: mem0.answer(client, value["answer_prompt"]) if native
        else generate_answer(client, qa["question"], value["context"]),
    )
    prediction = answer["value"]
    judgment = experiment.stage(
        directory / "judge",
        {
            "prediction": prediction,
            "reference": qa["reference"],
            "question": qa["question"],
            "judge": mem0.PROTOCOL if native else "mem0_2025",
        },
        client,
        lambda: mem0.judge(client, qa, prediction) if native
        else judge(client, qa["question"], qa["reference"], prediction, "mem0"),
    )
    # The search happened in the store's run; its calls are carried over, marked as replayed.
    searched = [{**c, "replayed": True, "reused_from_store": True} for c in retrieval["calls"]]
    calls = searched + answer["calls"] + judgment["calls"]
    query = searched + answer["calls"]
    row = {
        **qa,
        "prediction": prediction,
        "correct": judgment["value"],
        "f1": token_f1(prediction, qa["reference"]),
        "calls": calls,
        "generation_tokens": token_total(calls, "generation"),
        "embedding_tokens": token_total(calls, "embedding"),
        "judge_tokens": token_total(calls, "judge"),
        **context_metrics(value["context"]),
        "query_latency_s": None if answer["resumed"] else retrieval["active_seconds"] + answer["active_seconds"],
        "resumed": answer["resumed"],
        "query_cost_usd": standard_cost(query),
        "query_billed_cost_usd": billed_cost(query),
    }
    save(directory / "row.json", row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, required=True, help="finished Mem0 run whose retrievals are reused")
    p.add_argument("--answer-model", required=True)
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], required=True,
                   help="the answer model's effort, as in its own store run")
    p.add_argument("--protocol", choices=["shared", "native"], default="shared")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()

    store, output = args.store.resolve(), args.output.resolve()
    source = read(store / "summary.json")
    if source.get("status") != "complete" or source.get("method") != "mem0_oss":
        raise SystemExit(f"{store} is not a finished Mem0 run")
    questions = [qa for _, qa in load(DATA.read_bytes())]
    models = {"work": source["settings"]["roles"]["work"]["model"], "answer": args.answer_model,
              "judge": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    config = {
        "method": "mem0_oss",
        "operation": "reanswer",
        "store": str(store.relative_to(ROOT)) if store.is_relative_to(ROOT) else str(store),
        "store_manifest_sha256": hashlib.sha256((store / "manifest.json").read_bytes()).hexdigest(),
        "model": args.answer_model,
        "answer_protocol": mem0.PROTOCOL if args.protocol == "native" else "mem0_2025",
        "judge_protocol": mem0.PROTOCOL if args.protocol == "native" else "mem0_2025",
        "question_ids": [q["id"] for q in questions],
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "code_sha256": experiment.code_digest(),
        "roles": public_connections(connection_settings(args.env_file, models=models)),
        "reasoning_effort": args.reasoning_effort,
        "prices": PRICES,
    }
    print(f"{args.protocol} answers by {args.answer_model} ({args.reasoning_effort}) from {store} -> {output}", flush=True)
    if not args.run:
        print(json.dumps({k: v for k, v in config.items() if k != "question_ids"}, indent=2))
        print("Dry run; no API calls or writes.")
        return

    signal.signal(signal.SIGINT, lambda *_: experiment.STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: experiment.STOP.set())
    with run_lock(output):
        manifest = output / "manifest.json"
        if manifest.exists() and read(manifest) != config:
            raise SystemExit("Configuration or code changed since this run started; choose a new output")
        save(manifest, config)
        client = Client(args.env_file, models=models, reasoning_effort=args.reasoning_effort)
        rows, failures = [], []
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(evaluate, qa, store, client, args.protocol,
                                output / "questions" / qa["id"].replace(":", "_")): qa["id"]
                    for qa in questions
                }
                for future in as_completed(futures):
                    try:
                        rows.append(future.result())
                    except Exception as error:
                        failures.append({"id": futures[future], "error_type": type(error).__name__, "message": str(error)})
        finally:
            client.close()
        order = {q["id"]: i for i, q in enumerate(questions)}
        rows.sort(key=lambda r: order[r["id"]])
        result = {
            "dataset": "locomo",
            "method": "mem0_oss",
            "model": args.answer_model,
            "settings": config,
            "rows": rows,
            "failures": failures,
            "status": "complete" if len(rows) == len(questions) else "incomplete",
            "summary": summarize(rows) if rows else {},
            "resources": summarize_resources(rows) if rows else {},
            "ledger": experiment.ledger(output),
            "updated_at": now(),
        }
        save(output / "summary.json", result)
        print(f"{len(rows)}/{len(questions)} answered; {len(failures)} failed", flush=True)
        if failures:
            raise SystemExit("Unfinished questions; run the same command again to resume")


if __name__ == "__main__":
    main()
