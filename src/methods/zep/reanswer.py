"""Answer a Zep store's saved retrievals again with another model.

--store is a finished Zep run (results/zep/<build>-<build>). Its per-question retrievals
are reused, never searched again, so a cell differs from the store's run only in the
model that answers; the answer template and judge are the shared ones. Retrievals are
matched by question text, since an older store may number its questions differently.

Runs in the main environment (no Graphiti). Dry run by default; --run calls the API,
and the same command resumes.
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
from common.answer import generate_answer  # noqa: E402

ROOT = experiment.ROOT
DATA = ROOT / "data/locomo-conv26.json"


def saved_retrievals(store):
    """The store's checkpointed retrievals by question text, each checked against its digest."""
    found = {}
    for path in sorted((store / "questions").glob("*/retrieval/result.json")):
        record = read(path)
        if digest(record["result"]) != record["result_sha256"]:
            raise ValueError(f"Saved retrieval does not match its digest: {path}")
        if record["identity"]["group"] == "conv-26":
            found[record["identity"]["question"]] = record["result"]
    return found


def evaluate(qa, retrieval, client, directory):
    context = retrieval["value"]["context"]
    answer = experiment.stage(
        directory / "answer",
        {"question": qa["question"], "context_sha256": digest(context), "template": "mem0_2025"},
        client,
        lambda: generate_answer(client, qa["question"], context),
    )
    prediction = answer["value"]
    judgment = experiment.stage(
        directory / "judge",
        {"prediction": prediction, "reference": qa["reference"], "question": qa["question"], "judge": "mem0_2025"},
        client,
        lambda: judge(client, qa["question"], qa["reference"], prediction, "mem0"),
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
        **context_metrics(context),
        "query_latency_s": None if answer["resumed"] else retrieval["active_seconds"] + answer["active_seconds"],
        "resumed": answer["resumed"],
        "query_cost_usd": standard_cost(query),
        "query_billed_cost_usd": billed_cost(query),
    }
    save(directory / "row.json", row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, required=True, help="finished Zep run whose retrievals are reused")
    p.add_argument("--answer-model", required=True)
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], required=True,
                   help="the answer model's effort, as in its own store run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()

    store, output = args.store.resolve(), args.output.resolve()
    source = read(store / "summary.json")
    if source.get("status") != "complete" or source.get("method") != "zep_oss":
        raise SystemExit(f"{store} is not a finished Zep run")
    questions = [qa for _, qa in load(DATA.read_bytes())]
    retrievals = saved_retrievals(store)
    missing = [qa["id"] for qa in questions if qa["question"] not in retrievals]
    if missing:
        raise SystemExit(f"{store} has no retrieval for {len(missing)} questions: {missing[:5]}")
    models = {"work": source["settings"]["roles"]["work"]["model"], "answer": args.answer_model,
              "judge": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    config = {
        "method": "zep_oss",
        "operation": "reanswer",
        "store": str(store.relative_to(ROOT)) if store.is_relative_to(ROOT) else str(store),
        "store_manifest_sha256": hashlib.sha256((store / "manifest.json").read_bytes()).hexdigest(),
        "model": args.answer_model,
        "answer_protocol": "mem0_2025",
        "judge_protocol": "mem0_2025",
        "question_ids": [q["id"] for q in questions],
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "code_sha256": experiment.code_digest(),
        "roles": public_connections(connection_settings(args.env_file, models=models)),
        "reasoning_effort": args.reasoning_effort,
        "prices": PRICES,
    }
    print(f"answers by {args.answer_model} ({args.reasoning_effort}) from {store} -> {output}", flush=True)
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
                    pool.submit(evaluate, qa, retrievals[qa["question"]], client,
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
            "method": "zep_oss",
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
