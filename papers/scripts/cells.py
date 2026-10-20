#!/usr/bin/env python3
"""Copy this repository's runs, results/<dir>/<model>/summary.json, into data/conv26_cumulative.json.

    python papers/scripts/cells.py [results dir]

These are the memory methods with GPT-6 Luna at effort high and Claude Opus 5.5 (src/reproduce.py)
and the Harbor agents (src/methods/agent/run.py). A complete run replaces its cell; an unfinished
or missing one leaves the file as it was. Then every cell whose summaries can be read gets its
timing and usage: each question's seconds, turns and tokens, and the build's. The memory methods with
GPT-4o-mini, run first, are read from results/legacy-gpt-4o-mini/. Run conv26.py afterwards to redraw
the tables and figures.
"""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
# Paper method: results directory, and the model keys run here. The memory methods with GPT-4o-mini
# were run first and stay as they are.
MEMORY = {"Full context": "full_context", "BM25": "bm25", "Mem0": "mem0", "Zep": "zep"}
AGENTS = {"Pi": "pi", "Codex": "codex", "Claude Code": "claude-code"}
RUNS = [(method, d, model) for method, d in MEMORY.items() for model in ("luna-high", "opus")] + \
       [(method, d, model) for method, d in AGENTS.items() for model in ("mini", "luna-high", "opus")]
# Mem0 and Zep build a store with the same model that answers.
STORES = ("mem0", "zep")


def build(summary):
    """Build cost and successful build calls."""
    if "build_calls" in summary:  # Zep
        return summary["logical_construction_cost_usd"], len(summary["build_calls"])
    ledger = summary.get("ledger")
    if not ledger or not ledger["logical_construction_cost_usd"]:
        return None, 0
    return ledger["logical_construction_cost_usd"], ledger["phases"]["construction"]["successful_calls"]


def api(provider):
    """The API an agent's requests reach the model by, without the endpoint (manifest's endpoint field).
    Runs before that field named the endpoint inside: "<name> (<url>, <API>)"."""
    head, *rest = provider.split("; ")
    if "(" in head:
        head = "Anthropic Messages API" if "Messages" in head else "OpenAI Responses API"
    return "; ".join([head, *rest])


def seconds(row):
    """A question's wall time: end to end, or retrieval plus answer where the two were timed apart."""
    s = row.get("query_latency_s")
    return row.get("composed_query_latency_s") if s is None else s


def busy(calls):
    """Seconds with at least one of conv-26's build calls in flight: overlapping calls count once, and
    pauses between resumed runs not at all."""
    spans = sorted((datetime.fromisoformat(c["started_at"]), datetime.fromisoformat(c["ended_at"])) for c in calls
                   if c.get("phase") == "construction" and "conv-26/" in c.get("path", "") and c.get("started_at") and c.get("ended_at"))
    total, end = 0.0, None
    for a, b in spans:
        if end is None or a > end:
            total += (b - a).total_seconds()
            end = b
        elif b > end:
            total += (b - end).total_seconds()
            end = b
    return total


def timing(c, root):
    """Add each question's seconds and the build's to a cell, from its summaries under root. A question
    with no time (an interrupted one, resumed) takes the cell's median and is counted."""
    rows_file = root / c["rows"]
    if not rows_file.exists():
        return False
    rows = {r["id"]: r for r in json.loads(rows_file.read_text())["rows"]}
    secs = [seconds(rows[q["id"]]) for q in c["questions"]]
    median = statistics.median(s for s in secs if s is not None)
    for q, s in zip(c["questions"], secs):
        q["seconds"] = median if s is None else s
    c["seconds_imputed"] = sum(s is None for s in secs)
    build = root / c["build_source"] if c.get("build_source") else None
    if build and build.exists():
        summary = json.loads(build.read_text())
        c["build_s"] = busy((summary.get("ledger") or {}).get("calls") or summary.get("build_calls") or [])
    else:
        c["build_s"] = 0.0
    return True


def llm(calls):
    """Turns and tokens of these calls: the LLM calls, and their input plus output tokens, cached input
    included. Embedding and judge calls are left out."""
    calls = [c for c in calls if c.get("kind") == "generation" and c.get("role") != "judge" and c.get("phase") != "judge"]
    return len(calls), sum((c.get("input_tokens") or 0) + (c.get("output_tokens") or 0) for c in calls)


def usage(c, root):
    """Add each question's turns and tokens to a cell, and the build's, from its summaries under root."""
    rows_file = root / c["rows"]
    if not rows_file.exists():
        return False
    rows = {r["id"]: r for r in json.loads(rows_file.read_text())["rows"]}
    for q in c["questions"]:
        q["turns"], q["tokens"] = llm(rows[q["id"]].get("calls") or [])
    build = root / c["build_source"] if c.get("build_source") else None
    c["build_turns"], c["build_tokens"] = 0, 0
    if build and build.exists():
        summary = json.loads(build.read_text())
        calls = (summary.get("ledger") or {}).get("calls") or summary.get("build_calls") or []
        c["build_turns"], c["build_tokens"] = llm(k for k in calls if k.get("phase") == "construction" and "conv-26/" in k.get("path", ""))
    return True


def cell(method, directory, name, model, summary, reference):
    rows = {r["id"]: r for r in summary["rows"]}
    assert set(rows) == set(reference), (directory, name)
    assert all(rows[i]["query_cost_usd"] is not None for i in reference), (directory, name)
    settings = summary["settings"]
    path = f"results/{directory}/{name}/summary.json"
    out = {"method": method, "model": model, "effort": settings["reasoning_effort"], "rows": path}
    if directory in AGENTS.values():
        out["run"] = {"agent": settings["versions"]["agent"], "harbor": settings["versions"]["harbor"],
                      "provider": api(settings["provider"]), "code_sha256": settings["code_sha256"]}
        usd, calls = None, 0
    else:
        usd, calls = build(summary)
    out.update({"build_source": path if usd else None, "build_usd": usd, "build_calls": calls})
    out["questions"] = [{"id": i, "category": rows[i]["category"], "correct": bool(rows[i]["correct"]),
                         "cost_usd": rows[i]["query_cost_usd"]} for i in reference]
    return out


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER.parent / "results"
    path = PAPER / "data/conv26_cumulative.json"
    data = json.loads(path.read_text())
    reference = [q["id"] for q in data["cells"][0]["questions"]]
    cells = data["cells"]
    for method, directory, model in RUNS:
        name = f"{model}-{model}" if directory in STORES else model
        file = results / directory / name / "summary.json"
        summary = json.loads(file.read_text()) if file.exists() else {}
        if summary.get("status") != "complete":
            print(f"{method:<12} {model:<9} not complete, left as it was")
            continue
        cells = [c for c in cells if (c["method"], c["model"]) != (method, model)]
        cells.append(cell(method, directory, name, model, summary, reference))
        c = cells[-1]
        print(f"{method:<12} {model:<9} {sum(q['correct'] for q in c['questions'])}/{len(reference)} correct, "
              f"${(c['build_usd'] or 0) + sum(q['cost_usd'] for q in c['questions']):.4f} per rollout")
    for c in cells:
        root = results.parent
        if timing(c, root):
            print(f"{c['method']:<12} {c['model']:<9} build {c['build_s'] / 60:5.1f} min, questions "
                  f"{sum(q['seconds'] for q in c['questions']) / 60:6.1f} min, {c['seconds_imputed']} timed by median")
        if usage(c, root):
            print(f"{c['method']:<12} {c['model']:<9} build {c['build_turns']} turns, {c['build_tokens']:,} tokens; questions "
                  f"{sum(q['turns'] for q in c['questions'])} turns, {sum(q['tokens'] for q in c['questions']):,} tokens")
    data["cells"] = cells
    data["source"] = ("Memory methods with GPT-4o-mini: results/legacy-gpt-4o-mini/, run first with the code this "
                      "repository started from. Other cells: results/<method>/<model>/, the agents one Harbor task per question.")
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
