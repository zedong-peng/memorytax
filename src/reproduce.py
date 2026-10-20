"""Run each method on LoCoMo conv-26, one cell at a time, into results/<method>/<model>/.

    python src/reproduce.py                    status of every cell
    python src/reproduce.py mem0               status of the Mem0 cells
    python src/reproduce.py mem0 opus-mini     what would run for one cell
    python src/reproduce.py mem0 --run         run (or resume) the Mem0 cells

A cell's model is the answering model (mini, luna-high, opus), or <build>-<answer>
where a memory is built first. mini is GPT-4o-mini, luna-high GPT-6 Luna at reasoning effort high,
opus Claude Opus 5.5 at high. Cells run in table order, so a store is built before the
cells that reuse it. --run calls the API; running the same command again resumes
from the saved checkpoints. Credentials come from .env.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PYTHON = sys.executable
ZEP_PYTHON = ROOT / ".venv-zep/bin/python"  # Graphiti pins its own dependencies
MODELS = {"mini": ("gpt-4o-mini", "none"), "luna-high": ("gpt-6-luna", "high"),
          "opus": ("claude-opus-5-5", "high")}
FIXED = ["--judge-model", "gpt-4o-mini", "--embedding-model", "text-embedding-3-small",
         "--population", "full", "--seed", "0"]


def script(path):
    return [PYTHON, "-u", str(ROOT / "src" / path)]


def experiment(output, model, workers, *method):
    """src/run.py, which needs --resume to continue a started run."""
    name, effort = MODELS[model]
    command = [*script("run.py"), *method, "--work-model", name, "--answer-model", name,
               "--reasoning-effort", effort, *FIXED, "--workers", str(workers), "--output", str(output)]
    return [*command, "--resume"] if (output / "manifest.json").exists() else command


def cells():
    """(method, model, output, [commands]) for every table row, in table order."""
    rows = []

    def add(method, model, *commands):
        rows.append((method, model, RESULTS / method / model, commands))

    fixed = {"full_context": ["--method", "full_context"]}
    fixed["bm25"] = ["--method", "bm25", "--top-k", "8", "--window", "2"]
    for name, method in fixed.items():
        for model in MODELS:
            out = RESULTS / name / model
            add(name, model, lambda w, o=out, m=model, x=method: experiment(o, m, w, *x))
    for build in MODELS:
        store = RESULTS / "mem0" / f"{build}-{build}"
        add("mem0", f"{build}-{build}", lambda w, o=store, b=build: experiment(
            o, b, w, "--method", "mem0_oss", "--mem0-protocol", "shared", "--top-k", "200"))
    for build in MODELS:
        for answer in MODELS:
            if answer != build:
                out = RESULTS / "mem0" / f"{build}-{answer}"
                add("mem0", f"{build}-{answer}", lambda w, o=out, b=build, a=answer: [
                    *script("methods/mem0/reanswer.py"), "--store", str(RESULTS / "mem0" / f"{b}-{b}"),
                    "--answer-model", MODELS[a][0], "--reasoning-effort", MODELS[a][1],
                    "--workers", str(w), "--output", str(o)])
    for model in MODELS:
        out = RESULTS / "zep" / f"{model}-{model}"
        add("zep", f"{model}-{model}", lambda w, o=out, m=model: [
            str(ZEP_PYTHON), "-u", str(ROOT / "src/methods/zep/run.py"), "--model", MODELS[m][0],
            "--reasoning-effort", MODELS[m][1], "--workers", str(w), "--output", str(o)])
    for build in MODELS:
        for answer in MODELS:
            if answer != build:
                out = RESULTS / "zep" / f"{build}-{answer}"
                add("zep", f"{build}-{answer}", lambda w, o=out, b=build, a=answer: [
                    *script("methods/zep/reanswer.py"), "--store", str(RESULTS / "zep" / f"{b}-{b}"),
                    "--answer-model", MODELS[a][0], "--reasoning-effort", MODELS[a][1],
                    "--workers", str(w), "--output", str(o)])
    for agent in ("pi", "codex", "claude-code"):
        for model in MODELS:
            out = RESULTS / agent / model
            add(agent, model, lambda w, o=out, a=agent, m=model: [
                *script("methods/agent/run.py"), "--agent", a, "--model", m, "--judge-model", "gpt-4o-mini",
                "--population", "full", "--workers", str(w), "--output", str(o),
                *(["--resume"] if (o / "manifest.json").exists() else [])])
    return rows


def status(output):
    """complete, partial or missing, from the run's own summary files."""
    from json import loads

    def done(path):
        return path.exists() and loads(path.read_text()).get("status") == "complete"

    if not output.exists():
        return "missing"
    return "complete" if done(output / "summary.json") else "partial"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", nargs="?")
    parser.add_argument("model", nargs="?")
    parser.add_argument("--workers", type=int, default=8, help="concurrent questions per run")
    parser.add_argument("--run", action="store_true", help="call the API; repeat to resume")
    args = parser.parse_args()
    selected = [c for c in cells() if args.method in (None, c[0]) and args.model in (None, c[1])]
    if not selected:
        raise SystemExit(f"No cell {args.method} {args.model or ''}; run without arguments to list them")
    for method, model, output, commands in selected:
        state = status(output)
        print(f"{method:<13} {model:<10} {state:<9} {output.relative_to(ROOT)}", flush=True)
        if state == "complete" or not (args.run or args.model):
            continue
        for build in commands:
            # Every runner is a dry run without --run.
            command = [*build(args.workers), "--run"]
            if not args.run:
                print("  " + " ".join(str(Path(c).relative_to(ROOT)) if str(c).startswith(str(ROOT)) else str(c)
                                      for c in command))
                continue
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                raise SystemExit(f"{method} {model} stopped; run the same command again to resume")


if __name__ == "__main__":
    main()
