"""Pi, Codex or Claude Code on LoCoMo conv-26 through Harbor, one fresh container per question.

The container (Dockerfile) holds conv-26/conv.json in /workspace and nothing else of the dataset;
while the agent runs it can reach only its model endpoint. --model mini is GPT-4o-mini, luna-high
GPT-6 Luna at reasoning effort high, opus Claude Opus 5.5 at high; each model's endpoint comes from
.env as in common/llm.py. On the GPT models Pi and Codex call the Responses API and Claude Code calls
a local LiteLLM proxy that translates Anthropic Messages for it; on Opus, Claude Code and Pi call
Anthropic directly and the proxy translates Codex's Responses API instead. The agent's last message
is the answer, judged with the Mem0 2025 judge like every other cell.

    python src/methods/agent/run.py --agent pi --model mini --limit 2 --output results/smoke/pi
    ... --run              call the API; run the same command with --resume to continue

Dry run by default. A question whose trial fails runs again in a new Harbor job; only the
accepted trial is kept, in trials/<question>/.
"""

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import run as base  # noqa: E402
from common.artifacts import now, read, run_lock, save  # noqa: E402
from common.eval import judge, summarize, token_f1  # noqa: E402
from common.llm import DEFAULT_ENV, MODEL_CONNECTIONS, Client, api_root, public_connections, token_total  # noqa: E402
from common.usage import PRICES, standard_cost, summarize_resources  # noqa: E402

ROOT = SRC.parent
HERE = Path(__file__).resolve().parent
WORKSPACE = ROOT / "data/playground"
PROMPT = (HERE / "prompt.txt").read_text()
MODELS = {
    "mini": {"model": "openai/gpt-4o-mini", "effort": None},
    "luna-high": {"model": "openai/gpt-6-luna", "effort": "high"},
    "opus": {"model": "anthropic/claude-opus-5-5", "effort": "high", "api": "anthropic-messages"},
}
LIMITS = {"timeout_s": 1200, "max_turns": 100}
AGENTS = {
    # Codex has no turn cap; the timeout bounds it.
    "pi": {"version": "1.1.0", "options": {"model_api": "openai-responses", "max_turns": LIMITS["max_turns"]}},
    "codex": {"version": "0.159.2", "options": {"web_search": "disabled"}},
    "claude-code": {"version": "2.1.295",
                    "options": {"max_turns": LIMITS["max_turns"], "disable_web_search": "true"}},
}
HARBOR = "0.24.0"
LITELLM = "litellm[proxy]==1.104.2"
PROXY_HOST = "172.17.0.1"  # the docker bridge: reachable from containers only
ROUNDS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(*command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def image():
    """The question container, tagged by its Dockerfile and workspace so a change builds a new one."""
    tag = "memory-tax/conv26-agents:" + hashlib.sha256(
        (sha(HERE / "Dockerfile") + sha(WORKSPACE / "conv-26/conv.json")).encode()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode:
        subprocess.run(["docker", "build", "-q", "-t", tag, "-f", str(HERE / "Dockerfile"), str(WORKSPACE)],
                       check=True)
    return {"tag": tag, "id": output("docker", "image", "inspect", "--format", "{{.Id}}", tag)}


def task_name(qa):
    return qa["id"].replace(":", "_")


def write_tasks(directory, examples, tag):
    shutil.rmtree(directory, ignore_errors=True)
    for _, qa in examples:
        task = directory / task_name(qa)
        (task / "tests").mkdir(parents=True)
        (task / "environment").mkdir()  # Harbor recognizes a task by it; the image is prebuilt
        (task / "instruction.md").write_text(PROMPT.replace("{question}", qa["question"]))
        (task / "task.toml").write_text(
            'schema_version = "1.4"\n\n'
            f'[metadata]\nquestion_id = {json.dumps(qa["id"])}\n\n'
            f'[agent]\ntimeout_sec = {float(LIMITS["timeout_s"])}\nnetwork_mode = "allowlist"\nallowed_hosts = []\n\n'
            '[verifier]\ntimeout_sec = 60.0\n\n'
            f'[environment]\ndocker_image = "{tag}"\nworkdir = "/workspace"\n')
        # Verification is disabled; answers are judged after the run.
        (task / "tests/test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")
        (task / "tests/test.sh").chmod(0o755)


@contextlib.contextmanager
def proxy(model, upstream):
    """LiteLLM on the docker bridge, for the agent that does not speak the model's API; its master key
    exists only while it runs."""
    workdir = Path(tempfile.mkdtemp(prefix="mt-litellm-"))
    # A free port, so runs of different models can each have a proxy at once.
    with socket.socket() as probe:
        probe.bind((PROXY_HOST, 0))
        port = probe.getsockname()[1]
    master = "sk-local-" + secrets.token_hex(16)
    (workdir / "config.yaml").write_text(
        # Claude Code asks for the model by its full name, Pi without the provider.
        "model_list:\n" + "".join(
            f"  - model_name: {name}\n    litellm_params:\n      model: {model}\n"
            f"      api_base: {upstream['base_url']}\n      api_key: os.environ/UPSTREAM_API_KEY\n"
            for name in (model, model.split("/")[-1])) + "\n"
        "litellm_settings:\n  num_retries: 3\n  request_timeout: 600\n\n"
        f"general_settings:\n  master_key: {master}\n  forward_client_headers_to_llm_api: true\n")
    env = {**os.environ, "UPSTREAM_API_KEY": upstream["key"]}
    with (workdir / "proxy.log").open("wb") as log:
        process = subprocess.Popen(["uvx", "--from", LITELLM, "litellm", "--config", str(workdir / "config.yaml"),
                                    "--host", PROXY_HOST, "--port", str(port)], env=env, stdout=log, stderr=log)
    try:
        url = f"http://{PROXY_HOST}:{port}"
        for _ in range(120):
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(url + "/health/liveliness", timeout=2).status_code == 200:
                    break
            if process.poll() is not None:
                raise RuntimeError("LiteLLM exited: " + (workdir / "proxy.log").read_text()[-2000:])
            time.sleep(1)
        else:
            raise RuntimeError("LiteLLM did not start")
        yield {"ANTHROPIC_API_KEY": master, "ANTHROPIC_BASE_URL": url,
               "OPENAI_API_KEY": master, "OPENAI_BASE_URL": url + "/v1"}, PROXY_HOST
    finally:
        process.terminate()
        process.wait(timeout=30)
        shutil.rmtree(workdir, ignore_errors=True)


def agent_options(args):
    """The effort Harbor is given and the agent's options."""
    effort, extra = MODELS[args.model]["effort"], {}
    if args.agent == "pi" and claude(args):
        extra["model_api"] = "anthropic-messages"
    return effort, {**AGENTS[args.agent]["options"], **extra}


def claude(args):
    return MODELS[args.model].get("api") == "anthropic-messages"


def harbor_agent(args):
    """Harbor's agent, except Pi on Claude, which needs its model registered for adaptive thinking."""
    return "methods.agent.pi_adaptive:PiAdaptive" if args.agent == "pi" and claude(args) else args.agent


def provider(args):
    """The API the agent's requests reach the model by; the endpoint is in the manifest."""
    if claude(args):
        return "Anthropic Messages API" + ("; OpenAI Responses translated by LiteLLM" if args.agent == "codex" else "")
    return "OpenAI Responses API" + ("; Anthropic Messages translated by LiteLLM" if args.agent == "claude-code" else "")


def upstream(args):
    """The model's endpoint and key, from the same .env variables as common/llm.py."""
    values = {**dotenv_values(args.env_file), **os.environ}
    model = MODELS[args.model]
    prefix = MODEL_CONNECTIONS[model["model"].split("/")[-1]][0]
    url = api_root(values[prefix + "_BASE_URL"])
    # Anthropic's clients add /v1 themselves; Claude Code on Claude is given no base URL (run_job).
    url = url.removesuffix("/v1") if claude(args) else url
    url = "https://api.anthropic.com" if claude(args) and args.agent == "claude-code" else url
    return {"key": values[prefix + "_API_KEY"], "base_url": url, "host": urlsplit(url).hostname}


def harbor_command(args, tasks, jobs, name, host, include):
    effort, options = agent_options(args)
    options = [x for k, v in options.items() for x in ("--ak", f"{k}={v}")]
    effort = ["--effort", effort] if effort else []
    return ["harbor", "run", "-p", str(tasks), "-a", harbor_agent(args), "-m", MODELS[args.model]["model"], *effort,
            "--ak", f"version={AGENTS[args.agent]['version']}", *options,
            "-o", str(jobs), "--job-name", name, "-n", str(args.workers), "--max-retries", "2",
            "--disable-verification", "--allow-agent-host", host, "-y", "-q",
            *[x for t in include for x in ("-i", t)]]


def run_job(args, upstream, tasks, jobs, name, include):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "TERM", "DOCKER_HOST") if k in os.environ}
    env["PYTHONPATH"] = str(SRC)  # for a custom Harbor agent class (harbor_agent)
    with contextlib.ExitStack() as stack:
        model = MODELS[args.model]
        if claude(args) and args.agent == "codex":
            # Codex speaks only the Responses API; LiteLLM translates it to Anthropic Messages, its
            # effort high to adaptive thinking with output_config effort high (checked in its debug log).
            credentials, host = stack.enter_context(proxy(model["model"], upstream))
        elif claude(args):
            # With a base URL Harbor hands Claude Code the provider-prefixed name, which Anthropic
            # rejects; Pi needs one to register the model id, which its catalog does not know.
            credentials = {"ANTHROPIC_API_KEY": upstream["key"],
                           **({"ANTHROPIC_BASE_URL": upstream["base_url"]} if args.agent == "pi" else {})}
            host = upstream["host"]
        elif args.agent == "claude-code":
            credentials, host = stack.enter_context(proxy(model["model"], upstream))
        else:
            credentials = {"OPENAI_API_KEY": upstream["key"], "OPENAI_BASE_URL": upstream["base_url"]}
            host = upstream["host"]
        subprocess.run(harbor_command(args, tasks, jobs, name, host, include), env=env | credentials)


def trial_result(trial, model):
    """The answer and per-request usage of a finished trial, or None if the trial is not accepted."""
    result_path, trajectory_path = trial / "result.json", trial / "agent/trajectory.json"
    if not result_path.exists() or not trajectory_path.exists():
        return None
    result, trajectory = read(result_path), read(trajectory_path)
    steps = [s for s in trajectory["steps"] if s.get("source") == "agent"]
    answer = next((s["message"].strip() for s in reversed(steps) if (s.get("message") or "").strip()), "")
    if result.get("exception_info") or not answer:
        return None
    calls = [{"kind": "generation", "role": "work", "model": model, "requested_model": model.split("/")[-1],
              "input_tokens": m["prompt_tokens"], "output_tokens": m["completion_tokens"],
              "cached_input_tokens": m.get("cached_tokens"), "cost_usd": None, "phase": "query", "sequence": i}
             for i, m in enumerate((s["metrics"] for s in steps if s.get("metrics")), 1)]
    totals = trajectory["final_metrics"]
    # Every request's usage must be in the steps: the totals are what the agent itself reported.
    assert sum(c["input_tokens"] for c in calls) == totals["total_prompt_tokens"], trial
    assert sum(c["output_tokens"] for c in calls) == totals["total_completion_tokens"], trial
    extra = totals.get("extra") or {}
    # Output tokens include reasoning; LiteLLM does not pass the reasoning count on to Claude Code.
    reasoning = extra.get("reasoning_tokens", extra.get("reasoning_output_tokens"))
    started, ended = (datetime.fromisoformat(result["agent_execution"][k].replace("Z", "+00:00"))
                      for k in ("started_at", "finished_at"))
    return {"prediction": answer, "calls": calls, "trial": trial.name, "reasoning_tokens": reasoning,
            "agent_version": result["agent_info"]["version"], "active_seconds": (ended - started).total_seconds()}


def collect(directory, model):
    """Move each question's first accepted trial to trials/<question>; delete everything else Harbor wrote."""
    trials = directory / "trials"
    trials.mkdir(exist_ok=True)
    for job in sorted((directory / "harbor").glob("*")):
        for trial in sorted(p for p in job.iterdir() if p.is_dir()):
            task = trial.name.split("__")[0]
            if not (trials / task).exists() and trial_result(trial, model) is not None:
                trial.rename(trials / task)
        shutil.rmtree(job)
    with contextlib.suppress(OSError):
        (directory / "harbor").rmdir()


def evaluate(trial, qa, client, model):
    agent = trial_result(trial, model)
    judgment = base.stage(
        trial.parent.parent / "judge" / trial.name,
        {"prediction": agent["prediction"], "reference": qa["reference"], "question": qa["question"],
         "judge": "mem0_2025"},
        client,
        lambda: judge(client, qa["question"], qa["reference"], agent["prediction"], "mem0"),
    )
    calls = agent["calls"] + judgment["calls"]
    return {
        **qa,
        "prediction": agent["prediction"],
        "correct": judgment["value"],
        "f1": token_f1(agent["prediction"], qa["reference"]),
        "trial": agent["trial"],
        "agent_version": agent["agent_version"],
        "requests": len(agent["calls"]),
        "calls": calls,
        "input_tokens": token_total(agent["calls"], "generation", "input_tokens"),
        "output_tokens": token_total(agent["calls"], "generation", "output_tokens"),
        "cached_input_tokens": sum(c.get("cached_input_tokens") or 0 for c in agent["calls"]),
        "reasoning_tokens": agent["reasoning_tokens"],
        "generation_tokens": token_total(agent["calls"], "generation"),
        "judge_tokens": token_total(calls, "judge"),
        "query_latency_s": agent["active_seconds"],
        "query_cost_usd": standard_cost(agent["calls"]),
    }


def manifest(args, examples, client):
    return {
        "schema_version": 2,
        "dataset": "locomo",
        "method": args.agent,
        "model": MODELS[args.model]["model"],
        "reasoning_effort": MODELS[args.model]["effort"],
        "provider": provider(args),
        "endpoint": upstream(args)["base_url"],
        "answer_protocol": "agent final message; Mem0 2025 answer rules in the prompt",
        "judge_protocol": "mem0_2025",
        "population": args.population,
        "question_ids": [q["id"] for _, q in examples],
        "data_sha256": sha(args.data),
        "workspace": {"conv-26/conv.json": sha(WORKSPACE / "conv-26/conv.json")},
        "dockerfile_sha256": sha(HERE / "Dockerfile"),
        "prompt": PROMPT,
        "code_sha256": base.code_digest(),
        "judge": public_connections(client.connections)["judge"],
        "versions": {"harbor": HARBOR, "agent": AGENTS[args.agent]["version"],
                     **({"litellm": LITELLM.split("==")[1]}
                        if args.agent == ("codex" if claude(args) else "claude-code") else {})},
        "agent_options": agent_options(args)[1],
        **({"harbor_agent": harbor_agent(args)} if harbor_agent(args) != args.agent else {}),
        "isolation": "one Docker container per question; during the agent phase Harbor's egress control "
                     "allows only the model endpoint",
        "limits": LIMITS,
        "prices": PRICES,
        "price_basis": "Published list rates, every input token at the uncached rate",
    }


def run(args):
    examples = base.selection(args)
    # Only the judge role is used; the agents reach their model through Harbor.
    client = Client(args.env_file, models=dict.fromkeys(("work", "answer", "judge"), args.judge_model))
    config = manifest(args, examples, client)
    directory = args.output.resolve()
    print(f"{args.agent}: {len(examples)} questions -> {directory}", flush=True)
    if not args.run:
        print(json.dumps({k: v for k, v in config.items() if k not in ("question_ids", "prices", "prompt")}, indent=2))
        print("Dry run; no API calls or writes.")
        client.close()
        return config
    if output("harbor", "--version") != HARBOR:
        raise SystemExit(f"Harbor {HARBOR} is required")
    model, endpoint = MODELS[args.model], upstream(args)
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("Run exists. Use --resume for identical configuration or choose a new output.")
            previous = read(manifest_path)
            differences = sorted(k for k in set(config) | set(previous) if previous.get(k) != config.get(k))
            if differences:
                raise ValueError("Resume configuration differs: " + ", ".join(differences))
        else:
            save(manifest_path, config)
        save(directory / "status.json", {"status": "running", "updated_at": now()})
        tag = image()
        save(directory / "image.json", tag)
        write_tasks(directory / "tasks", examples, tag["tag"])
        collect(directory, model["model"])
        for round_ in range(1, ROUNDS + 1):
            missing = [task_name(qa) for _, qa in examples if not (directory / "trials" / task_name(qa)).exists()]
            if not missing or base.STOP.is_set():
                break
            print(f"round {round_}: {len(missing)} questions", flush=True)
            run_job(args, endpoint, directory / "tasks", directory / "harbor", f"round-{round_}",
                    [] if len(missing) == len(examples) else missing)
            collect(directory, model["model"])
        try:
            rows, failures = [], []
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(evaluate, directory / "trials" / task_name(qa), qa, client, model["model"]): qa["id"]
                           for _, qa in examples if (directory / "trials" / task_name(qa)).exists()}
                for future in as_completed(futures):
                    try:
                        rows.append(future.result())
                    except Exception as error:
                        failures.append({"id": futures[future], "error_type": type(error).__name__,
                                         "message": str(error)})
            failures += [{"id": qa["id"], "error_type": "NoAcceptedTrial", "message": f"none after {ROUNDS} rounds"}
                         for _, qa in examples if not (directory / "trials" / task_name(qa)).exists()]
            positions = {q["id"]: i for i, (_, q) in enumerate(examples)}
            rows.sort(key=lambda row: positions[row["id"]])
            result = {
                "dataset": "locomo", "method": args.agent, "model": model["model"], "settings": config,
                "rows": rows, "failures": failures, "expected_questions": len(examples),
                "status": "complete" if len(rows) == len(examples) else "incomplete",
                "summary": summarize(rows) if rows else {},
                "resources": summarize_resources(rows) if rows else {},
            }
            if rows:
                result["summary"]["coverage"] = len(rows) / len(examples)
                result["summary"]["fixed_denominator_accuracy"] = sum(r["correct"] for r in rows) / len(examples)
            save(directory / "summary.json", result)
            save(directory / "status.json", {"status": result["status"], "updated_at": now()})
            if failures:
                raise RuntimeError(f"{len(failures)} unfinished questions; resume this run to continue")
            return result
        except BaseException:
            save(directory / "status.json", {"status": "interrupted" if base.STOP.is_set() else "incomplete",
                                             "updated_at": now()})
            raise
        finally:
            client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--model", choices=sorted(MODELS), default="luna-high")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--data", type=Path, default=ROOT / "data/locomo-conv26.json")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--workers", type=int, default=6, help="concurrent Harbor trials")
    p.add_argument("--limit", type=int, help="smoke run: only the first N questions")
    p.add_argument("--population", choices=["smoke", "full"], default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise SystemExit("Counts must be positive")

    def stop(signum, frame):
        base.STOP.set()
        print("Stop requested: no new Harbor job starts.", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
