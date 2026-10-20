"""Check configured API keys and usage fields with small requests, without retries."""

import argparse
import json
import time
from pathlib import Path

import httpx

from common.llm import DEFAULT_ENV, ROLES, connection_settings, usage_record
from common.usage import standard_cost

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def check_usage(body, status, role, model, require_cost=False):
    usage = usage_record(
        body, "embedding" if role == "embedding" else "generation", model, role
    )
    fields = TOKEN_FIELDS[:1] if role == "embedding" else TOKEN_FIELDS
    missing = [
        field
        for field in fields
        if not isinstance(usage[field], int)
        or isinstance(usage[field], bool)
        or usage[field] < 0
    ]
    if usage["cost_usd"] is None and require_cost:
        missing.append("cost_usd")
    return {
        "ok": status == 200 and not missing,
        "missing": missing,
        "cost_usd": usage["cost_usd"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument(
        "--roles", nargs="+", choices=ROLES, default=["answer", "judge", "embedding"]
    )
    parser.add_argument(
        "--require-cost",
        action="store_true",
        help="Also require provider-reported USD cost",
    )
    parser.add_argument("--output", type=Path, default=Path("results/api-check"))
    args = parser.parse_args()
    output = args.output
    connections = connection_settings(args.env_file)
    secrets = [c["key"] for c in connections.values()]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(timeout=90) as http:
        for role in dict.fromkeys(args.roles):
            conn = connections[role]
            endpoints = (
                ["/embeddings"]
                if role == "embedding"
                else ["/chat/completions", "/responses"]
            )
            for endpoint in endpoints:
                for repeat in range(2):
                    payload = {"model": conn["model"]}
                    if endpoint == "/embeddings":
                        payload["input"] = ["Usage probe."]
                    elif endpoint == "/responses":
                        payload.update(
                            input="Reply with exactly OK.", max_output_tokens=32
                        )
                        if "luna" in conn["model"]:
                            payload["reasoning"] = {"effort": "none"}
                    else:
                        payload.update(
                            messages=[
                                {"role": "user", "content": "Reply with exactly OK."}
                            ],
                            max_completion_tokens=32,
                        )
                        if "luna" in conn["model"]:
                            payload["reasoning_effort"] = "none"
                    started = time.perf_counter()
                    try:
                        response = http.post(
                            conn["base_url"] + endpoint,
                            json=payload,
                            headers={"Authorization": "Bearer " + conn["key"]},
                        )
                        try:
                            body = response.json()
                        except ValueError:
                            body = {"non_json_body": response.text}
                        if not isinstance(body, dict):
                            body = {"invalid_response": body}
                        status = response.status_code
                    except httpx.HTTPError as error:
                        body, status = {"transport_error": type(error).__name__}, None
                    record = {
                        "role": role,
                        "base_url": conn["base_url"],
                        "endpoint": endpoint,
                        "repeat": repeat + 1,
                        "status": status,
                        "latency_s": time.perf_counter() - started,
                        "request": payload,
                        "response": body,
                    }
                    call = {
                        **usage_record(
                            body,
                            "embedding" if role == "embedding" else "generation",
                            conn["model"],
                            role,
                        ),
                        "base_url": conn["base_url"],
                    }
                    checks = check_usage(
                        body, status, role, conn["model"], args.require_cost
                    )
                    record.update(
                        checks=checks,
                        estimated_list_cost_usd=standard_cost([call]),
                    )
                    sanitized = json.dumps(record, indent=2, ensure_ascii=False)
                    for secret in secrets:
                        sanitized = sanitized.replace(secret, "[REDACTED]")
                    name = f"{role}-{endpoint.split('/')[-1]}-{repeat + 1}.json"
                    (output / name).write_text(sanitized + "\n")
                    results.append(
                        {
                            **checks,
                            "file": name,
                            "status": status,
                            "usage": json.loads(sanitized)["response"].get("usage"),
                            "latency_s": record["latency_s"],
                        }
                    )
                    state = "PASS" if checks["ok"] else "FAIL"
                    cost = checks["cost_usd"]
                    cost_text = (
                        "missing (estimate only)" if cost is None else f"{cost} USD"
                    )
                    print(
                        f"{name}: {state}; HTTP {status}; missing={checks['missing']}; cost={cost_text}",
                        flush=True,
                    )
    (output / "index.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
