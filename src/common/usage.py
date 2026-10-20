"""Comparable costs from provider token usage and a fixed uncached price schedule."""

from statistics import mean

from common.llm import ROLES

# Published list prices, USD per token; kept fixed per run.
PRICES = {
    # OpenAI model documentation, checked 2026-09-30; uncached Standard rates.
    "openai/gpt-6-luna": {"input": 0.10 / 1_000_000, "output": 0.50 / 1_000_000},
    # List prices, checked 2026-09-19.
    "openai/gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "openai/text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0.0},
    # Anthropic pricing, checked 2026-10-10; uncached input rate (cache reads are $0.20/MTok).
    "anthropic/claude-opus-5-5": {"input": 4.00 / 1_000_000, "output": 20.00 / 1_000_000},
}


def standard_cost(calls):
    total = 0.0
    for call in calls:
        model = call.get("requested_model", call.get("model", ""))
        if model.startswith(("gpt-", "text-embedding-")):
            model = "openai/" + model
        elif model.startswith("claude-"):
            model = "anthropic/" + model
        price = PRICES.get(model)
        if model == "openai/gpt-6-luna" and (call.get("input_tokens") or 0) > 272_000:
            price = {"input": price["input"] * 2, "output": price["output"] * 1.5}
        if (
            price is None
            or call.get("input_tokens") is None
            or (call.get("kind") != "embedding" and call.get("output_tokens") is None)
        ):
            return None
        total += (
            call["input_tokens"] * price["input"]
            + (0 if call.get("kind") == "embedding" else call["output_tokens"])
            * price["output"]
        )
    return total


def billed_cost(calls):
    values = [c.get("cost_usd") for c in calls]
    return sum(values) if all(v is not None for v in values) else None


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def summarize_resources(rows):
    result = {}
    for key in (
        "query_cost_usd",
        "query_billed_cost_usd",
        "query_latency_s",
        "retrieval_latency_s",
    ):
        values = [r.get(key) for r in rows]
        result[key] = mean(values) if all(v is not None for v in values) else None
    latency = [r.get("query_latency_s") for r in rows]
    for suffix, fraction in (("p50", 0.50), ("p95", 0.95)):
        result[f"query_latency_{suffix}_s"] = (
            percentile(latency, fraction)
            if all(v is not None for v in latency)
            else None
        )
    uninterrupted = [v for v in latency if v is not None]
    result["uninterrupted_latency_count"] = len(uninterrupted)
    result["missing_or_resumed_latency_count"] = len(latency) - len(uninterrupted)
    for suffix, fraction in (("p50", 0.50), ("p95", 0.95)):
        result[f"uninterrupted_latency_{suffix}_s"] = (
            percentile(uninterrupted, fraction) if uninterrupted else None
        )
    return result


def nullable_total(entries, field):
    values = [entry.get(field) for entry in entries]
    return sum(values) if all(value is not None for value in values) else None


def summarize_roles(calls, uncertain=()):
    """Physical requests only: disk ledger includes each committed response once."""
    result = {}
    for role in ROLES:
        entries = [c for c in calls if c.get("role") == role and not c.get("cache_hit")]
        failed = [c for c in uncertain if c.get("role") == role]

        result[role] = {
            "request_count": len(entries) + len(failed),
            "successful_requests": len(entries),
            "failed_or_unknown_requests": len(failed),
            **{
                field: nullable_total(entries + failed, field)
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                    "latency_s",
                )
            },
            "provider_reported_cost_usd": None if failed else billed_cost(entries),
            "estimated_list_cost_usd": None if failed else standard_cost(entries),
        }
    return result
