"""Shared LoCoMo set-token F1, with local and Mem0 2025 judge protocols."""

import json
import re
from statistics import mean

from common.mem0_judge import PROMPT as MEM0_JUDGE

JUDGE = """Label the generated answer as 'CORRECT' or 'WRONG' given the gold answer.
Be generous: if the generated answer captures the same core fact as gold, label CORRECT.
For dates/times, different formats of the same date count as CORRECT.

Question: {question}
Gold answer: {reference}
Generated answer: {prediction}

Return only JSON: {{"label": "CORRECT" or "WRONG"}}"""


def token_f1(prediction, reference):
    def tokens(text):
        return set(re.sub(r"[.,!?]", " ", str(text).lower()).split())

    predicted, expected = tokens(prediction), tokens(reference)
    return (
        2 * len(predicted & expected) / (len(predicted) + len(expected))
        if predicted and expected
        else 0.0
    )


def judge(client, question, reference, prediction, protocol="local"):
    if protocol == "mem0":
        raw = client.chat(
            None,
            MEM0_JUDGE.format(
                question=question, gold_answer=reference, generated_answer=prediction
            ),
            max_tokens=None,
            kind="judge",
            role="judge",
            seed=None,
            response_format={"type": "json_object"},
        )
    elif protocol == "local":
        raw = client.chat(
            "You evaluate answer correctness.",
            JUDGE.format(question=question, reference=reference, prediction=prediction),
            max_tokens=80,
            kind="judge",
            role="judge",
        )
    else:
        raise ValueError(f"Unknown judge protocol: {protocol}")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    label = json.loads(match[0] if match else raw)["label"].upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError("Invalid judge label")
    return int(label == "CORRECT")


def summarize(rows):
    if not rows:
        raise ValueError("No results to summarize")

    def average(field):
        values = [r.get(field) for r in rows]
        return mean(values) if all(v is not None for v in values) else None

    result = {
        "n": len(rows),
        "accuracy": average("correct"),
        "f1": average("f1"),
        "generation_tokens": average("generation_tokens"),
        "by_category": {
            str(c): summarize_category([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows}, key=str)
        },
    }
    if any("context_tokens" in r for r in rows):
        result["context_tokens"] = average("context_tokens")
    return result


def summarize_category(rows):
    return {
        "n": len(rows),
        "accuracy": mean(r["correct"] for r in rows),
        "f1": mean(r["f1"] for r in rows),
    }
