"""Official LoCoMo F1 (categories 1--4) of a run's saved answers.

    python src/common/locomo/score.py results/<method>/<model>/summary.json
"""

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.artifacts import read
from common.locomo.official import f1, f1_score

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "data/locomo-conv26.json").exists()
)


def official_f1(prediction, reference, category):
    if category == 3:
        reference = reference.split(";")[0].strip()
    if category == 1:
        return float(f1(prediction, reference))
    if category in (2, 3, 4):
        return float(f1_score(prediction, reference))
    if category == 5:
        return float(
            "no information available" in prediction.lower()
            or "not mentioned" in prediction.lower()
        )
    raise ValueError(f"Unknown LoCoMo category: {category}")


def aggregate(rows):
    return {
        "n": len(rows),
        "official_f1": mean(r["official_f1"] for r in rows),
        "by_category": {
            str(c): {
                "n": sum(r["category"] == c for r in rows),
                "official_f1": mean(
                    r["official_f1"] for r in rows if r["category"] == c
                ),
            }
            for c in sorted({r["category"] for r in rows})
        },
    }


def references(data):
    return {
        f"{c['sample_id']}:{i}": q for c in read(data) for i, q in enumerate(c["qa"])
    }


def score_rows(rows, gold):
    result = []
    for row in rows:
        q = gold[row["id"]]
        if (
            row.get("question", q["question"]) != q["question"]
            or int(row.get("category", q["category"])) != q["category"]
        ):
            raise ValueError(f"Question identity mismatch: {row['id']}")
        reference = str(q.get("answer", ""))
        if not isinstance(row["prediction"], str):
            raise ValueError(f"Non-text answer: {row['id']}")
        result.append(
            {
                "id": row["id"],
                "category": q["category"],
                "prediction_sha256": hashlib.sha256(
                    row["prediction"].encode()
                ).hexdigest(),
                "official_f1": official_f1(row["prediction"], reference, q["category"]),
            }
        )
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate question IDs")
    return result


if __name__ == "__main__":
    rows = read(sys.argv[1])["rows"]
    print(json.dumps(aggregate(score_rows(rows, references(ROOT / "data/locomo-conv26.json"))), indent=2))
