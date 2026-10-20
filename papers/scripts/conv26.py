#!/usr/bin/env python3
"""Tables 1 and 3 and the data of Figures 1 and 2 of index.html, from data/conv26_cumulative.json.

Table 1: per method and model, in METHODS and MODELS order, build cost, cost per question, and correct
answers per category. Table 3: the same correct rates split by whether a question's evidence is in a
turn that shares a photo, from the LoCoMo file.

Figure 1 follows HarnessTax's cost-success frontier: each cell's whole rollout,
cost on a log axis, with the cells no other beats marked. Figure 2 follows its
cumulative cost-success curves, with one rollout
being one run over conv-26: the build, then the 152 questions. Each cell's
questions are ordered from cheapest to most expensive, equal costs grouped, and
their costs and correct answers added up after the build; correct answers are
divided by 152. The page draws the curves from the JSON written here, smoothed
by a nine-point moving average with unsmoothed endpoints.

Both are written into index.html between their markers; run after changing the data.
"""

import json
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
CATEGORIES = {4: "Single", 1: "Multi", 2: "Temp.", 3: "Open"}
# Model key: name and reasoning effort, each model at its highest effort. Only these keys are shown.
MODELS = {"mini": ("GPT-4o-mini", None), "luna-high": ("GPT-6 Luna", "high"), "opus": ("Claude Opus 5.5", "high")}
label = lambda m: MODELS[m][0]
# Badge glyph and page key for each method shown.
METHODS = {"Full context": ("F", "full"), "BM25": ("B", "bm25"), "Mem0": ("M", "mem0"), "Zep": ("Z", "zep"),
           "Pi": ("P", "pi"), "Codex": ("C", "codex"), "Claude Code": ("CC", "claude")}
SMOOTH = 9


def sig(x, digits=2):
    """At least three decimals, and enough for two significant digits."""
    return f"{x:.{max(3, digits - 1 - math.floor(math.log10(x)))}f}"


def pct(qs):
    """Share of these questions judged correct, in whole percent, halves rounded up."""
    return int(100 * sum(q["correct"] for q in qs) / len(qs) + 0.5)


def table(cells):
    row = lambda cells, tag="td": "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
    qs = cells[0]["questions"]
    lines = [
        "<table>", "<thead>",
        '<tr><th colspan="3"></th><th colspan="2">Cost ($)</th><th colspan="' + str(len(CATEGORIES) + 1) + '">Correct (%)</th></tr>',
        row(["Method", "Model", "Effort", "Build", "Per q.", *CATEGORIES.values(), "All"], "th"),
        "</thead>", "<tbody>",
        '<tr class="count">' + row(["Questions", "", "", "", "", *(sum(q["category"] == c for q in qs) for c in CATEGORIES),
                                    len(qs)])[4:],
    ]
    for i, c in enumerate(cells):
        qs = c["questions"]
        query = sum(q["cost_usd"] for q in qs)
        first = i == 0 or cells[i - 1]["method"] != c["method"]
        cls = ' class="group"' if first and i else ""
        lines.append(f"<tr{cls}>" + row([c["method"] if first else "", MODELS[c["model"]][0], MODELS[c["model"]][1] or "–",
                                         sig(c["build_usd"]) if c["build_usd"] else "∘", sig(query / len(qs)),
                                         *(pct([q for q in qs if q["category"] == k]) for k in CATEGORIES), pct(qs)])[4:])
    return "\n".join(lines + ["</tbody>", "</table>"])


def photo_questions():
    """Turns that share a photo, and the questions with evidence in one. The official file holds each
    photo's caption, which the agents read; the memory methods were given the turn text only."""
    conv = json.loads((PAPER.parent / "data/locomo-conv26.json").read_text())
    conv = conv[0] if isinstance(conv, list) else conv
    turns = {t["dia_id"] for k, v in conv["conversation"].items() if k.startswith("session_") and isinstance(v, list)
             for t in v if "img_url" in t or "blip_caption" in t}
    qa = [q for q in conv["qa"] if q.get("category") != 5]
    return turns, {f"conv-26:{i}" for i, q in enumerate(qa) if any(e in turns for e in q.get("evidence", []))}


def photos(cells, photo):
    """Table 3: correct rate on the questions with photo evidence and on the others, per method and model."""
    at = {(c["method"], c["model"]): c["questions"] for c in cells}
    n = len(photo), len(cells[0]["questions"]) - len(photo)
    split = lambda i: ' class="split"' if i else ""
    lines = ["<table>", "<thead>",
             "<tr><th></th>" + "".join(f'<th colspan="2"{split(i)}>{label(k)}</th>' for i, k in enumerate(MODELS)) + "</tr>",
             "<tr><th>Method</th>" + "".join(f"<th{split(i)}>Photo ({n[0]})</th><th>Other ({n[1]})</th>" for i, _ in enumerate(MODELS))
             + "</tr>", "</thead>", "<tbody>"]
    for m in METHODS:
        cols = []
        for i, k in enumerate(MODELS):
            qs = at[(m, k)]
            cols.append(f"<td{split(i)}>{pct([q for q in qs if q['id'] in photo])}</td><td>{pct([q for q in qs if q['id'] not in photo])}</td>")
        lines.append(f"<tr><td>{m}</td>" + "".join(cols) + "</tr>")
    return "\n".join(lines + ["</tbody>", "</table>"])


def smooth(values):
    """Centred moving average, the window shrinking at the ends, endpoints kept."""
    half, out = SMOOTH // 2, []
    for i in range(len(values)):
        window = values[max(0, i - half): i + half + 1]
        out.append(sum(window) / len(window))
    out[0], out[-1] = values[0], values[-1]
    return out


def series(qs, value, build, build_s, fmt):
    """One rollout: the build, then the questions from the lowest value to the highest, equal values
    grouped. Each point: questions so far, correct so far, cumulative value, correct / n, highest
    question so far, and seconds so far, build included, the questions taken one after another in this
    order. fmt rounds the values of the points."""
    n, groups = len(qs), {}
    for q in qs:
        # Costs are sums of float prices; group those equal to 1e-10 dollars.
        groups.setdefault(round(value(q), 10), []).append(q)
    k = solved = 0
    total, secs = build, build_s
    points = [[0, 0, build, 0.0, 0, build_s]] if build else []
    for v, group in sorted(groups.items()):
        k += len(group)
        solved += sum(q["correct"] for q in group)
        total += v * len(group)
        secs += sum(q.get("seconds", 0.0) for q in group)
        points.append([k, solved, total, solved / n, v, secs])
    xs, ys = smooth([p[2] for p in points]), smooth([p[3] for p in points])
    return {
        "build": build,
        "points": [[a, b, fmt(x), round(y, 5), fmt(p), round(t, 1)] for a, b, x, y, p, t in points],
        "line": [[float(f"{x:.4g}"), round(y, 5)] for x, y in zip(xs, ys)],
    }


def curve(c):
    """A cell's curve ordered by cost, and in by the same by turns and by tokens."""
    qs, build_s = c["questions"], c.get("build_s", 0.0)
    glyph, key = METHODS[c["method"]]
    cost = series(qs, lambda q: q["cost_usd"], c["build_usd"] or 0.0, build_s, lambda x: float(f"{x:.4g}"))
    return {
        "model": c["model"], "modelName": label(c["model"]), "method": key, "methodName": c["method"],
        "glyph": glyph, "n": len(qs), "buildSeconds": round(build_s, 1), **cost,
        "by": {m: series(qs, lambda q: q[m], c[f"build_{m}"], build_s, int) for m in ("turns", "tokens")},
    }


def frontier(curves, get=lambda c: c):
    """Mark cells no other cell beats on both the total, cost unless get picks an ordering in by, and
    correct answers."""
    for c in curves:
        end = get(c)["points"][-1]
        get(c)["frontier"] = not any(
            o is not c and get(o)["points"][-1][2] <= end[2] and get(o)["points"][-1][1] >= end[1]
            and (get(o)["points"][-1][2], get(o)["points"][-1][1]) != (end[2], end[1])
            for o in curves)


def replace(text, name, body):
    begin, end = f"<!-- {name}: generated by scripts/conv26.py -->", f"<!-- end {name} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return head + begin + "\n" + body + "\n" + end + tail


def main():
    cells = [c for c in json.loads((PAPER / "data/conv26_cumulative.json").read_text())["cells"] if c["model"] in MODELS and c["method"] in METHODS]
    cells.sort(key=lambda c: (list(METHODS).index(c["method"]), list(MODELS).index(c["model"])))
    curves = [curve(c) for c in cells]
    frontier(curves)
    for m in ("turns", "tokens"):
        frontier(curves, lambda c: c["by"][m])
    benchmarks = [{"key": "locomo", "name": "LoCoMo", "title": "LoCoMo", "n": len(cells[0]["questions"]),
                   "curves": curves}]
    data = {"smooth": SMOOTH, "models": {k: label(k) for k in MODELS}, "methods": {k: [g, name] for name, (g, k) in METHODS.items()},
            "benchmarks": benchmarks}
    page = PAPER / "index.html"
    text = replace(page.read_text(), "conv26 table", table(cells))
    text = replace(text, "conv26 photos", photos(cells, photo_questions()[1]))
    script = '<script type="application/json" id="fig-data">' + json.dumps(data, separators=(",", ":")) + "</script>"
    page.write_text(replace(text, "conv26 curves", script))
    for c in curves:
        end = c["points"][-1]
        print(f"{c['methodName']:<13} {c['modelName']:<18} ${end[2]:.4f} per rollout, {end[1]}/{c['n']} correct"
              + ("  frontier" if c["frontier"] else ""))


if __name__ == "__main__":
    main()
