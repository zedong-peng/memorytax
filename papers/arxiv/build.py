#!/usr/bin/env python3
"""Figures and tables of the arXiv version (main.tex), from the same data as index.html.

    python3 build.py

Writes tables/*.tex, calls webfig.mjs for figures/*.pdf, and prints the numbers the text quotes so they can
be checked against it. The figures are index.html's own drawings, exported from the page by webfig.mjs:
Figure 1 the cost-correct frontier, Figure 2 the cumulative curves by cost, Figure 3 the writer and reader
of Mem0's and Zep's stores, and Figure 4 the cumulative curves by turns and by tokens.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER / "scripts"))
from conv26 import CATEGORIES, METHODS, MODELS, curve, frontier, pct, photo_questions, sig  # noqa: E402
import cross  # noqa: E402


def cells():
    data = json.loads((PAPER / "data/conv26_cumulative.json").read_text())["cells"]
    out = [c for c in data if c["model"] in MODELS and c["method"] in METHODS]
    return sorted(out, key=lambda c: (list(METHODS).index(c["method"]), list(MODELS).index(c["model"])))


def table_results(cs):
    """Table 2: per method and model, cost and correct answers per category."""
    qs0 = cs[0]["questions"]
    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Cost (\$)} & \multicolumn{5}{c}{Correct (\%)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-10}",
             "Method & Model & Effort & Build & Per q. & " + " & ".join(CATEGORIES.values()) + r" & All \\",
             r"\midrule",
             r"\textit{Questions} & & & & & " + " & ".join(str(sum(q["category"] == k for q in qs0)) for k in CATEGORIES)
             + f" & {len(qs0)} \\\\"]
    for i, c in enumerate(cs):
        qs = c["questions"]
        first = i == 0 or cs[i - 1]["method"] != c["method"]
        if first:
            lines.append(r"\midrule")
        name, effort = MODELS[c["model"]]
        cats = [str(pct([q for q in qs if q["category"] == k])) for k in CATEGORIES]
        lines.append(" & ".join([c["method"] if first else "", name, effort or "--",
                                 sig(c["build_usd"]) if c["build_usd"] else r"$\circ$",
                                 sig(sum(q["cost_usd"] for q in qs) / len(qs)), *cats, str(pct(qs))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables/results.tex").write_text("\n".join(lines) + "\n")


def table_cross(grid):
    """Table 3: correct rate with one model writing the store and another reading it."""
    keys = list(MODELS)
    at = {(c["method"], c["writer"], c["reader"]): c for c in grid}
    short = lambda k: MODELS[k][0].replace("Claude ", "")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Mem0, read by} & \multicolumn{3}{c}{Zep, read by} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             "Written by & " + " & ".join(short(r) for _ in cross.METHODS for r in keys) + r" \\", r"\midrule"]
    for w in keys:
        vals = []
        for m in cross.METHODS:
            for r in keys:
                c = at[(m, w, r)]
                v = str(int(100 * c["correct"] / c["n"] + 0.5))
                vals.append(rf"\textbf{{{v}}}" if w == r else v)
        lines.append(MODELS[w][0] + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables/cross.tex").write_text("\n".join(lines) + "\n")


def table_photos(cs, photo):
    """Table 4: correct rate on the questions with photo evidence and on the others."""
    keys = list(MODELS)
    at = {(c["method"], c["model"]): c["questions"] for c in cs}
    n = len(photo), len(cs[0]["questions"]) - len(photo)
    lines = [r"\begin{tabular}{l" + "rr" * len(keys) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MODELS[k][0]}}}" for k in keys) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(keys))),
             "Method" + f" & Photo ({n[0]}) & Other ({n[1]})" * len(keys) + r" \\", r"\midrule"]
    for m in METHODS:
        vals = []
        for k in keys:
            qs = at[(m, k)]
            vals += [str(pct([q for q in qs if q["id"] in photo])), str(pct([q for q in qs if q["id"] not in photo]))]
        lines.append(m + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables/photos.tex").write_text("\n".join(lines) + "\n")


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "tables").mkdir(exist_ok=True)
    cs = cells()
    curves = [curve(c) for c in cs]
    frontier(curves)
    for m in ("turns", "tokens"):
        frontier(curves, lambda c: c["by"][m])
    stored = {(c["method"], c["writer"], c["reader"]): c
              for c in json.loads((PAPER / "data/conv26_cross.json").read_text())["cells"]}
    grid = cross.grid(stored)
    subprocess.run(["node", str(HERE / "webfig.mjs")], check=True)
    table_results(cs)
    table_cross(grid)
    turns, photo = photo_questions()
    table_photos(cs, photo)
    # The numbers the text quotes.
    for c in curves:
        end = c["points"][-1]
        tags = [m for m, on in (("cost", c["frontier"]), ("turns", c["by"]["turns"]["frontier"]),
                                ("tokens", c["by"]["tokens"]["frontier"])) if on]
        print(f"{c['methodName']:<12} {c['modelName']:<16} ${end[2]:8.2f}  {end[1]:3}/{c['n']}  build ${c['build']:.2f}"
              f"  frontier: {', '.join(tags) or '-'}")
    print(f"{len(turns)} turns share a photo; {len(photo)} questions have evidence in one")
    full = {q["id"]: q["correct"] for q in next(c for c in cs if (c["method"], c["model"]) == ("Full context", "opus"))["questions"]}
    for c in cs:
        if c["model"] == "opus" and c["method"] != "Full context":
            won = [q["id"] for q in c["questions"] if q["correct"] and not full[q["id"]]]
            lost = [q["id"] for q in c["questions"] if full[q["id"]] and not q["correct"]]
            print(f"{c['method']} vs full context, Claude Opus 5.5: wins {len(won)} ({sum(i in photo for i in won)} photo),"
                  f" losses {len(lost)} ({sum(i in photo for i in lost)} photo)")
    for c in cs:
        if c["method"] == "Mem0":
            fc = next(f for f in cs if f["method"] == "Full context" and f["model"] == c["model"])
            per = lambda x: sum(q["cost_usd"] for q in x["questions"]) / len(x["questions"])
            print(f"Mem0 {c['model']}: build pays off after {c['build_usd'] / (per(fc) - per(c)):.0f} questions")


if __name__ == "__main__":
    main()
