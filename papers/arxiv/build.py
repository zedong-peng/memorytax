#!/usr/bin/env python3
"""Figures and tables of the arXiv version (main.tex), from the same data as index.html.

    python3 build.py

Needs matplotlib (3.8.2 was used). Writes figures/*.pdf and tables/*.tex, and prints the numbers the text
quotes so they can be checked against it. The figures are the still views of index.html's interactive
figures: Figure 1 the cost-correct frontier, Figure 2 the cumulative curves by cost, Figure 3 the writer
and reader of Mem0's and Zep's stores, and Figure 4 the cumulative curves by turns and by tokens.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER / "scripts"))
from conv26 import CATEGORIES, METHODS, MODELS, curve, frontier, pct, photo_questions, sig  # noqa: E402
import cross  # noqa: E402

COLOURS = {"mini": "#eb6834", "luna-high": "#1baf7a", "opus": "#eda100"}  # index.html's --m-* colours
AGENTS = {"Pi", "Codex", "Claude Code"}
FAINT = 0.35
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#888",
    "xtick.color": "#444", "ytick.color": "#444", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def cells():
    data = json.loads((PAPER / "data/conv26_cumulative.json").read_text())["cells"]
    out = [c for c in data if c["model"] in MODELS and c["method"] in METHODS]
    return sorted(out, key=lambda c: (list(METHODS).index(c["method"]), list(MODELS).index(c["model"])))


def badge(ax, x, y, c, alpha=1.0, filled=False, size=9.5, z=3):
    """A cell's badge: a circle in the model's colour holding the method's letter."""
    colour = COLOURS[c["model"]]
    glyph = METHODS[c["methodName"] if "methodName" in c else c["method"]][0]
    ax.plot([x], [y], "o", ms=size, mfc=colour if filled else "white", mec=colour, mew=1.1, alpha=alpha, zorder=z)
    ax.text(x, y, glyph, ha="center", va="center", fontsize=5.6 if len(glyph) > 1 else 6.2, fontweight="bold",
            color="white" if filled else colour, alpha=alpha, zorder=z + 0.1)


def model_legend(ax, **kw):
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=COLOURS[k], mew=1.2, label=name)
               for k, (name, _) in MODELS.items()]
    ax.legend(handles=handles, frameon=False, handletextpad=0.2, columnspacing=1.0, **kw)


def fig_frontier(curves):
    """Figure 1: each rollout's cost (log) and correct rate, with the stepped frontier."""
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ends = [(c["points"][-1][2], 100 * c["points"][-1][1] / c["n"], c) for c in curves]
    for x, y, c in ends:
        badge(ax, x, y, c, alpha=1.0 if c["frontier"] else 0.55)
    best, steps = -1, []
    for x, y, c in sorted(ends, key=lambda e: e[0]):
        if y > best:
            best = y
            steps.append((x, y, c))
    xs = [s[0] for s in steps] + [60]
    ys = [s[1] for s in steps] + [steps[-1][1]]
    ax.step(xs, ys, where="post", color="#333", lw=0.9, zorder=2)
    for i, (x, y, c) in enumerate(steps):  # the first label goes right, clear of the axis
        ax.annotate(f"{c['modelName']}\n{c['methodName']}", (x, y), xytext=(6 if i == 0 else -6, 6),
                    textcoords="offset points", ha="left" if i == 0 else "right", va="bottom", fontsize=6.8,
                    color="#222", linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 60)
    ax.set_ylim(0, 116)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xticks([0.1, 1, 10], ["$0.1", "$1", "$10"])
    ax.set_xlabel("cost per rollout (USD, log scale)")
    ax.set_ylabel("correct (share of 152 questions)")
    model_legend(ax, loc="lower right", ncol=3)
    fig.savefig(HERE / "figures/frontier.pdf")
    plt.close(fig)


def broken(limit=2.0, decade=1.0):
    """Linear up to limit dollars, then logarithmic, one decade taking `decade` dollars of width."""
    import numpy as np
    f = lambda x: np.where(x <= limit, x, limit + decade * np.log10(np.maximum(x, limit) / limit))
    g = lambda y: np.where(y <= limit, y, limit * 10 ** ((np.maximum(y, limit) - limit) / decade))
    return f, g


def cumulative(ax, curves, get, scale_x=lambda v: v):
    for c in curves:
        s = get(c)
        on = s["frontier"]
        xs = [scale_x(p[0]) for p in s["line"]]
        ys = [100 * p[1] for p in s["line"]]
        ax.plot(xs, ys, color=COLOURS[c["model"]], lw=1.4 if on else 0.9, alpha=1.0 if on else FAINT,
                ls=(0, (1, 1.4)) if c["methodName"] in AGENTS else "-", zorder=3 if on else 2)
        end = s["points"][-1]
        badge(ax, scale_x(end[2]), 100 * end[3], c, alpha=1.0 if on else 0.5, size=8.5, z=4 if on else 3)
    ax.set_ylim(-2, 104)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])


def fig_cumulative(curves):
    """Figure 2: cumulative cost-correct curves, the cost axis linear to $2 and logarithmic above."""
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.set_xscale("function", functions=broken())
    cumulative(ax, curves, lambda c: c)
    ax.axvline(2, color="#bbb", lw=0.6, zorder=1)
    ticks = [0, 0.5, 1, 1.5, 2, 10, 20, 40]
    ax.set_xticks(ticks, ["$0" if t == 0 else f"${t:g}" for t in ticks])
    ax.set_xlim(-0.05, 46)
    ax.set_xlabel("cumulative cost per rollout (USD; linear to $2, log above)")
    ax.set_ylabel("cumulative correct (share of 152 questions)")
    model_legend(ax, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(HERE / "figures/cumulative.pdf")
    plt.close(fig)


def fig_usage(curves):
    """Figure 4: the same curves by turns and by tokens, linear axes, each with its own frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
    for ax, m, label, unit in ((axes[0], "turns", "cumulative turns (LLM calls)", 1),
                               (axes[1], "tokens", "cumulative tokens (millions)", 1e6)):
        cumulative(ax, curves, lambda c: {**c["by"][m], "line": [[x / unit, y] for x, y in c["by"][m]["line"]],
                                           "points": [[a, b, x / unit, *rest] for a, b, x, *rest in c["by"][m]["points"]]})
        ax.set_xlabel(label)
        ax.set_xlim(left=-0.02 * max(c["by"][m]["points"][-1][2] / unit for c in curves))
    axes[0].set_ylabel("cumulative correct (share of 152 questions)")
    model_legend(axes[0], loc="lower right", ncol=1)
    fig.savefig(HERE / "figures/usage.pdf")
    plt.close(fig)


def fig_cross(grid):
    """Figure 3: each store a line of arrows from its writer reading it (filled) to the other readers."""
    keys = list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)
    for ax, (m, name) in zip(axes, cross.METHODS.items()):
        for w in keys:
            row = {c["reader"]: 100 * c["correct"] / c["n"] for c in grid if c["method"] == m and c["writer"] == w}
            x0, y0 = keys.index(w), row[w]
            for r in keys:
                if r != w:
                    ax.annotate("", (keys.index(r), row[r]), (x0, y0), arrowprops=dict(
                        arrowstyle="-|>", color=COLOURS[w], lw=1.2, shrinkA=6, shrinkB=1, mutation_scale=7))
            badge(ax, x0, y0, {"model": w, "method": name}, filled=True)
        ax.set_title(name, fontsize=8.5, loc="left", fontweight="bold")
        ax.set_xticks(range(3), [MODELS[k][0] for k in keys])
        ax.set_xlim(-0.4, 2.4)
        ax.set_xlabel("read by")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(30, 100)
    axes[0].set_yticks(range(30, 101, 10), [f"{v}%" for v in range(30, 101, 10)])
    axes[0].set_ylabel("correct")
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=COLOURS[k], mec=COLOURS[k], label=MODELS[k][0]) for k in keys]
    axes[0].legend(handles=handles, title="written by", title_fontsize=7, frameon=False, loc="lower right",
                   handletextpad=0.2)
    fig.savefig(HERE / "figures/cross.pdf")
    plt.close(fig)


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
    fig_frontier(curves)
    fig_cumulative(curves)
    fig_usage(curves)
    fig_cross(grid)
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
