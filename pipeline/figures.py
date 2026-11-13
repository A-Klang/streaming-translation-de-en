"""Report figures from results/summary.json (bootstrap.py) and results/report_table.csv
(report_table.py). Writes PDF (for the report) and PNG (for a quick look) to results/figures/.
From the repository folder:
    python pipeline/figures.py

fig1_quality_lag    COMET against LAAL, one panel per policy family, both arms (RQ1)
fig2_quality_compute COMET against compute per German word, one panel per arm (RQ1)
fig3_guessing_lag   committed and wrong guesses per 100 German words against LAAL (RQ2)
fig4_raw_guessing   share of mid-sentence replies that contain a guess, per setting (RQ2)
Error bars are 95% bootstrap intervals. LAAL is on the 712 report sentences throughout;
the guess counts are on the 100-sentence RQ2 sample.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bootstrap import ARMS, FAMILIES, FULL

OUT = Path("results/figures")
NAMES = {"hymt2-q4": "Hy-MT2", "gemma-e2b-q4": "Gemma"}
ARM_COLOR = {"hymt2-q4": "#2a78d6", "gemma-e2b-q4": "#eb6834"}  # validated pair (dataviz palette slots 1, 2)
FAM_COLOR = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # slots 1 to 4
FAM_MARKER = dict(zip(FAMILIES, ["o", "s", "^", "D"]))  # shape as the second channel
ARM_MARKER = {"hymt2-q4": "o", "gemma-e2b-q4": "s"}
SHORT = {"waitk1": "k1", "waitk3": "k3", "waitk5": "k5", "la-s1": "s1", "la-s2": "s2", "la-s3": "s3",
         "la-s1-verb": "s1", "la-s2-verb": "s2", "la-s3-verb": "s3",
         "grammar-d0": "d0", "grammar-d1": "d1", "grammar-d2": "d2"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200, "pdf.fonttype": 42,
})

S = json.loads(Path("results/summary.json").read_text())
T = {f"{r['arm']}_{r['setting']}": r for r in csv.DictReader(open("results/report_table.csv"))}


def table(arm, setting):
    name = NAMES[arm] + "_" + ("full sentence" if setting == FULL else setting)
    return {k: float(v) if v not in ("", None) and k not in ("arm", "setting") else v for k, v in T[name].items()}


def point(arm, setting, metric):
    m = S[f"{arm}_{setting}"][metric]
    return m["point"], m["point"] - m["lo"], m["hi"] - m["point"]


def errorbar(ax, xs, ys, color, marker, label=None, xerr=None, yerr=None):
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=color, marker=marker, markersize=5.5, linewidth=1.6,
                elinewidth=0.9, capsize=0, markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1():
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0), sharey=True)
    for ax, (fam, settings) in zip(axes, FAMILIES.items()):
        for arm in ARMS:
            pts = sorted((point(arm, s, "laal"), point(arm, s, "comet"), s) for s in settings)
            xs, ys = [p[0][0] for p in pts], [p[1][0] for p in pts]
            errorbar(ax, xs, ys, ARM_COLOR[arm], ARM_MARKER[arm], NAMES[arm],
                     xerr=[[p[0][1] for p in pts], [p[0][2] for p in pts]],
                     yerr=[[p[1][1] for p in pts], [p[1][2] for p in pts]])
            ax.axhline(S[f"{arm}_{FULL}"]["comet"]["point"], color=ARM_COLOR[arm], linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=1)
            if arm == ARMS[0]:  # setting labels once per panel, under the Hy-MT2 points
                for x, y, s in zip(xs, ys, [p[2] for p in pts]):
                    ax.annotate(SHORT[s], (x, y), xytext=(6, -10), textcoords="offset points",
                                ha="left", fontsize=7.5, color=MUTED)
        ax.set_title(fam, color=INK)
        ax.set_xlabel("LAAL (German words)")
    axes[0].set_ylabel("COMET")
    axes[-1].text(1.0, S[f"gemma-e2b-q4_{FULL}"]["comet"]["point"], " full\n sentence", transform=axes[-1].get_yaxis_transform(),
                  va="center", fontsize=7.5, color=MUTED)
    axes[0].legend(loc="lower right")
    save(fig, "fig1_quality_lag")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True, sharex=True)
    for ax, arm in zip(axes, ARMS):
        for fam, settings in FAMILIES.items():
            # in setting order (k1, k3, k5 ...), not sorted by compute, which would zigzag
            pts = [(table(arm, s)["ms_per_word"], point(arm, s, "comet"), s) for s in settings]
            errorbar(ax, [p[0] for p in pts], [p[1][0] for p in pts], FAM_COLOR[fam], FAM_MARKER[fam], fam,
                     yerr=[[p[1][1] for p in pts], [p[1][2] for p in pts]])
        full = table(arm, FULL)
        ax.plot(full["ms_per_word"], S[f"{arm}_{FULL}"]["comet"]["point"], marker="*", markersize=10,
                color=INK, linestyle="none", label="full sentence", zorder=4)
        ax.set_title(NAMES[arm], color=INK)
        ax.set_xlabel("compute per German word (ms)")
    axes[0].set_ylabel("COMET")
    for ax, arm in zip(axes, ARMS):  # mark where each wait-k and LA curve starts
        for fam in ("wait-k", "LA"):
            s = FAMILIES[fam][0]
            ax.annotate(SHORT[s], (table(arm, s)["ms_per_word"], point(arm, s, "comet")[0]), xytext=(6, -3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    save(fig, "fig2_quality_compute")


def fig3():
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), sharey="row", sharex="col")
    for row, (metric, label) in enumerate((("guesses", "committed guesses\nper 100 German words"),
                                            ("wrong", "wrong committed guesses\nper 100 German words"))):
        for ax, (fam, settings) in zip(axes[row], FAMILIES.items()):
            for arm in ARMS:
                pts = sorted((point(arm, s, "laal")[0], point(arm, s, metric), s) for s in settings)
                errorbar(ax, [p[0] for p in pts], [p[1][0] for p in pts], ARM_COLOR[arm], ARM_MARKER[arm], NAMES[arm],
                         yerr=[[p[1][1] for p in pts], [p[1][2] for p in pts]])
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(fam, color=INK)
            else:
                ax.set_xlabel("LAAL (German words)")
        axes[row][0].set_ylabel(label)
    axes[0][0].legend(loc="upper right")
    save(fig, "fig3_guessing_lag")


def fig4():
    settings = [s for f in FAMILIES.values() for s in f]
    labels = ["wait\nk1", "wait\nk3", "wait\nk5", "LA\ns1", "LA\ns2", "LA\ns3", "LA+verb\ns1", "LA+verb\ns2",
              "LA+verb\ns3", "Grammar\nd0", "Grammar\nd1", "Grammar\nd2"]
    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    w = 0.38
    for k, arm in enumerate(ARMS):
        vals = [point(arm, s, "raw") for s in settings]
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(settings))]
        ax.bar(xs, [v[0] for v in vals], width=w, color=ARM_COLOR[arm], label=NAMES[arm], zorder=2)
        ax.errorbar(xs, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]],
                    fmt="none", ecolor=INK, elinewidth=0.8, zorder=3)
    ax.set_xticks(range(len(settings)), labels, rotation=0, fontsize=7.5)
    ax.set_ylabel("replies with a guess (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "fig4_raw_guessing")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
    print("saved to", OUT)
