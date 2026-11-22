import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from engine import RESULTS, SETTINGS, results_dir  # noqa: E402

NAMES = {"edit": "EDIT-REP (Ours)", "gollapudi": "Online Conformal Selection", "conformal": "Conformal Alignment",
         "zliobaite": "Stream Active Learning", "thompson": "Thompson Sampling", "topk": "Top-K", "tau": "Fixed Threshold"}
COLOR = {"edit": "#ff7f0e", "gollapudi": "#d62728", "conformal": "#9467bd", "zliobaite": "#8c564b", "thompson": "#17becf",
         "topk": "#1f77b4", "tau": "#2ca02c"}
STYLE = {"edit": "-", "gollapudi": "-.", "conformal": "--", "zliobaite": ":", "thompson": "--", "topk": "--", "tau": ":"}
MARK = {"edit": "o", "gollapudi": "D", "conformal": "s", "zliobaite": "v", "thompson": "^", "topk": "P", "tau": "X"}
sns.set_theme(context="paper", style="darkgrid")
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold", "pdf.fonttype": 42})


def legend(fig, ax, y, ncol=None):
    h, lab = ax.get_legend_handles_labels()
    keep = [i for i, x in enumerate(lab) if not x.startswith("_")][::-1]
    fig.legend([h[i] for i in keep], [lab[i] for i in keep], loc="lower center", ncol=ncol or len(keep), frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, y), handlelength=2.6, columnspacing=1.4)


shift = pd.read_csv(RESULTS / "shift_results.csv")
rows = [("type_i", "Overall Type-I"), ("type_ii", "Overall Type-II")]
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
for c, s in enumerate(["MMLU-Pro shift", "PopQA shift"]):
    d = shift[shift.setting == s]
    for r, (col, lab) in enumerate(rows):
        ax = axes[r][c]
        for k in list(NAMES)[::-1]:
            x = d[d.method == k].sort_values("target")
            ax.plot(x.target, x[col], color=COLOR[k], ls=STYLE[k], marker=MARK[k], ms=6, lw=3.2 if k == "edit" else 1.9,
                    label=NAMES[k])
        if r == 0:
            ax.plot([0.1, 0.5], [0.1, 0.5], color="black", lw=1.4, ls="--", label="_target")
        if r == 0:
            ax.set_title(s)
        if c == 0:
            ax.set_ylabel(lab)
        if r == 1:
            ax.set_xlabel("Type-I target")
legend(fig, axes[0][0], -0.01, ncol=4)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(RESULTS / "shift_robustness.png", dpi=160, bbox_inches="tight")
fig.savefig(RESULTS / "shift_robustness.pdf", bbox_inches="tight")

fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
for r, model in enumerate(["llama", "qwen"]):
    m = pd.read_csv(results_dir(model) / "regret_mean.csv")
    for c, s in enumerate(SETTINGS):
        ax = axes[r][c]
        for k in list(NAMES)[::-1]:
            x = m[(m.setting == s) & (m.method == k)].sort_values("level")
            ax.plot(x.level, x.regb, color=COLOR[k], ls=STYLE[k], marker=MARK[k], ms=6, lw=3.2 if k == "edit" else 1.9,
                    label=NAMES[k])
        ax.set_title(f"{s} ({'Llama-3.3-70B' if model == 'llama' else 'Qwen3-32B'})")
        ax.set_xlabel("Type-I level")
        if c == 0:
            ax.set_ylabel("Missed edits / round vs. oracle")
legend(fig, axes[0][0], -0.04)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RESULTS / "oracle_regret.png", dpi=150, bbox_inches="tight")
fig.savefig(RESULTS / "oracle_regret.pdf", bbox_inches="tight")
