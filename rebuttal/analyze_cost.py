"""Comparison at matched realized review cost (unit cost: items reviewed per round, rounds >= 20)."""
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
sns.set_theme(context="paper", style="darkgrid")
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "font.weight": "bold",
                     "axes.titleweight": "bold", "axes.labelweight": "bold", "pdf.fonttype": 42})

COST_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0]
WINDOW = 0.25

rows = []
fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
for r, model in enumerate(["llama", "qwen"]):
    mean = pd.read_csv(results_dir(model) / "baselines_mean.csv")
    for c, s in enumerate(SETTINGS):
        ax = axes[r][c]
        for k in list(NAMES)[::-1]:
            d = mean[(mean.setting == s) & (mean.method == k)]
            for lvl in COST_LEVELS:
                w = d[(d.cost - lvl).abs() <= WINDOW + 1e-9]
                b = None if w.empty else w.loc[w.type_ii.idxmin()]
                rows.append(dict(model=model, setting=s, method=NAMES[k], cost_level=lvl,
                                 realized_cost=np.nan if b is None else b.cost,
                                 type_ii=np.nan if b is None else b.type_ii, type_i=np.nan if b is None else b.type_i))
            e = d.sort_values("cost")
            ax.plot(e.cost, e.type_ii.cummin(), drawstyle="steps-post", color=COLOR[k], ls=STYLE[k],
                    lw=3.4 if k == "edit" else 2.2, label=NAMES[k])
        ax.set_title(f"{s} ({'Llama-3.3-70B' if model == 'llama' else 'Qwen3-32B'})")
        ax.set_xlabel("Realized cost per round")
        ax.set_xlim(0, 5.2)
    axes[r][0].set_ylabel("Type-II error")
h, lab = axes[0][0].get_legend_handles_labels()
fig.legend(h[::-1], lab[::-1], loc="lower center", ncol=len(h), frameon=False, fontsize=15, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RESULTS / "matched_cost.png", dpi=150, bbox_inches="tight")
fig.savefig(RESULTS / "matched_cost.pdf", bbox_inches="tight")
pd.DataFrame(rows).to_csv(RESULTS / "matched_cost.csv", index=False)
