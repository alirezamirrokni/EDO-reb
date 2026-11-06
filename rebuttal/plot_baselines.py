import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from engine import RESULTS, SETTINGS  # noqa: E402

sns.set_theme(context="paper", style="darkgrid")
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold", "pdf.fonttype": 42})

NAMES = {"edit": "EDIT-REP (Ours)", "gollapudi": "Online Conformal Selection", "conformal": "Conformal Alignment",
         "zliobaite": "Stream Active Learning", "thompson": "Thompson Sampling", "topk": "Top-K", "tau": "Fixed Threshold"}
COLOR = {"edit": "#ff7f0e", "gollapudi": "#d62728", "conformal": "#9467bd", "zliobaite": "#8c564b", "thompson": "#17becf",
         "topk": "#1f77b4", "tau": "#2ca02c"}
STYLE = {"edit": "-", "gollapudi": "-.", "conformal": "--", "zliobaite": ":", "thompson": "--", "topk": "--", "tau": ":"}
KNOBS = {"edit": ["p_epsilon"], "gollapudi": ["p_epsilon", "p_dual_step"], "conformal": ["p_q"],
         "zliobaite": ["p_budget_per_batch", "p_randomized", "p_adj_step"], "thompson": ["p_budget_per_batch", "p_v"],
         "topk": ["p_budget_per_batch"], "tau": ["p_tau"]}
LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

raw = pd.read_csv(RESULTS / "baselines_raw.csv")
pcols = [c for c in raw.columns if c.startswith("p_")]
raw[pcols] = raw[pcols].fillna(-1)
mean = raw.groupby(["setting", "method"] + pcols).agg(type_i=("type_i", "mean"), type_ii=("type_ii", "mean"),
                                                      cost=("cost", "mean"), n_streams=("seed", "count")).reset_index()
mean.to_csv(RESULTS / "baselines_mean.csv", index=False)


def envelope(d):
    d = d.sort_values(["type_i", "type_ii"])
    return d[d.type_ii < d.type_ii.cummin().shift(1, fill_value=np.inf) - 1e-12]


levels, chosen = [], []
for s in SETTINGS:
    for k in NAMES:
        d = mean[(mean.setting == s) & (mean.method == k)]
        levels.append(dict(setting=s, method=NAMES[k], **{f"type_i<={l:.2f}": (d[d.type_i <= l + 1e-9].type_ii.min()
                                                                                if (d.type_i <= l + 1e-9).any() else np.nan) for l in LEVELS}))
        env = envelope(d)
        chosen.append(dict(setting=s, method=NAMES[k], envelope_points=len(env),
                           **{c[2:]: " ".join(f"{v:g}" for v in sorted(env[c].unique())) for c in KNOBS[k]}))
pd.DataFrame(levels).to_csv(RESULTS / "baselines_levels.csv", index=False)
pd.DataFrame(chosen).to_csv(RESULTS / "baselines_selected_hparams.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))
for ax, s in zip(axes, SETTINGS):
    for k in list(NAMES)[::-1]:
        d = mean[(mean.setting == s) & (mean.method == k)].sort_values("type_i")
        ax.plot(d.type_i, d.type_ii.cummin(), drawstyle="steps-post", color=COLOR[k], lw=3.4 if k == "edit" else 2.2,
                ls=STYLE[k], label=NAMES[k])
    ax.set_title(s)
    ax.set_xlabel("Type-I error")
    ax.set_xlim(0, mean[(mean.setting == s) & (mean.method == "edit")].type_i.max() + 0.03)
axes[0].set_ylabel("Type-II error")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], lab[::-1], loc="lower center", ncol=len(h), frameon=False, fontsize=15, bbox_to_anchor=(0.5, -0.07),
           handlelength=2.6, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(RESULTS / "baselines_frontier.png", dpi=170, bbox_inches="tight")
fig.savefig(RESULTS / "baselines_frontier.pdf", bbox_inches="tight")
