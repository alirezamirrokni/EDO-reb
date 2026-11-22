"""Create the baseline frontier figure and tables for the no-repeat runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from run_baselines_norepeat import SETTINGS

RESULTS = Path(__file__).resolve().parent / "results" / "norepeat"

NAMES = {
    "edit": "EDIT-REP (Ours)",
    "gollapudi": "Online Conformal Selection",
    "conformal": "Conformal Alignment",
    "zliobaite": "Stream Active Learning",
    "thompson": "Thompson Sampling",
    "topk": "Top-K",
    "tau": "Fixed Threshold",
}
COLOR = {
    "edit": "#ff7f0e",
    "gollapudi": "#d62728",
    "conformal": "#9467bd",
    "zliobaite": "#8c564b",
    "thompson": "#17becf",
    "topk": "#1f77b4",
    "tau": "#2ca02c",
}
STYLE = {
    "edit": "-",
    "gollapudi": "-.",
    "conformal": "--",
    "zliobaite": ":",
    "thompson": "--",
    "topk": "--",
    "tau": ":",
}
KNOBS = {
    "edit": ["p_epsilon", "p_alpha_step_size"],
    "gollapudi": ["p_epsilon", "p_dual_step"],
    "conformal": ["p_q"],
    "zliobaite": ["p_budget_per_batch", "p_randomized", "p_adj_step"],
    "thompson": ["p_budget_per_batch", "p_v"],
    "topk": ["p_budget_per_batch"],
    "tau": ["p_tau"],
}
LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

sns.set_theme(context="paper", style="darkgrid")
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
    }
)


def envelope(data):
    data = data.sort_values(["type_i", "type_ii"])
    return data[data.type_ii < data.type_ii.cummin().shift(1, fill_value=np.inf) - 1e-12]


def main():
    qwen17 = RESULTS / "baselines_raw_qwen17.csv"
    raw_path = qwen17 if qwen17.exists() else RESULTS / "baselines_raw.csv"
    raw = pd.read_csv(raw_path)
    print(f"[plot] source: {raw_path}")
    parameter_columns = [column for column in raw if column.startswith("p_")]
    raw[parameter_columns] = raw[parameter_columns].fillna(-1)
    mean = (
        raw.groupby(["setting", "method"] + parameter_columns)
        .agg(
            type_i=("type_i", "mean"),
            type_ii=("type_ii", "mean"),
            cost=("cost", "mean"),
            n_streams=("seed", "count"),
        )
        .reset_index()
    )
    if not mean.n_streams.eq(5).all():
        raise ValueError("Every frontier point must average exactly five no-repeat streams")
    mean.to_csv(RESULTS / "baselines_mean.csv", index=False)

    levels = []
    selected = []
    for setting in SETTINGS:
        for method in NAMES:
            data = mean[(mean.setting == setting) & (mean.method == method)]
            levels.append(
                {
                    "setting": setting,
                    "method": NAMES[method],
                    **{
                        f"type_i<={level:.2f}": (
                            data[data.type_i <= level + 1e-9].type_ii.min()
                            if (data.type_i <= level + 1e-9).any()
                            else np.nan
                        )
                        for level in LEVELS
                    },
                }
            )
            frontier = envelope(data)
            selected.append(
                {
                    "setting": setting,
                    "method": NAMES[method],
                    "envelope_points": len(frontier),
                    **{
                        column[2:]: " ".join(f"{value:g}" for value in sorted(frontier[column].unique()))
                        for column in KNOBS[method]
                    },
                }
            )
    pd.DataFrame(levels).to_csv(RESULTS / "baselines_levels.csv", index=False)
    pd.DataFrame(selected).to_csv(RESULTS / "baselines_selected_hparams.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))
    for axis, setting in zip(axes, SETTINGS):
        for method in list(NAMES)[::-1]:
            data = mean[(mean.setting == setting) & (mean.method == method)].sort_values("type_i")
            axis.plot(
                data.type_i,
                data.type_ii.cummin(),
                drawstyle="steps-post",
                color=COLOR[method],
                linewidth=3.4 if method == "edit" else 2.2,
                linestyle=STYLE[method],
                label=NAMES[method],
            )
        model, dataset = setting.split(" / ")
        axis.set_title(f"{dataset} ({model})")
        axis.set_xlabel("Type-I error")
        edit = mean[(mean.setting == setting) & (mean.method == "edit")]
        axis.set_xlim(0, min(1.0, edit.type_i.max() + 0.03))
    axes[0].set_ylabel("Type-II error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.07),
        handlelength=2.6,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "baselines_frontier.png", dpi=170, bbox_inches="tight")
    fig.savefig(RESULTS / "baselines_frontier.pdf", bbox_inches="tight")
    print(f"[done] wrote figures and tables to {RESULTS}")


if __name__ == "__main__":
    main()
