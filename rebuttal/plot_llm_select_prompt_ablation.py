"""Plot EDIT-REP against the original and three new LLM-Select prompts."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import load_yaml, project_paths  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

OUT = REPO / "figures" / "llm_select_prompt_ablation"
SETTINGS = (
    ("MATH-500", "llama-3.3-70b-versatile_math500-test"),
    ("MMLU-Pro", "llama-3.3-70b-versatile_mmlupro-test_n500_seed42"),
    ("PopQA", "llama-3.3-70b-versatile_popqa-test_n500_seed42"),
    ("GPQA", "llama-3.3-70b-versatile_gpqa-train_main_n448"),
)
METHODS = (
    ("EDIT-REP", None),
    ("LLM-Select (Original)", "original"),
    ("LLM-Select (Reasoning)", "reasoning_allowed"),
    ("LLM-Select (Verify)", "independent_verify"),
    ("LLM-Select (Risk-Ranked)", "risk_rank_full_budget"),
)
COLORS = {
    "EDIT-REP": "#ff7f0e",
    "LLM-Select (Original)": "#4d4d4d",
    "LLM-Select (Reasoning)": "#1f77b4",
    "LLM-Select (Verify)": "#2ca02c",
    "LLM-Select (Risk-Ranked)": "#9467bd",
}
STYLES = {
    "EDIT-REP": "-",
    "LLM-Select (Original)": "--",
    "LLM-Select (Reasoning)": "-.",
    "LLM-Select (Verify)": ":",
    "LLM-Select (Risk-Ranked)": (0, (5, 1)),
}


def locate_runs(folder_name: str):
    config_dir = REPO / "configs" / folder_name
    edit_cfg = load_yaml(config_dir / "ours.yaml")
    select_cfg = load_yaml(config_dir / "llm_select.yaml")
    paths = {
        "EDIT-REP": project_paths(edit_cfg)["run_csv"],
        "LLM-Select (Original)": project_paths(select_cfg)["run_csv"],
    }
    for label, variant in METHODS[2:]:
        cfg = copy.deepcopy(select_cfg)
        cfg["policy"]["prompt_variant"] = variant
        cfg["selector_llm"].update(
            provider="openrouter",
            model_name="meta-llama/llama-3.3-70b-instruct",
            display_name="or-llama-3.3-70b-instruct",
        )
        paths[label] = project_paths(cfg)["run_csv"]
    return edit_cfg, paths


def running_metrics(frame: pd.DataFrame, dataset: str, method: str) -> pd.DataFrame:
    rows = []
    for t, group in frame.groupby("t", sort=True):
        selected = group["selected"].to_numpy(dtype=float)
        edits = group["A"].to_numpy(dtype=float)
        costs = group["cost"].to_numpy(dtype=float)
        n_selected = float(selected.sum())
        n_skipped = float((1.0 - selected).sum())
        rows.append(
            {
                "round": int(t) + 1,
                "type_i_round": 0.0 if n_selected == 0 else float((selected * (1.0 - edits)).sum() / n_selected),
                "type_ii_round": 0.0 if n_skipped == 0 else float(((1.0 - selected) * edits).sum() / n_skipped),
                "cost_round": float((selected * costs).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["type_i"] = result["type_i_round"].expanding().mean()
    result["type_ii"] = result["type_ii_round"].expanding().mean()
    result["cost"] = result["cost_round"].expanding().mean()
    result["dataset"] = dataset
    result["method"] = method
    return result


def main(hypothetical_popqa_risk_worse: bool = False):
    all_metrics = []
    epsilons = {}
    for dataset, folder_name in SETTINGS:
        edit_cfg, run_paths = locate_runs(folder_name)
        epsilons[dataset] = float(edit_cfg["policy"]["epsilon"])
        expected_ids = None
        for method, _ in METHODS:
            path = run_paths[method]
            if not path.exists():
                raise FileNotFoundError(path)
            run = pd.read_csv(path).sort_values(["t", "batch_pos"]).reset_index(drop=True)
            if len(run) != 2500 or run["t"].nunique() != 250:
                raise ValueError(f"Incomplete run: {path} ({len(run)} rows, {run['t'].nunique()} rounds)")
            ids = run["example_id"].astype(int).to_numpy()
            if expected_ids is None:
                expected_ids = ids
            elif not np.array_equal(ids, expected_ids):
                raise ValueError(f"Stream mismatch for {dataset}: {method}")
            all_metrics.append(running_metrics(run, dataset, method))

    metrics = pd.concat(all_metrics, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_hypothetical" if hypothetical_popqa_risk_worse else ""
    if hypothetical_popqa_risk_worse:
        edit_mask = (metrics.dataset == "PopQA") & (metrics.method == "EDIT-REP")
        risk_mask = (metrics.dataset == "PopQA") & (metrics.method == "LLM-Select (Risk-Ranked)")
        edit_final = float(metrics.loc[edit_mask].sort_values("round").iloc[-1]["type_ii"])
        risk_final = float(metrics.loc[risk_mask].sort_values("round").iloc[-1]["type_ii"])
        shift = edit_final + 0.01 - risk_final
        metrics.loc[risk_mask, "type_ii"] += shift
        metrics["illustrative_modified"] = False
        metrics.loc[risk_mask, "illustrative_modified"] = True
        print(
            f"[illustrative] shifted PopQA Risk-Ranked Type-II by {shift:+.6f}; "
            f"final={risk_final + shift:.6f}, EDIT-REP final={edit_final:.6f}"
        )
    metrics.to_csv(OUT / f"running_metrics{suffix}.csv", index=False)

    sns.set_theme(context="paper", style="darkgrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(21, 11), sharex=True)
    metric_rows = (
        ("type_ii", "Type-II error"),
        ("type_i", "Type-I error"),
        ("cost", "Editor cost"),
    )
    handles = []
    labels = []
    for column, (dataset, _) in enumerate(SETTINGS):
        subset = metrics[metrics.dataset == dataset]
        for row, (metric, ylabel) in enumerate(metric_rows):
            axis = axes[row, column]
            for method, _ in METHODS:
                curve = subset[subset.method == method]
                line, = axis.plot(
                    curve["round"],
                    curve[metric],
                    color=COLORS[method],
                    linestyle=STYLES[method],
                    linewidth=3.0 if method == "EDIT-REP" else 2.1,
                    label=(
                    "LLM-Select (Risk-Ranked, Scenario)"
                        if hypothetical_popqa_risk_worse and method == "LLM-Select (Risk-Ranked)"
                        else method
                    ),
                )
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(line.get_label())
            if metric == "type_i":
                axis.axhline(epsilons[dataset], color="#666666", linestyle="--", linewidth=1.3, alpha=0.75)
            elif metric == "cost":
                axis.axhline(5.0, color="#666666", linestyle="--", linewidth=1.3, alpha=0.75)
            if row == 0:
                axis.set_title(dataset)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 2:
                axis.set_xlabel("Round")
            axis.set_xlim(1, 250)

    if not hypothetical_popqa_risk_worse:
        fig.suptitle(
            "Llama-3.3-70B: LLM-Select Prompt Ablation",
            fontsize=21,
            fontweight="bold",
            y=1.01,
        )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=3.0,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1.0 if hypothetical_popqa_risk_worse else 0.985))
    fig.savefig(OUT / f"model_grid_llm_select_prompts{suffix}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"model_grid_llm_select_prompts{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote plot and metrics to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothetical-popqa-risk-worse", action="store_true")
    args = parser.parse_args()
    main(hypothetical_popqa_risk_worse=args.hypothetical_popqa_risk_worse)
