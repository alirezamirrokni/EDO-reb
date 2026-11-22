"""Main RAGTruth figure: EDIT-REP (paper protocol, cold-start probe), EDIT-LLM, EDIT-RAND, LLM-Select.

EDIT-REP is tuned exactly as in run_ragtruth.py (same gamma / learning-rate / warm-up grid, five train orderings,
lowest train Type-II with train Type-I <= epsilon) for Qwen3-4B; EDIT-RAND uses Uniform(0, 1) scores with the
same controller and has gamma / warm-up tuned the same way as EDIT-LLM. epsilon is the value in selected.json.
"""
import itertools
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from run_ragtruth import (BUDGET, FIGURES, GAMMAS, LRS, RESULTS, SEEDS, WARMS, cumulative_rows,  # noqa: E402
                          simulate_rep, simulate_static)

ENCODERS = {"Qwen3-4B": "features_qwen4b_2048.npy"}

EPS = float(json.loads((RESULTS / "selected.json").read_text())["epsilon"])
G = {}


def orderings(n, training):
    out = []
    for seed in SEEDS:
        o = np.random.default_rng(seed).permutation(n)
        out.append(o[: len(o) - len(o) % 10] if training else o)
    return out


def rand_scores(n, seed):
    return np.random.default_rng(seed + 271828).uniform(0.0, 1.0, n)


def rep_cfg(args):
    enc, gamma, lr, warm = args
    m = np.mean([simulate_rep(y, X, gamma, lr, EPS, warm) for y, X in G[("train", enc)]], axis=0)
    return dict(method="rep", enc=enc, gamma=gamma, lr=lr, warm=warm, type_i=m[0], type_ii=m[1], cost=m[2])


def rand_cfg(args):
    gamma, warm = args
    m = np.mean([simulate_static(y, s, gamma, EPS, warm) for y, s in G["rand_train"]], axis=0)
    return dict(method="rand", gamma=gamma, warm=warm, type_i=m[0], type_ii=m[1], cost=m[2])


def pick(df):
    f = df[df.type_i <= EPS + 1e-12]
    return (f if len(f) else df.nsmallest(1, "type_i")).sort_values(["type_ii", "cost"]).iloc[0]


def main():
    train = pd.read_csv(RESULTS / "data_train.csv", dtype={"example_id": str})
    test = pd.read_csv(RESULTS / "data_test.csv", dtype={"example_id": str})
    ytr, yte = train.A.to_numpy(int), test.A.to_numpy(int)
    tr_o, te_o = orderings(len(train), True), orderings(len(test), False)
    for enc, f in ENCODERS.items():
        X = np.load(RESULTS / f).astype(np.float32)
        G[("train", enc)] = [(ytr[o], X[: len(train)][o]) for o in tr_o]
        G[("test", enc)] = [(yte[o], X[len(train):][o]) for o in te_o]
    G["rand_train"] = [(ytr[o], rand_scores(len(o), s)) for s, o in zip(SEEDS, tr_o)]

    jobs = [(e, g, lr, w) for e in ENCODERS for g, lr, w in itertools.product(GAMMAS, LRS, WARMS)]
    with Pool(8) as p:
        rep = pd.DataFrame(p.map(rep_cfg, jobs, chunksize=4))
        rnd = pd.DataFrame(p.map(rand_cfg, list(itertools.product(GAMMAS, WARMS))))
    rep.to_csv(RESULTS / "main_rep_tuning.csv", index=False)
    rnd.to_csv(RESULTS / "main_rand_tuning.csv", index=False)

    base = pd.read_csv(RESULTS / "running_metrics.csv")
    curves = [base[base.method.isin(["EDIT-LLM", "LLM-Select"])]]
    by_seed = [pd.read_csv(RESULTS / "test_summary_by_seed.csv").query("method in ['EDIT-LLM', 'LLM-Select']")]
    chosen = {}
    for enc in ENCODERS:
        b = pick(rep[rep.enc == enc])
        chosen[enc] = dict(gamma=b.gamma, lr=b.lr, warm=int(b.warm))
        for seed, (y, X) in zip(SEEDS, G[("test", enc)]):
            vals = np.array([r[:3] for r in simulate_rep(y, X, b.gamma, b.lr, EPS, int(b.warm), True)])
            label = f"EDIT-REP ({enc})"
            curves.append(cumulative_rows(label, seed, vals))
            by_seed.append(pd.DataFrame([dict(method=label, seed=seed, type_i=vals[:, 0].mean(),
                                              type_ii=vals[:, 1].mean(), cost=vals[:, 2].mean())]))
    b = pick(rnd)
    chosen["EDIT-RAND"] = dict(gamma=b.gamma, warm=int(b.warm))
    for seed, o in zip(SEEDS, te_o):
        vals = np.array([r[:3] for r in simulate_static(yte[o], rand_scores(len(o), seed + 1), b.gamma, EPS, int(b.warm), True)])
        curves.append(cumulative_rows("EDIT-RAND", seed, vals))
        by_seed.append(pd.DataFrame([dict(method="EDIT-RAND", seed=seed, type_i=vals[:, 0].mean(),
                                          type_ii=vals[:, 1].mean(), cost=vals[:, 2].mean())]))
    curves = pd.concat(curves, ignore_index=True)
    by_seed = pd.concat(by_seed, ignore_index=True)
    curves.to_csv(RESULTS / "main_running_metrics.csv", index=False)
    by_seed.to_csv(RESULTS / "main_summary_by_seed.csv", index=False)
    agg = by_seed.groupby("method").agg(type_i=("type_i", "mean"), type_ii=("type_ii", "mean"), cost=("cost", "mean"),
                                        type_i_sd=("type_i", "std"), type_ii_sd=("type_ii", "std")).reset_index()
    agg.to_csv(RESULTS / "main_summary.csv", index=False)
    (RESULTS / "main_chosen.json").write_text(json.dumps(chosen, indent=2, default=float))
    print(json.dumps(chosen, default=float))
    print(agg.round(3).to_string(index=False))


def plot(rep_label, extra_label=None, output_stem="ragtruth_typeii_typei_cost_main"):
    curves = pd.read_csv(RESULTS / "main_running_metrics.csv")
    sns.set_theme(context="paper", style="darkgrid")
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "axes.titleweight": "bold",
                         "axes.labelweight": "bold", "font.weight": "bold", "pdf.fonttype": 42})
    order = [(rep_label, "EDIT-REP", "#ff7f0e"), ("EDIT-LLM", "EDIT-LLM", "#1f77b4"),
             ("EDIT-RAND", "EDIT-RAND", "#2ca02c"), ("LLM-Select", "LLM-Select", "#d62728")]
    if extra_label:
        order.insert(1, (extra_label, extra_label.replace("Qwen3-4B, ", ""), "#9467bd"))
        order[0] = (rep_label, "EDIT-REP (last token)", "#ff7f0e")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (metric, title) in zip(axes, [("type_ii", "Type-II error"), ("type_i", "Type-I error"), ("cost", "Review cost")]):
        for key, label, color in order:
            g = curves[curves.method == key].groupby("round")[metric]
            mu, ci = g.mean(), 1.96 * g.std().fillna(0) / math.sqrt(len(SEEDS))
            ax.plot(mu.index, mu, label=label, color=color, linewidth=2.8)
            ax.fill_between(mu.index, mu - ci, mu + ci, color=color, alpha=0.15)
        if metric == "type_i":
            ax.axhline(EPS, color="#555555", linestyle="--", linewidth=1.4)
        if metric == "cost":
            ax.axhline(BUDGET, color="#555555", linestyle="--", linewidth=1.4)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Round")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{output_stem}.{ext}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(sys.argv[2] if len(sys.argv) > 2 else "EDIT-REP (Qwen3-4B)", sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None,
             sys.argv[4] if len(sys.argv) > 4 else "ragtruth_typeii_typei_cost_main")
    else:
        main()
