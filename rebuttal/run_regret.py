"""Regret against label-aware oracles (rounds >= 20, 5 streams).

For each method and Type-I level, the setting with the lowest mean Type-II among settings with mean
Type-I <= level is re-run and compared with two oracles that see the labels:
  same-spend oracle: reviews as many items as the method in every round, always true edits first;
  budget oracle:     reviews up to the per-round budget (5), always true edits first.
Regret = edits the oracle finds - edits the method finds, per round.
"""
import os

for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

import argparse
import ast
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import SETTINGS, Setting, results_dir
from run_baselines import SCORE_FROM_ROUND, SEEDS

LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BUDGET = 5
_cache = {}


def work(job):
    model, name, method, kw, seed, level = job
    if (model, name) not in _cache:
        _cache[(model, name)] = Setting(name, model)
    s = _cache[(model, name)]
    stream, sel = s.run(method, seed=seed, **kw)
    A = s.A[stream][SCORE_FROM_ROUND:]
    sel = sel[SCORE_FROM_ROUND:]
    edits, n_sel = A.sum(axis=1), sel.sum(axis=1)
    found = (sel * A).sum()
    T = len(A)
    return dict(model=model, setting=name, method=method, level=level, params=str(kw), seed=seed,
                edits_per_round=edits.sum() / T, found_per_round=found / T,
                oracle_same_spend_per_round=np.minimum(edits, n_sel).sum() / T,
                oracle_budget_per_round=np.minimum(edits, BUDGET).sum() / T, reviewed_per_round=n_sel.sum() / T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = []
    for model in ["llama", "qwen"]:
        mean = pd.read_csv(results_dir(model) / "baselines_mean.csv")
        pcols = [c for c in mean.columns if c.startswith("p_")]
        for s in SETTINGS:
            for method in mean.method.unique():
                d = mean[(mean.setting == s) & (mean.method == method)]
                for lvl in LEVELS:
                    f = d[d.type_i <= lvl + 1e-9]
                    if f.empty:
                        continue
                    b = f.loc[f.type_ii.idxmin()]
                    kw = {c[2:]: b[c] for c in pcols if b[c] != -1}
                    kw = {k: int(v) if k in ("warmup", "randomized") else float(v) for k, v in kw.items()}
                    jobs += [(model, s, method, kw, sd, lvl) for sd in SEEDS]
    print(len(jobs), "runs", flush=True)
    with Pool(args.workers) as p:
        out = pd.DataFrame(p.map(work, jobs, chunksize=4))
    out["regret_same_spend"] = out.oracle_same_spend_per_round - out.found_per_round
    out["regret_budget"] = out.oracle_budget_per_round - out.found_per_round
    for model in ["llama", "qwen"]:
        out[out.model == model].drop(columns="model").to_csv(results_dir(model) / "regret_raw.csv", index=False)
