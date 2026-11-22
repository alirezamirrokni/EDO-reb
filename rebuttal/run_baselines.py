import os

for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import SETTINGS, Setting, results_dir

SEEDS = [7, 17, 27, 37, 47]
SCORE_FROM_ROUND = 20
_cache = {}


def grid():
    g = []
    for e in np.round(np.arange(0.0, 0.801, 0.02), 2):
        for gam in [0.02, 0.05, 0.1]:
            g.append(("edit", dict(epsilon=float(e), alpha_step_size=gam)))
        for d in [0.063, 0.25, 1.0]:
            g.append(("gollapudi", dict(epsilon=float(e), dual_step=d)))
    for q in np.round(np.arange(0.02, 0.801, 0.02), 2):
        g.append(("conformal", dict(q=float(q), warmup=SCORE_FROM_ROUND)))
    for b in [1, 2, 3, 4, 5]:
        for r in [0, 1]:
            for s in [0.01, 0.05]:
                g.append(("zliobaite", dict(budget_per_batch=b, randomized=r, adj_step=s)))
        for v in [0.3, 1.0, 3.0]:
            g.append(("thompson", dict(budget_per_batch=b, v=v)))
        g.append(("topk", dict(budget_per_batch=float(b))))
    for t in np.round(np.arange(0.0, 0.981, 0.02), 2):
        g.append(("tau", dict(tau=float(t))))
    return g


def work(job):
    model, name, method, kw, seed = job
    if (model, name) not in _cache:
        _cache[(model, name)] = Setting(name, model)
    s = _cache[(model, name)]
    stream, sel = s.run(method, seed=seed, **kw)
    m = s.metrics(stream, sel, t0=SCORE_FROM_ROUND)
    return dict(setting=name, method=method, seed=seed, **{f"p_{k}": v for k, v in kw.items()},
                type_i=m["type_i"], type_ii=m["type_ii"], cost=m["mean_review_cost"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", choices=["llama", "qwen", "mistral"], default="llama")
    ap.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    args = ap.parse_args()
    for n in args.settings:
        Setting(n, args.model)
    jobs = [(args.model, n, m, kw, sd) for n in args.settings for (m, kw) in grid() for sd in SEEDS]
    print(f"{len(jobs)} runs", flush=True)
    rows, t0 = [], time.time()
    with Pool(args.workers) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 250 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"{i}/{len(jobs)}  elapsed {el / 60:.1f} min  eta {el / i * (len(jobs) - i) / 60:.1f} min", flush=True)
    out = results_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    path = out / "baselines_raw.csv"
    if path.exists() and set(args.settings) != set(SETTINGS):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.setting.isin(args.settings)], new], ignore_index=True)
    new.to_csv(path, index=False)
