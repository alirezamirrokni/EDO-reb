"""Distribution-shift robustness on the MMLU-Pro and PopQA shift streams (Llama-3.3-70B, paper stream).

Target-driven online methods receive the Type-I target directly: EDIT-REP runs at
epsilon = target - (1 + gamma) / (gamma * T) (Theorem 2 slack) and Online Conformal Selection
at success target 1 - target (its dual step is chosen on phase 1). Every other method has its
knobs chosen on phase 1 only (lowest phase-1 Type-II among settings with phase-1 Type-I <= target,
else the lowest phase-1 Type-I) and is then run unchanged on the whole stream.
All metrics are scored from round 20 onward, as in the frontier experiment.
"""
import os

for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import RESULTS, SHIFT_SETTINGS, Setting
from run_baselines import SCORE_FROM_ROUND, grid

TARGETS = [round(x, 2) for x in np.arange(0.10, 0.501, 0.05)]
EDIT_GAMMA = 0.1
CALIBRATED = ["conformal", "zliobaite", "thompson", "topk", "tau"]
_cache = {}


def setting(name):
    if name not in _cache:
        _cache[name] = Setting(name)
    return _cache[name]


def calibrate(job):
    name, method, kw = job
    s = setting(name)
    e1 = int(s.phase_edges[1])
    st, sel = s.run(method, rounds=e1, **kw)
    m = s.metrics(st, sel, t0=SCORE_FROM_ROUND)
    return dict(setting=name, method=method, kw=kw, type_i=m["type_i"], type_ii=m["type_ii"])


def final(job):
    name, target, method, kw = job
    s = setting(name)
    st, sel = s.run(method, **kw)
    m = s.metrics(st, sel, t0=SCORE_FROM_ROUND)
    row = dict(setting=name, target=target, method=method, params=str(kw), type_i=m["type_i"], type_ii=m["type_ii"],
               cost=m["mean_review_cost"], missed=m["absolute_missed_edits"])
    for k, ph in enumerate(s.phase_names):
        pm = s.metrics(st, sel, t0=max(SCORE_FROM_ROUND, int(s.phase_edges[k])), t1=int(s.phase_edges[k + 1]))
        row[f"type_i[{ph}]"], row[f"type_ii[{ph}]"] = pm["type_i"], pm["type_ii"]
    return row


def pick(rows, target):
    feas = [r for r in rows if r["type_i"] <= target + 1e-12]
    return min(feas, key=lambda r: (r["type_ii"], r["type_i"])) if feas else min(rows, key=lambda r: r["type_i"])


if __name__ == "__main__":
    knobs = {m: [kw for mm, kw in grid() if mm == m] for m in CALIBRATED}
    cal_jobs = [(n, m, kw) for n in SHIFT_SETTINGS for m in CALIBRATED for kw in knobs[m]]
    cal_jobs += [(n, "gollapudi", dict(epsilon=t, dual_step=d, horizon=setting(n).paper_stream.shape[0]))
                 for n in SHIFT_SETTINGS for t in TARGETS for d in [0.063, 0.25, 1.0]]
    with Pool(6) as p:
        cal = p.map(calibrate, cal_jobs, chunksize=4)
    pd.DataFrame(cal).assign(kw=lambda d: d.kw.astype(str)).to_csv(RESULTS / "shift_calibration.csv", index=False)

    fin_jobs = []
    for n in SHIFT_SETTINGS:
        T = setting(n).paper_stream.shape[0]
        for t in TARGETS:
            eps = max(0.0, t - (1 + EDIT_GAMMA) / (EDIT_GAMMA * T))
            fin_jobs.append((n, t, "edit", dict(epsilon=round(eps, 4), alpha_step_size=EDIT_GAMMA)))
            g = [r for r in cal if r["setting"] == n and r["method"] == "gollapudi" and r["kw"]["epsilon"] == t]
            fin_jobs.append((n, t, "gollapudi", pick(g, t)["kw"]))
            for m in CALIBRATED:
                fin_jobs.append((n, t, m, pick([r for r in cal if r["setting"] == n and r["method"] == m], t)["kw"]))
    with Pool(6) as p:
        out = pd.DataFrame(p.map(final, fin_jobs))
    ph_cols = [c for c in out.columns if c.startswith("type_i[")]
    out["max_phase_type_i"] = out[ph_cols].max(axis=1)
    out["violates_overall"] = out.type_i > out.target + 1e-12
    out["violates_any_phase"] = out.max_phase_type_i > out.target + 1e-12
    out.to_csv(RESULTS / "shift_results.csv", index=False)
