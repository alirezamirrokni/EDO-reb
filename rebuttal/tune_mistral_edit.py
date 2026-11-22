"""EDIT-REP (MiniLM) tuning on the recorded Mistral-Small-3.2 streams (seed 7, all 250 rounds) for MATH-500 and GPQA.

epsilon stays at the value shared with EDIT-LLM / EDIT-RAND; gamma, probe step size and warm start are searched.
Selection: lowest Type-II with Type-I <= epsilon. Writes results/mistral/edit_tuning.csv.
"""
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import pandas as pd  # noqa: E402

from engine import Setting, results_dir  # noqa: E402

GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
LRS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
WARMS = [0, 5, 10]
SETS = ["MATH-500", "GPQA"]
_cache = {}


def work(job):
    name, g, lr, w = job
    if name not in _cache:
        _cache[name] = Setting(name, "mistral")
    s = _cache[name]
    stream, sel = s.run("edit", seed=7, alpha_step_size=g, theta_step_size=lr, warm_start_batches=w)
    m = s.metrics(stream, sel)
    return dict(setting=name, epsilon=s.cfg["policy"]["epsilon"], gamma=g, theta_step_size=lr, warm=w,
                type_i=m["type_i"], type_ii=m["type_ii"], cost=m["mean_review_cost"])


if __name__ == "__main__":
    jobs = [(n, g, lr, w) for n in SETS for g, lr, w in itertools.product(GAMMAS, LRS, WARMS)]
    with Pool(6) as p:
        df = pd.DataFrame(p.map(work, jobs, chunksize=4))
    out = results_dir("mistral")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "edit_tuning.csv", index=False)
    for n in SETS:
        d = df[(df.setting == n) & (df.type_i <= df.epsilon + 1e-12)].sort_values(["type_ii", "cost"])
        print(n, "\n", d.head(8).round(4).to_string(index=False))
