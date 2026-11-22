"""Tune Qwen3-1.7B EDIT-REP for the no-repeat baseline frontier.

The literature baselines are reused from ``baselines_raw.csv``. This script
only reruns EDIT-REP with cached Qwen3-1.7B representations and the stronger
Type-II search protocol. Every point is averaged later over the same five
no-repeat streams as the competing methods.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import run_baselines_norepeat as base

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

base.FEATURE_ROOT = base.HERE / "results" / "norepeat_gamma" / "1.7B"

RESULTS = base.RESULTS
EDIT_RAW = RESULTS / "baselines_edit_qwen17_raw.csv"
COMBINED_RAW = RESULTS / "baselines_raw_qwen17.csv"

EPSILONS = np.round(np.arange(0.0, 0.801, 0.02), 2).tolist()
GAMMAS = [
    0.0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3,
    0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
]
THETA_STEP_SIZE = 4.0
WARM_START_BATCHES = 0

_cache: dict[str, base.NoRepeatSetting] = {}


def work(job):
    name, epsilon, gamma, seed = job
    if name not in _cache:
        _cache[name] = base.NoRepeatSetting(name)
    setting = _cache[name]
    stream, selected = setting.run(
        "edit",
        seed=seed,
        epsilon=epsilon,
        alpha_step_size=gamma,
        theta_step_size=THETA_STEP_SIZE,
        warm_start_batches=WARM_START_BATCHES,
    )
    metrics = setting.metrics(stream, selected)
    return {
        "setting": name,
        "method": "edit",
        "seed": seed,
        "p_epsilon": epsilon,
        "p_alpha_step_size": gamma,
        "p_theta_step_size": THETA_STEP_SIZE,
        "p_warm_start_batches": WARM_START_BATCHES,
        "type_i": metrics["type_i"],
        "type_ii": metrics["type_ii"],
        "cost": metrics["mean_review_cost"],
    }


def validate_qwen17_replay():
    """Match the stored seed-7 gamma=1 run before launching the sweep."""
    for name, run_key in base.SETTINGS.items():
        setting = base.NoRepeatSetting(name)
        epsilon = float(setting.cfg["policy"]["epsilon"])
        stream, selected = setting.run(
            "edit",
            seed=7,
            epsilon=epsilon,
            alpha_step_size=1.0,
            theta_step_size=1.0,
            warm_start_batches=0,
        )
        stored = pd.read_csv(base.FEATURE_ROOT / run_key / "gamma_1.csv").sort_values(
            ["t", "batch_pos"]
        )
        expected = stored["selected"].to_numpy(dtype=int).reshape(selected.shape)
        if not np.array_equal(selected, expected):
            raise AssertionError(f"{name}: Qwen3-1.7B replay does not match stored gamma=1 run")
        if not np.array_equal(setting.ids[stream].ravel(), stored["example_id"].astype(int)):
            raise AssertionError(f"{name}: Qwen3-1.7B replay stream does not match")
        print(f"[check] exact Qwen3-1.7B replay: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs for timing tests")
    args = parser.parse_args()

    validate_qwen17_replay()
    jobs = [
        (name, epsilon, gamma, seed)
        for name in base.SETTINGS
        for epsilon in EPSILONS
        for gamma in GAMMAS
        for seed in base.SEEDS
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"[run] {len(jobs)} Qwen3-1.7B EDIT frontier runs", flush=True)
    rows = []
    started = time.time()
    with Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(jobs):
                elapsed = time.time() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"[run] {index}/{len(jobs)} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )

    if args.limit is not None:
        print("[done] timing run; no results written", flush=True)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    edit = pd.DataFrame(rows)
    edit.to_csv(EDIT_RAW, index=False)
    old = pd.read_csv(RESULTS / "baselines_raw.csv")
    combined = pd.concat([old[old.method != "edit"], edit], ignore_index=True, sort=False)
    combined.to_csv(COMBINED_RAW, index=False)
    print(f"[done] wrote {EDIT_RAW}", flush=True)
    print(f"[done] wrote {COMBINED_RAW}", flush=True)


if __name__ == "__main__":
    main()
