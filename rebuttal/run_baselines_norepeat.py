"""Run the literature-baseline frontier on the four 2,500-item no-repeat streams.

Every method receives the same cached MiniLM features.  Seed 7 is the exact
recorded stream; seeds 17/27/37/47 are independent permutations of the same
pool, so every example appears exactly once in every evaluation stream.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from baselines import CLASSES
from methods.ours import OursSelection
from methods.topk import TopKSelection
from paper_analysis.common import compute_metrics, stream_from_existing_run
from run_baselines import SCORE_FROM_ROUND, SEEDS, grid
from engine import Features, FixedThreshold
from utils import load_yaml

RESULTS = HERE / "results" / "norepeat"
FEATURE_ROOT = HERE / "results" / "norepeat_gamma" / "minilm"

SETTINGS = {
    "Llama-3.3-70B / MATH": "or-llama-3.3-70b-instruct_math500-full_n2500_norepeat",
    "Llama-3.3-70B / PopQA": "or-llama-3.3-70b-instruct_popqa-test_n2500_seed42_norepeat",
    "Qwen3-32B / MATH": "or-qwen3-32b_math500-full_n2500_norepeat",
    "Qwen3-32B / PopQA": "or-qwen3-32b_popqa-test_n2500_seed42_norepeat",
}


class NoRepeatSetting:
    def __init__(self, name: str):
        self.name = name
        self.key = SETTINGS[name]
        self.cfg = load_yaml(REPO / "configs" / self.key / "ours.yaml")
        output_dir = REPO / "outputs" / self.key
        generation_path, = output_dir.glob("gen_*.csv")
        reference_path, = output_dir.glob("ours_*MiniLM*.csv")

        self.pool = (
            pd.read_csv(generation_path)
            .drop_duplicates("example_id", keep="last")
            .reset_index(drop=True)
        )
        self.ids = self.pool["example_id"].astype(int).to_numpy()
        self.A = self.pool["A"].astype(int).to_numpy()
        self.costs = np.ones(len(self.pool), dtype=float)
        if len(self.pool) != 2500 or len(np.unique(self.ids)) != 2500:
            raise ValueError(f"{name}: expected 2,500 unique examples")
        if self.cfg["policy"].get("cost_variant", "constant") != "constant":
            raise ValueError(f"{name}: expected constant review costs")

        reference = pd.read_csv(reference_path).sort_values(["t", "batch_pos"])
        self.paper_stream = stream_from_existing_run(reference_path, self.pool)
        if self.paper_stream.shape != (250, 10):
            raise ValueError(f"{name}: unexpected recorded stream shape {self.paper_stream.shape}")
        if len(np.unique(self.paper_stream)) != len(self.pool):
            raise ValueError(f"{name}: recorded stream is not no-repeat")

        feature_dir = FEATURE_ROOT / self.key
        meta = json.loads((feature_dir / "features.json").read_text())
        if int(meta["completed_rows"]) != len(reference):
            raise ValueError(f"{name}: incomplete MiniLM feature cache")
        stream_features = np.load(feature_dir / "features.npy", mmap_mode="r")
        reference_ids = reference["example_id"].astype(int).to_numpy()
        if len(np.unique(reference_ids)) != len(reference_ids):
            raise ValueError(f"{name}: feature source contains repeated IDs")
        feature_by_id = {int(example_id): stream_features[i] for i, example_id in enumerate(reference_ids)}
        self.features = np.stack([feature_by_id[int(example_id)] for example_id in self.ids]).astype(np.float32)
        self.reference = reference

    def stream(self, seed: int) -> np.ndarray:
        if seed == 7:
            stream = self.paper_stream
        else:
            stream = np.random.default_rng(seed).permutation(len(self.pool)).reshape(250, 10)
        if len(np.unique(stream)) != len(self.pool):
            raise AssertionError(f"{self.name}: seed {seed} repeats examples")
        return stream

    def run(self, method: str, seed: int, **params):
        cfg = copy.deepcopy(self.cfg)
        cfg["seed"] = 7
        for key, value in params.items():
            cfg["policy"][key] = float(value)
        cls = {"edit": OursSelection, "topk": TopKSelection, "tau": FixedThreshold, **CLASSES}[method]
        policy = cls(cfg, score_model=Features(self.features), state=None)
        stream = self.stream(seed)
        selected = np.zeros(stream.shape, dtype=int)
        for t, idx in enumerate(stream):
            batch = pd.DataFrame(
                {
                    "t": t,
                    "batch_pos": np.arange(len(idx)),
                    "pool_pos": idx,
                    "A": self.A[idx],
                    "cost": self.costs[idx],
                }
            )
            output, _ = policy.process_batch(batch, t=t)
            selected[t] = output["selected"].to_numpy(dtype=int)
        return stream, selected

    def metrics(self, stream: np.ndarray, selected: np.ndarray):
        rounds, batch_size = stream.shape
        frame = pd.DataFrame(
            {
                "t": np.repeat(np.arange(rounds), batch_size),
                "A": self.A[stream].ravel(),
                "cost": self.costs[stream].ravel(),
                "selected": selected.ravel(),
            }
        )
        return compute_metrics(frame[frame.t >= SCORE_FROM_ROUND])

    def validate_reference_replay(self):
        policy = self.cfg["policy"]
        stream, selected = self.run(
            "edit",
            seed=7,
            epsilon=float(policy["epsilon"]),
            alpha_step_size=float(policy["alpha_step_size"]),
        )
        expected = self.reference["selected"].to_numpy(dtype=int).reshape(selected.shape)
        if not np.array_equal(selected, expected):
            agreement = float(np.mean(selected == expected))
            raise AssertionError(f"{self.name}: reference replay agreement={agreement:.6f}")
        if not np.array_equal(self.ids[stream].ravel(), self.reference["example_id"].astype(int)):
            raise AssertionError(f"{self.name}: recorded stream IDs do not match")


_cache: dict[str, NoRepeatSetting] = {}


def work(job):
    name, method, parameters, seed = job
    if name not in _cache:
        _cache[name] = NoRepeatSetting(name)
    setting = _cache[name]
    stream, selected = setting.run(method, seed=seed, **parameters)
    result = setting.metrics(stream, selected)
    return {
        "setting": name,
        "method": method,
        "seed": seed,
        **{f"p_{key}": value for key, value in parameters.items()},
        "type_i": result["type_i"],
        "type_ii": result["type_ii"],
        "cost": result["mean_review_cost"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for name in SETTINGS:
        setting = NoRepeatSetting(name)
        setting.validate_reference_replay()
        print(f"[check] exact replay: {name}", flush=True)

    jobs = [
        (name, method, parameters, seed)
        for name in SETTINGS
        for method, parameters in grid()
        for seed in SEEDS
    ]
    print(f"[run] {len(jobs)} no-repeat baseline runs", flush=True)
    rows = []
    started = time.time()
    with Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            rows.append(row)
            if index % 250 == 0 or index == len(jobs):
                elapsed = time.time() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"[run] {index}/{len(jobs)} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "baselines_raw.csv", index=False)
    print(f"[done] wrote {RESULTS / 'baselines_raw.csv'}", flush=True)


if __name__ == "__main__":
    main()
