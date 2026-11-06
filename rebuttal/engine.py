from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from methods.costs import compute_cost, prepare_cost_context  # noqa: E402
from methods.ours import OursSelection  # noqa: E402
from methods.topk import TopKSelection  # noqa: E402
from paper_analysis.common import compute_metrics, stream_from_existing_run  # noqa: E402
from utils import load_yaml, sigmoid  # noqa: E402

FEAT_DIR = HERE / "features"
RESULTS = HERE / "results"
PREFIX = "llama-3.3-70b-versatile_"
SETTINGS = {
    "MATH-500": "math500-test",
    "MMLU-Pro": "mmlupro-test_n500_seed42",
    "PopQA": "popqa-test_n500_seed42",
    "GPQA": "gpqa-train_main_n448",
}


class Features:
    def __init__(self, features):
        self.features = features

    def encode_rows(self, rows):
        return self.features[np.fromiter((int(r["pool_pos"]) for r in rows), dtype=int)]


class FixedThreshold(OursSelection):
    """Same learned score and theta updates as EDIT-REP; review eta > tau, highest eta first, within budget."""

    def process_batch(self, batch_df, t):
        features = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(features.shape[1])
        costs = batch_df["cost"].to_numpy(dtype=float)
        A = batch_df["A"].to_numpy(dtype=float)
        budget = float(self.policy["budget_per_batch"])
        tau = float(self.policy["tau"])
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = sigmoid(features @ self.theta).astype(np.float32)
        selected = TopKSelection._budgeted_topk_selection(
            eta=np.where(eta > tau, eta, -np.inf).astype(float), costs=np.where(eta > tau, costs, 0.0), budget=budget, rng=rng
        )
        n_sel = float(selected.sum())
        if n_sel > 0:
            grad = ((selected * (eta - A))[:, None] * features).sum(axis=0) / n_sel
            self.theta = (self.theta - float(self.policy["theta_step_size"]) * grad).astype(np.float32)
        out = batch_df.copy()
        out["eta"] = eta
        out["selected"] = selected
        return out, {}


class Setting:
    """One unit-cost dataset: stored pool and labels, cached MiniLM features, and the paper's stream (seed 7)
    or a synthetic stream drawn with replacement from the pool (other seeds)."""

    def __init__(self, name):
        self.name = name
        self.dir = SETTINGS[name]
        self.cfg = load_yaml(REPO / f"configs/{PREFIX}{self.dir}/ours.yaml")
        out = REPO / f"outputs/{PREFIX}{self.dir}"
        gen = sorted(out.glob("gen_*.csv"))[0]
        self.pool = pd.read_csv(gen).drop_duplicates("example_id", keep="last").reset_index(drop=True)
        variant = self.cfg["policy"].get("cost_variant", "constant")
        ctx = prepare_cost_context(self.pool, variant)
        self.costs = np.array([compute_cost(r, variant, ctx) for r in self.pool.to_dict("records")], dtype=float)
        self.A = self.pool["A"].astype(int).to_numpy()
        self.paper_stream = stream_from_existing_run(sorted(out.glob("ours_*MiniLM*.csv"))[0], self.pool)
        self.features = self._features()

    def _features(self):
        FEAT_DIR.mkdir(exist_ok=True)
        path = FEAT_DIR / f"{PREFIX}{self.dir}.npz"
        ids = self.pool["example_id"].astype(int).to_numpy()
        if path.exists():
            z = np.load(path)
            if np.array_equal(z["example_id"], ids):
                return z["features"].astype(np.float32)
        from models.minilm import MiniLMScoreModel

        sc = dict(self.cfg["score_model"])
        sc["dataset_name"] = self.cfg["data"]["name"]
        feats = MiniLMScoreModel(sc).encode_rows(self.pool[["question", "model_answer"]].to_dict("records")).astype(np.float32)
        np.savez_compressed(path, example_id=ids, features=feats)
        return feats

    def stream(self, seed):
        if seed == 7:
            return self.paper_stream
        rng = np.random.default_rng(int(seed))
        return rng.choice(len(self.pool), size=self.paper_stream.shape, replace=True)

    def run(self, method, seed=7, **params):
        from baselines import CLASSES

        cfg = copy.deepcopy(self.cfg)
        cfg["seed"] = 7
        for k, v in params.items():
            cfg["policy"][k] = float(v)
        cls = {"edit": OursSelection, "topk": TopKSelection, "tau": FixedThreshold, **CLASSES}[method]
        m = cls(cfg, score_model=Features(self.features), state=None)
        stream = self.stream(seed)
        sel = np.zeros(stream.shape, dtype=int)
        for t in range(stream.shape[0]):
            idx = stream[t]
            batch = pd.DataFrame(
                {"t": t, "batch_pos": np.arange(len(idx)), "pool_pos": idx, "A": self.A[idx], "cost": self.costs[idx]}
            )
            out, _ = m.process_batch(batch, t=t)
            sel[t] = out["selected"].to_numpy(dtype=int)
        return stream, sel

    def metrics(self, stream, sel, t0=0):
        T, n = stream.shape
        df = pd.DataFrame(
            {"t": np.repeat(np.arange(T), n), "A": self.A[stream].ravel(), "cost": self.costs[stream].ravel(), "selected": sel.ravel()}
        )
        return compute_metrics(df[df.t >= t0])
