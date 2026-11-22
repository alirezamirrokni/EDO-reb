"""Replay EDIT-REP in the offline engine with Qwen3 encoder features instead of MiniLM.

Used where run_experiment.py cannot run because the benchmark itself cannot be downloaded.
The engine reproduces run_experiment.py selections item for item (checked with --check).
"""
import argparse
import sys

import numpy as np
import pandas as pd

from engine import REPO, Setting
from utils import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["llama", "qwen"], default="qwen")
ap.add_argument("--setting", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--check", help="stored run_experiment.py CSV to compare against instead of writing a new run")
args = ap.parse_args()

s = Setting(args.setting, args.model)
cfg = load_yaml(REPO / args.config)
sc = dict(cfg["score_model"])
sc["dataset_name"] = cfg["data"]["name"]
size = sc["model_name"].split("Qwen3-")[1]


class PoolRows:
    """Encodes each round's rows on demand, in stream order, exactly as run_experiment.py does."""

    def __init__(self, encoder, pool):
        self.encoder, self.pool = encoder, pool.to_dict("records")

    def encode_rows(self, rows):
        return self.encoder.encode_rows([self.pool[int(r["pool_pos"])] for r in rows])


from models.qwen3_embedding import Qwen3ScoreModel  # noqa: E402

encoder = PoolRows(Qwen3ScoreModel(sc), s.pool)
p = cfg["policy"]
stream, sel = s.run("edit", seed=7, score_model=encoder, epsilon=p["epsilon"], alpha_step_size=p["alpha_step_size"], theta_step_size=p["theta_step_size"])
T, n = stream.shape
eid = s.pool["example_id"].to_numpy()[stream]

def tag(x):
    return f"{x:g}".replace(".", "p")

name = f"ours_{s.prefix}{s.dir}_budget{tag(p['budget_per_batch'])}_Qwen-Qwen3-{size}_eps{tag(p['epsilon'])}_alpha{tag(p['alpha_step_size'])}_theta{tag(p['theta_step_size'])}"
if args.check:
    ref = pd.read_csv(REPO / args.check).sort_values(["t", "batch_pos"])
    rs = ref.selected.to_numpy().reshape(sel.shape)
    print(f"{args.check}: stream_match={np.array_equal(ref.example_id.to_numpy().reshape(eid.shape), eid)} "
          f"selection_agreement={np.mean(rs == sel):.4f}")
    sys.exit(0)
out = pd.DataFrame({"config": name, "method": "ours", "t": np.repeat(np.arange(T), n), "batch_pos": np.tile(np.arange(n), T),
                    "example_id": eid.ravel(), "A": s.A[stream].ravel(), "cost": s.costs[stream].ravel(), "selected": sel.ravel(),
                    "budget": float(p["budget_per_batch"]), "epsilon": p["epsilon"],
                    "main_llm_provider": cfg["main_llm"]["provider"], "main_llm": cfg["main_llm"]["model_name"],
                    "score_model_provider": sc["provider"], "score_model": sc["model_name"],
                    "selector_llm_provider": "none", "selector_llm": "none", "source": "rebuttal/replay_encoder.py"})
dest = REPO / f"outputs/{s.prefix}{s.dir}/{name}.csv"
out.to_csv(dest, index=False)
print("wrote", dest.relative_to(REPO))
