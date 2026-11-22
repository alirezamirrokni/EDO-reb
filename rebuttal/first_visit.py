import argparse

import numpy as np
import pandas as pd

from engine import PREFIXES, REPO, results_dir

SETS = {"llama": {"MATH-500": ("math500-test", 0.505), "MMLU-Pro": ("mmlupro-test_n500_seed42", 0.485),
        "PopQA": ("popqa-test_n500_seed42", 0.175), "GPQA": ("gpqa-train_main_n448", 0.505),
        "MATH-500 (review length)": ("math500-test_review_length", 0.535),
        "GPQA (review length)": ("gpqa-train_main_n448_review_length", 0.485)},
        "qwen": {"MATH-500": ("math500-test", 0.615), "MMLU-Pro": ("mmlupro-test_n500_seed42", 0.425),
                 "PopQA": ("popqa-test_n500_seed42", 0.0), "GPQA": ("gpqa-train_main_n448", 0.375)}}
COLS = ["t", "batch_pos", "example_id", "A", "cost", "selected"]


def tag(eps):
    return "eps" + f"{eps:g}".replace(".", "p") + "_"


def runs(prefix, d, eps):
    out = REPO / f"outputs/{prefix}{d}"
    f = {"EDIT-REP (MiniLM)": next(out.glob(f"ours_*MiniLM*{tag(eps)}*.csv"))}
    for q in ["0.6B", "1.7B", "4B"]:
        g = sorted(p for p in out.glob(f"ours_*Qwen-Qwen3-{q}_{tag(eps)}*.csv") if not p.name.endswith("_state.json"))
        for p in g:
            theta = p.stem.split("_theta")[-1].replace("p", ".")
            f[f"EDIT-REP (Qwen3-{q})" + (f" θ={theta}" if len(g) > 1 else "")] = p
    f["EDIT-LLM"] = next(out.glob("ours_llm_*.csv"))
    f["EDIT-RAND"] = next(out.glob("ours_random_*.csv"))
    f["LLM-Select"] = next(out.glob("llm_select_*.csv"))
    return f


def pooled(df):
    s, a = df.selected.to_numpy(float), df.A.to_numpy(float)
    return {"N": len(df), "Selected": int(s.sum()),
            "Type-I": (s * (1 - a)).sum() / max(s.sum(), 1), "Type-II": ((1 - s) * a).sum() / max((1 - s).sum(), 1),
            "Edit recall": (s * a).sum() / max(a.sum(), 1), "Edit precision": (s * a).sum() / max(s.sum(), 1),
            "Missed edits": int(((1 - s) * a).sum()), "Cost/round": (s * df.cost.to_numpy(float)).sum() / df.t.nunique()}


ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["llama", "qwen"], default="llama")
model = ap.parse_args().model
rows = []
for name, (d, eps) in SETS[model].items():
    for method, path in runs(PREFIXES[model], d, eps).items():
        df = pd.read_csv(path, usecols=COLS).sort_values(["t", "batch_pos"]).reset_index(drop=True)
        first = (~df.example_id.duplicated()).to_numpy()
        for subset, mask in [("first", first), ("all", np.ones(len(df), bool)), ("repeat", ~first)]:
            rows.append({"Dataset": name, "Method": method, "Subset": subset, **pooled(df[mask])})
out = results_dir(model)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "first_visit.csv", index=False)
