"""Fill a config's main-LLM generation cache with concurrent API calls.

Uses run_experiment.py's own data loading, stream sampling, prompts, grading and cache format, so
run_experiment.py --no_generate afterwards reads exactly the cache it would have written itself.
"""
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_experiment as rx  # noqa: E402
from data_wrappers import build_data_wrapper  # noqa: E402
from models import build_main_llm  # noqa: E402
from utils import load_yaml, project_paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

load_dotenv(REPO / ".env")
cfg = load_yaml(args.config)
paths = project_paths(cfg)
wrapper = build_data_wrapper(cfg["data"])
rx.apply_dataset_llm_defaults(cfg, wrapper)
records = rx.flatten_batches(rx.sample_batches(wrapper.load_records(), cfg))
cache_path = paths["generation_cache"]
cache = rx.read_csv_or_empty(cache_path, rx.GEN_COLUMNS)
done = set(cache["example_id"].astype(int)) if len(cache) else set()
todo = list({int(r["example_id"]): r for r in records if int(r["example_id"]) not in done}.values())
print(f"[gen] {cache_path} cached={len(done)} to_generate={len(todo)}", flush=True)

local = threading.local()
lock = threading.Lock()
rows = cache.to_dict("records") if len(cache) else []


def llm():
    if not hasattr(local, "llm"):
        local.llm = build_main_llm(dict(cfg["main_llm"]))
    return local.llm


def generate(rec):
    return rec, llm().generate(wrapper.build_prompt(rec["question"]))


def grade(rec, answer):
    pred = wrapper.parse_prediction(answer)
    return {"example_id": rec["example_id"], "question": rec["question"], "gold_answer": rec["gold_answer"],
            "gold_final": rec["gold_final"], "model_answer": answer, "pred_answer": pred,
            "A": int(wrapper.failure_label(pred, rec["gold_final"]))}


def flush():
    rx.write_csv_atomic(pd.DataFrame(rows, columns=rx.GEN_COLUMNS).drop_duplicates("example_id", keep="last"), cache_path)


with ThreadPoolExecutor(args.workers) as pool:
    futures = [pool.submit(generate, r) for r in todo]
    for i, fut in enumerate(as_completed(futures), 1):
        row = grade(*fut.result())
        with lock:
            rows.append(row)
            if i % 25 == 0 or i == len(futures):
                flush()
                acc = 1 - pd.DataFrame(rows)["A"].mean()
                print(f"[gen] {i}/{len(futures)} accuracy so far {acc:.3f}", flush=True)
flush()
print("[gen] done", flush=True)
