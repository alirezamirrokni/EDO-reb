"""Run one LLM-Select prompt arm on an already stored main-experiment stream.

This deliberately avoids reloading Hugging Face datasets: the historical
LLM-Select CSV already contains the exact questions, generations, labels,
costs, and batch order needed for a controlled prompt-only ablation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from methods.llm_select import LLMSelect
from run_experiment import RUN_COLUMNS, print_run_summary
from utils import load_json, load_yaml, project_paths, read_csv_or_empty, save_json_atomic, write_csv_atomic


def historical_run(output_dir: Path) -> Path:
    candidates = [
        path
        for path in output_dir.glob("llm_select_*.csv")
        if "_prompt" not in path.name
        and "_or-llama-3.3-70b-instruct_" not in path.name
        and ".failed" not in path.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one historical LLM-Select run in {output_dir}, found {candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prompt_variant",
        required=True,
        choices=["legacy_full_context", "reasoning_allowed", "independent_verify", "risk_rank_full_budget"],
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    cfg["policy"].update(
        prompt_variant=args.prompt_variant,
        max_prompt_chars=24000,
        fail_on_selector_error=True,
    )
    cfg["selector_llm"].update(
        provider="openrouter",
        model_name="meta-llama/llama-3.3-70b-instruct",
        display_name="or-llama-3.3-70b-instruct",
        min_seconds_between_calls=0.0,
        provider_routing={"only": ["CoreWeave"], "allow_fallbacks": False},
    )
    paths = project_paths(cfg)
    cfg["run_name"] = paths["run_name"]
    source_path = historical_run(paths["output_dir"])
    source = pd.read_csv(source_path).sort_values(["t", "batch_pos"])
    target = read_csv_or_empty(paths["run_csv"], RUN_COLUMNS)
    done = set(target["t"].astype(int)) if len(target) else set()
    state = load_json(paths["state_json"], default={})
    method = LLMSelect(cfg, state=state.get("method_state", {}))

    print(f"[replay] source={source_path}")
    print(f"[replay] target={paths['run_csv']}")
    print(f"[replay] completed={len(done)}/{source['t'].nunique()}")
    rows = target.to_dict("records") if len(target) else []
    rounds = sorted(int(value) for value in source["t"].astype(int).unique())
    for t in tqdm(rounds, desc=cfg["run_name"], dynamic_ncols=True):
        if t in done:
            continue
        batch = source[source["t"].astype(int) == t].copy()
        batch["config"] = cfg["run_name"]
        batch["selector_llm_provider"] = "openrouter"
        batch["selector_llm"] = "meta-llama/llama-3.3-70b-instruct"
        output, method_state = method.process_batch(batch, t=t)
        output = output[RUN_COLUMNS]
        rows.extend(output.to_dict("records"))
        current = pd.DataFrame(rows, columns=RUN_COLUMNS).sort_values(["t", "batch_pos"])
        write_csv_atomic(current, paths["run_csv"])
        save_json_atomic(
            {"config": cfg["run_name"], "next_batch": t + 1, "method_state": method_state},
            paths["state_json"],
        )
        done.add(t)
        print(f"[batch {t:03d}] selected={int(output['selected'].sum())}", flush=True)

    final = pd.DataFrame(rows, columns=RUN_COLUMNS).sort_values(["t", "batch_pos"])
    if final["t"].nunique() != source["t"].nunique() or len(final) != len(source):
        raise RuntimeError("Prompt replay did not produce a complete stream")
    print_run_summary(final)
    print(f"[done] wrote {paths['run_csv']}")


if __name__ == "__main__":
    main()
