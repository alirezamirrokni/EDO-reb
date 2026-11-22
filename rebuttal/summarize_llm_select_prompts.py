"""Summarize the four Llama-3.3-70B LLM-Select prompt arms."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paper_analysis.common import compute_metrics

RESULTS = Path(__file__).resolve().parent / "results"
DATASETS = {
    "MATH-500": "llama-3.3-70b-versatile_math500-test",
    "MMLU-Pro": "llama-3.3-70b-versatile_mmlupro-test_n500_seed42",
    "PopQA": "llama-3.3-70b-versatile_popqa-test_n500_seed42",
    "GPQA": "llama-3.3-70b-versatile_gpqa-train_main_n448",
}
VARIANTS = (
    "legacy_original",
    "legacy_full_context",
    "reasoning_allowed",
    "independent_verify",
    "risk_rank_full_budget",
)


def locate(folder: Path, variant: str) -> Path:
    if variant == "legacy_original":
        candidates = [
            p for p in folder.glob("llm_select_*.csv")
            if "_prompt" not in p.name and "_or-llama-3.3-70b-instruct_" not in p.name
        ]
    else:
        candidates = list(folder.glob(f"llm_select_*_prompt{variant}_seed7.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {variant} run in {folder}, found {len(candidates)}")
    return candidates[0]


def main():
    rows = []
    for dataset, folder_name in DATASETS.items():
        folder = REPO / "outputs" / folder_name
        for variant in VARIANTS:
            path = locate(folder, variant)
            run = pd.read_csv(path)
            metrics = compute_metrics(run)
            rows.append(
                {
                    "dataset": dataset,
                    "prompt_variant": variant,
                    "rows": len(run),
                    "rounds": run["t"].nunique(),
                    **metrics,
                    "source_csv": str(path),
                }
            )
    result = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "llm_select_prompt_ablation.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
