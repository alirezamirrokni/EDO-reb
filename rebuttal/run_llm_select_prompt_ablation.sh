#!/usr/bin/env bash
set -euo pipefail

cd /home/mll/alireza/EDO-reb
log=rebuttal/results/llm_select_prompt_ablation.log
mkdir -p rebuttal/results
: > "$log"

# The tmux server can retain stale credentials from the environment in which it
# was first created. Force run_experiment.py/load_dotenv() to read this repo's
# current .env instead of inheriting an older provider key.
unset OPENROUTER_API_KEY GROQ_API_KEY

configs=(
  configs/llama-3.3-70b-versatile_math500-test/llm_select.yaml
  configs/llama-3.3-70b-versatile_mmlupro-test_n500_seed42/llm_select.yaml
  configs/llama-3.3-70b-versatile_popqa-test_n500_seed42/llm_select.yaml
  configs/llama-3.3-70b-versatile_gpqa-train_main_n448/llm_select.yaml
)
variants=(legacy_full_context reasoning_allowed independent_verify risk_rank_full_budget)

for variant in "${variants[@]}"; do
  for config in "${configs[@]}"; do
    echo "[ablation] variant=$variant config=$config" | tee -a "$log"
    /home/mll/alireza/or_runs/venv/bin/python -u rebuttal/run_llm_select_prompt_replay.py \
      --config "$config" \
      --prompt_variant "$variant" 2>&1 | tee -a "$log"
  done
done

/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/summarize_llm_select_prompts.py 2>&1 | tee -a "$log"
