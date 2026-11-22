#!/usr/bin/env bash
set -euo pipefail

cd /home/mll/alireza/EDO-reb
mkdir -p rebuttal/results

# Reload current credentials from .env; retain the proxy inherited by this
# dedicated tmux server.
unset OPENROUTER_API_KEY GROQ_API_KEY

configs=(
  configs/llama-3.3-70b-versatile_math500-test/llm_select.yaml
  configs/llama-3.3-70b-versatile_mmlupro-test_n500_seed42/llm_select.yaml
  configs/llama-3.3-70b-versatile_popqa-test_n500_seed42/llm_select.yaml
  configs/llama-3.3-70b-versatile_gpqa-train_main_n448/llm_select.yaml
)
names=(math500 mmlupro popqa gpqa)
variants=(legacy_full_context reasoning_allowed independent_verify risk_rank_full_budget)
pids=()

for index in "${!configs[@]}"; do
  config=${configs[$index]}
  name=${names[$index]}
  log="rebuttal/results/llm_select_prompt_ablation_${name}.log"
  (
    : > "$log"
    for variant in "${variants[@]}"; do
      echo "[ablation] dataset=$name variant=$variant" | tee -a "$log"
      /home/mll/alireza/or_runs/venv/bin/python -u rebuttal/run_llm_select_prompt_replay.py \
        --config "$config" \
        --prompt_variant "$variant" 2>&1 | tee -a "$log"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[failed] at least one dataset worker failed" >&2
  exit "$status"
fi

/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/summarize_llm_select_prompts.py \
  2>&1 | tee rebuttal/results/llm_select_prompt_ablation.log
