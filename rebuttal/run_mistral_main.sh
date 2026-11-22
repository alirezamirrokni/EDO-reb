#!/usr/bin/env bash
set -uo pipefail
cd /home/mll/alireza/EDO-reb
export http_proxy="http://127.0.0.1:10809" https_proxy="http://127.0.0.1:10809"
export HTTP_PROXY="http://127.0.0.1:10809" HTTPS_PROXY="http://127.0.0.1:10809"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
py=/home/mll/alireza/or_runs/venv/bin/python
name=or-mistral-small-3.2-24b
mkdir -p logs/mistral

one() {
  local ds=$1 cfg=configs/${name}_$1 log=logs/mistral/$1.log
  {
    echo "[$(date -u +%FT%TZ)] $ds start"
    "$py" -u generate_cache.py --config "$cfg/generate.yaml" || { echo "GEN_FAILED $ds"; return 1; }
    "$py" -u run_experiment.py --config "$cfg/llm_select.yaml" --fail_on_selector_error || { echo "SELECT_FAILED $ds"; return 1; }
    "$py" -u rebuttal/mistral_set_eps.py "$cfg" || return 1
    for m in ours_llm random ours_random; do
      "$py" -u run_experiment.py --config "$cfg/$m.yaml" --no_generate || echo "FAILED $ds $m"
    done
    for m in ours ours_qwen3_0p6b ours_qwen3_1p7b ours_qwen3_4b; do
      flock logs/mistral/gpu.lock "$py" -u run_experiment.py --config "$cfg/$m.yaml" --no_generate || echo "FAILED $ds $m"
    done
    echo "[$(date -u +%FT%TZ)] $ds DONE"
  } >> "$log" 2>&1
}

curl -s --max-time 30 -o /dev/null -w "%{http_code}" https://openrouter.ai/api/v1/models | grep -q 200 || { echo "OPENROUTER_UNREACHABLE"; exit 1; }
for ds in math500-test popqa-test_n500_seed42 mmlupro-test_n500_seed42 gpqa-train_main_n448; do
  one "$ds" &
done
wait
echo "[$(date -u +%FT%TZ)] MISTRAL_ALL_DONE" | tee -a logs/mistral/all.log
