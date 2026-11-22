#!/usr/bin/env bash
set -uo pipefail
cd /home/mll/alireza/EDO-reb
export http_proxy="http://127.0.0.1:10809" https_proxy="http://127.0.0.1:10809"
export HTTP_PROXY="http://127.0.0.1:10809" HTTPS_PROXY="http://127.0.0.1:10809"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
py=/home/mll/alireza/or_runs/venv/bin/python
name=or-mistral-small-3.2-24b
one() {
  local ds=$1 cfg=configs/${name}_$1 log=logs/mistral/$1.log
  {
    echo "[$(date -u +%FT%TZ)] $ds resume after epsilon"
    for m in ours_llm random ours_random; do
      "$py" -u run_experiment.py --config "$cfg/$m.yaml" --no_generate || echo "FAILED $ds $m"
    done
    for m in ours ours_qwen3_0p6b ours_qwen3_1p7b ours_qwen3_4b; do
      flock logs/mistral/gpu.lock "$py" -u run_experiment.py --config "$cfg/$m.yaml" --no_generate || echo "FAILED $ds $m"
    done
    echo "[$(date -u +%FT%TZ)] $ds DONE"
  } >> "$log" 2>&1
}
for ds in math500-test popqa-test_n500_seed42 mmlupro-test_n500_seed42 gpqa-train_main_n448; do one "$ds" & done
wait
echo "[$(date -u +%FT%TZ)] MISTRAL_REST_DONE" | tee -a logs/mistral/all.log
