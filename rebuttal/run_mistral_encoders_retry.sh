#!/usr/bin/env bash
set -uo pipefail
cd /home/mll/alireza/EDO-reb
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
py=/home/mll/alireza/or_runs/venv/bin/python
name=or-mistral-small-3.2-24b
until grep -q MISTRAL_REST_DONE logs/mistral/all.log 2>/dev/null; do sleep 30; done
for ds in math500-test popqa-test_n500_seed42 mmlupro-test_n500_seed42 gpqa-train_main_n448; do
  for m in 0.6B:0p6b 1.7B:1p7b 4B:4b; do
    size=${m%%:*} tag=${m##*:}
    if ! ls outputs/${name}_$ds/ours_${name}_${ds}_budget5_Qwen-Qwen3-${size}_*theta1.csv >/dev/null 2>&1; then
      echo "[$(date -u +%FT%TZ)] retry $ds Qwen3-$size"
      "$py" -u run_experiment.py --config configs/${name}_$ds/ours_qwen3_$tag.yaml --no_generate --reset || echo "FAILED $ds $size"
    fi
  done
done
echo "[$(date -u +%FT%TZ)] ENCODERS_RETRY_DONE"
