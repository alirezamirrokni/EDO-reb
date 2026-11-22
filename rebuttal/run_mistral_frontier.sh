#!/usr/bin/env bash
set -uo pipefail
cd /home/mll/alireza/EDO-reb/rebuttal
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
py=/home/mll/alireza/or_runs/venv/bin/python
log=../logs/mistral/frontier.log
until grep -q MISTRAL_REST_DONE ../logs/mistral/all.log 2>/dev/null; do sleep 60; done
{
  echo "[$(date -u +%FT%TZ)] frontier start"
  "$py" -u run_baselines.py --model mistral --workers 8 && "$py" -u plot_baselines.py --model mistral && echo "[$(date -u +%FT%TZ)] FRONTIER_DONE"
} >> "$log" 2>&1
