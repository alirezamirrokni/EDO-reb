#!/usr/bin/env bash
set -euo pipefail
cd /home/mll/alireza/EDO-reb
py=/home/mll/alireza/or_runs/venv/bin/python
# Extraction is resumable and must not share a wall-clock limit with downloading.
"$py" -u rebuttal/embed_qwen17_remaining.py --llama-primary
"$py" -u rebuttal/run_review_quality_llama.py --output rebuttal/results/review_quality_llama_qwen17_v1 --workers 8
