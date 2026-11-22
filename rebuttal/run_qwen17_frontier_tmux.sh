#!/usr/bin/env bash
set -euo pipefail

cd /home/mll/alireza/EDO-reb/rebuttal
log=results/norepeat/qwen17_frontier.log

/home/mll/alireza/or_runs/venv/bin/python -u run_baselines_norepeat_qwen17.py --workers 8 2>&1 | tee "$log"
/home/mll/alireza/or_runs/venv/bin/python -u plot_baselines_norepeat.py 2>&1 | tee -a "$log"
