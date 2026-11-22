#!/bin/bash
cd /home/mll/alireza/EDO-reb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export http_proxy=http://127.0.0.1:10809 https_proxy=http://127.0.0.1:10809
export HF_HUB_DOWNLOAD_TIMEOUT=60
/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/sweep_norepeat_gamma.py >> rebuttal/results/norepeat_gamma/run.log 2>&1
printf 'Sweep exited with code %s\n' "$?"
exec bash
