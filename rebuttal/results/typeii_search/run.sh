#!/bin/bash
cd /home/mll/alireza/EDO-reb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/search_typeii.py --workers 6 >> rebuttal/results/typeii_search/run.log 2>&1
printf 'Type-II search exited with code %s\n' "$?"
exec bash
