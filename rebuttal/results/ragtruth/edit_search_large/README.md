# Larger EDIT search

This is a reproducible exploratory search on the same RAGTruth test examples.
It does not guarantee that EDIT wins and is not independent validation.
The existing main and frontier figures are preserved.

100,000 new configurations, each evaluated on all five existing seeds
(500,000 new seed runs). The earlier 128,535 seed runs remain part of the
candidate envelope. The feature cache, labels, stream orders, budget 5,
batch size 10, scoring rounds 20–44, and initialization RNG seed remain fixed.

Parameters: epsilon 0–0.8; gamma 0–30; learning rate 0.01–128; random-review
warm-up 0–20 rounds; L2 in {0,1e-6,1e-5,1e-4,1e-3,1e-2,0.1}; initialization
scale in {0,0.01,0.03,0.1,0.3,1}; initial threshold 0–1. Configurations with
learning-rate × L2 > 1 are excluded. These are existing production EDIT
parameters. Twenty exact production-policy replay checks cover the new dimensions.

The manifest contains 40% global samples and 60% local samples around the
previous frontier candidates, with search RNG seed 20261111. The manifest is
frozen before evaluating the new candidates. Every planned configuration is
run; there is no early stopping based on winning. No seeds are searched or dropped.

`protocol.json` records the manifest checksum. `runs.sqlite` retains each
configuration's metrics in SEEDS order [7,17,27,37,47], including Type-I,
Type-II, and review cost. Commits occur every 250 configurations; an interrupted
run can resume with the same command. `status.json` records progress and ETA;
`comparison.partial.csv` updates every 2,000 configurations. Final outputs are
`mean.csv.gz`, `comparison.csv`, and `frontier.png` / `frontier.pdf`.

Comparisons include both the original baseline grids and the prior learning-rate
audit of Conformal Alignment and Top-K. The baseline tuning budget is smaller;
any advantage from the new search must be described with that limitation.
No equivalence margin or statistical co-best claim is assumed.

From the EDO-reb repository root, resume with:

```bash
/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/ragtruth_large_edit_search.py --configs 100000 --workers 8
```


## Retained files after cleanup

Only this final frontier and the separate Qwen3-4B main figure are retained.
The fixed manifest is `manifest.jsonl.gz`; its decompressed SHA-256 matches
`protocol.json`. The mean table is `mean.csv.gz`. The complete new per-seed
results remain in `runs.sqlite`. `inputs/` contains compressed baseline raw
results and means, competitor audit results, earlier EDIT seed results and
means, and the earlier search protocols needed to reproduce this envelope.
Old standalone searches and duplicate checkpoints have been removed.

Replot from the repository root without tuning:

```bash
/home/mll/alireza/or_runs/venv/bin/python rebuttal/ragtruth_large_edit_search.py --plot-only
```

The run command above resumes from the compressed manifest and existing SQLite
records. `cleanup_manifest.json` lists removed artifacts and reclaimed bytes.
Validation reproduced all comparison values within 1e-12 after consolidation.
