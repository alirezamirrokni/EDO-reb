# Rebuttal experiments

Offline experiments on the four unit-cost settings (Llama-3.3-70B responses on MATH-500, MMLU-Pro, PopQA, GPQA). They reuse the stored generations and labels in `outputs/` and the cached MiniLM features in `features/`, so they need no API key and no GPU.

## Reproduce

```bash
cd rebuttal
python run_baselines.py --workers 8   # ~25 min on 8 CPU cores -> results/baselines_raw.csv
python plot_baselines.py              # figure and tables in results/
python first_visit.py                 # results/first_visit.csv
```

## Baselines against EDIT-REP

Every method uses the same MiniLM features, logistic probe, probe update on reviewed items, initialization, per-round budget and streams: the paper stream (seed 7) and four streams drawn with replacement from the pool (seeds 17, 27, 37, 47). Metrics are the paper's per-round Type-I/II, computed from round 20 onward so that Conformal Alignment's labelled warm-up is excluded for all methods. Each curve is the lower envelope over the method's grid, averaged over the five streams.

| Method | Reference | Swept | Fixed |
|---|---|---|---|
| EDIT-REP (Ours) | — | ε ∈ {0.00, 0.02, …, 0.80} × α step γ ∈ {0.02, 0.05, 0.1} | budget 5 |
| Online Conformal Selection | Gollapudi et al., 2026 (arXiv 2605.14953) | ε ∈ {0.00, …, 0.80} (target 1−ε) × dual step ∈ {0.063, 0.25, 1.0} | threshold arms τ ∈ {0.00, 0.05, …, 0.95} plus "review nothing", budget 5 |
| Conformal Alignment | Gui, Jin & Ren, NeurIPS 2024 | q ∈ {0.02, 0.04, …, 0.80} | 20 fully labelled warm-up rounds (10 train, 10 calibration), probe frozen afterwards, budget 5 |
| Stream Active Learning | Žliobaitė et al., IEEE TNNLS 2014 | budget ∈ {1, …, 5} × randomized ∈ {no, yes} × step ∈ {0.01, 0.05} | initial threshold 1.0 |
| Thompson Sampling | Chapelle & Li, NeurIPS 2011 | budget ∈ {1, …, 5} × posterior scale ∈ {0.3, 1.0, 3.0} | Laplace posterior, prior precision 1.0 |
| Top-K | — | budget ∈ {1, …, 5} | — |
| Fixed Threshold | — | τ ∈ {0.00, 0.02, …, 0.98} | budget 5 |

Adaptations to this setting: Online Conformal Selection uses the observed round precision (1 − Type-I) as its success signal and budget minus edits found as its cost, with "review the top-K by score" as the aggressive boundary arm. Stream Active Learning applies its budget per round.

Outputs in `results/`:

- `baselines_frontier.{png,pdf}`: Type-I vs Type-II per dataset.
- `baselines_raw.csv`: one row per (dataset, method, setting, stream).
- `baselines_mean.csv`: mean over the five streams.
- `baselines_levels.csv`: best Type-II at Type-I ≤ level.
- `baselines_selected_hparams.csv`: grid values on each envelope.

## First-visit metrics

`first_visit.py` reads the stored paper runs and reports pooled Type-I, Type-II, edit recall, edit precision and missed edits on first visits (an item's first appearance in the stream), on repeat visits and on all visits. It covers EDIT-REP (MiniLM and Qwen3-0.6B/1.7B/4B encoders), EDIT-LLM, EDIT-RAND and LLM-Select, on the four unit-cost settings and the two review-length settings.

## RAGTruth with Qwen3-4B

The retained main figure is `../figures/ragtruth/ragtruth_typeii_typei_cost_main.{png,pdf}`.
Its EDIT-REP curve uses `features_qwen4b_2048.npy` (2,560-dimensional Qwen3-4B
last-token features, 2,048-token input limit), with train-selected learning rate
0.5, controller step 0.2, and zero warm-up. Replot with
`python ragtruth_main_figure.py plot`; rerun main-figure tuning with
`python ragtruth_main_figure.py`.

The retained frontier is
`results/ragtruth/edit_search_large/frontier.{png,pdf}`. Recreate it without
rerunning tuning:

```bash
python ragtruth_large_edit_search.py --plot-only
```

This final frontier includes the 100,000-configuration EDIT search and earlier
candidates, with original-baseline and learning-rate-audit panels. Compressed
inputs live in `edit_search_large/inputs/`; the complete new seed-level results
remain in `runs.sqlite`. The manifest and mean table are gzip-compressed.
See `edit_search_large/README.md` for the search protocol and reproducibility.

`run_ragtruth_frontier.py` retains the shared data adapter and can regenerate
the baseline inputs; `ragtruth_audit_baseline_lr.py` regenerates the competitor
learning-rate audit. Superseded frontier figures, launchers, exploratory
scripts, duplicate checkpoints, and unused 1.7B/4K feature caches were removed.
The retained main figure uses the Qwen3-4B 2,048-token feature cache.

## Requested epsilon versus achieved Type-I

Run `python plot_epsilon_tracking.py` from `rebuttal/` to reproduce
`results/norepeat/epsilon_tracking.{png,pdf}` from the saved Qwen3-1.7B EDIT
frontier runs. The four panels fix learning rate 4, zero warm-up, and gamma
in {0.02, 0.5, 5, 10}; epsilon varies from 0 to 0.80 in steps of 0.02.
Lines show five-ordering means and bands show one sample standard deviation.
The script validates all 3,280 selected seed rows and exports 656 aggregated
points, an excess-risk summary, source-hash metadata, and a paper-ready caption.
This is a post-hoc descriptive diagnostic; the original frontier grid is unchanged.
