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
| EDIT-REP (Ours) | — | ε ∈ {0.00, 0.02, …, 0.80} | γ = 0.05, budget 5 |
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
