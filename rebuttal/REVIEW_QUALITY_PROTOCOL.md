# Locked full-task controller study

Launch: `tmux attach -t review_quality`

Reproduction (use a new output directory for an independent rerun):

```bash
cd /home/mll/alireza/EDO-reb
/home/mll/alireza/or_runs/venv/bin/python -u rebuttal/run_review_quality.py --output rebuttal/results/review_quality_v1 --workers 8
```

## Scope

17 settings: all four original datasets (MATH-500, MMLU-Pro, PopQA, GPQA), both Llama and Qwen response models, MiniLM; additionally no-repeat MATH and PopQA with both response models and both MiniLM/Qwen3-1.7B; RAGTruth with its available Qwen3-4B feature cache. No primary task is omitted. Qwen3-1.7B is not available for every task; encoder coverage is explicitly reported, not imputed.

Historical data were already examined in prior experiments. This is a prospectively locked reanalysis, not a new untouched-test confirmation. The old MATH-only ablation and every existing result are preserved.

## Separation and calibration

Five grouped outer folds cover every record of each primary/no-repeat pool. A normalized question stays in one fold. For each held-out fold, two other folds fit the logistic probe, one fits a Platt calibrator and one selects policy parameters. These four roles are disjoint. For RAGTruth, official test sources stay held out; official training sources are partitioned into fit/calibration/validation. Probe C=1 and calibrator C=1 are fixed, not tuned on test. Five probe fits comprise the original fit and four bootstrap resamples of training records, with fixed calibration/validation partitions; each has its own development-only parameter selection. This measures variability of the fitted-and-selected pipeline, not purely optimizer randomness. Cross-fitting evaluates the deployed procedure with fold-specific policies, not one single global policy.

No source documents are available for reliable document-level grouping in all datasets; primary grouping uses normalized question text. RAGTruth uses source ID.

## Policies and tuning

Primary: adaptive versus same-initial-alpha frozen EDIT. The original production threshold selector and alpha update are reused. No test probe updates; the only difference is whether gamma is active. Threshold initialization is selected to optimize the frozen baseline on validation. Gamma is then selected with that threshold fixed; zero gamma is permitted. Fixed Threshold independently uses the same validation criterion and thus may coincide with Frozen-alpha. Top-K uses five reviews per full batch. A secondary fixed threshold matches adaptive development review cost as closely as the grid permits; test residual cost differences are reported, not assumed away.

Existing threshold grid: 0 to 0.98 in steps of 0.02. Existing gamma grid: 0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10. Primary epsilon 0.40; sensitivity 0.20 and 0.60. Three validation orders, seeds 101–103. Minimum paper-defined Type-II subject to mean validation Type-I <= epsilon; infeasible configurations explicitly flagged. All settings and bootstrap-probe variants finish tuning before any test evaluation begins.

## Deployment accounting

Budget ceiling five per batch of ten; all rounds count. Final partial batches are retained, with budget at most five and at most their size. Policies share all initial offline labels; the count of fit, calibration and validation labels is saved alongside online reviews. Repeated label reuse in offline tuning does not imply newly acquired labels per replay. Adaptive feedback only uses selected labels. No full-label startup advantage is provided to a baseline.

Type-I is mean roundwise false confirmations / reviewed records; Type-II is mean roundwise missed edits / unreviewed records, with empty denominators zero, verified against paper code. Cost, pooled precision, coverage and raw counts are saved. Undefined pooled precision for no reviews is NaN, never treated as perfect precision.

## Uncertainty and reporting

Thirty paired order seeds 1000–1029. Grouped held-out fold results are aggregated before inference; folds are not counted as independent observations. Paired percentile bootstrap CIs (20,000 resamples) and two-sided sign-flip tests (50,000 resamples); the sign-flip interpretation assumes symmetry/exchangeability under the null, with a sign-test sensitivity provided. Holm correction covers every primary same-alpha comparison, including all 17 settings.

Source-resampling diagnostic: 200 cluster bootstraps of held-out question/source groups, with a fresh full policy replay per resample. Bootstrap repeats are resampling for uncertainty, not repeated records in the primary deployment streams. This component includes source and order variability; it is not a pure variance decomposition. Probe-training sensitivity uses the other four bootstrapped fits, reports all outcomes, and does not pool these as independent datasets with the order seeds. The baseline and adaptive policy always share the probe and sampled stream within each comparison.

Joint ordering evidence requires positive lower paired effect CI, adjusted p<0.05, development feasibility, and a one-sided 95% ordering-bootstrap upper bound for adaptive Type-I <= epsilon. Separate source-bootstrap bounds are reported. These marginal bounds are not a simultaneous familywise confidence guarantee for feasibility and all effects. No practically meaningful non-inferiority margin is invented. Cost efficiency is not claimed when realized expenditures differ. All tolerance sensitivities, null effects and infeasible points are retained.

## Artifacts

Log: `rebuttal/review_quality_v1.log`. Outputs: `rebuttal/results/review_quality_v1/`. Per-setting split IDs, probe weights, calibrated scores, complete validation grids and locked policies are retained. Main deliverables: `primary_table.tex`, `primary_joint_results.csv`, `paired_effects.csv`, `quality_cost_precision_coverage.csv`, `record_bootstrap.csv`, `probe_training_sensitivity.csv`, `academic_paragraph.txt`, `openreview_response.md`. These are generated only after execution finishes; no result is presumed favorable.
