# Ordered shared-gamma search

Scope: the four OpenRouter n2500 no-repeat runs (Llama/Qwen × MATH/PopQA).

1. Sweep MiniLM over the 25 gamma values in plan.json.
2. Unless a shared gamma is best/co-best on all four combinations, repeat with frozen Qwen3-0.6B, then Qwen3-1.7B, then Qwen3-4B. Stop at the first successful stage or after the grid is exhausted.
3. Rank shared gammas by number of feasible best/co-best combinations, number feasible, mean epsilon excess, mean Type-II, then mean cost. Never pick separate gammas per combination.

Feasibility: final mean per-round Type-I <= configured epsilon (numerical tolerance 1e-9), and no round exceeds its budget. Comparison is to the lowest Type-II among feasible existing plotted methods; if none is feasible, a feasible candidate is the first eligible method. Co-best tolerance is 0.005 absolute Type-II. Metrics use all 250 rounds, as in the plotting notebook. No warm-up rounds are excluded. Note Qwen/PopQA epsilon is zero, so an initial reviewed correct answer prevents strict feasibility.

All existing hyperparameters and the recorded stream remain fixed except gamma and, in later stages, the encoder. MiniLM's original gamma is replayed with an exact selection-agreement check before sweeping each combination. Frozen features are saved by original batches, and trials are checkpointed. Generation APIs are not used. Missing encoder weights may be downloaded into model_cache.

Artifacts: status.json, baselines.csv, per-stage sweep.csv, ranking.csv, best.json, best_shared_gamma.png/pdf, and per-setting cached features and trial CSVs. Original run outputs and the original combined figure are unchanged. Plots retain the original four methods and add the best shared-gamma candidate for the stage.

These are exploratory results selected on these same four streams, not held-out estimates. All tested settings are retained.

Monitor: tmux attach -t edo_gamma (progress window), or tail -f run.log.
Resume from the workspace: bash EDO-reb/rebuttal/results/norepeat_gamma/run.sh
