# Type-II-first search

User constraint: each EDIT method's final running-average Type-I must not exceed LLM-Select's by more than **0.03 absolute**. Epsilon is shared between EDIT-REP, EDIT-LLM and EDIT-RAND within each combination; it may differ between combinations. Every selected EDIT run must satisfy Type-I <= epsilon <= LLM-Select Type-I + 0.03, plus per-round budget <= 5.

Objective: minimize mean Type-II over the four combinations. Among exactly tied results, prefer lower mean review cost, then lower mean epsilon. No reward for pushing Type-I unnecessarily below epsilon. There is no requirement to produce a winning comparison if the data do not support it.

Encoder order: MiniLM, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B. All stages are evaluated to seek the strongest result. The selected EDIT-REP uses a separately selected gamma for each model–dataset combination, with a shared predictor learning rate and warm-start length across the four combinations. Epsilon is selected per combination, from positive multiples of 0.025 plus the exact Type-I ceiling.

Grid: 19 gamma values from 0 to 10, 8 predictor learning rates from 0.05 to 8, and 0/5/10 random-review warm-start rounds. Warm starts consume the normal budget and count in all metrics. For each epsilon, EDIT-LLM and EDIT-RAND independently minimize Type-II using the same gamma/warm-start grid; their gamma and warm-up can differ between combinations. This gives the comparison methods a strong tuning budget. LLM-Select is unchanged and does not use epsilon.

Metrics: mean per-round Type-I and Type-II over all 250 rounds, exactly as in the plotting notebook. Per-round budgets are verified. All frozen features and LLM scores come from existing caches; no model downloads or API calls. The accelerated replay passed 24 original/modified-policy exact selection checks against stored runs and policy classes; see replay_checks.json.

Results are exploratory tuning on the existing streams, **not held-out evaluation**. Preserve all sweeps and report the chosen epsilon values and parameters when presenting a figure.

Artifacts:
- status.json and run.log: progress.
- trials.sqlite: resumable atomic job checkpoints.
- baselines_sweep.csv and baseline_best_by_epsilon.csv: comparison-method tuning.
- *_sweep.csv, *_ranking.csv, *_epsilon_frontier.csv: every encoder's results.
- best_overall.json, selected_summary.csv, selected_runs/: current winning full comparison.
- figures/norepeat_typeii/model_grid_llama_qwen_combined.png and .pdf: one current winner EDIT-REP, plus EDIT-LLM, EDIT-RAND and LLM-Select. Replaced only when an encoder improves the objective; no extra MiniLM curve remains if Qwen wins.

Monitor: tmux attach -t edo_typeii
Resume from workspace: bash EDO-reb/rebuttal/results/typeii_search/run.sh
