# Experimental details

## 1. Baseline frontiers

### Protocol

- Datasets: MATH-500, MMLU-Pro, PopQA, and GPQA; response models: Llama-3.3-70B and Qwen3-32B.
- Five streams: recorded seed 7 plus seeds 17, 27, 37, and 47 sampled with replacement from the response pool.
- Unit costs, batch size 10, 250 rounds, budget 5. Metrics exclude rounds 0--19 and are averaged over the five streams.
- Score-based methods use frozen `all-MiniLM-L6-v2` features (512 tokens) and the same online probe (learning rate 1.0, initialization scale 0.1).
- Hyperparameters were selected on the displayed streams; there was no separate validation set.

### Grids

| Method | Grid |
|---|---|
| EDIT-REP | epsilon = 0.00, 0.02, ..., 0.80; gamma in {0.02, 0.05, 0.1} |
| Online Conformal Selection | epsilon = 0.00, 0.02, ..., 0.80; dual step in {0.063, 0.25, 1.0} |
| Conformal Alignment | q = 0.02, 0.04, ..., 0.80; 20 labelled warm-up rounds (10 training + 10 calibration) |
| Stream Active Learning | budget in {1, 2, 3, 4, 5}; randomized in {false, true}; step in {0.01, 0.05} |
| Thompson Sampling | budget in {1, 2, 3, 4, 5}; posterior scale in {0.3, 1.0, 3.0} |
| Top-K | budget in {1, 2, 3, 4, 5} |
| Fixed Threshold | threshold = 0.00, 0.02, ..., 0.98 |

Each curve is the lower Type-II envelope over its grid after five-stream averaging; hence there is no single final tuple. Retained values are in [`results/baselines_selected_hparams.csv`](results/baselines_selected_hparams.csv) and [`results/qwen/baselines_selected_hparams.csv`](results/qwen/baselines_selected_hparams.csv).

### Oracle regret, matched cost, and robustness

- **Oracle regret:** for each Type-I ceiling in {0.15, 0.20, ..., 0.50}, select the frontier configuration with minimum mean Type-II subject to mean Type-I <= the ceiling, then rerun it on five streams. The same-spend oracle reviews the method's number of items per round; the budget oracle reviews up to five; both prioritize true edits. Exact selections: [`results/regret_raw.csv`](results/regret_raw.csv) and [`results/qwen/regret_raw.csv`](results/qwen/regret_raw.csv).
- **Matched cost:** reuse the frontier grid. At each target cost in {1, 2, 3, 4, 5}, select minimum Type-II among configurations within 0.25 realized reviews/round of the target. Results: [`results/matched_cost.csv`](results/matched_cost.csv).
- **Robustness:** fixed Llama MMLU-Pro (200 rounds) and PopQA (180 rounds) shift streams; targets {0.10, 0.15, ..., 0.50}. Select non-EDIT hyperparameters on phase 1 only using minimum phase-1 Type-II subject to phase-1 Type-I <= target (otherwise minimum Type-I), then freeze them. EDIT-REP fixes gamma = 0.1 and uses `epsilon = max(0, target - (1 + gamma)/(gamma T))`. Online Conformal Selection selects its dual step from {0.063, 0.25, 1.0} on phase 1. Exact selections: [`results/shift_results.csv`](results/shift_results.csv).

## 2. No-repeat MATH and PopQA

### Protocol and frontier

- Four combinations: Llama-3.3-70B and Qwen3-32B on MATH and PopQA.
- Each pool has 2,500 unique examples in 250 batches of 10 without replacement. The five frontier streams are seed 7 plus four independent permutations (17, 27, 37, 47).
- Non-EDIT methods use the Section 1 grid with MiniLM. EDIT-REP uses Qwen3-1.7B and sweeps epsilon = 0.00, 0.02, ..., 0.80 and gamma in {0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10}, with probe learning rate 4.0 and no warm start.
- Metrics exclude rounds 0--19 and curves are five-stream lower envelopes. Retained values: [`results/norepeat/baselines_selected_hparams.csv`](results/norepeat/baselines_selected_hparams.csv).

### Type-II-tuned combined figure

This search uses the four recorded seed-7 streams and all 250 rounds. Within each model--dataset combination, EDIT-REP, EDIT-LLM, and EDIT-RAND use the same epsilon and must satisfy Type-I <= epsilon <= LLM-Select Type-I + 0.03 and per-round cost <= 5.

Grids:

- encoder in {MiniLM, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B};
- gamma in {0, 0.005, 0.01, 0.02, 0.05, 0.068, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10};
- EDIT-REP learning rate in {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8};
- warm start in {0, 5, 10};
- epsilon at positive multiples of 0.025 up to the combination-specific cap, plus the exact cap.

EDIT-REP uses per-combination gamma/epsilon but a shared learning rate and warm start. EDIT-LLM and EDIT-RAND are independently tuned over gamma and warm start at the same epsilon.

The displayed figure uses Qwen3-1.7B and EDIT-REP learning rate 4.0:

| Model | Dataset | Method | epsilon | gamma | warm start |
|---|---|---|---:|---:|---:|
| Llama | MATH | EDIT-REP / EDIT-LLM / EDIT-RAND | 0.675 | 0.1 / 0.068 / 1.5 | 0 / 0 / 5 |
| Llama | PopQA | EDIT-REP / EDIT-LLM / EDIT-RAND | 0.200 | 2.0 / 0.4 / 3.0 | 0 / 0 / 0 |
| Qwen | MATH | EDIT-REP / EDIT-LLM / EDIT-RAND | 0.635 | 0.2 / 0.15 / 10.0 | 0 / 0 / 10 |
| Qwen | PopQA | EDIT-REP / EDIT-LLM / EDIT-RAND | 0.1202 | 0.4 / 0.4 / 5.0 | 0 / 0 / 10 |

This was exploratory tuning on the displayed streams. Qwen3-1.7B was chosen because it led on three combinations; Qwen3-4B had the lower four-combination mean Type-II. Full results: [`results/typeii_search`](results/typeii_search).

## 3. LLM-Select prompt sensitivity

### Protocol

- Same stored Llama-3.3-70B responses, labels, costs, and order for all four datasets; 250 batches of 10 and budget 5.
- Controlled arms use `meta-llama/llama-3.3-70b-instruct` through OpenRouter/CoreWeave, temperature 0, no fallback, and a 24,000-character cap.
- `legacy_original` is the historical Groq run; `legacy_full_context` reruns the original prompt under the controlled setup.

The legacy prompt asks for the budget-feasible items most likely to require correction using only the question/options and model answer, forbids using the gold answer, forbids solving, and requires `{"selected_indices": [...]}`. Exact dataset-specific wording: [`methods/llm_select.py`](../methods/llm_select.py).

The three sensitivity variants insert:

### Reasoning allowed

```text
Reasoning instruction:
- You may solve the questions or reason through them privately when useful.
- Use that private reasoning to assess whether each model answer is correct.
- Do not reveal your reasoning; the final response must still contain only the JSON object.
```

### Independent verification

```text
Verification procedure:
- For every item, independently derive or verify the correct answer before judging the model answer.
- Compare your result with both the model's final answer and any supporting reasoning.
- Treat substantive reasoning errors, contradictions, invalid options, and instruction failures as evidence that correction is needed.
- Select the budget-feasible items with the highest probability of requiring correction.
- Perform the verification privately; return only the JSON object.
```

### Risk ranking with full budget

```text
Risk-ranking procedure:
- Independently solve or fact-check every item, then compare your conclusion with the model answer.
- Rank all items from highest to lowest probability of requiring a substantive correction.
- Spend as much of the available budget as possible without exceeding it; for equal unit costs, select exactly as many highest-risk items as the budget permits.
- List selected indices in descending order of estimated error risk.
- Keep all analysis private and return only the JSON object.
```

Each variant also permits private reasoning in the system prompt while requiring JSON-only output. Results: [`results/llm_select_prompt_ablation.csv`](results/llm_select_prompt_ablation.csv).
