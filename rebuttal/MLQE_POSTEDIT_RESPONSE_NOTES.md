# Human post-edit experiment: protocol and current scope

Status: launched in tmux `mlqe_postedit`; results are pending. Do not claim a performance gain before completion.

We evaluate on the authors' MLQE-PE English-to-German and English-to-Chinese human-post-edit data (https://github.com/sheffieldnlp/mlqe-pe). Each interaction consists of a source sentence and its stored machine translation. The binary edit label is one iff the whitespace-normalized translation differs from its human post-edit. Provided HTER measures edit severity; it is not measured editing time.

The experiment uses official development and test splits. Exact normalized source duplicates are removed within each; development sources occurring in test are excluded from development. This is source-sentence deduplication, not a verified document-level separation. Official training data is archived for provenance but not used for probe fitting: this is cold-start online deployment. No human post-edit, HTER, or gold label is included in encoder input. Feedback labels become available to each online method only for the examples it reviews.

The current launched comparison includes EDIT-REP, fixed threshold, Top-K, and random review. All representation methods use identical frozen Qwen3-4B last-token, L2-normalized features (maximum 512 tokens) and online logistic probes. Qwen-4B is used because the cached Qwen-1.7B entries point to missing weights. No LLM-Select or EDIT-LLM API runs are included in this launch.

Streams are without replacement, with batches of 10, maximum review budget 5, and all rounds included. Up to nine trailing records are omitted to form complete batches, with the retained records determined by each permutation. Thirty test orders use seeds 1000–1029. Development tuning uses seeds 101–103.

Target epsilon grid: 0.05, 0.10, 0.20, 0.30, 0.45. EDIT gamma: 0.02, 0.10, 0.50; probe learning rate: 0.1, 1, 4. Fixed thresholds: 21 equally spaced values from 0 to 1, crossed with the same three learning rates. Top-K budgets: 1–5 crossed with those learning rates. Random budgets: 1–5. Initial alpha is zero; no fully labeled calibration period. Search counts differ by method and must be disclosed.

Configurations are selected using development metrics only: lowest paper-defined Type-II among configurations with round-averaged Type-I at most the target, breaking ties by cost. EDIT's internal epsilon equals the target. If none is feasible, the least-violating fallback is explicitly flagged. Configurations are locked before test simulation.

Metrics: mean roundwise unnecessary confirmations / reviewed items (Type-I), mean roundwise missed edits / unreviewed items (Type-II), review cost, edit recall, fraction of total HTER captured, and captured HTER per reviewed example. Severity is used only for evaluation, not selection or feedback. Repeated stream orders quantify ordering robustness, not independent population samples.

This experiment addresses whether EDIT identifies actual human corrections and captures correction magnitude. It does not measure elapsed editor time, subjective disagreement, downstream model improvement, or a causal controller-only effect. It evaluates stored NMT outputs, not newly generated Llama responses. Results must be reported even if EDIT does not lead.

Runner: `rebuttal/run_mlqe_postedit.py`
Log: `rebuttal/mlqe_postedit.log`
Outputs: `rebuttal/results/mlqe_postedit/`
