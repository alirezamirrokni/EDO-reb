**Requested tolerance versus achieved Type-I error.** EDIT-REP uses frozen
Qwen3-1.7B representations, probe learning rate 4.0, and no warm start on the
four no-repeat model–dataset settings. Each curve fixes the controller step
size γ while sweeping ε from 0 to 0.80 in increments of 0.02. Lines show the
mean and shaded bands show ±1 sample standard deviation over five orderings
(seeds 7, 17, 27, 37, and 47) of the same 2,500 examples. Bands are clipped to
[0, 1]. Type-I is the average per-round fraction of reviewed items not requiring
edits; rounds 0–19 are excluded and a round with no reviews contributes zero.
All runs use batches of 10 and a review budget of 5. The dashed diagonal marks
achieved Type-I equal to ε; points below it are conservative relative to the
requested tolerance, and points above it exceed that tolerance. ε is specified
independently of LLM-Select, and no configuration is selected separately at
each ε. This is a descriptive replay of saved runs, not independent held-out
validation or a claim that every finite-run average lies below ε.

The four γ values were chosen for a compact post-hoc diagnostic; the original
frontier experiments used the full 19-value grid. Shading describes variation
across orderings, not uncertainty from independently sampled datasets.
