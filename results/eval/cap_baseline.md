# Baseline with a 32,768-token output cap (snapshot 2026-11-20)

Question: how much of the failure at our 8,192-token cap is the cap? Qwen's model cards recommend 32,768 output tokens for most queries and 81,920 for math benchmarks, and both cards warn that the model "is more prone to entering thinking loops". Runs: `harness.target::cap` (`/runs/eval/cap/`). Models as downloaded, no edits; MATH-500, first 250 problems × 2 answers; the same prompts, sampling and seeds as every other score in this repo; only the cap differs.

| | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|
| Accuracy at the 8,192 cap (cached run) | 7.8% | 38.0% |
| Accuracy at the 32,768 cap | 8.4% | 67.4% |
| Paired difference, 32K − 8K | +0.6 ± 0.9 | +29.4 ± 2.6 |
| Answers still cut at 32,768 tokens | 93.4% | 32.4% |
| Answers finished within 8,192 / 16,384 / 32,768 tokens | 5.2 / 6.2 / 6.6% | 36.2 / 52.2 / 67.6% |
| Correct among finished answers | 100% | 98.5% |
| Mean tokens per answer | 30,962 | 17,558 |
| Generated tokens (500 answers) | 15.5M | 8.8M |

Reading:
- Qwen3.5-0.8B does not finish with more room: 93% of answers still run to 32,768 tokens, and accuracy is unchanged (8.4% against 7.8%). Its loop is non-termination, not slow reasoning. The editor's edits (about 40% at the 8,192 cap) fix something a larger budget does not.
- Qwen3.5-2B mostly needs more room: accuracy rises from 38.0% to 67.4% with four times the budget, and the share of finished answers keeps growing with the cap (36 → 52 → 68%). A third of its answers still do not finish at 32,768 tokens.
- On the 2B, the best edit we have (64.2% at the 8,192 cap, from 2 untrained-editor rollouts) is at the level of the unedited model with a 32,768 cap (67.4%). At 8,192 tokens the edit and the larger budget buy about the same accuracy; the edit does it with about a fifth of the tokens (3,505 per answer against 17,558).
- On both models nearly every finished answer is correct, at either cap.

Not measured: the 81,920-token cap; edited models at 32,768 tokens. The paired comparison with the 8,192 run is approximate: with the same seed the server's batching still changes some samples (in the 0.8B run, 12 answers went wrong → right and 9 right → wrong).
