# Evaluation results (snapshot 16 November 2026, 10:40)

All scores are on held-out sets that the editor never sees and that were never used to choose an edit.
Gains are in percentage points over the unedited model. Small samples: this is a proof of concept, not a measured difference.

## (i) Does RL make the editor better?

Untrained editor (a fresh LoRA on Inkling 2 Small, no RL) vs trained editor (after 8 RL steps), each doing editing sessions on the same model; each submitted edit scored on the held-out set. "Best" = the edit with the highest reward-set score, scored on the held-out set. A session that submits no edit counts as 0.

| Setting | Sessions per editor | Untrained: per edit | mean / best | Trained: per edit | mean / best |
|---|---|---|---|---|---|
| Qwen3.5-0.8B, MATH-500 (original 6.6%) | 4, 90-min cap | +17.1, 0*, +13.8, +16.2 | +11.8 / 23.7% | +18.2, +4.9, +13.5, +18.9 | +13.9 / 24.8% |
| Qwen3.5-0.8B, MATH-500: editor after 4 steps of run `cur4g16` (16 sessions per step, Dr. GRPO) | 4, 90-min cap | | | +16.4, +28.8, +14.0, +12.2 | +17.9 / 35.4% |
| Qwen3.5-2B, MATH-500, 250 problems (original 38.0%) | 2, 60-min cap | +26.2, +12.4 | +19.3 / 64.2% | 0 (no edit), +15.6 | +7.8 / 53.6% |

\* this session's edit scored +19.1 on the reward set, but it ended at the time limit and its edit files were not saved, so the held-out score is 0 (a harness bug, counted as 0).

Reading: on MATH-500 the step-4 `cur4g16` editor has the best numbers (mean +17.9, best edit 35.4%), but on the reward set the untrained and step-4 editors tie (17.8 vs 17.7); most of the MATH-500 gap comes from the untrained session whose edit was lost to the time-limit bug (untrained mean without it: +15.7). So no RL benefit is established. Earlier reading: no RL benefit was shown. On the 0.8B the trained editor is about 2 points ahead (within noise); on the 2B the untrained editor did better, with 2 sessions each.

Earlier edits from the training runs, scored on GSM8K test (Qwen3.5-0.8B, original 12.5%; 8 edits each): untrained step 1 mean +20.1 (best +47.3); step 4 (editor after near-zero updates) mean +31.4 (best +46.0).

## (ii) Does the edited (child) model get better?

Qwen3.5-0.8B as downloaded vs the child models (Qwen with the best edits from earlier steps applied on top of each other). Scoring only.

| Held-out set | Original | Best single edit, no RL | Child 1 (2 stacked edits) | Child 2 (3 stacked edits) |
|---|---|---|---|---|
| MATH-500 (1,000 answers) | 6.6% | 39.4% | 42.5% | 42.5% |
| GSM8K test (1,000 answers) | 12.1–12.5% | 59.8% | 53.8% | 58.0% |
| NuminaMath cn_k12 (500 answers) | 8.2% | 39.2% | 44.6% | 41.4% |

Children vs original: almost all changes are wrong→right (MATH-500 360 vs 1; GSM8K 464 vs 5; NuminaMath 188 vs 6). The child models were built from edits chosen on MATH training problems; GSM8K test and NuminaMath were never used. The child's gain comes mostly from the editing method plus picking the best edit of each step; against the best single edit with no RL it is +3 points on MATH-500, −2 on GSM8K and +2 to +5 on NuminaMath.

Raw outputs: `/eval/<name>/summary.json` and `scores.json` on the `glimmer-runs` Modal volume.
