# Evaluation results

## Fine-tuning vs the trained editor, by compute (18 November 2026)

Qwen3.5-0.8B as downloaded (7.8% on these problems). Gain on MATH-500, first 250 problems × 2 answers. Compute is on Qwen only.

| Route | What is counted | Qwen FLOPs | Relative | MATH-500 gain |
|---|---|---|---|---|
| Fine-tuning, no editor | forward + backward passes of training | 2.4e16 | 1× | +23.6 |
| Trained editor, rollouts only | 3 rollouts, best by reward | 6e16 | 2.5× | +27.3 |
| Trained editor, training + rollouts | 128 training rollouts + 3 rollouts | 1.7e18 | 70× | +27.3 |

**Claim.** On this model and this failure, a trained editor's targeted edits match or slightly beat a small fine-tune (+27.3 against +23.6, within noise), and its best edits beat it clearly (40–43% accuracy against 31.4%; 31–48% of answers still cut at the token cap against 69%). This does not show that editing beats fine-tuning in general: the fine-tune was small (122 answers, one run), and the editor costs 2.5× the compute, or 70× with its training counted.

- **Fine-tuning, no editor.** Qwen is trained on 122 of its own correct answers (3 learning rates × 144 steps × 4 answers). No editor is involved. The best of 9 checkpoints, chosen on practice problems, is scored.
- **Trained editor, rollouts only.** The editor after 8 RL steps edits Qwen's weights in 3 separate rollouts. The edit with the highest reward-set score is kept. The cost of training the editor is not counted.
- **Trained editor, training + rollouts.** The same 3 rollouts, plus the 128 rollouts on Qwen that trained the editor.

Not counted: for fine-tuning, generating the 2,048 answers the training set was picked from (2.4e16) and scoring the 9 checkpoints (1.4e16); for the editor, the editor model's own compute. FLOPs are estimates (1.5 GFLOPs per generated token, 4.5 per trained token), good to about ±30%. The +27.3 is the average over every way of drawing 3 of the 12 rollouts without the gradient request (`grad_tool_armC.md`). Fine-tuning: `sft_baseline.md`.

# Earlier results (snapshot 16 November 2026, 19:00)

All scores are on held-out sets that the editor never sees and that were never used to choose an edit.
Gains are in percentage points over the unedited model. Small samples: this is a proof of concept, not a measured difference.

## (i) Does RL make the editor better?

Untrained editor (a fresh LoRA on Inkling 2 Small, no RL) vs trained editor (after 8 RL steps), each doing editing sessions on the same model; each submitted edit scored on the held-out set. "Best" = the edit with the highest reward-set score, scored on the held-out set. A session that submits no edit counts as 0.

| Setting | Sessions per editor | Untrained: per edit | mean / best | Trained: per edit | mean / best |
|---|---|---|---|---|---|
| Qwen3.5-0.8B, MATH-500 (original 6.6%) | 4, 90-min cap | +17.1, 0*, +13.8, +16.2 | +11.8 / 23.7% | +18.2, +4.9, +13.5, +18.9 | +13.9 / 24.8% |
| Qwen3.5-0.8B, MATH-500: editor after 4 steps of run `cur4g16` (16 sessions per step, Dr. GRPO) | 4, 90-min cap | | | +16.4, +28.8, +14.0, +12.2 | +17.9 / 35.4% |
| Qwen3.5-0.8B, MATH-500: untrained editor rerun with the save fix | 4, 90-min cap | +12.2, +18.5, +17.9, +0.5 | +12.3 / 25.1% | | |
| Qwen3.5-0.8B, MATH-500: editor after 8 steps of run `cur4g16` | 4, 90-min cap | | | +20.8, +31.9, +19.0, +23.5 | +23.8 / 38.5% |
| Qwen3.5-2B, MATH-500, 250 problems (original 38.0%) | 2, 60-min cap | +26.2, +12.4 | +19.3 / 64.2% | 0 (no edit), +15.6 | +7.8 / 53.6% |

\* this session's edit scored +19.1 on the reward set, but it ended at the time limit and its edit files were not saved, so the held-out score is 0 (a harness bug, counted as 0).

Reading (16 November, 19:00): pooling both untrained runs (8 sessions; 7 edits scored on MATH-500, one lost to the since-fixed save bug), the untrained editor gains +13.7 per edit on MATH-500 and +14.7 on the reward set. The step-4 editor: +17.9 and +17.7. The step-8 editor: +23.8 and +24.5, every edit above +19, best edit 38.5% against 25.1% (best untrained, rerun). The step-8 editor is ahead by about 10 points on both sets; 4 sessions for the trained editors, so a first positive result rather than an established one. Earlier reading: no RL benefit was shown. On the 0.8B the trained editor is about 2 points ahead (within noise); on the 2B the untrained editor did better, with 2 sessions each.

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
