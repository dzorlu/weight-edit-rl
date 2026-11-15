# hero3 step 1: did the first update move the editor toward the high-advantage rollouts? (2026-11-14)

Script: `scripts/update_check.py` (Modal, CPU only, `tinker` secret). Per-turn sums: `update_check.json` in this folder. Log-probabilities are Tinker `compute_logprobs` (no sampling) under the pre-update policy (a fresh rank-32 LoRA on `thinkingmachines/Inkling-2-Small:peft:262144`, made as `rl/loop.py` made it; hero3 had no `start_from`, and Tinker lists no checkpoint in its training run before `hero3-step1`) and under `.../sampler_weights/hero3-step1`. Changes are post minus pre, in nats; negative means the step-1 tokens became less likely.

## Answer

The update did not move the policy in the intended direction on the data it was trained on. Under the prompt the rollouts were sampled and trained with, all 8 rollouts became less likely, the high-advantage ones included: r4 (advantage +1.71) by 45.3 nats, r7 (+1.18) by 8.9 nats, r0 (−1.06) by 27.8, r5 (−1.16) by 20.3, r6 (−0.93) by 71.0. The correlation between advantage and the change in total log-likelihood is −0.02 (Spearman −0.12). The surrogate the step optimised, Σ advantage × change in log-likelihood over all sampled tokens, went from 0 to −5.7 nats, i.e. slightly down. The mean change is 0.0006 nats per token (NLL 0.4395 → 0.4401). Individual tokens moved much more than that (mean |change| 0.033 nats per token, 30% of tokens moved by more than 0.01), but the moves cancel within each rollout.

This change is about 50 times smaller than the drop in sampling entropy from step 1 to nolib1 (0.436 → 0.407, −0.029 nats per token), and of the opposite sign on the step-1 tokens. The entropy drop therefore does not come from the update making step-1 behaviour more likely; it reflects the different samples and contexts of nolib1 (whether the update sharpened the policy on its own new samples is not determinable from this check).

## Per rollout (training prompt, with the web/library section)

Totals are summed over the editor's sampled tokens; per-token values in parentheses.

- r0, adv −1.060, 86,934 tokens: −27.78 (−0.00032). save_edit turns +5.66 (+0.00059, 9,634 tokens); final reply +0.42 (1,204 tokens); turns issuing generate calls +4.17 (28,877 tokens).
- r1, adv −0.358, 55,916 tokens: −50.89 (−0.00091). save_edit −9.78 (2,385 tokens); final −2.95 (1,325); generate −2.45 (13,753).
- r2, adv +0.443, 117,649 tokens: −86.59 (−0.00074). save_edit +0.40 (5,087); final −5.82 (1,097); generate −3.78 (24,349).
- r3, adv +0.175, 90,846 tokens: −93.41 (−0.00103). save_edit −14.40 (9,841); final +0.47 (935); generate −5.18 (24,996).
- r4, adv +1.711, 87,034 tokens: −45.29 (−0.00052). save_edit +2.22 (2,909); final +1.98 (1,264); generate −12.74 (24,145).
- r5, adv −1.160, 25,016 tokens: −20.31 (−0.00081). No save_edit, final or generate turns.
- r6, adv −0.927, 129,463 tokens: −70.98 (−0.00055). save_edit −6.44 (10,164); final +3.11 (1,016); generate −8.20 (26,135).
- r7, adv +1.177, 85,982 tokens: −8.93 (−0.00010). save_edit +2.32 (8,358); final −5.65 (1,638); generate +8.06 (20,437).

Groups: high (r4, r7) −54.22 in total, −0.00031 per token; low (r0, r5, r6) −119.07, −0.00049 per token; middle (r1, r2, r3) −230.90, −0.00087 per token. The high group fell less per token than the low group, and the per-token correlation with advantage is +0.26, but no rollout became more likely.

By turn type, summed over rollouts:
- (a) save_edit turns, 48,378 tokens: −20.03 (−0.00041 per token); Σ adv × change +7.66; correlation with advantage +0.15.
- (b) final replies, 8,479 tokens: −8.43 (−0.00099); Σ adv × change −8.03; correlation −0.32.
- (c) turns issuing generate calls (problem ids typed inline), 162,692 tokens: −20.13 (−0.00012); Σ adv × change −10.83; correlation −0.22.
- all other turns, 459,291 tokens: −355.59 (−0.00077); Σ adv × change +5.53; correlation +0.03.

## Scored with nolib1's prompt (no web/library section, no web tools)

Same tokens, prompt and tool declaration from `../nolib1/config.json`, status lines without "web calls left". Here the update points the intended way, weakly. NLL 0.4418 → 0.4418 (+0.00007 per token). Totals: r0 −6.54, r1 −52.13, r2 +42.99, r3 −7.56, r4 +0.65, r5 −23.33, r6 −13.05, r7 +11.10. Correlation with advantage +0.47 (per token +0.53, Spearman +0.67); Σ adv × change +96.65. High group +11.75 (+0.00007 per token), low group −42.92 (−0.00018). Final replies: +12.79 in total, correlation +0.75; r7's final reply +7.09, r4's +3.93. save_edit turns −17.25, correlation +0.39.

So the same update reads as a broad decrease in likelihood under the prompt it was trained on, and as a small advantage-aligned change under nolib1's prompt. In both cases r4, the best rollout (+33.6 reward), gains at most 0.65 nats over 87,034 tokens: its probability as a whole is essentially unchanged.

## How exact the rebuild is

- Each rollout was replayed from its transcript through `rl/loop.py`'s renderer (tinker-cookbook 0.6.1, tml-renderers 0.4.1, tinker 0.39.0, the same image definition as the loop), tool declaration (`TOOLS` + web tools, read from `rl/loop.py`), prompt (`config.json`), effort 0.9, status lines and nag messages. Every one of the 599 rebuilt prompts has exactly the logged prompt length, and 593 of 599 replies have exactly the logged number of sampled tokens (678,840 rebuilt vs 678,730 logged).
- Rebuilt by inference, not read from logs: where the log joins thinking and a text part, the split was put at the last blank line (chosen because it reproduces the logged length); the editor's own key order for 7 forward calls (the history re-renders tool JSON with sorted keys; the position of `token_logprobs`/`topk`/`hidden_states`/`mlp_act` in the editor's original order is not logged); the special-token form of 9 malformed replies (verified by length); the "minutes left" number in status lines (estimated from turn times; token count verified).
- Not exact: the two repaired tool calls (r1 turn 14, r5 turn 10; 1 token longer than logged) and two malformed r5 replies (turns 25 and 26, 54 tokens longer) are scored in their re-rendered form. Predicted training sequences: 28 against 30 in `steps.jsonl`, so 2 turns differ from the training data in a way the logs do not show.
- Check against the log: mean NLL of the sampled tokens under the pre-update policy is 0.4395 here and 0.4357 in `steps.jsonl` (the sampler's own log-probabilities). `compute_logprobs` reproduces the sampler's log-probabilities exactly (tested at prompts of 5k, 86k and 183k tokens: difference 0.0), and two fresh LoRAs and a repeat call give identical sums, so there is no numerical noise. The 0.0038 nats-per-token gap (about 2,600 nats) therefore comes from token content that differs from what was sampled while keeping the same lengths; where it is is not determinable from the logs. It is 6 times the measured mean effect of the update, so the conclusions rest on the pre/post comparison on identical sequences, which is exact for these sequences, and on the assumption that the training sequences differ from them only in a few places.
- Not known: why the effect differs so much between the two prompts; why one first Adam step (lr 2e-5, which moves every LoRA B entry by about the learning rate in the sign of its gradient regardless of gradient size) lowered rather than raised its own surrogate. The data are consistent with a step whose sign-normalised size spreads into many low-gradient directions, lowering the likelihood of the editor's samples broadly, with the advantage-aligned part smaller than that.

Spend: about 6M tokens of `compute_logprobs` and 2,000 sampled tokens, roughly $2 at the loop's prefill price ($0.30 per million); Tinker's billing for `compute_logprobs` was not checked.
