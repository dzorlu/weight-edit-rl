# Gradient ablation, arm C: full weight gradients, gradient edits allowed (snapshot 2026-11-18)

Editor: the step-8 editor of run `cur4g16` (`tinker://80fdacdc-ec4e-5c2e-afaf-de75995e07e4:train:0/sampler_weights/cur4g16-step8-keep`), no training. Starting model: Qwen3.5-0.8B as downloaded (MATH-500 6.6%). 4 sessions per arm, 90-minute cap, 2 sessions per GPU, 32 forward passes. Gains are over the starting model: reward set (256 answers) and MATH-500 (first 250 problems × 2). Runs: `/runs/eval/gradC-armA`, `/runs/eval/gradC-armC`.

- Arm A: current tools. Both arms can pass texts from a file in /work (`texts_file`).
- Arm C: the same plus the forward tool's `gradient` request: G = d(target)/dW for all 318 block tensors saved to /work, with the top singular pair (u, v) of each 2-D gradient; up to 8 texts of 8,192 tokens. The prompt allows edits built from gradients. The request does not change the weights.

| | Arm A (current tools) | Arm C (+ weight gradients) |
|---|---|---|
| MATH-500 gain per session | +7.8, +12.0, −0.2, +11.0 (mean +7.7, SE 2.8) | +24.8, +31.2, +17.2, +14.4 (mean +21.9, SE 3.8) |
| reward-set gain per session | +7.8, +14.8, +3.9, +10.9 (mean +9.4) | +30.5, +32.4, +18.4, +20.7 (mean +25.5) |
| answers cut at the 8,192-token cap (start 93%) | 85, 79, 90, 81% | 55, 40, 71, 75% |
| forward passes used | 21, 18, 9, 17 | 12, 9, 8, 6 |
| sessions that called the gradient request | – | 2 of 4 (r0, r1), one call each |

How the gradients were used:
- C r0 (+24.8): zeroed about 27 "Wait" MLP units found by logit attribution, then took the gradient of the `</think>` logit at the 8 positions right before `</think>` in finished answers, on the edited model, and added a rank-1 gradient edit `W += 0.01·outer(u, v)` (top singular pair of G) to 14 writer matrices. Doses 0.02 and 0.5 made the model close thinking too early and were rejected by test.
- C r1 (+31.2): took the same gradient on the original model, found that the rank-1 output direction of the layer-23 `down_proj` gradient has cosine 0.97 with the `</think>` readout direction, and then applied the readout amplification `W += α·outer(d, d @ W)` by hand to 13 writer matrices. The gradient confirmed the direction and picked the layers; the edit itself is not a gradient step.
- C r2 (+17.2), C r3 (+14.4): no gradient request; readout amplification on 48 writers, and 11 zeroed "Wait" units.
- No session took a plain gradient step `W + ηG` on full matrices or used the `loss` target.

Arm A's edits: zeroed "Wait" units in 1–8 tensors (r0, r1, r3) and a small edit on 5 tensors (r2).

## Rerun with 8 sessions per arm, 120-minute cap (`/runs/eval/gradC8-armA`, `/runs/eval/gradC8-armC`)

Same editor, starting model and scoring. Harness fixes for both arms before this run: tests named by split range, texts up to 9,216 tokens, one-line errors for malformed requests (no test failed on a mistyped id; at most 2 failed requests per session).

| | Arm A (current tools) | Arm C (+ weight gradients) |
|---|---|---|
| MATH-500 gain per session | +35.2, +29.8, +22.6, +29.2, +19.6, +25.6, +5.0, 0.0 (mean +20.9, SE 4.4) | +34.8, +10.2, +23.0, +9.0, +14.0, +19.6, −0.6, +29.2 (mean +17.4, SE 4.1) |
| reward-set gain, mean | +22.8 | +18.1 |
| sessions that called the gradient request | – | 2 of 8 (r5, r6), one call each |

- C r5 (+19.6): gradient of the `</think>` logit at the closing position of 6 finished answers; the consensus of the per-tensor top singular vectors u has cosine 0.89 with the `</think>` readout; the edit is the hand-built amplification `W += 0.5·outer(d, d @ W)` along that consensus direction on 18 writers.
- C r6 (−0.6): the only plain gradient step in either run: `W + ηG`, η = 1e-7, on the 165 tensors with the largest gradient, for the same target. Reward 0.0; 93% of answers still cut at the cap (start 93%).
- Arm A r7 submitted no edit (0.0).

Both runs pooled (12 sessions per arm): arm A mean +16.5, arm C mean +18.9. Within arm C, the 4 sessions that called the gradient request average +18.8 (+24.8, +31.2, +19.6, −0.6) and the 8 that did not average +19.0.

Reading: the gradient request does not change the editor's result. It is called in 4 of 12 sessions, once each, always for the `</think>` logit; three of those used it to find or confirm the `</think>` readout direction, which sessions without gradients find from the embedding matrix directly. The 14-point gap of the first run was variation between sessions. Fine-tuning without an editor (+23.6 ± 2.2) is at the level of either arm's mean (+20.9 and +17.4 here).

## Reading of the first run (4 sessions per arm, 90-minute cap), superseded by the rerun above

Arm C is ahead by 14 points on MATH-500 and 16 on the reward set, with 4 sessions per arm. The two sessions that used gradients are the two best of the eight. The two arm-C sessions that did not use gradients (mean +15.8) are also above arm A (mean +7.7), so part of the gap is session-to-session variation, not the gradient request: the same editor on the same starting model scored +23.8 (4 sessions, 120-minute cap) on 16 November without gradients. Both gradient sessions still fix the same failure, the missing `</think>`.

Fine-tuning baseline, no editor (`sft_baseline.md`): +23.6 ± 2.2 on the same 250 problems.
