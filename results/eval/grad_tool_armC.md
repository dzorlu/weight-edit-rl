# Gradient ablation, arm C: full weight gradients, gradient edits allowed (snapshot 2026-11-17)

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

Reading: arm C is ahead by 14 points on MATH-500 and 16 on the reward set, with 4 sessions per arm. The two sessions that used gradients are the two best of the eight. The two arm-C sessions that did not use gradients (mean +15.8) are also above arm A (mean +7.7), so part of the gap is session-to-session variation, not the gradient request: the same editor on the same starting model scored +23.8 (4 sessions, 120-minute cap) on 16 November without gradients. Both gradient sessions still fix the same failure, the missing `</think>`.

The fine-tuning baseline (`harness/sft.py`, no editor) is reported in `sft_baseline.md` when it finishes.
