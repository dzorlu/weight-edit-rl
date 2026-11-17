# Gradient-tool ablation, arm B: what went wrong, and fixes for arm C (snapshot 2026-11-17)

Arm B: step-8 editor on `cur4g16-s8best` with the forward tool's read-only `gradient` request. Results (reward-set gain over the starting model): +1.6, 0, 0, 0; MATH-500 mean +1.4. Arm A (no gradient tool): +3.9, +0.4, +1.6, (one session pending). Four sessions per arm cannot separate them. Logs: `/eval/gradabl-{A,B}/` on the `glimmer-runs` volume.

## What happened
- All four B sessions diagnosed the same failure: the starting model's earlier edit amplified `</think>`, so answers open with `</think>` and 31% fall into a `</think></think>…` loop (p ≈ 1.0). Every session read the prompt's rule "do not undo, reverse or 'repair'" the earlier edit as forbidding the obvious fix.
- Only one session (B1) called the gradient tool, 3 times: call 1 passed a Python expression as the text (scored junk, charged a pass); call 2 ran out of GPU memory on a 4,000-token text (documented limit 4,096); call 3 used 1,000 tokens of pure loop with a log-prob objective on `</think>`, which is saturated (p ≈ 0.986), so attributions were near zero. It then built edits without the attribution at α ±1 with no sweep (−20.7, −41, −21) and stopped with 26 of 32 passes left.
- B0's gain (+5.6 on MATH-500) came from a gradient-free difference-of-means direction (answer-commit positions minus loop positions at layer 24, orthogonal to `</think>`), α 1 on 48 writers. B2 read guide item 11 ("needs gradients … not allowed here") as a ban; B3 lost 3 requests to made-up problem ids and stopped with 29 passes left.
- The prompt still said "Gradients are disabled" and RULES (a) "no backward passes, autograd"; the gradient paragraph was appended after HOW THE SESSION RUNS. Output is cut at 6,000 characters although the prompt says 10,000.

## Root causes, ranked
1. The starting model's main failure comes from the earlier edit, which the editor is told not to undo.
2. Texts and problem ids must be pasted inline (6 passes wasted, 8 failed id requests).
3. The prompt contradicts the tool in three places.
4. Gradient output design: saturated log-prob targets give near-zero attributions; layer-23 residual attribution is ≈ 0 by construction (final RMSNorm is scale-invariant); a mislabelled key; "full lists" are top-k only; memory limit lower than documented.
5. No guidance from attribution to an edit.
6. Sessions stop with most of the budget unused.

## Fixes for arm C
1. Prompt: delete "Gradients are disabled"; RULES (a) → "no weight updates from gradients (no optimizer steps, no iterative fitting); the forward tool's gradient request is allowed for diagnosis"; fix guide item 11; move the gradient text into WHAT YOU HAVE and the guide. Clarify that a position-conditional edit that weakens a loop caused by the earlier edit is allowed (or start from Qwen as downloaded).
2. Inputs from files: `texts_file` / `chats_file` (JSON in /work) for teacher-forced and gradient requests; `split` + slice or `problems_file` for generation; echo token counts; make the display limit and the prompt agree.
3. Gradient objectives: `logit_diff` (target vs baseline tokens); positions by condition, e.g. where the next token is the loop's first `</think>`; contrast files returning attribution A − B.
4. Gradient outputs: per-block attention and MLP output × gradient (direct logit attribution, Elhage et al. 2021) instead of residual × gradient; full per-unit arrays saved; fix the key name; state the real memory limit.
5. Kernel helpers (analysis only): `logit_lens(vec)`; `dom(...)` difference of means (Turner et al. 2023; Rimsky et al. 2024); `orth_edit(names, d, alpha)` (Arditi et al. 2024); `rank1(name, key, value, alpha, null_keys)` with keys from mean down_proj inputs at loop-onset vs neutral positions (ROME, Meng et al. 2022) and null-space projection against neutral-text keys (AlphaEdit, Fang et al. 2024).
6. Forward requests: `kl` (both models on a neutral texts file, mean and max KL); `edited_dir` to test saved candidate edits in an α sweep without rebuilding them.
7. A recipe paragraph: contrastive target → localise → check by ablating in a teacher-forced pass → read with the logit lens → build rank-1 → sweep α ∈ {0.25, 0.5, 1, 2} → KL guard → paired generation; warn that boosting one token's readout creates a loop of that token (seen in A and B).
8. Say that unused forward passes should go to more candidates.

## Arm C design
- Arms: A′ (all fixes, no gradient tool) vs C (all fixes + gradient), 6 sessions each, same seeds; starting model Qwen3.5-0.8B as downloaded, or `cur4g16-s8best` with fix 1's clarification.
- Measure per session: MATH-500 gain (separately from reward), gradient calls and successes, whether a gradient-ranked component is in the submitted edit, failed requests, passes used, minutes, KL.
- Before launch: grep each arm's rendered prompt for "disabled", "backward", "not allowed here"; smoke-test a gradient request from a texts file on a 2,048-token loop-onset text (no OOM, non-zero attributions at onset positions); check the tool schema lists the new parameters; dry-run every helper and the `kl` request.
