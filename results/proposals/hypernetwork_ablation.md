# Proposed ablation: a one-shot hypernetwork against the reasoning editor (plan, 18 November 2026; not built)

## Question

On the same model and failure, does a hypernetwork that writes a weight update in one forward pass, trained by a gradient through the base model, match an LLM editor that measures, tests and revises over many steps?

## What exists (Qwen3.5-0.8B as downloaded, 7.8%; gain on MATH-500, first 250 problems × 2 answers; `results/eval/SUMMARY.md`)

| Route | Qwen FLOPs | MATH-500 gain |
|---|---|---|
| Fine-tuning all block tensors on 122 of Qwen's own correct answers | 2.4e16 (training only; 6.2e16 with data generation and checkpoint scoring) | +23.6 |
| Untrained editor (Inkling 2 Small agent, python sandbox + forward passes), best of 3 rollouts by reward | 6e16 | +26.3 |
| RL-trained editor (8 steps × 16 rollouts), best of 3 rollouts | 6e16 (1.7e18 with its training) | +30.5 |

The failure: about 94% of Qwen's answers run to the 8,192-token cap in a "Wait, ..." loop and never emit `</think>`; answers that finish are almost all correct. Every successful edit so far stops the loop (e.g. `W += α·outer(d, d @ W)` along the `</think>` readout direction on 13–48 writer matrices).

## New arms

**H1. Hypernetwork, supervised through Qwen.**
- Input: the problem text only. It is run through frozen Qwen; the hypernetwork reads the activations `Z` of shape [24 layers, N tokens, 1024].
- Writer (Doc-to-LoRA design, arXiv 2602.15902, Eqs. 5–8): for each layer l, a Perceiver shared across layers (r learned queries cross-attending to `Z[l−1]`, B blocks, width d) gives [r, d]; a per-layer linear head maps each query to one row of `A_l` [r, 3584] and one column of `B_l` [1024, r]; `ΔW_l = B_l @ A_l` on `mlp.down_proj`.
- Sizes considered: small (d 512, r 8, 8 blocks, down_proj only, about 80M trainable, 0.1× Qwen); medium (d 1024, r 16, down_proj + attention output projections, about 290M); large (d 2048, 12 blocks, about 1B).
- Training: for a practice problem with a correct, finished answer from Qwen itself, apply the written LoRA, compute cross-entropy on that answer's tokens, backpropagate through frozen Qwen into the writer. No Tinker; all on Modal.
- Data: Qwen's own answers to the 3,700 practice problems × 4 samples (about 117M generated tokens), keeping the correct, finished ones (expected about 1,000). Practice problems are disjoint from MATH-500; select and confirm splits are left out of training.
- Evaluation: for each MATH-500 problem, the writer reads Qwen's activations on the problem text (no answer), writes a LoRA for that problem, Qwen answers with it. Second check on GSM8K test and the NuminaMath held-out sample.
- Controls: (a) junk input: give the writer another problem's activations at test time (if the score does not drop, the writer is a constant update; AdapterBench reports this check on Text-to-LoRA: +0.029 and −0.001); (b) the fine-tuning row as the fixed-update reference.

**H2. Hypernetwork, reward (only if H1 uses its input).** Same writer; Qwen generates answers with the written update, each answer gets r = +1 (correct) or −1, and the loss is the reward-weighted log-likelihood of the generated tokens, backpropagated through Qwen into the writer (the form of Transformer-Squared, arXiv 2501.06252, Eq. 1).

## Compute estimate for H1 (small)

Data generation 1.8e17 FLOPs (about $6); training on about 1,000 answers × 5 passes 6.8e16 (about $2); three scoring runs with per-problem LoRAs about $7. Total about 2.5e17 FLOPs on Qwen, $15–20: about 4× the fine-tuning route and the editor's 3 rollouts, one seventh of the editor's RL training.

## Open choices

1. Training target: Qwen's own correct answers (default) or KL to a teacher's token distribution (Doc-to-LoRA's objective; teacher candidates: Qwen with "Wait" banned at decoding, 22.9%; the best edited model, about 40%).
2. Writer size, and whether to match compute to the fine-tune or to the RL run.
3. Simple pooling first, or the Perceiver from the start.

## Independent review (18 November 2026)

A reviewer read this plan, the results files, `harness/sft.py` and the texts of Doc-to-LoRA, Text-to-LoRA, Transformer-Squared and AdapterBench. Verified points are from those files; the rest is the reviewer's judgement.

1. Conditioning has nothing it must learn: every successful edit is one fix applied to all problems, and the problem text is also in Qwen's prompt (the setting where AdapterBench finds matched-minus-junk near zero: +0.029, −0.001). The per-layer head biases alone are a fixed rank-8 LoRA. A "no drop" result under the junk-input control cannot separate "nothing to learn" from "too little data". Fix: a fixed LoRA of the same shape trained on identical data and steps (AdapterBench's "matched minus static").
2. Data is 3–4 orders of magnitude short: Doc-to-LoRA 3.2M contexts, 101M triplets, 100K steps, 309M writer; Text-to-LoRA 479 tasks × 128 descriptions, 5–55M writers; this plan about 1,000 answers (nearer 750 after the 6,000-token limit in `sft.py`, which kept 122 of 162). Judgement: an 80M writer will fit a constant; larger writers add nothing.
3. Selection bias: only about 7% of answers finish, so the writer never sees activations of problems Qwen fails; the fine-tune trained this way leaves 69% of answers cut at the cap against 35% for the best edit.
4. Train/test mismatch: teacher-forced loss on finished answers against free generation to 8,192 tokens. Fix: choose checkpoints by the cut-at-cap rate on generated answers.
5. Comparison: editor rows are best of 3 with tests inside the rollout; compare with single-rollout means (+16.6 untrained, +20.9 trained); gaps under about 8 points are unresolved at these sample sizes.
6. Evaluation: 250 distinct LoRAs need per-request adapters or batched hooks in HF generate; the scoring cost is a lower bound.
7. `sft.py` reuse: generated answers are not saved to disk; each sequence runs alone; a failed `gradient_checkpointing_enable` is not fatal.
8. H2 as specified is not sound: about 92% of samples would get −1, and minimising their log-likelihood is unbounded. Transformer-Squared (Eq. 1) adds a KL term to the original model, picks λ on validation, stops early, trains only singular-value scales, and uses a −0.1 reward in one setting "for training stability". Fix: r ∈ {0, 1} with a KL-to-original term.

Suggested first experiment (about 6 GPU-hours, about $11): on the same saved data, a fixed rank-8 LoRA against a mean-pool writer against the junk-input writer, gated on teacher-forced loss over held-out finished answers; generate only if the matched writer beats both controls. Cut: H2, the medium and large writers, the Perceiver, the GSM8K and NuminaMath checks. A planted-fault family (known rank-1 damage; the writer reads the damaged model; KL on short texts to the undamaged one) would force conditioning; about 3–5 GPU-hours, a new build.
