# Hypernetwork that writes a LoRA per question (snapshot 2026-11-19)

Code: `harness/hyper.py`. Plan and independent review: `results/proposals/hypernetwork_ablation.md`. Runs: `/runs/hyper/hyper1`, `/runs/hyper/static1`, `/runs/hyper/_original`.

## Setup

- Inner model: Qwen3.5-0.8B as downloaded, frozen.
- Hypernetwork (59.25M trainable): the question is run through frozen Qwen; the activations going into each of the 24 layers [24, N, 1024] are projected to width 512 and pooled by 8 learned queries to [24, 8, 512]; a per-layer head (512 → 4,608) writes one row of A [8, 3584] and one column of B [1024, 8] per query: a rank-8 LoRA on every `mlp.down_proj`.
- Training (the "use the base model's loss" scheme): cross-entropy of one of Qwen's own correct, finished answers with the LoRA applied, backpropagated through frozen Qwen into the hypernetwork. 3 passes, 591 steps of 4 answers, learning rate 1e-4.
- Control `static`: the pooled vectors are a learned constant, so the question is ignored: one fixed LoRA of the same shape, same data, same steps.
- Data: 14,928 answers from Qwen (3,732 practice problems outside select and confirm × 4 samples, 6,000-token cap; 87M tokens); 835 correct and finished (5.6%), from 491 problems; about 1/16 of the problems held out for the validation loss.
- Scoring: MATH-500, first 250 problems × 1 answer, 8,192-token cap, generated in PyTorch with one LoRA per question (same sampling settings as the harness; different seeds from the vLLM scores elsewhere, so unmodified Qwen is scored through the same path).

## Results

| | Loss on held-out finished answers | MATH-500 (250 answers) | Answers cut at the cap |
|---|---|---|---|
| Qwen as downloaded (same generation path) | 0.3676 | 9.6% | 91.6% |
| Fixed LoRA (`static1`) | 0.3148 | 15.6% | 85.2% |
| Hypernetwork, each question's own activations (`hyper1`) | 0.3150 | 16.4% | 84.0% |

Gains over Qwen through the same path: fixed LoRA +6.0, hypernetwork +6.8 (standard error of each accuracy about 2 points at 250 answers). For reference, on the same 250 problems with 2 answers each through vLLM (original 7.8%): fine-tuning all block tensors +23.6; untrained editor +16.6 per rollout, +26.3 best of 3; trained editor +20.9 per rollout, +30.5 best of 3 (`SUMMARY.md`).

## Reading

- The hypernetwork does not use the question: its held-out loss equals the fixed LoRA's at every pass (0.3176/0.3155/0.3150 against 0.3174/0.3155/0.3148) and its MATH-500 score is within noise of it (16.4% against 15.6%). With one failure shared by all questions, a constant update is the optimum.
- Both gain far less than fine-tuning on the same kind of data (+6 to +7 against +23.6). They differ from that baseline in several ways at once: a rank-8 update on `down_proj` only against 279 full tensors, 835 answers against 122, 591 steps at one learning rate with no grid and no checkpoint selection on generated answers. Which of these accounts for the gap is not separated here.
- The junk-input scoring (another question's activations) was not run: the loss comparison already shows no dependence on the question.

## Cost

Training answers about $6.50 (3.6 GPU-hours); two training runs about $5; scoring about $6.50, plus about $13 for a first set of scoring runs whose answers were lost (network-blocked workers could not return them; they now write to the volume). About 2.3e17 FLOPs on Qwen for data and training.
