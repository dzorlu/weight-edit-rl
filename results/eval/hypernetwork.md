# Hypernetwork that writes a LoRA per question (snapshot 2026-11-20)

Code: `harness/hyper.py`. Plan and independent review: `results/proposals/hypernetwork_ablation.md`. Runs: `/runs/hyper/hyper1`, `/runs/hyper/static1`, `/runs/hyper/_original`.

## Setup

- Inner model: Qwen3.5-0.8B as downloaded, frozen.
- Hypernetwork (59.25M trainable): the question is run through frozen Qwen; the activations going into each of the 24 layers [24, N, 1024] are projected to width 512 and pooled by 8 learned queries to [24, 8, 512]; a per-layer head (512 → 4,608) writes one row of A [8, 3584] and one column of B [1024, 8] per query: a rank-8 LoRA on every `mlp.down_proj`.
- Training (the "use the base model's loss" scheme): cross-entropy of one of Qwen's own correct, finished answers with the LoRA applied, backpropagated through frozen Qwen into the hypernetwork. 3 passes, 591 steps of 4 answers, learning rate 1e-4.
- Control `static`: the pooled vectors are a learned constant, so the question is ignored: one fixed LoRA of the same shape, same data, same steps.
- Data: 14,928 answers from Qwen (3,732 practice problems outside select and confirm × 4 samples, 6,000-token cap; 87M tokens); 835 correct and finished (5.6%), from 491 problems; about 1/16 of the problems held out for the validation loss.
- Scoring: MATH-500, first 250 problems × 1 answer, 8,192-token cap, generated in PyTorch with one LoRA per question (same sampling settings as the harness; different seeds from the vLLM scores elsewhere, so unmodified Qwen is scored through the same path).

## How it works

Sizes for Qwen3.5-0.8B: 24 layers, hidden 1024, `mlp.down_proj` is [1024, 3584]. Hypernetwork width 512, rank 8.

```
HyperNet(                                              # 59.25M trainable
  proj     : Linear(1024 -> 512) + LayerNorm(512)      # shared across layers
  queries  : Parameter[8, 512]                         # 8 learned queries, shared
  mlp      : Linear(512 -> 2048) -> SiLU -> Linear(2048 -> 512)   # shared, residual
  out_norm : LayerNorm(512), then a 1/512 scale
  head     : Parameter[24, 512, 4608]                  # one Linear(512 -> 4608) per Qwen layer
)
```

One training step (`harness/hyper.py::train_hyper`):

```python
q_ids  : [N]        # the question, in the chat template
qa_ids : [T]        # the question followed by one of Qwen's own correct, finished answers

# 1. read the question with frozen Qwen: no gradient, no LoRA
X = Qwen.hidden_states(q_ids)[0:24]                    # [24, N, 1024]  activations going into each layer

# 2. hypernetwork (the only trainable part)
H   = norm(proj(X))                                    # [24, N, 512]
att = softmax(queries @ H.T / sqrt(512))               # [24, 8, N]
P   = att @ H                                          # [24, 8, 512]   8 pooled vectors per layer
P   = out_norm(P + mlp(P)) / 512                       # [24, 8, 512]
O   = einsum('lrd,ldk->lrk', P, head)                  # [24, 8, 4608]
A, B = O[:, :, :3584], O[:, :, 3584:].transpose(1, 2)  # A [24, 8, 3584], B [24, 1024, 8]

# 3. run frozen Qwen on question + answer with the LoRA applied
#    inside layer l's MLP, for x [T, 3584]:  out = x @ W_l.T + (x @ A[l].T) @ B[l].T      -> [T, 1024]
logits = Qwen(qa_ids, lora=(A, B))                     # [T, vocabulary]
loss   = cross_entropy(logits at the answer positions, the answer tokens)

# 4. backward: the gradient goes from the loss through Qwen's 24 layers to A and B,
#    and from there into head, mlp, queries and proj. Qwen's own weights are frozen: used, never changed.
loss.backward()
optimizer.step()                                       # updates the hypernetwork only
```

Two passes through Qwen per training step; the gradient runs through the second only:

```
pass 1:  question → Qwen → X [24, N, 1024] ──────┐   (no gradient; X is fixed data)
                                                 ▼
                                           hypernetwork → A [24, 8, 3584], B [24, 1024, 8]
                                                 │
pass 2:  question + answer → Qwen + (A, B) → loss
                                                 ▲
         loss.backward(): loss → Qwen's 24 layers → A, B → hypernetwork
```

`X[l]` holds the vectors going into layer `l + 1` (index 0 is the token embeddings); the LoRA of a layer is written from the vectors going into that same layer.

| | Pass 1 | Pass 2 |
|---|---|---|
| Input | the question | the question + the answer |
| LoRA applied | no | yes |
| Purpose | `X`, the hypernetwork's input | the loss |
| Gradient | none | `loss.backward()` runs through this pass |

At test time: steps 1 and 2 on the held-out question, then Qwen generates its answer with (A, B) applied. The `static` control replaces `P` in step 2 with a learned Parameter[24, 8, 512], so the question is not read.

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

## Fixed LoRA, tuned (`harness/hyper.py::lora_tune`, runs `/runs/hyper/lora-*`, summary `/runs/hyper/lora_tune.json`)

Question: is the small gain above a matter of the update's shape and of tuning? A plain LoRA (A and B trained directly) on the same 835 answers (785 for training), the same loss and backward pass through frozen Qwen, 2 passes (394 steps of 4 answers). Each pass's merged weights are scored on 128 select problems × 1 answer through vLLM (Qwen as downloaded: 7.0%), and the best of the 8 checkpoints on MATH-500 with `heldout_eval`.

| Config | Trainable | Held-out loss, pass 1 / 2 (no LoRA 0.3676) | Select accuracy, pass 1 / 2 | Cut at the cap, pass 1 / 2 |
|---|---|---|---|---|
| rank 32, `down_proj` + attention output (48 matrices), lr 1e-4 | 5.9M | 0.3135 / 0.3120 | 18.8 / 23.4 | 81 / 77% |
| same, lr 3e-4 | 5.9M | 0.3160 / 0.3153 | 14.1 / 18.8 | 84 / 80% |
| same, lr 1e-3 | 5.9M | 0.3457 / 0.3304 | **31.3** / 18.0 | 63 / 80% |
| rank 8, `down_proj` only (24 matrices), lr 1e-3 | 0.9M | 0.3170 / 0.3145 | 18.8 / 18.8 | 81 / 80% |

Best checkpoint (rank 32, lr 1e-3, pass 1) on MATH-500, first 250 problems × 2, paired with the cached original: 7.8% → 23.2%, **+15.4 ± 1.9** (84 answers wrong → right, 7 right → wrong); answers cut at the cap 94.4% → 74.6%.

Reading:
- Tuning and checkpoint selection move the fixed LoRA from about +6 (first run, one learning rate, last checkpoint, PyTorch scoring) to +15.4. That is at the level of one untrained-editor rollout (+16.6) and 8 points below fine-tuning all block tensors (+23.6).
- The held-out loss does not predict accuracy: the best checkpoint on select has the worst loss of the eight (0.3457). Checkpoints have to be chosen on generated answers.
- The grid is noisy (128 answers per cell, standard error about 4 points): seven of eight cells lie between 14 and 23%, and the best cell's 31.3% on select became 23.2% on MATH-500. The small shape (rank 8, `down_proj` only) at the higher learning rate is within noise of the rank-32 runs at lower rates, so this grid does not show that the larger shape matters.

## Qwen3.5-2B (runs `/runs/hyper/hyper2b`, `/runs/hyper/lora2b-*`, summary `/runs/hyper/lora2b_tune.json`)

Base model changed to Qwen3.5-2B (reasons: CHANGELOG, 20 November). Sizes: 24 layers, hidden 2048, `mlp.down_proj` [2048, 6144]; the hypernetwork writes A [24, 8, 6144] and B [24, 2048, 8] and has 103.8M trainable parameters. Data: the 2B's own answers to 3,732 practice problems × 1 sample at a 6,000-token cap (18.4M tokens); 1,280 correct and finished (34%); 1,275 used for training after the validation problems are held out. Settings from the 0.8B grid: learning rate 1e-3, 2 passes (638 steps of 4 answers), a checkpoint after each pass.

| | Trainable | Held-out loss, pass 1 / 2 (no LoRA 0.2842) | Select accuracy, pass 1 / 2 (128 answers) |
|---|---|---|---|
| Hypernetwork (`hyper2b`) | 103.8M | 0.2395 / 0.2389 | not scored |
| Fixed LoRA, rank 8 on `down_proj` (the hypernetwork's output shape) | 1.6M | 0.2430 / 0.2426 | 53.9 / 55.5 |
| Fixed LoRA, rank 32 on 48 matrices | 9.4M | 0.2797 / 0.2635 | 51.6 / 55.5 |

Best fixed LoRA (rank 8, pass 2) on MATH-500, first 250 problems × 2, 8,192-token cap, paired with the cached original: 38.0% → 51.2%, **+13.2 ± 1.8** (73 answers wrong → right, 7 right → wrong); answers cut at the cap 62.0% → 49.2%.

References on the same problems: unedited 2B at the 8,192 cap 38.0%, at the 32,768 cap 67.4% (`cap_baseline.md`); editor rollouts on the 2B at 8,192 (2 per editor, 16 November): untrained +26.2 and +12.4, step-8 0 and +15.6.

Reading:
- The fixed LoRA gains about the same on the 2B (+13.2) as on the 0.8B (+15.4), from 1,275 answers instead of 785. It is below the unedited model with a 32,768-token cap (67.4%) and below the best editor rollout (64.2%).
- The hypernetwork's held-out loss is slightly lower than the same-shape fixed LoRA's (0.2389 against 0.2426). On the 0.8B the loss did not predict accuracy, so this does not show that reading the question helps; the per-question scoring that would show it has not been run.
- The two fixed-LoRA shapes are again within noise of each other on select (55.5% both after pass 2).
- Both LoRA training runs logged pass 1 twice: the container was restarted once and the run started over (billed time about 1.5 times the training time).

## Cost

Training answers about $6.50 (3.6 GPU-hours); two training runs about $5; scoring about $6.50, plus about $13 for a first set of scoring runs whose answers were lost (network-blocked workers could not return them; they now write to the volume). About 2.3e17 FLOPs on Qwen for data and training. Billed (Modal, app `weight-edit`): hypernetwork on the 0.8B $57.69 including the lost scoring runs (the dollar figures earlier in this paragraph were estimates at the GPU rate and are about half the billed cost); the fixed-LoRA grid on the 0.8B $12.87; the 2B runs above $18.98.
