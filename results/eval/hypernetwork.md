# Hypernetwork that writes a LoRA per question (snapshot 2026-11-19)

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

## Cost

Training answers about $6.50 (3.6 GPU-hours); two training runs about $5; scoring about $6.50, plus about $13 for a first set of scoring runs whose answers were lost (network-blocked workers could not return them; they now write to the volume). About 2.3e17 FLOPs on Qwen for data and training.
