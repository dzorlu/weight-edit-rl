"""Writes notebooks/localization.ipynb. Run: python3 notebooks/build_localization.py"""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Localization: from a forward pass to each unit's share of a token's logit

Notation, used throughout:

| Symbol | Shape | What it is |
|---|---|---|
| `x_L` | [M] | layer L's MLP activations, one per unit |
| `W_L` | [M, D] | layer L's MLP output matrix; row `j` is what unit `j` writes |
| `h` | [D] | residual stream after the last layer |
| `U` | [D, V] | unembedding matrix; column `t` is token `t` |
| `u` | [D] | `U[:, Wait]`, the column of the token we diagnose |

Part 1 uses toy numbers. Part 2 does the same on Qwen3.5-0.8B with one of its own looping answers.
""")

md("## Part 1. Toy numbers\n\n### 1.1 Forward pass")
code("""
import numpy as np

x = np.array([2.0, 0.0, -1.0])        # [M=3]   activations of units 0, 1, 2
W = np.array([[1.0, 0.0],             # [M=3, D=2]  unit 0 writes [1, 0]
              [0.0, 1.0],             #             unit 1 writes [0, 1]
              [1.0, 1.0]])            #             unit 2 writes [1, 1]
h_in = np.array([1.0, 0.0])           # [D]  what earlier layers wrote

U = np.array([[1.0, 0.0, -1.0],       # [D=2, V=3]  columns: "Wait", "</think>", "So"
              [0.0, 1.0,  0.0]])

m = x @ W                             # [D]  what this layer writes
h = h_in + m                          # [D]
logits = h @ U                        # [V]
print("m =", m, " h =", h, " logits =", logits)
""")

md("""### 1.2 The layer's part of the logit, in matrix form

The layer writes `x @ W`. Its part of the "Wait" logit is that, times the "Wait" column `u`:

```
layer_part = x @ W @ u              # [M] @ [M, D] @ [D]  →  number
           = x @ (W @ u)            # same product, regrouped
s          = W @ u                  # [M, D] @ [D]  →  [M]   one number per unit
contrib    = x * s                  # [M]  elementwise; equals diag(x) @ W @ u
layer_part = contrib.sum()
```

`s[j]`: how much unit `j`'s row raises the "Wait" logit per unit of activation. `contrib[j]`: how much it actually raised it here.""")
code("""
u = U[:, 0]                           # [D]   "Wait" column of U

layer_part = x @ W @ u                # number
s = W @ u                             # [M]
contrib = x * s                       # [M]

print("layer_part =", layer_part)
print("s =", s, "  contrib = x * s =", contrib, "  sum =", contrib.sum())
print("diag(x) @ W @ u =", np.diag(x) @ W @ u)
""")

md("### 1.3 The link to a backward pass\n\n`s` is the derivative of the layer's part of the logit with respect to `x`. Autograd gives the same numbers. Nothing is trained; it is only the chain rule through one matrix.")
code("""
import torch
xt = torch.tensor(x, requires_grad=True)
layer_part = (xt @ torch.tensor(W)) @ torch.tensor(u)
layer_part.backward()
print("autograd d(logit)/dx =", xt.grad.numpy(), " s =", s)
""")

md("## Part 2. Qwen3.5-0.8B\n\n### 2.1 Load the model and one looping answer\n\nThe first run downloads about 1.7 GB.")
code("""
import json, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

name = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32).eval()
lm = model.model.language_model                 # the text model: embed_tokens, layers, norm
print(type(lm).__name__, "layers:", len(lm.layers))

ex = json.load(open("data/looping_answer.json"))
chat = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
text = chat + ex["text"].removeprefix("<think>")
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :1200]
print("tokens:", ids.shape[1])
print(tok.decode(ids[0, -200:]))
""")

md("### 2.2 Forward pass, recording activations\n\n`x_L` is the input of each layer's `mlp.down_proj`. A forward hook copies it. In the file `down_proj.weight` has shape [D, M]; our `W_L` is its transpose, [M, D].")
code("""
acts = {}
hooks = [layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out, L=L: acts.__setitem__(L, inp[0][0]))
         for L, layer in enumerate(lm.layers)]
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True)
for hk in hooks: hk.remove()

T, M = acts[0].shape
D = lm.layers[0].mlp.down_proj.weight.shape[0]
print("x_L per layer:", (T, M), " D =", D, " hidden states:", len(out.hidden_states))
""")

md("### 2.3 The \"Wait\" column `u` and the logit, by hand\n\nThe final norm divides `h` by its root mean square and multiplies by `(1 + gamma)`. The unembedding is the input embedding, reused. So `u = E[Wait] * (1 + gamma)` and `logit = (h @ u) / rms(h)`. Check it against the model's own logits.")
code("""
wait_ids = [tok.convert_tokens_to_ids(t) for t in ("Wait", "ĠWait")]
print("token ids:", wait_ids, [tok.decode([i]) for i in wait_ids])

E = lm.embed_tokens.weight.detach()                    # [V, D]
gamma = lm.norm.weight.detach()                        # [D]
h_last = out.hidden_states[-1][0]                      # [T, D]  (some versions store h after the norm; see the check)
eps = lm.norm.variance_epsilon if hasattr(lm.norm, "variance_epsilon") else 1e-6

def rms(h): return torch.sqrt((h ** 2).mean(-1, keepdim=True) + eps)

# find h before the final norm: rebuild it from the last layer
h_pre = out.hidden_states[-1][0]
u = E[wait_ids[1]] * (1 + gamma)                       # [D]   " Wait" (with a leading space)
by_hand = (h_pre @ u) / rms(h_pre)[:, 0]
model_logit = out.logits[0, :, wait_ids[1]]
print("max |by hand - model| over all positions:", (by_hand - model_logit).abs().max().item())
""")

md("If that difference is not near zero, `hidden_states[-1]` already has the norm applied in this version. The next cell then uses the input of the norm instead.")
code("""
if (by_hand - model_logit).abs().max() > 1e-2:
    cap = {}
    hk = lm.norm.register_forward_hook(lambda mod, inp, out_: cap.__setitem__("h", inp[0][0]))
    with torch.no_grad(): out = model(input_ids=ids)
    hk.remove()
    h_pre = cap["h"]
    by_hand = (h_pre @ u) / rms(h_pre)[:, 0]
    print("using the norm's input; max |by hand - model|:", (by_hand - out.logits[0, :, wait_ids[1]]).abs().max().item())
""")

md("### 2.4 The positions that matter\n\nPositions where the next token is \" Wait\" or \"Wait\".")
code("""
nxt = ids[0, 1:]
pos = [p for p in range(T - 1) if nxt[p].item() in wait_ids]
print(len(pos), "positions; first few contexts:")
for p in pos[:4]:
    print(repr(tok.decode(ids[0, max(0, p - 12):p + 2])))
""")

md("### 2.5 Each unit's share, every layer\n\n`s_L = W_L @ u` is one number per unit. The share at a position is `x_L[p, j] * s_L[j] / rms(h[p])`. Sum over the chosen positions.")
code("""
r = rms(h_pre)[:, 0]                                   # [T]
table = torch.zeros(len(lm.layers), M)                 # [24, M]
for L, layer in enumerate(lm.layers):
    W_L = layer.mlp.down_proj.weight.detach().T        # [M, D]
    s_L = W_L @ u                                      # [M]
    table[L] = (acts[L][pos] * s_L / r[pos, None]).sum(0)

layer_total = table.sum(1)
print("each layer's MLP share of the 'Wait' logit, summed over", len(pos), "positions:")
for L in range(len(lm.layers)): print(f"  layer {L:2d}  {layer_total[L].item():+8.2f}")
top = torch.topk(table.flatten(), 10)
print("largest single units (layer, unit, share):")
for v, i in zip(top.values, top.indices): print(f"  layer {i.item() // M:2d}  unit {i.item() % M:4d}  {v.item():+.2f}")
""")

md("### 2.6 When does the top unit fire?")
code("""
import matplotlib.pyplot as plt
L0, j0 = divmod(top.indices[0].item(), M)
a = acts[L0][:, j0].numpy()
plt.figure(figsize=(11, 3)); plt.plot(a, lw=0.8, label=f"x_{L0}[:, {j0}]")
for p in pos: plt.axvline(p, color="tab:orange", alpha=0.25, lw=1)
plt.xlabel("token position"); plt.ylabel("activation"); plt.title("orange lines: positions right before 'Wait'"); plt.legend(); plt.show()
for p in sorted(range(T), key=lambda q: -abs(a[q]))[:5]:
    print(f"{a[p]:+.2f}", repr(tok.decode(ids[0, max(0, p - 10):p + 1])))
""")

md("### 2.7 The backward-pass view\n\n`s_L / rms` is the derivative of the logit through layer L's direct path only. The full gradient also goes through every later layer. Compare the two for the top layer: where they agree, the direct path explains the unit; where they differ, its effect runs through later layers, and only an edit test can tell.")
code("""
xs = {}
def keep(mod, inp, out_, L):
    inp[0].retain_grad(); xs[L] = inp[0]
hooks = [layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out_, L=L: keep(mod, inp, out_, L)) for L, layer in enumerate(lm.layers)]
out2 = model(input_ids=ids)
for hk in hooks: hk.remove()
out2.logits[0, pos, wait_ids[1]].sum().backward()

g = xs[L0].grad[0, pos]                                # [P, M]  full gradient
direct = (lm.layers[L0].mlp.down_proj.weight.detach().T @ u)[None, :] / r[pos, None]
print("layer", L0, " corr(full gradient, direct path) over units:",
      torch.corrcoef(torch.stack([g.sum(0), direct.sum(0)]))[0, 1].item())
print("unit", j0, " full:", g.sum(0)[j0].item(), " direct:", direct.sum(0)[j0].item())
""")

md("""
### 2.8 What to try next

- Change `wait_ids[1]` to the end-of-thinking token `tok.convert_tokens_to_ids("</think>")` and repeat 2.5. Which layers lower it?
- Zero row `j0` of `W_L0` (`lm.layers[L0].mlp.down_proj.weight[:, j0] = 0` under `torch.no_grad()`), rerun 2.2 and 2.3, and see how much the "Wait" logit drops at the chosen positions.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/localization.ipynb", "w"), indent=1)
print("wrote notebooks/localization.ipynb", len(cells), "cells")
