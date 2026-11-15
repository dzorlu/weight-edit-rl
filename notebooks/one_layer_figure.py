"""Figure 1 of "How to diagnose": a one-layer network with the same x, W and u as figures 2 and 3.

    python3 notebooks/one_layer_figure.py      # writes notebooks/figs/diag1.svg
Colours are CSS variables, so the page theme applies. Every number is computed and checked here.
"""
import os
import numpy as np

h0 = np.array([0.5, 0.0, 0.0])
x = np.array([1.0, 2.0, 1.5, 0.5])
W = np.array([[0.5, 0, 1], [-1, 0.5, 0], [2, 1, 0.5], [0, -1, 1]])
U = np.array([[1.0, 0, -1], [0.5, 1, 0], [0, 0, 0.5]])
TOK = ["Wait", "&lt;/think&gt;", "So"]
m = x @ W; h = h0 + m; logits = h @ U; u = U[:, 0]
assert np.allclose(u, [1, 0.5, 0]) and np.isclose(m @ u, 2.5) and np.argmax(logits) == 0
f = lambda v: f"{v:g}".replace("-", "−")
vec = lambda v: "[" + ", ".join(f(a) for a in v) + "]"

o = []
def t(x_, y, s, cls="s", a="start"): o.append(f'<text x="{x_}" y="{y}" class="{cls}" text-anchor="{a}" xml:space="preserve">{s}</text>')
def arrow(x1, y1, x2, y2): o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ar" marker-end="url(#hd)"/>')
def cell(x_, y, v, vmax=2.0, w=38, h=26):
    k = "pos" if v > 1e-9 else "neg" if v < -1e-9 else "zero"
    op = 0.22 + 0.78 * min(1, abs(v) / vmax) if k != "zero" else 1
    o.append(f'<rect x="{x_}" y="{y}" width="{w}" height="{h}" rx="3" class="{k}" fill-opacity="{op:.2f}"/>')
    t(x_ + w / 2, y + h / 2 + 4, f(v), "num", "middle")

SX = 90                                     # residual stream
t(SX, 18, "residual stream", "h", "middle")
o.append(f'<rect x="{SX - 70}" y="28" width="140" height="30" rx="5" class="box"/>'); t(SX, 48, f"h0 = {vec(h0)}", "code", "middle")
o.append(f'<line x1="{SX}" y1="58" x2="{SX}" y2="318" class="stream"/>')
# x
Y = 100
t(200, Y - 14, "x [4]", "s", "middle")
for j, v in enumerate(x):
    cell(181, Y + j * 30, v, 2.0)
    t(172, Y + j * 30 + 17, f"unit {j}", "tiny", "end")

arrow(225, Y + 55, 262, Y + 55)
# W
t(332, Y - 14, "W [4 × 3]: one row per unit", "s", "middle")
o.append(f'<rect x="266" y="{Y - 6}" width="136" height="128" rx="6" class="box"/>')
for j in range(4):
    for d in range(3):
        cell(273 + d * 42, Y + j * 30, W[j, d])
arrow(402, Y + 55, 440, Y + 55)
t(446, Y + 51, "m = x @ W", "code"); t(446, Y + 69, f"  = {vec(m)}", "code")
o.append(f'<path d="M 470 {Y + 80} C 470 {Y + 190}, 300 {Y + 190}, {SX + 13} {Y + 190}" class="ar" fill="none" marker-end="url(#hd)"/>')
o.append(f'<circle cx="{SX}" cy="{Y + 190}" r="11" class="addc"/>'); t(SX, Y + 196, "+", "op", "middle")
t(SX + 18, Y + 214, f"h = h0 + m = {vec(h)}", "code")
arrow(SX, 318, SX, 344)
# U
UY = 360
t(SX + 10, UY - 8, "U [3 × 3]: one column per token", "s")
o.append(f'<rect x="{SX - 70}" y="{UY}" width="250" height="122" rx="6" class="box"/>')
for k, name in enumerate(TOK):
    t(SX - 40 + k * 80 + 19, UY + 18, name, "tiny", "middle")
    for d in range(3):
        cell(SX - 40 + k * 80, UY + 26 + d * 30, U[d, k], 1.0)
o.append(f'<rect x="{SX - 44}" y="{UY + 22}" width="46" height="96" rx="4" class="sel"/>')
t(SX - 70, UY + 140, "u = U[:, Wait] = " + vec(u) + ", the outlined column", "tiny")
# logits as their own vector box, below U
LY = UY + 222
arrow(SX + 55, UY + 152, SX + 55, LY - 24)
t(SX + 66, UY + 181, "h @ U", "code")
t(SX - 70, LY - 8, "logits [3]: one per token", "s")
o.append(f'<rect x="{SX - 70}" y="{LY}" width="250" height="80" rx="6" class="box"/>')
for k, name in enumerate(TOK):
    t(SX - 46 + k * 80 + 25, LY + 18, name, "tiny", "middle")
    cell(SX - 46 + k * 80, LY + 26, logits[k], 3.0, w=50)
o.append(f'<rect x="{SX - 50}" y="{LY + 22}" width="58" height="34" rx="4" class="sel"/>')
t(SX - 70, LY + 108, f"h [3] @ U [3 × 3] = logits [3] = {vec(logits)}", "code")
t(SX - 70, LY + 130, f"\"Wait\" scores highest. Its logit is h · u = h0·u + m·u = {f(h0 @ u)} + {f(m @ u)} = {f(logits[0])}.", "tiny")
t(SX - 70, LY + 148, f"m·u = {f(m @ u)} is the layer's part. Figure 2 splits it over the units and weights.", "tiny")

STYLE = open(os.path.join(os.path.dirname(__file__), "two_layer_figure.py")).read().split('STYLE = """', 1)[1].split('"""', 1)[0]
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 730" role="img"><title>One layer: activations x times W give m, added to the residual stream; U turns the result into one logit per token</title>'
       f'{STYLE}{"".join(o)}</svg>')
os.makedirs(os.path.join(os.path.dirname(__file__), "figs"), exist_ok=True)
open(os.path.join(os.path.dirname(__file__), "figs", "diag1.svg"), "w").write(svg)
print("m", m, "h", h, "logits", logits)
