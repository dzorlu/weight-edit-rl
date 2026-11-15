"""Three figures for the "How to diagnose" section: a two-layer toy network, where a token's logit comes from,
and what an edit changes. Every number is computed here and the identities are asserted.

    python3 notebooks/two_layer_figure.py      # writes notebooks/figs/diag{1,2,3}.svg
Colours are CSS variables (--ink, --muted, --line, --code, --accent, --pos, --neg, --bg) so the page theme applies.
"""
import os
import numpy as np

# ---------------------------------------------------------------- the toy network: D = 2, M = 3 units, V = 3 tokens
h0 = np.array([1.0, 0.0])
x1 = np.array([2.0, 0.0, -1.0]); W1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
x2 = np.array([0.0, 1.0, 3.0]);  W2 = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]])
U = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
TOK = ["Wait", "&lt;/think&gt;", "So"]
m1, m2 = x1 @ W1, x2 @ W2
h1, h2 = h0 + m1, h0 + m1 + m2
logits = h2 @ U
u = U[:, 0]
p0, p1, p2 = h0 @ u, m1 @ u, m2 @ u
s1, s2 = W1 @ u, W2 @ u
c1, c2 = x1 * s1, x2 * s2
assert np.isclose(p0 + p1 + p2, logits[0]) and np.isclose(c1.sum(), p1) and np.isclose(c2.sum(), p2)
# edits on layer 2
W2_zero = W2.copy(); W2_zero[2] = 0.0
W2_dir = W2 - np.outer(W2 @ u, u)                    # remove direction u (|u| = 1) from what layer 2 writes
edits = [("original", W2), ("zero unit 2", W2_zero), ("remove direction u", W2_dir)]
for name, W in edits:
    s = W @ u; c = x2 * s
    assert np.isclose(c.sum(), (x2 @ W) @ u)
assert np.allclose(W2_dir @ u, 0)

f = lambda v: (f"{v:g}" if abs(v - round(v, 2)) < 1e-9 else f"{v:.2f}").replace("-", "−")
vec = lambda v: "[" + ", ".join(f(a) for a in v) + "]"

STYLE = """<style>
text{font-family:-apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--ink,#1d1d1b)}
.s{font-size:13px}.tiny{font-size:11.5px;fill:var(--muted,#5f5e5a)}.num{font-size:11.5px;font-family:ui-monospace,Menlo,monospace}
.code{font-size:13px;font-family:ui-monospace,Menlo,monospace}.h{font-size:14px;font-weight:600}.op{font-size:17px;font-weight:600}
.stream{stroke:var(--muted,#5f5e5a);stroke-width:3}.ar{stroke:var(--muted,#5f5e5a);stroke-width:1.4;fill:none}.hd{fill:var(--muted,#5f5e5a)}
.axis{stroke:var(--muted,#5f5e5a);stroke-width:1}.box{fill:none;stroke:var(--muted,#5f5e5a);stroke-width:1.2}
.sel{fill:none;stroke:var(--accent,#2a78d6);stroke-width:2.2}.addc{fill:var(--bg,#fff);stroke:var(--muted,#5f5e5a);stroke-width:1.4}
.pos{fill:var(--pos,#2a6fc9)}.neg{fill:var(--neg,#d2601f)}.zero{fill:var(--code,#e9e8e4);stroke:var(--line,#d9d7d1);stroke-width:.6}
.gray{fill:var(--muted,#5f5e5a)}
</style>
<defs><marker id="hd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="hd"/></marker></defs>"""


class Svg:
    def __init__(self): self.o = []
    def add(self, s): self.o.append(s)
    def t(self, x, y, s, cls="s", a="start"): self.add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{a}" xml:space="preserve">{s}</text>')
    def arrow(self, x1, y1, x2, y2): self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ar" marker-end="url(#hd)"/>')
    def cell(self, x, y, w, h, v, vmax=3.0):
        k = "pos" if v > 1e-9 else "neg" if v < -1e-9 else "zero"
        op = 0.22 + 0.78 * min(1.0, abs(v) / vmax) if k != "zero" else 1
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" class="{k}" fill-opacity="{op:.2f}"/>')
        self.t(x + w / 2, y + h / 2 + 4, f(v), "num", "middle")
    def hbar(self, x0, y, v, scale, h=18, cls=None):
        k = cls or ("pos" if v > 1e-9 else "neg" if v < -1e-9 else "zero")
        w = abs(v) * scale
        self.add(f'<rect x="{x0 + min(0, v * scale)}" y="{y}" width="{max(w, 1.5)}" height="{h}" rx="2" class="{k}" fill-opacity="0.85"/>')
    def done(self, w, h, title):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img"><title>{title}</title>{STYLE}{"".join(self.o)}</svg>')


# ---------------------------------------------------------------- figure 1: the network
g = Svg()
SX = 200                                                   # x of the residual stream
g.t(SX, 20, "residual stream", "h", "middle")
g.add(f'<rect x="{SX - 62}" y="32" width="124" height="30" rx="5" class="box"/>'); g.t(SX, 52, f"h0 = {vec(h0)}", "code", "middle")
g.t(SX - 70, 51, "embedding", "tiny", "end")
g.add(f'<line x1="{SX}" y1="62" x2="{SX}" y2="472" class="stream"/>')


def draw_layer(y, L, x, W, m, h_after):
    g.t(330, y - 34, f"layer {L} MLP", "h")
    g.add(f'<path d="M {SX} {y + 13} L 324 {y + 13}" class="ar" stroke-dasharray="4 3" marker-end="url(#hd)"/>')
    g.t(266, y + 6, "gate, up", "tiny", "middle")
    g.t(330, y - 16, f"x{L} [3]: activations", "tiny")
    for j, v in enumerate(x):
        g.cell(330 + j * 36, y, 32, 26, v)
        g.t(330 + j * 36 + 16, y + 42, f"unit {j}", "tiny", "middle")
    g.arrow(438, y + 13, 472, y + 13)
    g.add(f'<rect x="478" y="{y - 22}" width="96" height="102" rx="6" class="box"/>')
    g.t(526, y - 28, f"W{L} [3 × 2]", "s", "middle")
    for j in range(3):
        for d in range(2):
            g.cell(488 + d * 40, y - 14 + j * 30, 36, 26, W[j, d], 1.5)
        g.t(578, y - 14 + j * 30 + 17, f"row {j}", "tiny")
    g.arrow(526, y + 80, 526, y + 104)
    g.t(536, y + 100, f"m{L} = x{L} @ W{L} = {vec(m)}", "code")
    g.add(f'<path d="M 526 {y + 114} C 526 {y + 134}, 380 {y + 134}, {SX + 13} {y + 134}" class="ar" marker-end="url(#hd)"/>')
    g.add(f'<circle cx="{SX}" cy="{y + 134}" r="11" class="addc"/>'); g.t(SX, y + 140, "+", "op", "middle")
    g.t(SX - 18, y + 158, f"h{L} = {vec(h_after)}", "code", "end")


draw_layer(120, 1, x1, W1, m1, h1)
draw_layer(300, 2, x2, W2, m2, h2)
g.arrow(SX, 472, SX, 502)
g.add(f'<rect x="{SX - 62}" y="506" width="300" height="96" rx="6" class="box"/>')
g.t(SX + 12, 498, "U [2 × 3]: one column per token", "s")
for k, name in enumerate(TOK):
    g.t(SX - 18 + k * 96 + 22, 524, name, "tiny", "middle")
    for d in range(2):
        g.cell(SX - 18 + k * 96, 530 + d * 30, 44, 26, U[d, k], 1.0)
g.add(f'<rect x="{SX - 22}" y="526" width="52" height="66" rx="4" class="sel"/>')
g.t(SX - 4, 624, f"logits = h2 @ U = {vec(logits)}", "code")
g.t(SX - 4, 644, f"\"Wait\" scores highest, {f(logits[0])}. u = U[:, Wait] = {vec(u)}, outlined.", "tiny")
FIG1 = g.done(780, 660, "A two-layer network: activations x feed each layer's output matrix W; the outputs m are added to the residual stream; U turns the final vector into one logit per token")

# ---------------------------------------------------------------- figure 2: where the "Wait" logit comes from
g = Svg()
g.t(0, 20, "1. h2 is a sum, so its dot product with u is a sum of parts", "h")
g.t(0, 46, "logit(Wait) = h2 · u = h0·u + m1·u + m2·u", "code")
g.t(0, 66, f"             = {f(p0)}  +  {f(p1)}  +  {f(p2)}  =  {f(logits[0])}", "code")
x0, sc = 0, 70
labels = [("embedding part", p0, "zero"), ("layer 1 part", p1, "pos"), ("layer 2 part", p2, "pos")]
xx = x0
for k, (lab, v, cls) in enumerate(labels):
    w = v * sc
    g.add(f'<rect x="{xx}" y="84" width="{w}" height="28" rx="2" class="{cls}" fill-opacity="{0.45 + 0.25 * k}"/>')
    g.t(xx + w / 2, 103, f(v), "num", "middle")
    g.t(max(xx + w / 2, 42), 130 + 15 * (k % 2), lab, "tiny", "middle")
    xx += w
g.t(xx + 10, 103, f"= {f(logits[0])}", "code")
g.t(0, 172, "Each part is one number: m_L and u are both vectors of length 2, and their dot product is a number.", "tiny")

g.t(0, 200, "2. Split each layer part over the layer's units", "h")
g.t(0, 226, "m_L = x_L @ W_L, so   layer part = m_L · u = x_L @ (W_L @ u) = x_L @ s_L", "code")
g.t(0, 248, "s_L = W_L @ u        [3]  each unit's worth to \"Wait\" per unit of activation", "code")
g.t(0, 268, "contrib_L = x_L * s_L  [3]  what each unit added here; sum(contrib_L) = layer part", "code")


def contrib_rows(y, L, x, s, c, part):
    g.t(0, y, f"layer {L}", "h")
    g.t(0, y + 18, f"x{L} = {vec(x)}   s{L} = W{L} @ u = {vec(s)}", "code")
    zero = 120
    for j in range(3):
        yy = y + 32 + j * 26
        g.t(0, yy + 14, f"unit {j}", "tiny")
        g.hbar(zero, yy, c[j], 60)
        g.t(zero + max(c[j], 0) * 60 + 8, yy + 14, f"contrib = {f(x[j])} × {f(s[j])} = {f(c[j])}", "tiny")
    g.add(f'<line x1="{zero}" y1="{y + 28}" x2="{zero}" y2="{y + 112}" class="axis"/>')
    g.t(0, y + 130, f"sum = {f(c.sum())} = layer {L} part", "s")


contrib_rows(300, 1, x1, s1, c1, p1)
contrib_rows(462, 2, x2, s2, c2, p2)
FIG2 = g.done(720, 610, "The Wait logit split into an embedding part and one part per layer, and each layer part split into one contribution per unit")

# ---------------------------------------------------------------- figure 3: what an edit changes (layer 2)
g = Svg()
g.t(0, 20, "An edit changes s = W @ u, so it changes every contribution that uses it", "h")
g.t(0, 42, f"Layer 2 only. x2 = {vec(x2)} stays the same; layer 1 and the embedding still give {f(p0 + p1)}.", "tiny")
colw = 250
for k, (name, W) in enumerate(edits):
    X = k * colw
    s = W @ u; c = x2 * s; part = c.sum(); lg = p0 + p1 + part
    g.t(X, 76, name, "h")
    for j in range(3):
        for d in range(2):
            g.cell(X + d * 40, 88 + j * 30, 36, 26, W[j, d], 1.5)
        g.t(X + 86, 88 + j * 30 + 17, f"row {j}", "tiny")
    if name == "zero unit 2":
        g.add(f'<rect x="{X - 4}" y="{88 + 2 * 30 - 4}" width="84" height="34" rx="4" class="sel"/>')
    if name == "remove direction u":
        g.add(f'<rect x="{X - 4}" y="84" width="44" height="94" rx="4" class="sel"/>')
    g.t(X, 200, f"s2 = W2 @ u = {vec(s)}", "code")
    zero = X + 46
    for j in range(3):
        yy = 214 + j * 24
        g.t(X, yy + 13, f"unit {j}", "tiny")
        g.hbar(zero, yy, c[j], 48, h=16)
        g.t(zero + max(c[j], 0) * 48 + 6, yy + 13, f(c[j]), "num")
    g.add(f'<line x1="{zero}" y1="210" x2="{zero}" y2="286" class="axis"/>')
    g.t(X, 310, f"layer 2 part = {f(part)}", "s")
    g.t(X, 330, f"logit(Wait) = {f(lg)}", "s")
g.t(0, 368, "Zeroing a row sets one entry of s to 0. Removing direction u, W − (W @ u) uᵀ, sets every entry of s to 0:", "tiny")
g.t(0, 386, "layer 2 can no longer change the \"Wait\" logit at all, whatever its activations.", "tiny")
FIG3 = g.done(760, 400, "Two edits to layer 2: zeroing unit 2's row removes its contribution; removing direction u from every row makes s zero for all units")

os.makedirs(os.path.join(os.path.dirname(__file__), "figs"), exist_ok=True)
for i, svg in enumerate((FIG1, FIG2, FIG3), 1):
    open(os.path.join(os.path.dirname(__file__), "figs", f"diag{i}.svg"), "w").write(svg)
print("logits", logits, "parts", p0, p1, p2, "s", s1, s2, "contrib", c1, c2,
      "edits", [(n, float(((x2 @ W) @ u))) for n, W in edits])
