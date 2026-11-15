"""Figure 1 of "How to diagnose": the forward pass of one layer, as one line.

    python3 notebooks/one_layer_figure.py      # writes notebooks/figs/diag1.svg
Same x, W and U as figures 2 and 3. Colours are CSS variables, so the page theme applies.
"""
import os
import numpy as np

h0 = np.array([0.5, 0.0, 0.0])
x = np.array([1.0, 2.0, 1.5, 0.5])
W = np.array([[0.5, 0, 1], [-1, 0.5, 0], [2, 1, 0.5], [0, -1, 1]])
U = np.array([[1.0, 0, -1], [0.5, 1, 0], [0, 0, 0.5]])
m = x @ W; h = h0 + m; logits = h @ U; u = U[:, 0]
assert np.isclose(h0 @ u + m @ u, logits[0]) and np.isclose(m @ u, 2.5)
f = lambda v: f"{v:g}".replace("-", "−")
vec = lambda v: "[" + ", ".join(f(a) for a in v) + "]"

o = []
def t(x_, y, s, cls="s", a="middle"): o.append(f'<text x="{x_}" y="{y}" class="{cls}" text-anchor="{a}" xml:space="preserve">{s}</text>')
def arrow(x1, y1, x2, y2): o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ar" marker-end="url(#hd)"/>')
def vbox(cx, cy, name, v, w, cls="box"):
    o.append(f'<rect x="{cx - w / 2}" y="{cy - 22}" width="{w}" height="44" rx="6" class="{cls}"/>')
    t(cx, cy - 4, name, "s"); t(cx, cy + 14, vec(v), "code")
def obox(cx, cy, label, w):
    o.append(f'<rect x="{cx - w / 2}" y="{cy - 16}" width="{w}" height="32" rx="16" class="op2"/>')
    t(cx, cy + 5, label, "code")

# where x is read: the token before "Wait"
o.append('<rect x="20" y="10" width="196" height="30" rx="5" class="tokbg"/>')
t(30, 30, "… so the answer is 12.", "code", "start")
o.append('<rect x="175" y="14" width="34" height="22" rx="4" class="sel"/>')
t(226, 30, "x is read here, where the next token is \"Wait\"", "tiny", "start")
arrow(160, 40, 160, 76)

TOP, BOT = 100, 190
vbox(110, TOP, "h0 [3]", h0, 130)
o.append(f'<line x1="175" y1="{TOP}" x2="549" y2="{TOP}" class="stream"/>')
t(360, TOP - 8, "residual: passes through unchanged", "tiny")
o.append(f'<circle cx="560" cy="{TOP}" r="11" class="addc"/>'); t(560, TOP + 6, "+", "op", "middle")
arrow(571, TOP, 596, TOP)
vbox(660, TOP, "h [3]", h, 124)
arrow(722, TOP, 746, TOP); obox(775, TOP, "@ U", 54); arrow(802, TOP, 826, TOP)
vbox(896, TOP, "logits [3]", logits, 136)
t(896, TOP + 38, "Wait, &lt;/think&gt;, So", "tiny")
# MLP branch
arrow(110, TOP + 22, 110, BOT - 18)
obox(110, BOT, "gate, up", 92)
arrow(156, BOT, 186, BOT)
vbox(248, BOT, "x [4]", x, 134, "box xbox")
arrow(311, BOT, 332, BOT)
obox(362, BOT, "@ W", 60)
arrow(393, BOT, 406, BOT)
vbox(468, BOT, "m [3]", m, 120)
o.append(f'<path d="M 529 {BOT} L 560 {BOT} L 560 {TOP + 13}" class="ar" fill="none" marker-end="url(#hd)"/>')
t(310, BOT + 44, "MLP: up to 4 units, then back down to 3 through W", "tiny")
t(362, BOT + 62, "figure 2 zooms into this box", "tiny accent")

STYLE = """<style>
text{font-family:-apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--ink,#1d1d1b)}
.s{font-size:12.5px;fill:var(--muted,#5f5e5a)}.tiny{font-size:11.5px;fill:var(--muted,#5f5e5a)}.accent{fill:var(--accent,#2a78d6)}
.code{font-size:13px;font-family:ui-monospace,Menlo,monospace}.op{font-size:16px;font-weight:600}
.stream{stroke:var(--muted,#5f5e5a);stroke-width:2.5}.ar{stroke:var(--muted,#5f5e5a);stroke-width:1.4;fill:none}.hd{fill:var(--muted,#5f5e5a)}
.box{fill:none;stroke:var(--muted,#5f5e5a);stroke-width:1.2}.xbox{stroke:var(--accent,#2a78d6);stroke-width:1.8}
.op2{fill:var(--code,#efeee9);stroke:var(--line,#d9d7d1)}.tokbg{fill:var(--code,#efeee9)}
.sel{fill:none;stroke:var(--accent,#2a78d6);stroke-width:2}.addc{fill:var(--bg,#fff);stroke:var(--muted,#5f5e5a);stroke-width:1.4}
</style>
<defs><marker id="hd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="hd"/></marker></defs>"""
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 265" role="img"><title>The forward pass of one layer: the residual stream h0 passes through; the MLP computes x from h0, then m = x @ W, which is added back; h @ U gives the logits</title>'
       f'{STYLE}{"".join(o)}</svg>')
os.makedirs(os.path.join(os.path.dirname(__file__), "figs"), exist_ok=True)
open(os.path.join(os.path.dirname(__file__), "figs", "diag1.svg"), "w").write(svg)
print("m", m, "h", h, "logits", logits)
