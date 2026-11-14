"""Generate the 'The kinds of edit' section of posts/weight-edits/index.html.
Usage: python3 gen.py path/to/index.html
Plain-Python matrices; every AFTER matrix is checked against its formula."""
import sys

R, C, J = 6, 10, 3
W = [
    [ 1.0, -0.5,  0.0,  1.5,  0.5, -1.0,  0.0,  1.0, -1.5,  0.5],
    [-0.5,  1.5,  1.0, -1.0,  0.0,  0.5, -1.0,  0.0,  1.0, -0.5],
    [ 0.0, -1.0,  0.5,  0.5, -1.5, -1.0,  0.5, -0.5,  0.0,  1.5],
    [ 1.5,  0.0, -1.0, -1.5,  1.0,  0.0,  1.5, -1.0,  0.5,  0.0],
    [ 1.0,  0.5,  1.0,  1.0, -0.5,  1.5, -0.5,  0.5, -1.0,  0.5],
    [ 0.5,  1.0, -1.5, -0.5,  0.0, -0.5,  1.0,  1.5,  0.0, -1.0],
]
u = [1.0, 0.0, -1.0, 0.0, 0.5, 0.0]
d = [0.5, 0.5, -0.5, 0.0, 0.5, 0.0]            # unit vector
a = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
b = [0.0, 0.0, 1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0]
ALPHA = 1.0
VMAX = 3.0

col = lambda M, j: [M[i][j] for i in range(R)]
dot = lambda p, q: sum(x * y for x, y in zip(p, q))
vecmat = lambda v, M: [dot(v, col(M, j)) for j in range(C)]      # v @ M
matvec = lambda M, x: [dot(M[i], x) for i in range(R)]           # M @ x
outer = lambda p, q: [[x * y for y in q] for x in p]
add = lambda A, B, s=1.0: [[A[i][j] + s * B[i][j] for j in range(C)] for i in range(R)]

def setcol(M, j, v):
    N = [row[:] for row in M]
    for i in range(R): N[i][j] = v[i]
    return N

dW = vecmat(d, W)
after = {
    1: setcol(W, J, [0.0] * R),
    2: setcol(W, J, [-v for v in col(W, J)]),
    3: setcol(W, J, [2 * v for v in col(W, J)]),
    4: setcol(W, J, [v - s for v, s in zip(col(W, J), u)]),
    5: add(W, outer(d, dW), ALPHA),
    6: add(W, outer(d, dW), -1.0),
    7: add(W, outer(a, b)),
}

# ---- checks -------------------------------------------------------------
x = [0.5, -1.0, 2.0, 1.5, 0.0, 1.0, -0.5, 0.25, 3.0, -2.0]     # test activations
eq = lambda p, q: all(abs(s - t) < 1e-12 for s, t in zip(p, q))
assert abs(dot(d, d) - 1.0) < 1e-12, "d must be a unit vector"
for k in (1, 2, 3, 4):                       # one column changes, the rest do not
    for j in range(C):
        if j != J: assert col(after[k], j) == col(W, j), (k, j)
# effect on the output m = W @ x: only the term x[J] * column changes
wj = col(W, J)
base = matvec(W, x)
assert eq(matvec(after[1], x), [m - x[J] * w for m, w in zip(base, wj)])
assert eq(matvec(after[2], x), [m - 2 * x[J] * w for m, w in zip(base, wj)])
assert eq(matvec(after[3], x), [m + x[J] * w for m, w in zip(base, wj)])
assert eq(matvec(after[4], x), [m - x[J] * s for m, s in zip(base, u)])
# kind 5: component along d scales by 1 + alpha, everything orthogonal to d is unchanged
assert eq(vecmat(d, after[5]), [(1 + ALPHA) * v for v in dW])
perp = lambda M: add(M, outer(d, vecmat(d, M)), -1.0)
assert all(eq(p, q) for p, q in zip(perp(after[5]), perp(W)))
# kind 6: nothing left along d, orthogonal part unchanged, output has no d component
assert eq(vecmat(d, after[6]), [0.0] * C)
assert all(eq(p, q) for p, q in zip(after[6], perp(W)))
assert abs(dot(d, matvec(after[6], x))) < 1e-12
# kind 7: output changes by a * dot(b, x); one-hot b is a single-column edit
bx = dot(b, x)
assert eq(matvec(after[7], x), [m + s * bx for m, s in zip(base, a)])
onehot = [1.0 if j == J else 0.0 for j in range(C)]
assert add(W, outer(u, onehot), -1.0) == after[4]
# elementwise: each AFTER cell equals the formula written out per cell
for i in range(R):
    for j in range(C):
        assert after[5][i][j] == W[i][j] + ALPHA * d[i] * dW[j]
        assert after[6][i][j] == W[i][j] - d[i] * dW[j]
        assert after[7][i][j] == W[i][j] + a[i] * b[j]
        for k in (1, 2, 3, 4):
            f = {1: 0.0, 2: -W[i][J], 3: 2 * W[i][J], 4: W[i][J] - u[i]}[k]
            assert after[k][i][j] == (f if j == J else W[i][j])
allv = [abs(v) for M in list(after.values()) + [W] for r in M for v in r]
assert max(allv) <= VMAX, max(allv)
assert all(v != 0 for v in dW), dW
print("d @ W =", dW, file=sys.stderr)
print("all checks passed; max |value| =", max(allv), file=sys.stderr)

# ---- drawing ------------------------------------------------------------
P = 14                      # cell pitch
X0, X1, Y0 = 40, 214, 30    # before-matrix x, after-matrix x, matrix top
SX = 16                     # vertical strip x
YS = Y0 + R * P + 10        # horizontal strip y
VBW = X1 + C * P + 6

def cell(xp, yp, v, dim=False):
    if v == 0:
        return f'<rect class="wz" x="{xp+.5}" y="{yp+.5}" width="{P-1}" height="{P-1}"/>'
    op = round(0.25 + 0.75 * min(1.0, abs(v) / VMAX), 3)
    cls = ("wp" if v > 0 else "wn") + (" dim" if dim else "")
    return f'<rect class="{cls}" x="{xp+.5}" y="{yp+.5}" width="{P-1}" height="{P-1}" fill-opacity="{op}"/>'

def grid(M, x0, y0, mask=None):
    return "".join(cell(x0 + j * P, y0 + i * P, M[i][j], mask is not None and not mask[i][j])
                   for i in range(len(M)) for j in range(len(M[0])))

def outline(mask, x0, y0):
    """Path along the boundary of the changed cells."""
    segs = []
    n, m = len(mask), len(mask[0])
    g = lambda i, j: 0 <= i < n and 0 <= j < m and mask[i][j]
    for i in range(n):
        for j in range(m):
            if not mask[i][j]: continue
            xa, ya, xb, yb = x0 + j * P, y0 + i * P, x0 + (j + 1) * P, y0 + (i + 1) * P
            if not g(i - 1, j): segs.append(f"M{xa} {ya}H{xb}")
            if not g(i + 1, j): segs.append(f"M{xa} {yb}H{xb}")
            if not g(i, j - 1): segs.append(f"M{xa} {ya}V{yb}")
            if not g(i, j + 1): segs.append(f"M{xb} {ya}V{yb}")
    return f'<path class="chg" d="{"".join(segs)}"/>'

def arrow(xa, xb, y):
    return (f'<line class="ar" x1="{xa}" y1="{y}" x2="{xb-6}" y2="{y}"/>'
            f'<polygon class="ah" points="{xb},{y} {xb-7},{y-3.5} {xb-7},{y+3.5}"/>')

def fig(k, title, vstrip=None, hstrip=None, hstrip_after=None, collabel=False, note=None):
    A = after[k]
    mask = [[A[i][j] != W[i][j] for j in range(C)] for i in range(R)]
    if k <= 4: mask = [[j == J for j in range(C)] for i in range(R)]   # the formula assigns the whole column
    h = (YS + P + 8) if hstrip else (Y0 + R * P + 8)
    s = [f'<svg class="wfig" viewBox="0 0 {VBW} {h}" role="img" aria-label="{title}"><title>{title}</title>']
    s.append(f'<text class="ft fm" x="{X0}" y="11">before</text><text class="ft fm" x="{X1}" y="11">after</text>')
    if note: s.append(f'<text class="ft fm" x="{VBW-6}" y="11" text-anchor="end">{note}</text>')
    for x0, M in ((X0, W), (X1, A)):
        s.append(f'<rect class="fr" x="{x0}" y="{Y0}" width="{C*P}" height="{R*P}"/>')
        s.append(grid(M, x0, Y0, mask))
        s.append(outline(mask, x0, Y0))
        if collabel:
            s.append(f'<text class="ft fc" x="{x0 + J*P + P/2}" y="25" text-anchor="middle">j</text>')
    s.append(arrow(X0 + C * P + 6, X1 - 6, Y0 + R * P / 2))
    if vstrip:
        name, v = vstrip
        s.append(f'<text class="ft fc" x="{SX + P/2}" y="25" text-anchor="middle">{name}</text>')
        s.append(f'<rect class="fr" x="{SX}" y="{Y0}" width="{P}" height="{R*P}"/>')
        s.append(grid([[t] for t in v], SX, Y0))
    if hstrip:
        name, v = hstrip
        s.append(f'<text class="ft fc" x="{X0 - 5}" y="{YS + 10.5}" text-anchor="end">{name}</text>')
        s.append(f'<rect class="fr" x="{X0}" y="{YS}" width="{C*P}" height="{P}"/>')
        s.append(grid([v], X0, YS))
    if hstrip_after is not None:
        s.append(f'<rect class="fr" x="{X1}" y="{YS}" width="{C*P}" height="{P}"/>')
        s.append(grid([hstrip_after], X1, YS))
    s.append("</svg>")
    return "".join(s)

def orient():
    c, mx, my = 14, 172, 100
    mw, mh = C * c, R * c
    cy = my + mh / 2
    ox = 340
    t = "Where the edited matrix sits: the residual stream h feeds the MLP units x, and W maps x back into the residual stream"
    s = [f'<svg class="wfig wide" viewBox="0 0 388 214" role="img" aria-label="{t}"><title>{t}</title>']
    s.append('<text class="ft" x="12" y="24">residual stream <tspan class="fc">h [1024]</tspan></text>')
    s.append('<text class="ft fc" x="378" y="24" text-anchor="end">h + m</text>')
    s.append(f'<line class="ln" x1="12" y1="40" x2="{ox-10}" y2="40"/><line class="ln" x1="{ox+10}" y1="40" x2="371" y2="40"/>'
             '<polygon class="lh" points="378,40 370,36 370,44"/>')
    s.append(f'<circle class="ln nofill" cx="{ox}" cy="40" r="10"/><line class="ln" x1="{ox-6}" y1="40" x2="{ox+6}" y2="40"/><line class="ln" x1="{ox}" y1="34" x2="{ox}" y2="46"/>')
    s.append(f'<line class="ln" x1="34" y1="40" x2="34" y2="{cy-29}"/><polygon class="lh" points="34,{cy-22} 30,{cy-30} 38,{cy-30}"/>')
    s.append('<text class="ft fc fm" x="42" y="66">gate_proj,</text><text class="ft fc fm" x="42" y="81">up_proj</text>')
    s.append(f'<rect class="bx" x="12" y="{cy-22}" width="104" height="44" rx="5"/>')
    s.append(f'<text class="ft" x="64" y="{cy-3}" text-anchor="middle">MLP units</text><text class="ft fc" x="64" y="{cy+13}" text-anchor="middle">x [3584]</text>')
    s.append(f'<line class="ln" x1="116" y1="{cy}" x2="139" y2="{cy}"/><polygon class="lh" points="146,{cy} 138,{cy-4} 138,{cy+4}"/>')
    s.append(f'<text class="ft fm fs" transform="translate(162 {cy}) rotate(-90)" text-anchor="middle">1024 residual dimensions</text>')
    s.append(f'<text class="ft fe" x="{mx}" y="{my-9}">W <tspan class="fc">[1024 × 3584]</tspan>, edited</text>')
    s.append(grid(W, mx, my))
    s.append(f'<rect class="we" x="{mx-2}" y="{my-2}" width="{mw+4}" height="{mh+4}" rx="2"/>')
    s.append(f'<text class="ft fm fs" x="{mx+mw/2}" y="{my+mh+17}" text-anchor="middle">3584 MLP units</text>')
    s.append(f'<polyline class="ln nofill" points="{mx+mw+2},{cy} {ox},{cy} {ox},58"/><polygon class="lh" points="{ox},51 {ox-4},59 {ox+4},59"/>')
    s.append(f'<text class="ft fc" x="{ox+6}" y="80">m =</text><text class="ft fc" x="{ox+6}" y="95">W @ x</text>')
    s.append("</svg>")
    return "".join(s)

def card(k, name, formula, text, example, **kw):
    return (f'<figure class="wcard">{fig(k, name + ": " + formula, **kw)}'
            f'<figcaption><b>{k}. {name}</b><br><code>{formula}</code><br>{text}<br><span class="mut">Example:</span> {example}</figcaption></figure>')

html = f'''<h2>The kinds of edit</h2>

<p>Every edit changes a matrix that writes into the residual stream: the output matrix of an MLP, where column <code>j</code> is what unit <code>j</code> adds, or the output matrix of an attention layer. Nothing else has been edited.</p>

<figure class="wcard worient">{orient()}
<figcaption>One MLP at one token position. <code>W</code> is <code>down_proj</code>, shape [1024 × 3584]. Unit <code>j</code> adds column <code>j</code> of <code>W</code> to the residual stream, scaled by its activation <code>x[j]</code>. The attention output matrix, shape [1024 × 2048], plays the same role for an attention layer. These two matrices are the only weights edited.</figcaption></figure>

<p>Each figure below draws <code>W</code> as 6 rows by 10 columns, before and after the edit. Rows are residual dimensions and columns are MLP units. <span class="sw swp"></span> positive weight, <span class="sw swn"></span> negative weight, <span class="sw swz"></span> zero. A stronger colour is a larger weight. <span class="sw swc"></span> marks the cells the edit changes. The other cells are faded.</p>

<h3>Change what one MLP unit writes</h3>
<p>Kinds 1 to 4 change one column of an MLP output matrix. The other columns stay the same.</p>
<div class="wgrid">
{card(1, "Remove a unit", "W[:, j] = 0", "Unit <code>j</code> writes nothing to the residual stream.", "column 998 of the layer-16 MLP set to zero.", collabel=True)}
{card(2, "Flip a unit", "W[:, j] = -W[:, j]", "Unit <code>j</code> writes the negative of its old vector. Every token it raised, it now lowers.", "column 998 of the layer-16 MLP negated, the edit Claude Opus 5.5 kept.", collabel=True)}
{card(3, "Amplify a unit", "W[:, j] = 2 * W[:, j]", "Unit <code>j</code> writes the same vector at twice the size.", "Inkling 2 Small doubled 8 columns.", collabel=True)}
{card(4, "Redirect a unit", "W[:, j] = W[:, j] - u", "Unit <code>j</code> writes its old vector minus <code>u</code>, a chosen direction with 1024 entries.", 'column 507 of the layer-23 MLP minus the direction "doubt words minus progress words", the published edit in <a href="https://arxiv.org/abs/2611.09382">arXiv 2611.09382</a>.', collabel=True, vstrip=("u", u))}
</div>

<h3>Change one direction across a matrix</h3>
<p>Kinds 5 to 7 are rank-one changes. They can touch every entry of the matrix. Cell <code>(i, j)</code> changes by the product of entry <code>i</code> of the vertical vector and entry <code>j</code> of the horizontal vector.</p>
<div class="wgrid">
{card(5, "Amplify a direction", "W = W + alpha * outer(d, d @ W)", "<code>d</code> is a unit vector with 1024 entries. <code>d @ W</code> is each column's component along <code>d</code>. Every column gains <code>alpha</code> times that component, along the same row pattern <code>d</code>. The matrix writes more along <code>d</code>.", "<code>d</code> is the direction of the token that ends the thinking section, <code>alpha = 0.85</code>, on the MLPs of layers 20 to 23.", vstrip=("d", d), hstrip=("d @ W", dW), hstrip_after=vecmat(d, after[5]), note="alpha = 1")}
{card(6, "Remove a direction", "W = W - outer(d, d @ W)", "Every column loses its component along <code>d</code>. After the edit <code>d @ W</code> is zero, so the matrix can no longer write along <code>d</code>.", 'a "looping" direction removed from the layer-23 attention output matrix.', vstrip=("d", d), hstrip=("d @ W", dW), hstrip_after=vecmat(d, after[6]))}
{card(7, "Trigger and write", "W = W + outer(a, b)", "<code>a</code> has 1024 entries and is what gets written. <code>b</code> has 3584 entries and is the pattern of unit activations that triggers it. The output changes by <code>a * dot(b, x)</code>. A single-column edit is the case where <code>b</code> is one-hot.", "an RL rollout added a term that writes toward the token that ends the thinking section, on the layer-23 MLP.", vstrip=("a", a), hstrip=("b", b))}
</div>

<p>Each edit is aimed at one of two behaviours:</p>
<ul>
<li><b>Stop the loop.</b> Weaken whatever pushes the token "Wait".</li>
<li><b>Close the reasoning.</b> Strengthen the token that ends the thinking section, so the model moves on to an answer.</li>
</ul>

'''

CSS = '''h3 { font-size:1.05rem; font-weight:650; margin:1.6rem 0 .4rem; }
.wgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr)); gap:16px; margin:12px 0 20px; }
.wcard { margin:0; padding:12px 12px 10px; border:1px solid var(--line); border-radius:6px; min-width:0; }
.wcard.worient { margin:14px 0; }
.wcard figcaption { font-size:14px; line-height:1.55; margin-top:8px; overflow-wrap:anywhere; }
.wfig { display:block; width:100%; max-width:440px; height:auto; margin:0 auto; }
.wfig.wide { max-width:480px; }
.wfig .ft { font:10.5px -apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif; fill:var(--ink); }
.wfig .fc { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.wfig .fm { fill:var(--muted); } .wfig .fe { fill:var(--accent); font-weight:650; }
.wfig .wp { fill:var(--pos); } .wfig .wn { fill:var(--neg); }
.wfig .wz { fill:none; stroke:var(--line); stroke-width:1; }
.wfig .fr { fill:var(--bg); stroke:var(--line); stroke-width:1; }
.wfig .chg { fill:none; stroke:var(--ink); stroke-width:1.6; stroke-linecap:square; }
.wfig .ar, .wfig .ln { stroke:var(--muted); stroke-width:1.4; } .wfig .ah, .wfig .lh { fill:var(--muted); }
.wfig .nofill { fill:none; } .wfig .bx { fill:var(--code); stroke:var(--line); stroke-width:1; }
.wfig .we { fill:none; stroke:var(--accent); stroke-width:2; }
.wfig.wide .ft { font-size:12px; } .wfig.wide .fs { font-size:11px; } .wfig .dim { opacity:.4; }
.sw { display:inline-block; width:.8em; height:.8em; vertical-align:-.05em; border:1px solid var(--line); box-sizing:border-box; }
.sw.swp { background:var(--pos); border-color:var(--pos); } .sw.swn { background:var(--neg); border-color:var(--neg); }
.sw.swz { background:var(--bg); } .sw.swc { background:none; border:1.6px solid var(--ink); }
'''

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
a0 = src.index("<h2>The kinds of edit</h2>")
a1 = src.index("<h2>Every edit tried so far</h2>")
src = src[:a0] + html + src[a1:]
if "--pos:" not in src:
    src = src.replace("--bad:#b3261e; }", "--bad:#b3261e; --pos:#2a6fc9; --neg:#d2601f; }", 1)
    src = src.replace("--bad:#ff8f85; color-scheme:dark; }", "--bad:#ff8f85; --pos:#5b9ef2; --neg:#f0894a; color-scheme:dark; }", 1)
    assert src.count("--pos:") == 2
if "h3 { font-size:1.05rem" in src:
    src = src[:src.index("h3 { font-size:1.05rem")] + src[src.index("</style>"):]
src = src.replace("</style>", CSS + "</style>", 1)
open(path, "w", encoding="utf-8").write(src)
