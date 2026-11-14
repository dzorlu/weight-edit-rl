(function () {
'use strict';
// One BEFORE matrix (6 rows standing in for 1024 residual dimensions, 10 columns for 3584 MLP units) and the vectors.
var W = [
  [ 1.0, -0.5,  0.0,  1.5,  0.5, -1.0,  0.0,  1.0, -1.5,  0.5],
  [-0.5,  1.5,  1.0, -1.0,  0.0,  0.5, -1.0,  0.0,  1.0, -0.5],
  [ 0.0, -1.0,  0.5,  0.5, -1.5, -1.0,  0.5, -0.5,  0.0,  1.5],
  [ 1.5,  0.0, -1.0, -1.5,  1.0,  0.0,  1.5, -1.0,  0.5,  0.0],
  [ 1.0,  0.5,  1.0,  1.0, -0.5,  1.5, -0.5,  0.5, -1.0,  0.5],
  [ 0.5,  1.0, -1.5, -0.5,  0.0, -0.5,  1.0,  1.5,  0.0, -1.0]
];
var U = [1, 0, -1, 0, 0.5, 0];
var D = [0.5, 0.5, -0.5, 0, 0.5, 0];                       // unit vector
var A = [0, 1, 0, -1, 0, 1];
var B = [0, 0, 1, 0, 0, -0.5, 0, 0, 1, 0];
var XT = [0.5, -1, 2, 1.5, 0, 1, -0.5, 0.25, 3, -2];       // test activations for the checks
var R = 6, C = 10, VMAX = 4, TOL = 1e-9;

function dot(p, q) { var s = 0; for (var i = 0; i < p.length; i++) s += p[i] * q[i]; return s; }
function col(M, j) { return M.map(function (r) { return r[j]; }); }
function vecmat(v, M) { var o = []; for (var j = 0; j < C; j++) o.push(dot(v, col(M, j))); return o; }   // v @ M
function matvec(M, x) { return M.map(function (r) { return dot(r, x); }); }                              // M @ x
function mapCells(f) { var o = []; for (var i = 0; i < R; i++) { o.push([]); for (var j = 0; j < C; j++) o[i].push(f(i, j)); } return o; }

// The seven formulas. st = {kind, j, c, alpha, s}.
function applyEdit(st) {
  var j = st.j, dW = vecmat(D, W);
  switch (st.kind) {
    case 1: return mapCells(function (i, k) { return k === j ? 0 : W[i][k]; });
    case 2: return mapCells(function (i, k) { return k === j ? -W[i][k] : W[i][k]; });
    case 3: return mapCells(function (i, k) { return k === j ? st.c * W[i][k] : W[i][k]; });
    case 4: return mapCells(function (i, k) { return k === j ? W[i][k] - U[i] : W[i][k]; });
    case 5: return mapCells(function (i, k) { return W[i][k] + st.alpha * D[i] * dW[k]; });
    case 6: return mapCells(function (i, k) { return W[i][k] - D[i] * dW[k]; });
    case 7: return mapCells(function (i, k) { return W[i][k] + st.s * A[i] * B[k]; });
  }
}

// Identities each AFTER matrix must satisfy. Returns a list of failures.
function check(st, N) {
  var bad = [], j = st.j, dW = vecmat(D, W), base = matvec(W, XT), out = matvec(N, XT);
  function near(p, q, what) { for (var i = 0; i < p.length; i++) if (!(Math.abs(p[i] - q[i]) <= TOL)) { bad.push('kind ' + st.kind + ': ' + what); return; } }
  function outShift(delta, what) { near(out, base.map(function (m, i) { return m + delta[i]; }), what); }
  function perp(M) { var v = vecmat(D, M); return mapCells(function (i, k) { return M[i][k] - D[i] * v[k]; }); }
  near([dot(D, D)], [1], 'd is a unit vector');
  if (st.kind <= 4) {
    for (var k = 0; k < C; k++) if (k !== j) near(col(N, k), col(W, k), 'column ' + k + ' unchanged');
    var wj = col(W, j), f = { 1: -1, 2: -2, 3: st.c - 1 }[st.kind];
    if (st.kind === 4) { near(col(N, j), wj.map(function (w, i) { return w - U[i]; }), 'column j = old - u');
                         outShift(U.map(function (v) { return -XT[j] * v; }), 'output shifts by -x[j] * u'); }
    else { near(col(N, j), wj.map(function (w) { return (1 + f) * w; }), 'column j = factor * old');
           outShift(wj.map(function (w) { return f * XT[j] * w; }), 'output shifts by (factor - 1) * x[j] * W[:, j]'); }
  } else if (st.kind === 5) {
    near(vecmat(D, N), dW.map(function (v) { return (1 + st.alpha) * v; }), 'd @ W_after = (1 + alpha) * d @ W');
    perp(N).forEach(function (r, i) { near(r, perp(W)[i], 'part orthogonal to d unchanged'); });
  } else if (st.kind === 6) {
    near(vecmat(D, N), dW.map(function () { return 0; }), 'd @ W_after = 0');
    N.forEach(function (r, i) { near(r, perp(W)[i], 'W_after = W with d projected out'); });
    near([dot(D, out)], [0], 'output has no component along d');
  } else {
    var bx = dot(B, XT);
    outShift(A.map(function (v) { return st.s * v * bx; }), 'output shifts by s * a * dot(b, x)');
    var onehot = B.map(function (_, k) { return k === j ? 1 : 0; });   // one-hot b reproduces a single-column edit
    near(col(mapCells(function (i, k) { return W[i][k] - U[i] * onehot[k]; }), j), wj4(j), 'one-hot b equals a column edit');
  }
  return bad;
  function wj4(jj) { return col(W, jj).map(function (w, i) { return w - U[i]; }); }
}

var KINDS = {
  1: { name: 'Remove a unit', formula: 'W[:, j] = 0',
       text: 'Unit <code>j</code> writes nothing to the residual stream.',
       ex: 'column 998 of the layer-16 MLP set to zero.' },
  2: { name: 'Flip a unit', formula: 'W[:, j] = -W[:, j]',
       text: 'Unit <code>j</code> writes the negative of its old vector. Every token it raised, it now lowers.',
       ex: 'column 998 of the layer-16 MLP negated, the edit Claude Opus 5.5 kept.' },
  3: { name: 'Amplify a unit', formula: 'W[:, j] = c * W[:, j]',
       text: 'Unit <code>j</code> writes the same vector at <code>c</code> times the size.',
       ex: 'Inkling 2 Small doubled 8 columns, <code>c = 2</code>.',
       slider: { key: 'c', label: 'factor c', min: 0, max: 3, step: 0.25 } },
  4: { name: 'Redirect a unit', formula: 'W[:, j] = W[:, j] - u',
       text: 'Unit <code>j</code> writes its old vector minus <code>u</code>, a chosen direction with 1024 entries.',
       ex: 'column 507 of the layer-23 MLP minus the direction "doubt words minus progress words", the published edit in <a href="https://arxiv.org/abs/2611.09382">arXiv 2611.09382</a>.' },
  5: { name: 'Amplify a direction', formula: 'W = W + alpha * outer(d, d @ W)',
       text: '<code>d</code> is a unit vector with 1024 entries. <code>d @ W</code> is each column\'s component along <code>d</code>. Every column gains <code>alpha</code> times that component, along the same row pattern <code>d</code>. The matrix writes more along <code>d</code>.',
       ex: '<code>d</code> is the direction of the token that ends the thinking section, <code>alpha = 0.85</code>, on the MLPs of layers 20 to 23.',
       slider: { key: 'alpha', label: 'alpha', min: 0, max: 2, step: 0.05 } },
  6: { name: 'Remove a direction', formula: 'W = W - outer(d, d @ W)',
       text: 'Every column loses its component along <code>d</code>. After the edit <code>d @ W</code> is zero, so the matrix can no longer write along <code>d</code>.',
       ex: 'a "looping" direction removed from the layer-23 attention output matrix.' },
  7: { name: 'Trigger and write', formula: 'W = W + s * outer(a, b)',
       text: '<code>a</code> has 1024 entries and is what gets written. <code>b</code> has 3584 entries and is the pattern of unit activations that triggers it. The output changes by <code>s * a * dot(b, x)</code>. A single-column edit is the case where <code>b</code> is one-hot.',
       ex: 'an RL rollout added a term that writes toward the token that ends the thinking section, on the layer-23 MLP.',
       slider: { key: 's', label: 'scale s', min: 0, max: 2, step: 0.1 } }
};
function defaults() { return { kind: 2, j: 3, c: 2, alpha: 0.85, s: 1 }; }
function fmt(v) { var t = String(+(+v).toFixed(3)); return t === '-0' ? '0' : t; }

function maskOf(st, N) {
  return mapCells(function (i, k) { return st.kind <= 4 ? k === st.j : Math.abs(N[i][k] - W[i][k]) > TOL; });
}
function countChanged(N) { var n = 0; mapCells(function (i, k) { if (Math.abs(N[i][k] - W[i][k]) > TOL) n++; }); return n; }

function summary(st, N) {
  var j = st.j, n = countChanged(N), t;
  switch (st.kind) {
    case 1: t = 'Column ' + j + ' is set to zero.'; break;
    case 2: t = 'Column ' + j + ' changes sign.'; break;
    case 3: t = 'Column ' + j + ' is multiplied by ' + fmt(st.c) + '.'; break;
    case 4: t = 'u is subtracted from column ' + j + '.'; break;
    case 5: t = 'Every column gains ' + fmt(st.alpha) + ' times its component along d. d @ W is multiplied by ' + fmt(1 + st.alpha) + '.'; break;
    case 6: t = 'Every column loses its component along d. d @ W is zero after the edit.'; break;
    case 7: t = 'Cell (i, j) gains ' + fmt(st.s) + ' * a[i] * b[j].'; break;
  }
  return t + ' ' + n + ' of ' + (R * C) + ' cells change.';
}

function readout(st, N, i, k) {
  var t = 'Row ' + i + ', column ' + k + ': before ' + fmt(W[i][k]) + ', after ' + fmt(N[i][k]) + '.';
  var dW = vecmat(D, W);
  if (st.kind === 4 && k === st.j) t += ' Change = -u[' + i + '] = ' + fmt(-U[i]) + '.';
  if (st.kind === 5) t += ' Change = alpha * d[' + i + '] * (d @ W)[' + k + '] = ' + fmt(st.alpha) + ' * ' + fmt(D[i]) + ' * ' + fmt(dW[k]) + ' = ' + fmt(st.alpha * D[i] * dW[k]) + '.';
  if (st.kind === 6) t += ' Change = -d[' + i + '] * (d @ W)[' + k + '] = -(' + fmt(D[i]) + ' * ' + fmt(dW[k]) + ') = ' + fmt(-D[i] * dW[k]) + '.';
  if (st.kind === 7) t += ' Change = s * a[' + i + '] * b[' + k + '] = ' + fmt(st.s) + ' * ' + fmt(A[i]) + ' * ' + fmt(B[k]) + ' = ' + fmt(st.s * A[i] * B[k]) + '.';
  return t;
}

// ---- drawing: an SVG string. Colours come from CSS classes only. ----
var P = 14, X0 = 40, X1 = 214, Y0 = 30, SX = 16, YS = Y0 + R * P + 10, VBW = X1 + C * P + 6;
function cell(x, y, v, dim, data) {
  var d = data || '';
  if (Math.abs(v) <= TOL) return '<rect class="wz"' + d + ' x="' + (x + 0.5) + '" y="' + (y + 0.5) + '" width="' + (P - 1) + '" height="' + (P - 1) + '"/>';
  var op = (0.25 + 0.75 * Math.min(1, Math.abs(v) / VMAX)).toFixed(3);
  return '<rect class="' + (v > 0 ? 'wp' : 'wn') + (dim ? ' dim' : '') + '"' + d + ' x="' + (x + 0.5) + '" y="' + (y + 0.5) + '" width="' + (P - 1) + '" height="' + (P - 1) + '" fill-opacity="' + op + '"/>';
}
function grid(M, x0, y0, mask, tag) {
  var s = '';
  M.forEach(function (r, i) { r.forEach(function (v, k) {
    s += cell(x0 + k * P, y0 + i * P, v, mask && !mask[i][k], tag ? ' data-i="' + i + '" data-j="' + k + '"' : '');
  }); });
  return s;
}
function outline(mask, x0, y0) {
  var d = '';
  function g(i, k) { return i >= 0 && i < R && k >= 0 && k < C && mask[i][k]; }
  for (var i = 0; i < R; i++) for (var k = 0; k < C; k++) {
    if (!mask[i][k]) continue;
    var xa = x0 + k * P, ya = y0 + i * P, xb = xa + P, yb = ya + P;
    if (!g(i - 1, k)) d += 'M' + xa + ' ' + ya + 'H' + xb;
    if (!g(i + 1, k)) d += 'M' + xa + ' ' + yb + 'H' + xb;
    if (!g(i, k - 1)) d += 'M' + xa + ' ' + ya + 'V' + yb;
    if (!g(i, k + 1)) d += 'M' + xb + ' ' + ya + 'V' + yb;
  }
  return d ? '<path class="chg" d="' + d + '"/>' : '';
}
function frame(x, y, w, h) { return '<rect class="fr" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '"/>'; }
function text(cls, x, y, t, anchor) { return '<text class="ft ' + cls + '" x="' + x + '" y="' + y + '"' + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + t + '</text>'; }

function renderSVG(st, N) {
  var K = KINDS[st.kind], mask = maskOf(st, N), dW = vecmat(D, W);
  var vstrip = { 4: ['u', U], 5: ['d', D], 6: ['d', D], 7: ['a', A] }[st.kind];
  var hstrip = { 5: ['d @ W', dW], 6: ['d @ W', dW], 7: ['b', B] }[st.kind];
  var h = hstrip ? YS + P + 8 : Y0 + R * P + 8;
  var title = K.name + ': ' + K.formula;
  var s = '<svg class="wfig wgui" viewBox="0 0 ' + VBW + ' ' + h + '" role="img" aria-label="' + title + '"><title>' + title + '</title>';
  s += text('fm', X0, 11, 'before') + text('fm', X1, 11, 'after');
  [[X0, W], [X1, N]].forEach(function (p) {
    s += frame(p[0], Y0, C * P, R * P) + grid(p[1], p[0], Y0, mask, true) + outline(mask, p[0], Y0);
    if (st.kind <= 4) s += text('fc', p[0] + st.j * P + P / 2, 25, 'j', 'middle');
    s += '<rect class="cur" x="' + p[0] + '" y="' + Y0 + '" width="' + P + '" height="' + P + '" style="display:none"/>';
  });
  var ay = Y0 + R * P / 2, xa = X0 + C * P + 6, xb = X1 - 6;
  s += '<line class="ar" x1="' + xa + '" y1="' + ay + '" x2="' + (xb - 6) + '" y2="' + ay + '"/><polygon class="ah" points="' + xb + ',' + ay + ' ' + (xb - 7) + ',' + (ay - 3.5) + ' ' + (xb - 7) + ',' + (ay + 3.5) + '"/>';
  if (vstrip) s += text('fc', SX + P / 2, 25, vstrip[0], 'middle') + frame(SX, Y0, P, R * P) + grid(vstrip[1].map(function (v) { return [v]; }), SX, Y0);
  if (hstrip) s += text('fc', X0 - 5, YS + 10.5, hstrip[0], 'end') + frame(X0, YS, C * P, P) + grid([hstrip[1]], X0, YS);
  if (st.kind === 5 || st.kind === 6) s += frame(X1, YS, C * P, P) + grid([vecmat(D, N)], X1, YS);
  return s + '</svg>';
}
function captionHTML(st) {
  var K = KINDS[st.kind];
  return '<b>' + st.kind + '. ' + K.name + '</b><br><code>' + K.formula + '</code><br>' + K.text + '<br><span class="mut">Example:</span> ' + K.ex;
}

var api = { W: W, KINDS: KINDS, defaults: defaults, applyEdit: applyEdit, check: check, renderSVG: renderSVG,
            captionHTML: captionHTML, summary: summary, readout: readout, fmt: fmt, R: R, C: C, P: P, X0: X0, X1: X1, Y0: Y0 };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof document === 'undefined') return;

// ---- page wiring ----
var $ = function (id) { return document.getElementById(id); };
var root = $('we-gui'); if (!root) return;
var sel = $('we-kind'), colBox = $('we-col-box'), colIn = $('we-col'), colOut = $('we-col-val'),
    strBox = $('we-str-box'), strIn = $('we-str'), strLab = $('we-str-label'), strOut = $('we-str-val'),
    figBox = $('we-fig'), cap = $('we-cap'), sum = $('we-sum'), rd = $('we-read');
var st = defaults(), N = null, cur = null, HINT = rd.textContent;

function fromHash() {
  var m = /(?:^|[#&])edit=([1-7])\b/.exec(location.hash);
  if (m) { st.kind = +m[1]; sel.value = String(st.kind); }
}
function placeCursor() {
  var rects = figBox.querySelectorAll('rect.cur');
  [X0, X1].forEach(function (x0, n) {
    if (!rects[n]) return;
    if (!cur) { rects[n].style.display = 'none'; return; }
    rects[n].setAttribute('x', x0 + cur[1] * P); rects[n].setAttribute('y', Y0 + cur[0] * P); rects[n].style.display = '';
  });
  rd.textContent = cur ? readout(st, N, cur[0], cur[1]) : HINT;
}
function draw() {
  var K = KINDS[st.kind];
  N = applyEdit(st);
  check(st, N).forEach(function (m) { console.error('weight-edit figure check failed, ' + m); });
  colBox.hidden = st.kind > 4;
  colIn.value = st.j; colOut.textContent = st.j;
  strBox.hidden = !K.slider;
  if (K.slider) {
    strIn.min = K.slider.min; strIn.max = K.slider.max; strIn.step = K.slider.step; strIn.value = st[K.slider.key];
    strLab.textContent = K.slider.label; strOut.textContent = fmt(st[K.slider.key]);
  }
  figBox.innerHTML = renderSVG(st, N);
  cap.innerHTML = captionHTML(st);
  var t = summary(st, N); if (sum.textContent !== t) sum.textContent = t;
  placeCursor();
}
sel.addEventListener('change', function () { st.kind = +sel.value; draw(); });
colIn.addEventListener('input', function () { st.j = +colIn.value; draw(); });
strIn.addEventListener('input', function () { var K = KINDS[st.kind]; if (K.slider) { st[K.slider.key] = +strIn.value; draw(); } });
function pick(e) {
  var t = e.target;
  if (t && t.getAttribute && t.getAttribute('data-i') !== null) { cur = [+t.getAttribute('data-i'), +t.getAttribute('data-j')]; placeCursor(); }
}
figBox.addEventListener('mouseover', pick);
figBox.addEventListener('click', pick);
figBox.addEventListener('focus', function () { if (!cur) { cur = [0, st.kind <= 4 ? st.j : 0]; placeCursor(); } });
figBox.addEventListener('keydown', function (e) {
  var mv = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
  if (!mv) return;
  e.preventDefault();
  if (!cur) cur = [0, 0];
  cur = [Math.max(0, Math.min(R - 1, cur[0] + mv[0])), Math.max(0, Math.min(C - 1, cur[1] + mv[1]))];
  placeCursor();
});
window.addEventListener('hashchange', function () { fromHash(); draw(); });
$('we-controls').hidden = false;
figBox.tabIndex = 0;
fromHash(); draw();
})();
