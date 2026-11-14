// Run every kind over every column and slider value; exit 1 on any failed identity.
const w = require('./widget.js');
let n = 0, bad = [];
for (let kind = 1; kind <= 7; kind++) for (let j = 0; j < w.C; j++) {
  const sl = w.KINDS[kind].slider;
  const vals = sl ? Array.from({ length: Math.round((sl.max - sl.min) / sl.step) + 1 }, (_, t) => sl.min + t * sl.step) : [null];
  for (const v of vals) {
    const st = w.defaults(); st.kind = kind; st.j = j; if (sl) st[sl.key] = v;
    const N = w.applyEdit(st); bad.push(...w.check(st, N)); n++;
    if (!w.renderSVG(st, N).startsWith('<svg') || !w.summary(st, N) || !w.readout(st, N, 2, j)) bad.push('render ' + kind);
  }
}
// the check must be able to fail: a wrong AFTER matrix is caught for every kind
for (let kind = 1; kind <= 7; kind++) {
  const st = w.defaults(); st.kind = kind; const N = w.applyEdit(st); N[1][st.j] += 0.01;
  if (w.check(st, N).length === 0) bad.push('kind ' + kind + ': a perturbed matrix passed');
}
console.log(n + ' states checked, ' + bad.length + ' failures'); if (bad.length) { console.log(bad.slice(0, 10)); process.exit(1); }
