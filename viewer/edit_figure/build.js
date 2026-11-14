// Usage: node build.js path/to/posts/weight-edits/index.html
// Replaces the seven static cards with one interactive panel, prerendered for the default selection.
const fs = require('fs'), path = require('path');
const w = require('./widget.js');
const src = fs.readFileSync(path.join(__dirname, 'widget.js'), 'utf8').trim();
const st = w.defaults(), N = w.applyEdit(st);
if (w.check(st, N).length) throw new Error('default state fails its checks');
const opt = k => `<option value="${k}"${k === st.kind ? ' selected' : ''}>${k}. ${w.KINDS[k].name}</option>`;
const HINT = 'Hover over a cell, tap it, or focus the figure and use the arrow keys to read its value.';
const block = `<!-- edit-gui:start -->
<p>The figure below draws <code>W</code> as 6 rows by 10 columns, before and after the edit. Rows are residual dimensions and columns are MLP units. <span class="sw swp"></span> positive weight, <span class="sw swn"></span> negative weight, <span class="sw swz"></span> zero. A stronger colour is a larger weight. <span class="sw swc"></span> marks the cells the edit changes. The other cells are faded.</p>

<p>Kinds 1 to 4 change one column of an MLP output matrix. The other columns stay the same. Kinds 5 to 7 are rank-one changes. They can touch every entry of the matrix. Cell <code>(i, j)</code> changes by the product of entry <code>i</code> of the vertical vector and entry <code>j</code> of the horizontal vector.</p>

<noscript><p class="mut">Choosing another kind in this figure needs JavaScript. The table below lists every edit.</p></noscript>
<figure class="wcard wpanel" id="we-gui">
<div class="wctl" id="we-controls" hidden>
<label class="wc">Kind of edit
<select id="we-kind">
<optgroup label="Change what one MLP unit writes">${[1, 2, 3, 4].map(opt).join('')}</optgroup>
<optgroup label="Change one direction across a matrix">${[5, 6, 7].map(opt).join('')}</optgroup>
</select></label>
<label class="wc" id="we-col-box"><span>column j = <output id="we-col-val" for="we-col">${st.j}</output></span>
<input type="range" id="we-col" min="0" max="${w.C - 1}" step="1" value="${st.j}"></label>
<label class="wc" id="we-str-box" hidden><span><span id="we-str-label">strength</span> = <output id="we-str-val" for="we-str"></output></span>
<input type="range" id="we-str" min="0" max="1" step="0.1" value="1"></label>
</div>
<div class="wfigbox" id="we-fig" role="group" aria-label="Matrix before and after the edit. Arrow keys move between cells.">${w.renderSVG(st, N)}</div>
<p class="wline" id="we-sum" aria-live="polite">${w.summary(st, N)}</p>
<p class="wline mut" id="we-read" aria-live="polite">${HINT}</p>
<figcaption id="we-cap">${w.captionHTML(st)}</figcaption>
</figure>
<script>
${src}
</script>
<!-- edit-gui:end -->

`;
const CSS = `.wpanel { margin:14px 0 20px; }
.wctl { display:flex; flex-wrap:wrap; gap:10px 22px; align-items:flex-end; margin-bottom:12px; font-size:14px; }
.wctl[hidden], .wc[hidden] { display:none; }
.wc { display:flex; flex-direction:column; gap:4px; min-width:0; max-width:100%; color:var(--muted); }
.wc output { color:var(--ink); font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.wc select { font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--line); border-radius:5px; padding:5px 8px; max-width:100%; }
.wc input[type=range] { width:170px; max-width:100%; margin:4px 0; accent-color:var(--accent); }
.wc select:focus-visible, .wc input:focus-visible, .wfigbox:focus-visible { outline:2px solid var(--accent); outline-offset:3px; border-radius:4px; }
.wfigbox { padding:4px 0; }
.wfig.wgui { max-width:520px; margin:0; }
.wfig .cur { fill:none; stroke:var(--accent); stroke-width:2.2; pointer-events:none; }
.wline { font-size:14px; line-height:1.5; margin:6px 0 0; min-height:1.5em; overflow-wrap:anywhere; }
`;
const file = process.argv[2];
let html = fs.readFileSync(file, 'utf8');
const cut = (a, b) => { const i = html.indexOf(a), j = html.indexOf(b); if (i < 0 || j < i) throw new Error('anchor not found: ' + a); return [i, j]; };
const [a0, a1] = html.includes('<!-- edit-gui:start -->')
  ? [html.indexOf('<!-- edit-gui:start -->'), html.indexOf('<p>Each edit is aimed at')]
  : cut('<p>Each figure below draws', '<p>Each edit is aimed at');
html = html.slice(0, a0) + block + html.slice(a1);
// CSS: drop the rules only the old cards used, add the panel rules once
html = html.replace(/^h3 \{ font-size:1\.05rem[^\n]*\n/m, '').replace(/^\.wgrid \{[^\n]*\n/m, '');
if (html.includes('.wpanel {')) html = html.slice(0, html.indexOf('.wpanel {')) + html.slice(html.indexOf('</style>'));
html = html.replace('</style>', CSS + '</style>');
fs.writeFileSync(file, html);
console.log('wrote', file);
