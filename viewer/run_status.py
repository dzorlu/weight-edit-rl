"""Write the run pages: <out_dir>/index.html links to one page per run, <out_dir>/<run>/index.html.

    python3 viewer/run_status.py <out_dir> hero3 nolib1:hero3@1
An argument `name:parent@k` marks a run branched from `parent` after its step k. Its page lists the parent's steps
1 to k (marked as shared) and then its own steps, numbered from k + 1.
"""
import glob, html, json, os, sys, time

out_dir, specs = sys.argv[1], sys.argv[2:]
e = lambda x: html.escape(str(x))
sgn = lambda v: "" if v is None else f"{v * 100:+.1f}"
NOTES = {"hero3": "with the shared notes library and web tools",
         "nolib1": "no notes library, no web tools",
         "rebase2": "No notes, no web tools, learning rate 2e-4; the starting model is updated to the best edit so far every 2 steps.",
         "rebase2b": "Continues rebase2 from its last checkpoint and starting model (rebase2-k2). No notes, no web tools, learning rate 2e-4; the starting model is updated every 2 steps.",
         "rebase1": "No notes, no web tools, learning rate 2e-4; the starting model is updated to the best edit so far every step."}


def load(exp):
    d = os.path.join("results", "rl", exp)
    cfg = json.load(open(os.path.join(d, "config.json")))
    steps = [json.loads(l) for l in open(os.path.join(d, "steps.jsonl"))] if os.path.exists(os.path.join(d, "steps.jsonl")) else []
    test = json.load(open(os.path.join(d, "test.json"))) if os.path.exists(os.path.join(d, "test.json")) else {}
    return d, cfg, steps, test


def rollout_rows(d, exp, step):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"{exp}-s{step}-r*.jsonl")), key=lambda p: int(p.rsplit("-r", 1)[1].split(".")[0])):
        recs = [json.loads(l) for l in open(f)]
        m = [r for r in recs if "t" in r]; t = [r for r in recs if "tool" in r]; s = [r for r in recs if "summary" in r]
        best = None
        for r in t:
            if r["tool"] == "forward" and not r["result"].startswith("["):
                try:
                    j = json.loads(r["result"]); p = j.get("paired_edited_minus_original"); n = j["original"]["n"]
                    if p and n >= 64 and (best is None or p["diff_pp"] > best[0]):
                        best = (p["diff_pp"], p["se_pp"], n)
                except Exception:
                    pass
        rid = os.path.basename(f)[:-6]
        rows.append(dict(rid=rid, n=int(rid.rsplit("-r", 1)[1]) + 1, turns=len(m), minutes=round(m[-1]["t"] / 60) if m else 0,
                         fwd=t[-1]["used_f"] if t else 0, web=sum(r["tool"] in ("web_search", "web_read") for r in t),
                         restarts=max([r.get("target_restarts", 0) for r in t] or [0]), best=best, done=s[0]["summary"] if s else None))
    return rows


runs = []
for spec in specs:
    name, _, parent = spec.partition(":")
    par, _, k = parent.partition("@")
    d, cfg, steps, test = load(name)
    runs.append(dict(name=name, parent=par or None, offset=int(k or 0), d=d, cfg=cfg, test=test,
                     done=[s for s in steps if s.get("event") == "step"], ended=any(s.get("event") in ("done", "stopped") for s in steps)))

byname = {r["name"]: r for r in runs}
cfg0 = runs[0]["cfg"]
STAMP = time.strftime('%H:%M %Z, %d %b %Y')
INTRO = (f"<p>Each step: {cfg0['group_size']} rollouts from the current checkpoint, {cfg0['forward_budget']} forward passes and {cfg0['rollout_minutes']} minutes each, "
         "then one training update. Reward: accuracy of the submitted edit minus the original weights' 7.4%, on 256 hidden answers (the reward set), in points. "
         "Test: the same edit on MATH-500 (original 6.6%), which training never uses.</p>")
HEAD = "<div class='tw'><table><tr><th>Step</th><th>Mean reward</th><th>Rewards</th><th>Mean test</th><th>Test per rollout</th><th>Entropy</th><th>Note</th></tr>"


def state_of(r):
    return "stopped" if (r["ended"] or r["name"] == "hero3") else "running"


def step_row(r, s, n, note):
    rids = [x["rid"] for x in s.get("rollouts", [])]
    tests = [r["test"][i]["paired"]["diff_pp"] for i in rids if i in r["test"] and r["test"][i].get("paired")]
    mt = f"{sum(tests) / len(tests):+.1f} ({len(tests)} edits)" if tests else "being scored"
    tl = ", ".join(f"{r['test'][i]['paired']['diff_pp']:+.1f}" if i in r["test"] and r["test"][i].get("paired") else ("no edit" if i in r["test"] else "–") for i in rids)
    rw = s.get("rewards") or []
    return (f"<tr><td>{n}</td><td>{sgn(s.get('mean_reward'))}</td><td>{', '.join(sgn(x) for x in rw)}</td>"
            f"<td>{mt}</td><td>{tl if tests else ''}</td><td>{e(s.get('entropy', '–'))}</td><td>{note}</td></tr>")


def step_detail(r, s, n, note):
    """One row per rollout of a finished step, then the mean over rollouts."""
    ro = s.get("rollouts", [])
    if not ro:
        return []
    def test_of(rid):
        t = r["test"].get(rid)
        return t["paired"]["diff_pp"] if t and t.get("paired") else None
    cols = ["Rollout", "Reward", "Test", "Edited accuracy", "Hit cap", "Tensors edited", "Turns", "Minutes", "Forward passes", "Python calls", "Web calls", "Ended"]
    p = [f"<h3>Step {n}{f', {note}' if note else ''}</h3><div class='tw'><table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"]
    vals = []
    for x in ro:
        t = test_of(x["rid"])
        v = dict(reward=x.get("reward"), test=t, acc=x.get("edited_accuracy"), cap=x.get("edited_hit_cap"),
                 tens=len(x.get("edited_tensors") or []), turns=x.get("turns"), minutes=x.get("minutes"),
                 fwd=x.get("forward_used"), py=x.get("python_used"), web=x.get("web_used"))
        vals.append(v)
        tt = f"{t:+.1f}" if t is not None else ("no edit" if x["rid"] in r["test"] else "being scored")
        pct = lambda a: "–" if a is None else f"{a * 100:.1f}%"
        p.append(f"<tr><td><a href='../../traces/#rl-{e(x['rid'])}'>{int(x['rid'].rsplit('-r', 1)[1]) + 1}</a></td><td>{sgn(v['reward'])}</td><td>{tt}</td>"
                 f"<td>{pct(v['acc'])}</td><td>{pct(v['cap'])}</td><td>{v['tens']}</td><td>{v['turns']}</td><td>{v['minutes']}</td>"
                 f"<td>{v['fwd']}</td><td>{'–' if v['py'] is None else v['py']}</td><td>{v['web']}</td><td>{e(x.get('ended', ''))}</td></tr>")
    def mean(k):
        xs = [v[k] for v in vals if v[k] is not None]
        return sum(xs) / len(xs) if xs else None
    f1 = lambda k: "–" if mean(k) is None else f"{mean(k):.1f}"
    tm = mean("test")
    p.append(f"<tr class='avg'><td>Mean</td><td>{sgn(mean('reward'))}</td><td>{'being scored' if tm is None else f'{tm:+.1f}'}</td>"
             f"<td>{'–' if mean('acc') is None else f'{mean(chr(97)+chr(99)+chr(99)) * 100:.1f}%'}</td><td>{'–' if mean('cap') is None else f'{mean(chr(99)+chr(97)+chr(112)) * 100:.1f}%'}</td>"
             f"<td>{f1('tens')}</td><td>{f1('turns')}</td><td>{f1('minutes')}</td><td>{f1('fwd')}</td><td>{f1('py')}</td><td>{f1('web')}</td><td></td></tr>")
    p.append("</table></div>")
    return p


COLS = ["Rollout", "Reward", "Test", "Edited accuracy", "Hit cap", "Tensors edited", "Turns", "Minutes", "Forward passes", "Python calls", "Ended"]


def step_table(r, rows, title):
    """rows: per-rollout dicts (finished summaries or live progress). One row each, then the mean."""
    pct = lambda a: "–" if a is None else f"{a * 100:.1f}%"
    def test_of(rid):
        t = r["test"].get(rid)
        if t and t.get("paired"):
            return t["paired"]["diff_pp"]
        return None
    out = [f"<h2>{title}</h2><div class='tw'><table><tr>" + "".join(f"<th>{c}</th>" for c in COLS) + "</tr>"]
    vals = []
    for x in rows:
        done = x.get("reward") is not None
        t = test_of(x["rid"])
        tt = f"{t:+.1f}" if t is not None else ("no edit" if x["rid"] in r["test"] else "–")
        v = dict(reward=x.get("reward"), test=t, acc=x.get("edited_accuracy"), cap=x.get("edited_hit_cap"),
                 tens=len(x["edited_tensors"]) if done else None, turns=x.get("turns"), minutes=x.get("minutes"),
                 fwd=x.get("forward_used"), py=x.get("python_used"))
        vals.append(v)
        n = int(x["rid"].rsplit("-r", 1)[1]) + 1
        out.append(f"<tr><td><a href='../../traces/#rl-{e(x['rid'])}'>{n}</a></td><td>{sgn(v['reward']) if done else 'running'}</td><td>{tt}</td>"
                   f"<td>{pct(v['acc'])}</td><td>{pct(v['cap'])}</td><td>{'–' if v['tens'] is None else v['tens']}</td><td>{v['turns']}</td>"
                   f"<td>{v['minutes']}</td><td>{v['fwd']}</td><td>{'–' if v['py'] is None else v['py']}</td><td>{e(x.get('ended', '')) if done else ''}</td></tr>")
    def mean(k):
        xs = [v[k] for v in vals if v[k] is not None and v["reward"] is not None]
        return sum(xs) / len(xs) if xs else None
    f1 = lambda k: "–" if mean(k) is None else f"{mean(k):.1f}"
    nd = sum(v["reward"] is not None for v in vals)
    lab = "Mean" if nd == len(vals) else f"Mean of {nd} finished"
    tm = mean("test")
    out.append(f"<tr class='avg'><td>{lab}</td><td>{sgn(mean('reward'))}</td><td>{'–' if tm is None else f'{tm:+.1f}'}</td>"
               f"<td>{pct(mean('acc'))}</td><td>{pct(mean('cap'))}</td><td>{f1('tens')}</td><td>{f1('turns')}</td><td>{f1('minutes')}</td>"
               f"<td>{f1('fwd')}</td><td>{f1('py')}</td><td></td></tr></table></div>")
    return out


def live_rows(r, step):
    rows = []
    for x in rollout_rows(r["d"], r["name"], step):
        if x["done"]:
            rows.append(x["done"])
        else:
            rows.append(dict(rid=x["rid"], turns=x["turns"], minutes=x["minutes"], forward_used=x["fwd"], python_used=None))
    return rows


def ancestor_steps(r):
    """The steps a run inherits: its parent's steps up to the branch point, including what the parent inherited."""
    if not r["parent"] or r["parent"] not in byname:
        return []
    par = byname[r["parent"]]
    out = [(o, s) for o, s in ancestor_steps(par) if s["step"] + 1 + o["offset"] <= r["offset"]]
    out += [(par, s) for s in par["done"] if s["step"] + 1 + par["offset"] <= r["offset"]]
    return out


def run_page(r):
    head = (f"Branched from {e(r['parent'])} after step {r['offset']}; steps 1 to {r['offset']} are shared. {NOTES.get(r['name'], '')}"
            if r["parent"] else "With the shared notes library and web tools.")
    p = [f"<h1>{e(r['name'])}</h1><div class='meta'>{head} Updated {STAMP} · <a href='../'>all runs</a></div>"]
    for owner, s in ancestor_steps(r):
        p += step_table(owner, s.get("rollouts", []), f"Step {s['step'] + 1 + owner['offset']} (shared with {e(owner['name'])})")
    for s in r["done"]:
        p += step_table(r, s.get("rollouts", []), f"Step {s['step'] + 1 + r['offset']}")
    cur = len(r["done"])
    if state_of(r) == "running" and cur < r["cfg"]["steps"]:
        rows = live_rows(r, cur)
        if rows:
            p += step_table(r, rows, f"Step {cur + 1 + r['offset']} (running)")
    p.append("<p class='meta'>Reward: edited minus original accuracy on the reward set (256 answers, original 7.4%), in points. "
             "Test: the same on MATH-500 (original 6.6%), not used for training; scored after a step finishes, – until then. Hit cap: share of answers reaching 8,192 tokens (original 93%).</p>")
    return p, "../../../"


def index_page():
    p = ["<h1>RL runs</h1><ul>"]
    for r in runs:
        p.append(f"<li><a href='{e(r['name'])}/'>{e(r['name'])}</a></li>")
    p.append("</ul>")
    return p, "../../"


style = """:root { --ink:#1d1d1b; --muted:#5f5e5a; --line:#e4e3df; --bg:#fcfcfb; --th:#f0efec; --accent:#2a78d6; }
html.dark { --ink:#e8e6e1; --muted:#9c9a92; --line:#2c2c2a; --bg:#151514; --th:#242422; --accent:#7fb2f0; color-scheme:dark; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 -apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.topbar { height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 2.1rem; border-bottom:1px solid var(--line); }
.topbar .brand { font-size:1.25rem; font-weight:650; color:var(--ink); }
main { max-width:900px; margin:0 auto; padding:32px 20px 80px; }
h1 { font-size:1.8rem; margin:0 0 .3rem; } h2 { font-size:1.25rem; margin:2rem 0 .6rem; }
.meta { color:var(--muted); font-size:.85rem; }
.tw { overflow-x:auto; } table { border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; }
th, td { border:1px solid var(--line); padding:5px 9px; text-align:left; } th { background:var(--th); }\ntr.avg td { font-weight:600; background:var(--th); } h3 { font-size:1.05rem; margin:1.4rem 0 .4rem; }"""
theme = """<script>(function () { try { let v = localStorage.getItem('quarto-color-scheme');
if (v == null && window.matchMedia('(prefers-color-scheme: dark)').matches) v = 'alternate';
if (v === 'alternate') document.documentElement.classList.add('dark'); } catch (e) {} })();</script>"""
def write(path, parts, brand):
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<meta http-equiv='refresh' content='600'><title>RL runs</title><style>{style}</style>{theme}</head><body>"
            f"<header class='topbar'><a class='brand' href='{brand}'>Tinkerings</a></header><main>{''.join(parts)}</main></body></html>\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(page)


write(os.path.join(out_dir, "index.html"), *index_page())
for r in runs:
    write(os.path.join(out_dir, r["name"], "index.html"), *run_page(r))
print("wrote", out_dir, [r["name"] for r in runs])
