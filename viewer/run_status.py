"""Write the live status page for the RL runs: a combined step table across runs, then each run's latest step.

    python3 viewer/run_status.py out.html hero3 nolib1:hero3@1
An argument `name:parent@k` marks a run branched from `parent` after its step k: its first step is shown as step k + 1.
"""
import glob, html, json, os, sys, time

out, specs = sys.argv[1], sys.argv[2:]
e = lambda x: html.escape(str(x))
sgn = lambda v: "" if v is None else f"{v * 100:+.1f}"
NOTES = {"hero3": "with the shared notes library and web tools",
         "nolib1": "no notes library, no web tools"}


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

cfg0 = runs[0]["cfg"]
parts = ["<h1>Live RL runs</h1>",
         f"<div class='meta'>Updated {time.strftime('%H:%M %Z, %d %b %Y')}. Inkling 2 Small learns to edit Qwen3.5-0.8B. "
         "<a href='../'>map of the edits</a> · <a href='../traces/'>trajectories</a> · <a href='../changelog/'>changelog</a></div>",
         f"<p>Each step: {cfg0['group_size']} rollouts from the current checkpoint, {cfg0['forward_budget']} forward passes and {cfg0['rollout_minutes']} minutes each, "
         "then one training update. Reward: accuracy of the submitted edit minus the original weights' 7.4%, on 256 hidden answers (the reward set), in points. "
         "Test: the same edit on MATH-500 (original 6.6%), which training never uses.</p>"]

# the runs and how they relate
parts.append("<h2>Runs</h2><ul>")
for r in runs:
    rel = f"branched from {e(r['parent'])} after step {r['offset']}, so its first step is step {r['offset'] + 1}" if r["parent"] else "starts from the untrained model"
    state = "stopped" if (r["ended"] or r["name"] == "hero3") else "running"
    parts.append(f"<li><b>{e(r['name'])}</b>: {NOTES.get(r['name'], '')}; {rel}; {state}. "
                 f"<a href='https://github.com/dzorlu/weight-edit-rl/tree/main/results/rl/{e(r['name'])}'>logs</a></li>")
parts.append("</ul>")

# step tables: steps shared by every run first, then each run's own steps
def step_row(r, s, label):
    n = s["step"] + 1 + r["offset"]
    rids = [x["rid"] for x in s.get("rollouts", [])]
    tests = [r["test"][i]["paired"]["diff_pp"] for i in rids if i in r["test"] and r["test"][i].get("paired")]
    mt = f"{sum(tests) / len(tests):+.1f} ({len(tests)} edits)" if tests else "pending"
    tl = ", ".join(f"{r['test'][i]['paired']['diff_pp']:+.1f}" if i in r["test"] and r["test"][i].get("paired") else ("no edit" if i in r["test"] else "–") for i in rids)
    rw = s.get("rewards") or []
    return (f"<tr><td>{n}</td><td>{label}</td><td>{sgn(s.get('mean_reward'))}</td><td>{', '.join(sgn(x) for x in rw)}</td>"
            f"<td>{mt}</td><td>{tl if tests else ''}</td><td>{e(s.get('entropy', '–'))}</td></tr>")

HEAD = "<div class='tw'><table><tr><th>Step</th><th>Run</th><th>Mean reward</th><th>Rewards</th><th>Mean test</th><th>Test per rollout</th><th>Entropy</th></tr>"
root = next((r for r in runs if not r["parent"]), None)
shared_k = min([r["offset"] for r in runs if r["parent"]] or [0])
if root and shared_k:
    parts.append(f"<h2>Step{'s' if shared_k > 1 else ''} 1{'' if shared_k == 1 else f' to {shared_k}'}, shared by both runs</h2>")
    parts.append(f"<p class='meta'>These steps ran once, in {e(root['name'])}. Its checkpoint after step {shared_k} is where the other run{'s' if len(runs) > 2 else ''} branched off.</p>" + HEAD)
    for s in root["done"][:shared_k]:
        parts.append(step_row(root, s, "shared"))
    parts.append("</table></div>")
for r in runs:
    own = r["done"][shared_k:] if (r is root and shared_k) else r["done"]
    first = (shared_k if r is root else r["offset"]) + 1
    state = "stopped" if (r["ended"] or r["name"] == "hero3") else "running"
    parts.append(f"<h2>{e(r['name'])}: steps from {first}, {NOTES.get(r['name'], '')} ({state})</h2>" + HEAD)
    for s in own:
        parts.append(step_row(r, s, e(r["name"])))
    cur = len(r["done"])
    if state == "running" and cur < r["cfg"]["steps"]:
        parts.append(f"<tr><td>{cur + 1 + r['offset']}</td><td>{e(r['name'])}</td><td colspan='5'>in progress</td></tr>")
    parts.append("</table></div>")

# in-progress step of each running run
for r in runs:
    cur = len(r["done"])
    if r["ended"] or r["name"] == "hero3" or cur >= r["cfg"]["steps"]:
        continue
    rr = rollout_rows(r["d"], r["name"], cur)
    parts.append(f"<h2>{e(r['name'])}, step {cur + 1 + r['offset']}, in progress</h2>")
    if not rr:
        parts.append("<p>Rollouts are starting.</p>"); continue
    parts.append("<div class='tw'><table><tr><th>Rollout</th><th>Turns</th><th>Minutes</th><th>Forward passes</th><th>Best paired test so far</th><th>State</th></tr>")
    for x in rr:
        best = f"{x['best'][0]:+.1f} ± {x['best'][1]}, {x['best'][2]} answers per side" if x["best"] else "–"
        st = f"done: reward {sgn(x['done']['reward'])}, {e(x['done']['ended'])}" if x["done"] else "running"
        if x["restarts"]:
            st += f", target restarted {x['restarts']}×"
        parts.append(f"<tr><td><a href='../traces/#rl-{e(x['rid'])}'>{x['n']}</a></td><td>{x['turns']}</td><td>{x['minutes']}</td>"
                     f"<td>{x['fwd']} / {r['cfg']['forward_budget']}</td><td>{best}</td><td>{st}</td></tr>")
    parts.append("</table></div><p class='meta'>A paired test runs the original and the edited weights on the same train problems. Tests on fewer than 64 answers are not shown.</p>")

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
th, td { border:1px solid var(--line); padding:5px 9px; text-align:left; } th { background:var(--th); }"""
theme = """<script>(function () { try { let v = localStorage.getItem('quarto-color-scheme');
if (v == null && window.matchMedia('(prefers-color-scheme: dark)').matches) v = 'alternate';
if (v === 'alternate') document.documentElement.classList.add('dark'); } catch (e) {} })();</script>"""
page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<meta http-equiv='refresh' content='600'><title>Live RL runs</title><style>{style}</style>{theme}</head><body>"
        f"<header class='topbar'><a class='brand' href='../../../'>Tinkerings</a></header><main>{''.join(parts)}</main></body></html>\n")
open(out, "w").write(page)
print("wrote", out)
