"""Write a live status page for one RL run from its step record and rollout logs.

    python3 viewer/run_status.py hero3 out.html
"""
import glob, html, json, os, sys, time

exp, out = sys.argv[1], sys.argv[2]
d = os.path.join("results", "rl", exp)
cfg = json.load(open(os.path.join(d, "config.json")))
steps = [json.loads(l) for l in open(os.path.join(d, "steps.jsonl"))] if os.path.exists(os.path.join(d, "steps.jsonl")) else []
base = next((s for s in steps if s.get("event") == "reward_base"), None)
done_steps = [s for s in steps if s.get("event") == "step"]
e = lambda x: html.escape(str(x))
sgn = lambda v: "" if v is None else f"{v * 100:+.1f}"


def rollout_rows(step):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"{exp}-s{step}-r*.jsonl")), key=lambda p: int(p.rsplit("-r", 1)[1].split(".")[0])):
        recs = [json.loads(l) for l in open(f)]
        m = [r for r in recs if "t" in r]; t = [r for r in recs if "tool" in r]; s = [r for r in recs if "summary" in r]
        best = None
        for r in t:
            if r["tool"] == "forward" and not r["result"].startswith("["):
                try:
                    p = json.loads(r["result"]).get("paired_edited_minus_original")
                    n = json.loads(r["result"])["original"]["n"]
                    if p and n >= 64 and (best is None or p["diff_pp"] > best[0]):
                        best = (p["diff_pp"], p["se_pp"], n)
                except Exception:
                    pass
        rid = os.path.basename(f)[:-6]
        rows.append(dict(rid=rid, n=int(rid.rsplit("-r", 1)[1]) + 1, turns=len(m), minutes=round(m[-1]["t"] / 60) if m else 0,
                         fwd=t[-1]["used_f"] if t else 0, web=sum(r["tool"] in ("web_search", "web_read") for r in t),
                         restarts=max([r.get("target_restarts", 0) for r in t] or [0]), best=best,
                         done=s[0]["summary"] if s else None))
    return rows


cur = len(done_steps)
parts = [f"<h1>Live RL run: {e(exp)}</h1>",
         f"<div class='meta'>Updated {time.strftime('%H:%M %Z, %d %b %Y')}. Inkling 2 Small learns to edit Qwen3.5-0.8B; the reward is the accuracy gain of its submitted edit on the reward set. "
         f"<a href='../'>map of the edits</a> · <a href='../traces/'>trajectories</a> · <a href='../changelog/'>changelog</a> · <a href='https://github.com/dzorlu/weight-edit-rl/tree/main/results/rl/{e(exp)}'>raw logs</a></div>",
         f"<p>Settings: {cfg['steps']} steps of {cfg['group_size']} rollouts, {cfg['forward_budget']} forward passes and {cfg['rollout_minutes']} minutes per rollout, "
         f"reward on {cfg['reward_items']} problems × {cfg.get('reward_n', 2)} samples, web search {'on' if cfg.get('web') else 'off'}. "
         + (f"Original weights on the reward set: {base['accuracy'] * 100:.1f}% correct, {base['hit_cap'] * 100:.0f}% hit the token cap." if base else "") + "</p>",
         "<h2>Steps</h2>"]
if done_steps:
    parts.append("<div class='tw'><table><tr><th>Step</th><th>Mean reward</th><th>Best</th><th>Rewards</th><th>KL sampler–trainer</th><th>Entropy</th><th>Minutes</th></tr>")
    for s in done_steps:
        rw = s.get("rewards") or []
        parts.append(f"<tr><td>{s['step'] + 1}</td><td>{sgn(s.get('mean_reward'))}</td><td>{sgn(max(rw)) if rw else ''}</td><td>{', '.join(sgn(r) for r in rw)}</td>"
                     f"<td>{e(s.get('kl_sample_train_v1', '–'))}</td><td>{e(s.get('entropy', '–'))}</td><td>{e(s.get('rollout_minutes'))}</td></tr>")
    parts.append("</table></div><p class='meta'>Rewards are in points: accuracy with the edit minus accuracy of the original weights, on 256 paired answers.</p>")
else:
    parts.append("<p>No step has finished yet.</p>")
if cur < cfg["steps"]:
    rows = rollout_rows(cur)
    parts.append(f"<h2>Step {cur + 1}, in progress</h2>")
    if rows:
        parts.append("<div class='tw'><table><tr><th>Rollout</th><th>Turns</th><th>Minutes</th><th>Forward passes</th><th>Web calls</th><th>Best paired test so far</th><th>State</th></tr>")
        for r in rows:
            best = f"{r['best'][0]:+.1f} ± {r['best'][1]}, {r['best'][2]} answers per side" if r["best"] else "–"
            st = f"done: reward {sgn(r['done']['reward'])}, {e(r['done']['ended'])}" if r["done"] else "running"
            if r["restarts"]:
                st += f", target restarted {r['restarts']}×"
            parts.append(f"<tr><td><a href='../traces/#rl-{e(r['rid'])}'>{r['n']}</a></td><td>{r['turns']}</td><td>{r['minutes']}</td><td>{r['fwd']} / {cfg['forward_budget']}</td><td>{r['web']}</td><td>{best}</td><td>{st}</td></tr>")
        parts.append("</table></div><p class='meta'>A paired test runs the original and the edited weights on the same problems. Tests on fewer than 64 answers are not shown.</p>")
    else:
        parts.append("<p>Rollouts are starting.</p>")

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
        f"<meta http-equiv='refresh' content='600'><title>Live RL run</title><style>{style}</style>{theme}</head><body>"
        f"<header class='topbar'><a class='brand' href='../../../'>Tinkerings</a></header><main>{''.join(parts)}</main></body></html>\n")
open(out, "w").write(page)
print("wrote", out)
