"""Summarise the gradient-tool ablation (gradabl-A: current tools, gradabl-B: + gradient tool).
Downloads the eval folders from the glimmer-runs volume into /tmp/gradabl and prints a table plus gradient-tool usage.
    python3 results/eval/grad_tool_ablation.py
"""
import glob, json, os, re, subprocess

T = "/tmp/gradabl"
os.makedirs(T, exist_ok=True)


def get(path, dst):
    subprocess.run(["uvx", "-q", "modal", "volume", "get", "--force", "glimmer-runs", path, dst], capture_output=True)


for arm in ("gradabl-A", "gradabl-B"):
    d = f"{T}/{arm}"
    os.makedirs(d, exist_ok=True)
    ls = subprocess.run(["uvx", "-q", "modal", "volume", "ls", "glimmer-runs", f"/eval/{arm}"], capture_output=True, text=True).stdout.split()
    for p in ls:
        b = os.path.basename(p)
        if b.endswith((".jsonl", ".json")):
            get(f"/eval/{arm}/{b}", f"{d}/{b}")


def se_of(p):
    return p.get("se_pp") or 0


for arm in ("gradabl-A", "gradabl-B"):
    d = f"{T}/{arm}"
    summ = json.load(open(f"{d}/summary.json")) if os.path.exists(f"{d}/summary.json") else {}
    scores = json.load(open(f"{d}/scores.json")) if os.path.exists(f"{d}/scores.json") else []
    scores = scores if isinstance(scores, list) else scores.get("scores", [])
    print(f"== {arm}: MATH-500 gains {summ.get('gains_pp')}, mean {summ.get('gain_mean_pp')}, reward-set mean {summ.get('reward_mean_pp')}")
    for f in sorted(glob.glob(f"{d}/{arm}-s0-r*.jsonl")):
        R = [json.loads(l) for l in open(f)]
        s = [r for r in R if "summary" in r]
        s = s[0]["summary"] if s else {}
        tools = [r for r in R if "tool" in r]
        grads = [r for r in tools if r["tool"] == "forward" and '"gradient"' in r["input"]]
        ok_grads = [r for r in grads if not r["result"].startswith("[")]
        best_z = None
        for r in tools:
            if r["tool"] != "forward" or r["result"].startswith("["):
                continue
            try:
                p = json.loads(r["result"]).get("paired_edited_minus_original")
            except Exception:
                continue
            if p and se_of(p):
                z = p["diff_pp"] / se_of(p)
                best_z = z if best_z is None else max(best_z, z)
        fin = [r for r in R if r.get("final")]
        txt = fin[-1]["final"] if fin else ""
        i = txt.lower().find("what i changed")
        print(f" {os.path.basename(f)[:-6]}: reward {100 * (s.get('reward') or 0):+.1f}, tensors {len(s.get('edited_tensors') or [])}, "
              f"ended {s.get('ended')}, gradient calls {len(grads)} ({len(ok_grads)} ok), best paired z {best_z if best_z is None else round(best_z, 1)}")
        print("   edit:", (txt[i:i + 300] if i >= 0 else txt[:300]).replace("\n", " "))
        for r in ok_grads[:2]:
            nxt = [m for m in R if m.get("turn", -1) == r["turn"] + 1 and "t" in m]
            print("   gradient request:", r["input"][:240].replace("\n", " "))
            if nxt:
                print("   next turn reasoning:", (nxt[0].get("reasoning") or "")[:400].replace("\n", " "))
