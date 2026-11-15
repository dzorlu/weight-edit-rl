"""Build the trace viewer: viewer/site/index.html plus one data file per trajectory.

    python3 viewer/build.py        # from the repo root
    open viewer/site/index.html    # needs a local web server for the data files: python3 -m http.server -d viewer/site
"""
import json, os

RUNS = [
    dict(key="opus", group="Searches on Qwen3.5-0.8B", dir="searches/opus-5.5", editor="Claude Opus 5.5", prompt="base", note="The driver was interrupted at turn 135, as the editor was finalizing; the edit saved at that moment is the one evaluated."),
    dict(key="inkling-1", group="Searches on Qwen3.5-0.8B", dir="searches/inkling-2-small-1", editor="Inkling 2 Small, first search", prompt="methods", note="Prompt includes a section describing the methods Opus used, without naming the token, layer or unit it found."),
    dict(key="inkling-2", group="Searches on Qwen3.5-0.8B", dir="searches/inkling-2-small-2", editor="Inkling 2 Small, second search", prompt="methods", note="Same prompt as the first Inkling search. It found a different edit: 11 'Wait' columns zeroed and 8 end-of-thinking columns doubled."),
]
import glob, re
STEP_OFFSET = {"nolib1": 1}  # nolib1 branched from hero3 after step 1, so its first step is step 2
for exp in sorted(os.path.basename(p) for p in glob.glob(os.path.join("results", "rl", "*")) if os.path.exists(os.path.join(p, "config.json"))):  # RL runs: one entry per rollout
    cfg = os.path.join("results", "rl", exp, "config.json")
    for f in sorted(glob.glob(os.path.join("results", "rl", exp, f"{exp}-s*-r*.jsonl"))):
        st, ro = re.search(r"-s(\d+)-r(\d+)\.jsonl$", f).groups()
        RUNS.append(dict(key=f"rl-{exp}-s{st}-r{ro}", group=f"RL run {exp} (Inkling 2 Small{', no notes, branched from hero3 after step 1' if exp == 'nolib1' else ', with notes' if exp == 'hero3' else ''})", transcript=f, config=cfg,
                         editor=f"RL step {int(st) + 1 + STEP_OFFSET.get(exp, 0)}, rollout {int(ro) + 1}", budget=json.load(open(cfg)).get("forward_budget", 32),
                         note="A rollout from an RL run. Step 1 uses the untrained Inkling; later steps use the checkpoint after each update. The reward is the accuracy gain on hidden answers."))
import ast


def const(path, name):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


PROMPT, METHODS, PROTOCOL = const("harness/prompts.py", "PROMPT"), const("editors/tinker_openai.py", "METHODS"), const("editors/tinker_openai.py", "PROTOCOL")
TOOL_NOTE = "\n\n[The two tools, python and forward, were also given to this editor as function definitions.]"


def prompt_for(run, d):
    if "config" in run:
        return json.load(open(run["config"]))["prompt"] + TOOL_NOTE
    return PROMPT + METHODS + PROTOCOL if run["prompt"] == "methods" else PROMPT + TOOL_NOTE


os.makedirs("viewer/site/data", exist_ok=True)
index = []
for run in RUNS:
    d = os.path.join("results", run["dir"]) if "dir" in run else os.path.dirname(run["transcript"])
    load = lambda f: json.load(open(os.path.join(d, f))) if ("dir" in run and os.path.exists(os.path.join(d, f))) else None
    turns, rl_summary = {}, None
    for line in open(run.get("transcript") or os.path.join(d, "transcript.jsonl")):
        r = json.loads(line)
        if "summary" in r:
            rl_summary = r
        if "turn" not in r or "error" in r:
            continue
        t = turns.setdefault(r["turn"], {"turn": r["turn"]})
        if "t" in r:
            t.update(raw=r.get("raw", ""), reasoning=r.get("reasoning"), final=r.get("final"), minutes=round(r["t"] / 60, 1),
                     prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"])
        else:
            t.update(tool=r["tool"], input=r["input"], result=r["result"], used_f=r["used_f"], used_py=r["used_py"])
    tl = [turns[k] for k in sorted(turns)]
    s, h = load("search_summary.json"), load("heldout.json")
    tools = [t for t in tl if "tool" in t]
    meta = dict(key=run["key"], group=run["group"], budget=run.get("budget", 50), editor=run["editor"], note=run["note"], turns=len(tl),
                forward=tools[-1]["used_f"] if tools else 0, python=tools[-1]["used_py"] if tools else 0,
                minutes=round(max((t.get("minutes") or 0) for t in tl)), finished=bool(s), ended=(s or {}).get("ended"))
    if h:
        meta["heldout"] = dict(original=h["original"]["accuracy_pct"], edited=(h.get("edited") or {}).get("accuracy_pct"),
                               paired=h.get("paired_edited_minus_original"), tensors=h.get("edited_tensors"))
    if "transcript" in run:
        meta["finished"] = bool(rl_summary)
        if rl_summary:
            meta["reward"] = rl_summary["summary"]["reward"]; meta["ended"] = rl_summary["summary"]["ended"]
            meta["edited_tensors"] = len(rl_summary["summary"].get("edited_tensors") or [])
    index.append(meta)
    json.dump({"meta": meta, "prompt": prompt_for(run, d), "turns": tl, "final_text": (s or {}).get("final_text") or (rl_summary or {}).get("final_text")}, open(f"viewer/site/data/{run['key']}.json", "w"))
    print(run["key"], len(tl), "turns", round(os.path.getsize(f"viewer/site/data/{run['key']}.json") / 1e6, 2), "MB", meta.get("heldout"))

page = open("viewer/template.html").read().replace("__INDEX__", json.dumps(index).replace("</", "<\\/"))
open("viewer/site/index.html", "w").write(page)
print("viewer/site/index.html", round(os.path.getsize("viewer/site/index.html") / 1e3), "KB")
