"""Compare editor behaviour in hero3 step 1 (s0), step 2 (s1) and the Opus 5.5 standalone search.

Reads the rollout logs only. Prints markdown tables and per-rollout details.
Run from anywhere:  python3 results/rl/hero3/analyze_steps.py [--detail]
"""
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPUS = HERE.parent.parent / "searches" / "opus-5.5" / "transcript.jsonl"

# Split of a paired test, from its first problem id (constant across rollouts of hero3;
# the cached original accuracy on the full split is 6.25 / 12.5 / 9.38 %).
HERO_SPLITS = {
    "math/geometry/551": "select",
    "math/algebra/1277": "confirm_1",
    "math/counting_and_probability/406": "confirm_2",
}
# Opus built its own splits (notes.md): sel = selection set, conf / conf2 = held-out sets.
OPUS_SPLITS = {
    "math/counting_and_probability/98": "select",
    "math/prealgebra/367": "confirm_1",
    "math/geometry/593": "confirm_1",
    "math/algebra/862": "confirm_2",
    "math/algebra/180": "diagnosis",
}

# Final edits, annotated by hand from each rollout's FINAL SUBMISSION text (final_text).
# form: col = column zero/flip/scale; dir+ = W + a*d(d^T W) with a > 0; dir- = removal (a < 0);
# dirs: wait = ' Wait' readout, think = '</think>' readout, loop = difference-of-means direction.
EDITS = {
    "opus": dict(form="col flip + dir+", dirs="think (whitened, specific) + L16 unit 998 (Wait unit)",
                 alpha="col998 x(-1); think +2.5 on 9 writers L15-22", basis="own"),
    "s0-r0": dict(form="col zero", dirs="Wait units (diff-of-means + attribution)", alpha="7 cols of L23 down_proj = 0", basis="own"),
    "s0-r1": dict(form="dir+", dirs="think (specific vs top-80 tokens, 5 PCs)", alpha="think +0.45, 48 writers", basis="own"),
    "s0-r2": dict(form="col zero + col scale", dirs="Wait-loop units / anti-Wait units (attribution)", alpha="16 cols = 0, 5 cols x2-3", basis="own"),
    "s0-r3": dict(form="col scale + col zero", dirs="ready-to-stop units (attribution) / loop units", alpha="18 cols x5, 4 cols = 0", basis="own"),
    "s0-r4": dict(form="dir- + dir+", dirs="wait (specific, top-600) + think (specific, top-600)", alpha="wait -1.0, think +2.0, 48 writers", basis="own"),
    "s0-r5": dict(form="none", dirs="-", alpha="-", basis="-"),
    "s0-r6": dict(form="dir-", dirs="loop (diff of means, L23 input)", alpha="-1 on L23 o_proj only", basis="own"),
    "s0-r7": dict(form="dir- + dir+", dirs="wait (plain readout) on 48 + think (plain readout) on L18-23 MLP", alpha="wait -1.0 (48), think +0.6 (6 MLP)", basis="own"),
    "s1-r0": dict(form="dir+ + dir-", dirs="think + wait (specific, top-600)", alpha="think +1.75, wait -1.0, 48 writers", basis="library"),
    "s1-r1": dict(form="dir+ + dir-", dirs="think + wait (specific, top-600)", alpha="think +1.5, wait -1.0, 48 writers", basis="library"),
    "s1-r2": dict(form="dir- + dir+", dirs="wait + think (specific, top-600)", alpha="wait -1.0, think +1.0, 48 writers", basis="library"),
    "s1-r3": dict(form="dir+", dirs="think (specific, top-600)", alpha="think +1.75, 48 writers", basis="library"),
    "s1-r4": dict(form="dir+", dirs="think (specific, top-600)", alpha="think +1.75, 48 writers", basis="library"),
    "s1-r5": dict(form="dir+ (unfinished)", dirs="think (specific)", alpha="think +2.5 being tested at turn 49", basis="library"),
    "s1-r6": dict(form="dir+", dirs="think (specific, top-600)", alpha="think +2.0, 48 writers", basis="library"),
    "s1-r7": dict(form="dir+", dirs="think (specific, top-600)", alpha="think +2.0, 48 writers", basis="library"),
}

# Step 1 MATH-500 test scores (points), from the run notes; r5 submitted nothing.
TEST = {"s0-r0": 0.6, "s0-r1": 11.1, "s0-r2": 16.7, "s0-r3": 15.2, "s0-r4": 32.8, "opus": 27.2}

# Per-unit logit attribution: named, or a readout vector multiplied with down_proj columns.
RE_ATTR = re.compile(r"attrib|contrib|\bdla\b|logit.lens|"
                     r"r_\w+\s*@\s*(Wd|W_down|Wdown|down|W)\b|(Wd|W_down|Wdown|down)\w*(\.T)?\s*@\s*r_", re.I)
# Difference of means: named, or two means subtracted (mean(..) - mean(..), aL - aN, mu_a - mu_b, m_loop - m_fin).
RE_DOM = re.compile(r"diff\w*.{0,4}of.{0,4}means|\bdm\b\s*=|mean\([^)\n]*\)\s*-\s*[\w\[\]\.]*mean\(|"
                    r"\b(a|m|mu|mean)_?[A-Za-z]\w{0,8}\s*-\s*(a|m|mu|mean)_?[A-Za-z]\w{0,8}\b(?=.*\n?)", re.I)
RE_DIR_BUILD = re.compile(r"248069|top.?600|specific")


def load(path):
    return [json.loads(l) for l in open(path)]


def parse_json(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def analyse(path, name, splits):
    recs = load(path)
    turns = [r for r in recs if "reasoning" in r]
    tools = [r for r in recs if "tool" in r]
    summ = None
    for r in recs:
        if "summary" in r:
            summ = r["summary"]
            if isinstance(summ, str):
                summ = ast.literal_eval(summ)
    turn_t = {int(r["turn"]): int(r["t"]) for r in turns}
    out = dict(name=name, finished=summ is not None)

    # outcome
    if summ:
        out.update(reward=summ.get("reward"), ended=summ.get("ended"), minutes=summ.get("minutes"),
                   turns=summ.get("turns"), fwd=summ.get("forward_used"), py=summ.get("python_used"),
                   web=summ.get("web_used", 0), hit_cap=summ.get("edited_hit_cap"),
                   tensors=summ.get("edited_tensors") or [])
    else:
        last = tools[-1] if tools else {}
        out.update(reward=None, ended="unfinished (log ends at turn %s)" % turns[-1]["turn"],
                   minutes=round(int(turns[-1]["t"]) / 60, 1), turns=int(turns[-1]["turn"]),
                   fwd=int(last.get("used_f", 0)), py=int(last.get("used_py", 0)),
                   web=sum(1 for t in tools if t["tool"].startswith("web")), hit_cap=None, tensors=[])
    if name == "opus":
        s = json.load(open(OPUS.parent / "search_summary.json"))
        h = json.load(open(OPUS.parent / "heldout.json"))
        out.update(ended=s["ended"][:40], minutes=round(s["seconds"] / 60, 1), turns=s["turns"],
                   fwd=s["forward_used"], py=s["python_used"], web=0, tensors=h["edited_tensors"],
                   reward=None)
    tn = out["tensors"]
    out["n_tensors"] = len(tn)
    out["kinds"] = dict(down=sum("down_proj" in t for t in tn), o_proj=sum("o_proj" in t for t in tn),
                        out_proj=sum("out_proj" in t and "o_proj" not in t for t in tn))

    # process
    py_calls = [t for t in tools if t["tool"] == "python"]
    fw_calls = [t for t in tools if t["tool"] == "forward"]
    save_calls = [t for t in py_calls if "save_edit(" in t["input"]]
    first_save = int(save_calls[0]["turn"]) if save_calls else None
    out["first_save_turn"] = first_save
    out["first_save_min"] = round(turn_t.get(first_save, 0) / 60, 1) if first_save else None
    out["save_calls"] = len(save_calls)

    paired, tf, failed_fw = [], [], 0
    tf_layers = set()
    edited_gens = 0
    for t in fw_calls:
        inp = parse_json(t["input"])
        res = t["result"]
        if res.startswith("[request failed") or inp is None:
            failed_fw += 1
            continue
        typ = inp.get("type") or (inp.get("args") or {}).get("type")
        if typ == "teacher_forced":
            tf.append(t)
            ret = inp.get("return") or {}
            for k in ("hidden_states", "mlp_act", "attn_out", "mlp_out"):
                v = ret.get(k)
                if isinstance(v, list):
                    tf_layers.update((k, x) for x in v)
            continue
        if typ != "generate":
            continue
        r = parse_json(res) or {}
        probs = inp.get("problems") or []
        if not isinstance(probs, list) or not probs:
            continue
        model = inp.get("model")
        if model in ("both", "edited"):
            edited_gens += 1
        split = splits.get(probs[0], "other(%d)" % len(probs))
        e = r.get("edited") or {}
        o = r.get("original") or {}
        p = r.get("paired_edited_minus_original") or {}
        paired.append(dict(turn=int(t["turn"]), model=model, split=split, size=len(probs) * (inp.get("n") or 1),
                           orig=o.get("accuracy_pct"), edit=e.get("accuracy_pct"), diff=p.get("diff_pp"),
                           cap=e.get("cut_at_max_tokens_pct"), toks=e.get("mean_tokens")))
    out["paired"] = paired
    out["n_both"] = sum(1 for p in paired if p["model"] == "both")
    out["n_edited_gens"] = edited_gens
    sel = [p for p in paired if p["split"] == "select" and p["model"] in ("both", "edited")]
    out["sel_tests"] = len(sel)
    out["sweep"] = len(sel) >= 3
    out["confirm"] = sorted({p["split"] for p in paired if p["split"].startswith("confirm") and p["model"] in ("both", "edited")})
    out["n_tf"] = len(tf)
    layers = {}
    for k, l in tf_layers:
        layers.setdefault(k, set()).add(l)
    out["tf_layers"] = {k: (len(v), min(v), max(v)) for k, v in layers.items()}
    out["failed_fw"] = failed_fw
    out["tracebacks"] = sum("Traceback" in t["result"] for t in py_calls)
    out["repaired"] = sum(r.get("repaired") == "True" for r in turns)
    out["no_call_turns"] = sum(1 for r in turns if r.get("call") in (None, "None", "")
                               and not (r.get("final") or "").lstrip().startswith("FINAL"))
    out["ctx_end"] = int(turns[-1]["prompt_tokens"]) + int(turns[-1]["completion_tokens"])
    code = "\n".join(t["input"] for t in py_calls)
    out["attribution"] = sum(bool(RE_ATTR.search(t["input"])) for t in py_calls)
    out["diff_means"] = sum(bool(RE_DOM.search(t["input"])) and "mean" in t["input"] for t in py_calls)
    out["read_baseline"] = "/practice/baseline" in code
    out["read_splits"] = "splits.json" in code
    lib_reads = [int(t["turn"]) for t in py_calls if re.search(r"/library/(?!INDEX)|listdir\('/library'\)|for f in os.listdir\(.?/library", t["input"])
                 and "write" not in t["input"]]
    out["lib_read_turn"] = lib_reads[0] if lib_reads else None
    out["lib_write"] = any("/library/" in t["input"] and ("open(" in t["input"]) and ("'w'" in t["input"] or '"w"' in t["input"] or "'a'" in t["input"] or '"a"' in t["input"] or "write_text" in t["input"]) for t in py_calls)
    dir_turns = [int(t["turn"]) for t in py_calls if RE_DIR_BUILD.search(t["input"])]
    out["first_think_dir_turn"] = dir_turns[0] if dir_turns else None
    out["tf_before_save"] = sum(1 for t in tf if first_save and int(t["turn"]) < first_save)
    out["py_before_save"] = sum(1 for t in py_calls if first_save and int(t["turn"]) < first_save)
    # final edit quality on confirm splits: last both/edited test on each confirm split
    fin = {}
    for p in paired:
        if p["split"].startswith("confirm") and p["model"] in ("both", "edited"):
            fin[p["split"]] = p
    out["final_confirm"] = fin
    out.update(EDITS.get(name, {}))
    out["test"] = TEST.get(name)
    return out


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def table(rows):
    hdr = ("| rollout | reward (pts) | test (pts) | ended | min | turns | fwd | py | web | tensors (down/o/out) "
           "| form | dirs | strengths | 1st save turn (min) | save calls | 'both' calls (edited tests on select) | sweep | confirm "
           "| TF passes (before 1st save) | attr / DoM calls | failed fwd | tracebacks | repaired calls | turns w/o parsed call | ctx end |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for o in rows:
        k = o["kinds"]
        lines.append("| " + " | ".join([
            o["name"], fmt(o["reward"] * 100 if o["reward"] is not None else None), fmt(o["test"]),
            o["ended"], fmt(o["minutes"]), fmt(o["turns"]), fmt(o["fwd"]), fmt(o["py"]), fmt(o["web"]),
            f"{o['n_tensors']} ({k['down']}/{k['o_proj']}/{k['out_proj']})",
            o.get("form", ""), o.get("dirs", ""), o.get("alpha", ""),
            f"{fmt(o['first_save_turn'])} ({fmt(o['first_save_min'])})", fmt(o["save_calls"]),
            f"{o['n_both']} ({o['sel_tests']})", "yes" if o["sweep"] else "no", ",".join(o["confirm"]) or "no",
            f"{o['n_tf']} ({o['tf_before_save']})", f"{o['attribution']} / {o['diff_means']}",
            fmt(o["failed_fw"]), fmt(o["tracebacks"]), fmt(o["repaired"]), fmt(o["no_call_turns"]), fmt(o["ctx_end"]),
        ]) + " |")
    return "\n".join(lines)


def group_stats(rows, label):
    fin = [o for o in rows if o["reward"] is not None]
    import statistics as st
    rw = [o["reward"] * 100 for o in fin]
    def m(key, rs=rows):
        v = [o[key] for o in rs if o[key] is not None]
        return round(st.mean(v), 1) if v else None
    print(f"\n{label}: finished {len(fin)}/{len(rows)}; reward mean {st.mean(rw):.1f} sd {st.pstdev(rw):.1f} "
          f"min {min(rw):.1f} max {max(rw):.1f}; >=20 pts: {sum(r >= 20 for r in rw)}")
    print(f"  mean minutes {m('minutes', fin)}, turns {m('turns', fin)}, fwd {m('fwd', fin)}, py {m('py', fin)}, "
          f"first save turn {m('first_save_turn')}, first save min {m('first_save_min')}, TF passes {m('n_tf')}, "
          f"TF before save {m('tf_before_save')}, paired tests {m('n_both')}, select tests {m('sel_tests')}, "
          f"failed fwd {sum(o['failed_fw'] for o in rows)}, tracebacks {sum(o['tracebacks'] for o in rows)}, "
          f"repaired {sum(o['repaired'] for o in rows)}, ctx end {m('ctx_end')}")
    print(f"  hit-cap on reward set (edited): {[o['hit_cap'] for o in rows]}")
    print(f"  48-tensor edits: {sum(o['n_tensors'] == 48 for o in rows)}; forms: {[o.get('form') for o in rows]}")
    print(f"  library read turn: {[o['lib_read_turn'] for o in rows]}; library write: {[o['lib_write'] for o in rows]}")
    print(f"  first </think>-direction code turn: {[o['first_think_dir_turn'] for o in rows]}")
    print(f"  read baseline: {[o['read_baseline'] for o in rows]}; read splits: {[o['read_splits'] for o in rows]}")


def main():
    detail = "--detail" in sys.argv
    groups = {
        "step 1 (s0)": [analyse(HERE / f"hero3-s0-r{i}.jsonl", f"s0-r{i}", HERO_SPLITS) for i in range(8)],
        "step 2 (s1)": [analyse(HERE / f"hero3-s1-r{i}.jsonl", f"s1-r{i}", HERO_SPLITS) for i in range(8)],
        "opus": [analyse(OPUS, "opus", OPUS_SPLITS)],
    }
    for g, rows in groups.items():
        print(f"\n## {g}\n")
        print(table(rows))
        if g != "opus":
            group_stats(rows, g)
    if detail:
        for g, rows in groups.items():
            for o in rows:
                print(f"\n### {o['name']}  tf_layers={o['tf_layers']}")
                for p in o["paired"]:
                    print("   ", p)


if __name__ == "__main__":
    main()
