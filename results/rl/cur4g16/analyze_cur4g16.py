"""How did the editor's behaviour change over the 8 steps of RL run cur4g16?

Reads the rollout logs (cur4g16-s{0..7}-r{0..15}.jsonl), steps.jsonl, the practice splits and the two held-out
evaluation folders (untrained editor vs editor after 4 updates, 4 sessions each, Qwen as downloaded, no 128 cap).
Prints markdown tables used in analysis.md. Edit kinds, targets and strengths are annotated by hand from each
FINAL SUBMISSION text (EDITS below); everything else is computed from the logs. Regex counts are approximate.

Run:  python3 results/rl/cur4g16/analyze_cur4g16.py [--splits PATH] [--eval DIR] [--detail]
Files not in the repo are fetched from the glimmer-runs Modal volume into /tmp/c4g if missing:
  /data/math_practice/splits.json  ->  /tmp/c4g/splits/splits.json
  /eval/<name>/{steps.jsonl,scores.json,<name>-s0-r{0..3}.jsonl} -> /tmp/c4g/eval/<name>/
"""
import ast
import json
import math
import random
import re
import statistics as st
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_NAMES = {"untrained": "untrained-orig-math500", "step4": "cur4g16-s4-orig-math500"}


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SPLITS_PATH = Path(arg("--splits", "/tmp/c4g/splits/splits.json"))
EVAL_DIR = Path(arg("--eval", "/tmp/c4g/eval"))


def fetch(remote, local):
    local = Path(local)
    if local.exists():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uvx", "-q", "modal", "volume", "get", "--force", "glimmer-runs", remote, str(local)],
                   check=True, stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------------------------------------------------------
# Hand annotation of each submitted edit, from its FINAL SUBMISSION text.
# kinds: RR = token readout direction removed from writer matrices (W - a d d^T W, a > 0; a = 2 is a sign flip)
#        RA = token readout direction amplified (W + a d d^T W, a > 0)
#        UZ / UF / US = down_proj column(s) zeroed / sign-flipped / scaled by a positive factor
#        UW = a token readout vector added to one down_proj column (the unit then writes that token)
#        RW = rank-one write W + c * outer(r(T), b) with b a difference-of-means direction in the matrix's input
#             (the token T is written only when the input lies along b)
#        LD = a difference-of-means residual direction (loop positions minus other positions) removed from writers
#        H  = one attention head's output block zeroed
#        none = no net change (empty /work/edited, or tensors saved identical to the starting weights)
# target: wait = " Wait"/"Wait" readout; think = "</think>" readout; think- = </think> suppressed; loop = a
#         difference-of-means loop direction; anti-wait = units that write against Wait, amplified; other = named.
# ---------------------------------------------------------------------------------------------------------------
E = lambda kinds, target, alpha: dict(kinds=kinds, target=target, alpha=alpha)  # noqa: E731
EDITS = {
    # step 1 (s0), untrained editor
    "s0-r0": E("RA", "think", "+0.75 on 24 down_proj"),
    "s0-r1": E("LD+UZ", "loop+wait", "loop dir -1 on 48 writers; 7 Wait units = 0"),
    "s0-r2": E("US", "think", "14 </think>-writing units x2.5"),
    "s0-r3": E("RA", "other", "'boxed'-specific readout +1.5 on L23 o_proj"),
    "s0-r4": E("LD+RR", "loop+wait", "L23 attn loop dir and (Wait - </think>) axis removed, L23 o_proj"),
    "s0-r5": E("UZ", "wait", "8 Wait units = 0"),
    "s0-r6": E("UZ", "wait", "17 Wait/But/Oh units = 0"),
    "s0-r7": E("RR+RA", "wait+think", "Wait (2 tokens) -1 on 24 down_proj; </think> +3 on 15 unit columns"),
    "s0-r8": E("UW", "think", "L16 col 172 += 1.0 r(</think>)"),
    "s0-r9": E("RA", "think", "+0.5 one-sided (ReLU) on 24 down_proj"),
    "s0-r10": E("UZ", "wait", "16 Wait units = 0"),
    "s0-r11": E("RR", "wait", "Wait (2 tokens) -2 (sign flip) on L23 o_proj"),
    "s0-r12": E("RR+RA", "wait+think", "Wait -1, </think> +0.6 on 24 down_proj"),
    "s0-r13": E("RA", "think", "</think>-specific +2.5 on 16 writers L16-23"),
    "s0-r14": E("RW", "think", "outer(r(</think>), b_loop), scale 1/S, 22 down_proj"),
    "s0-r15": E("LD+RA", "loop+think", "loop dir -2 (reflection) on L23 o_proj; </think>-specific +1 on 24 down_proj"),
    # step 2 (s1)
    "s1-r0": E("RR", "other", "(</think> - indent token) direction -1 on 48 writers"),
    "s1-r1": E("RR+RA", "wait+think", "Wait-specific -1, </think>-specific +1.5 on L23 o_proj"),
    "s1-r2": E("RR", "wait", "Wait -1 on 48 writers"),
    "s1-r3": E("LD", "loop", "final-residual loop dir -1 on 24 down_proj"),
    "s1-r4": E("RR", "wait", "Wait (2 tokens) -1 on 30 writers"),
    "s1-r5": E("RA", "think", "(</think> - Wait) direction +1.0 on 48 writers"),
    "s1-r6": E("RA", "think", "+0.7 on 10 writers L16-23"),
    "s1-r7": E("RA", "think", "</think>-specific +1.5 on 13 writers"),
    "s1-r8": E("RW", "think", "outer(0.12 r(</think>), b_loop) on L23 down_proj"),
    "s1-r9": E("RR+RA", "wait+think", "Wait -1, </think> +0.25 on 48 writers"),
    "s1-r10": E("LD+RR", "loop+wait", "loop dir -1 on 30 writers; Wait -1 on 6 o_proj"),
    "s1-r11": E("H", "other", "L23 head 6 output block = 0"),
    "s1-r12": E("RR", "wait", "Wait -4 on L23 o_proj"),
    "s1-r13": E("RR+RA", "wait+think", "Wait -1, </think> +1 on L23 o_proj"),
    "s1-r14": E("LD", "loop", "per-layer attention loop dir -1 on 24 attention writers"),
    "s1-r15": E("UZ+US+LD", "wait+loop", "6 Wait units = 0; L22 u573 x2; loop dir -1 on L23 o_proj"),
    # step 3 (s2), tests capped at 128 answers from here on
    "s2-r0": E("LD+UZ", "loop", "loop dir -1 on 48 writers; 7 loop units = 0"),
    "s2-r1": E("RA", "think", "+0.1 on L23 o_proj"),
    "s2-r2": E("RA", "think", "+1.0 / +0.75 on 4 writers (L19, L23)"),
    "s2-r3": E("UZ", "wait", "14 Wait units = 0"),
    "s2-r4": E("RR", "wait", "Wait (2 tokens) -1 on 48 writers"),
    "s2-r5": E("UZ", "wait", "14 Wait units = 0"),
    "s2-r6": E("LD", "loop", "per-layer loop dir -1 on 7 attention writers"),
    "s2-r7": E("RR+RA", "wait+think", "Wait -1, </think> +0.3 on 48 writers"),
    "s2-r8": E("UZ+LD", "wait+loop", "2 units = 0; loop dir -0.5 on L23 o_proj"),
    "s2-r9": E("RW", "think", "outer(r(</think>), b_pre-close) +0.035 on 4 down_proj"),
    "s2-r10": E("RA", "think", "+1.0 on L23 down_proj"),
    "s2-r11": E("RW", "think", "outer(r(</think>), b_loop) +0.08 on 24 down_proj"),
    "s2-r12": E("RR", "wait", "Wait -1 on 8 writers"),
    "s2-r13": E("RR", "wait", "Wait-specific -1 on 48 writers"),
    "s2-r14": E("RA+US", "think", "</think>-specific +1.0 on 24 down_proj; 14 units x3"),
    "s2-r15": E("US", "anti-wait", "L22 u573 x5"),
    # step 4 (s3)
    "s3-r0": E("US", "anti-wait", "L22 u573 x3"),
    "s3-r1": E("RA", "think", "+5 on L23 o_proj"),
    "s3-r2": E("UZ", "wait", "5 Wait units = 0"),
    "s3-r3": E("RR", "wait", "Wait -1 on 48 writers"),
    "s3-r4": E("UZ+US", "wait+anti-wait", "3 units = 0; L22 u573 and L23 u211 x2"),
    "s3-r5": E("RA", "think", "</think>-specific +5.5 on L23 o_proj"),
    "s3-r6": E("RR+UZ", "wait", "Wait -1 on 48 writers; 5 units = 0"),
    "s3-r7": E("RR", "wait", "Wait -1 on 30 writers"),
    "s3-r8": E("US+UZ", "anti-wait+wait", "L22 u573 x8; 6 units = 0"),
    "s3-r9": E("US", "anti-wait", "4 Wait-suppressing units x3"),
    "s3-r10": E("UZ+US", "wait+think", "L16 u998 = 0; L19 u31 x8"),
    "s3-r11": E("LD", "loop", "L23 residual loop dir -1 on 48 writers"),
    "s3-r12": E("UZ+RA", "wait+think", "~27 Wait units = 0; </think>-specific +1.0 on 12 down_proj"),
    "s3-r13": E("RR", "wait", "Wait -1 on 24 down_proj"),
    "s3-r14": E("RR", "wait", "Wait (2 tokens) -3 on L23 o_proj"),
    "s3-r15": E("RR", "wait", "Wait -1 on 48 writers"),
    # step 5 (s4), starting model cur4g16-k1
    "s4-r0": E("none", "-", "329 tensors saved identical to the starting weights"),
    "s4-r1": E("none", "-", "-"),
    "s4-r2": E("RW", "other", "outer(r(To)+r(The), b post-</think> spam) +0.1 on 10 down_proj"),
    "s4-r3": E("RW", "wait", "outer(r(Wait), b_loop) -0.10 on 11 down_proj"),
    "s4-r4": E("none", "-", "-"),
    "s4-r5": E("RR", "wait", "Wait -1 on L23 o_proj"),
    "s4-r6": E("UZ", "wait+think-", "25 Wait-loop and </think>-repeat units = 0"),
    "s4-r7": E("none", "-", "-"),
    "s4-r8": E("RW", "think-+other", "outer(r(</think>), b) -0.01 on L22-23; outer(r(boxed), b) +0.005 on L0-22"),
    "s4-r9": E("none", "-", "-"),
    "s4-r10": E("none", "-", "-"),
    "s4-r11": E("RR", "wait", "Wait -1 on 48 writers"),
    "s4-r12": E("none", "-", "-"),
    "s4-r13": E("none", "-", "-"),
    "s4-r14": E("none", "-", "-"),
    "s4-r15": E("UZ", "think-", "11 </think>-restart units = 0"),
    # step 6 (s5)
    "s5-r0": E("RW", "think-", "outer(r(</think>), b answer-phase) -0.08 on 24 down_proj"),
    "s5-r1": E("UZ+US", "wait+anti-wait", "11 Wait units = 0; L20 u405, L22 u573 x2"),
    "s5-r2": E("none", "-", "-"),
    "s5-r3": E("none", "-", "-"),
    "s5-r4": E("none", "-", "-"),
    "s5-r5": E("none", "-", "-"),
    "s5-r6": E("none", "-", "-"),
    "s5-r7": E("RW", "wait", "outer(r(Wait), b_loop) -0.04 on 24 down_proj"),
    "s5-r8": E("RR", "think-", "</think> -1 on L23 o_proj"),
    "s5-r9": E("RW", "other", "outer(r(The), b_loop) +0.04 on 24 down_proj"),
    "s5-r10": E("none", "-", "-"),
    "s5-r11": E("RW", "wait", "outer(r(Wait), b_loop) -0.15 on 24 down_proj"),
    "s5-r12": E("none", "-", "-"),
    "s5-r13": E("UZ", "wait", "16 loop units = 0"),
    "s5-r14": E("failed", "-", "session failed (no summary)"),
    "s5-r15": E("none", "-", "-"),
    # step 7 (s6)
    "s6-r0": E("none", "-", "-"),
    "s6-r1": E("none", "-", "-"),
    "s6-r2": E("none", "-", "-"),
    "s6-r3": E("none", "-", "-"),
    "s6-r4": E("none", "-", "-"),
    "s6-r5": E("none", "-", "-"),
    "s6-r6": E("RW", "wait", "outer(r(Wait), b_loop) -0.05 on 24 down_proj"),
    "s6-r7": E("none", "-", "-"),
    "s6-r8": E("none", "-", "-"),
    "s6-r9": E("none", "-", "-"),
    "s6-r10": E("none", "-", "-"),
    "s6-r11": E("RR", "wait", "Wait -1 on 24 attention writers"),
    "s6-r12": E("none", "-", "-"),
    "s6-r13": E("RW", "wait", "outer(r(Wait)+r( Wait), b post-</think>) -0.1 on 4 down_proj"),
    "s6-r14": E("none", "-", "-"),
    "s6-r15": E("none", "-", "24 tensors saved identical to the starting weights (reported)"),
    # step 8 (s7)
    "s7-r0": E("UZ", "think-", "7 </think>-repeat units = 0"),
    "s7-r1": E("failed", "-", "session failed (no summary)"),
    "s7-r2": E("RR", "think-", "</think> -1 on 6 o_proj, -0.5 on 18 out_proj"),
    "s7-r3": E("RR", "wait", "Wait -1 on 49 writers"),
    "s7-r4": E("none", "-", "-"),
    "s7-r5": E("none", "-", "-"),
    "s7-r6": E("none", "-", "-"),
    "s7-r7": E("UZ", "wait", "4 Wait units = 0"),
    "s7-r8": E("none", "-", "24 tensors saved identical to the starting weights (reported)"),
    "s7-r9": E("RR", "wait", "Wait (2 tokens) -0.5 on 48 writers"),
    "s7-r10": E("RW", "wait", "outer(r(Wait)s, b_loop) -0.05 on 24 down_proj"),
    "s7-r11": E("none", "-", "-"),
    "s7-r12": E("RR", "wait", "Wait -2 on 8 down_proj L16-23"),
    "s7-r13": E("RR", "wait", "Wait (2 tokens) -0.6 on 48 writers"),
    "s7-r14": E("none", "-", "-"),
    "s7-r15": E("none", "-", "-"),
    # held-out evaluation sessions (Qwen as downloaded, MATH-500 scoring, 90-minute cap, no 128 cap)
    "untrained-r0": E("US", "anti-wait", "L22 u573 x2"),
    "untrained-r1": E("UZ", "wait", "13 Wait units = 0"),
    "untrained-r2": E("RR+RA", "wait+think", "Wait -1, </think> +1.5 on L23 o_proj"),
    "untrained-r3": E("RR", "wait", "Wait -1 on 24 attention writers"),
    "step4-r0": E("RA", "think", "+5 on L23 o_proj"),
    "step4-r1": E("RR+RW", "wait+think", "Wait -1 on 48 writers; outer(r(</think>), b_pre-close) on down_proj"),
    "step4-r2": E("UZ", "wait", "21 Wait units = 0"),
    "step4-r3": E("UZ", "wait", "9 Wait units = 0"),
}

# Python code: per-unit logit attribution, difference of means, logit lens (readout through E), random direction
RE_ATTR = re.compile(r"attrib|contrib|\bdla\b|logit.lens|"
                     r"r_\w+\s*@\s*(Wd|W_down|Wdown|down|W)\b|(Wd|W_down|Wdown|down)\w*(\.T)?\s*@\s*r_", re.I)
RE_DOM = re.compile(r"diff\w*.{0,4}of.{0,4}means|\bdm\b\s*=|mean\([^)\n]*\)\s*-\s*[\w\[\]\.]*mean\(|"
                    r"\b(a|m|mu|mean)_?[A-Za-z]\w{0,8}\s*-\s*(a|m|mu|mean)_?[A-Za-z]\w{0,8}\b", re.I)
RE_LENS = re.compile(r"logit.?lens|embed_tokens|\bE\s*@|@\s*E\.T|E\[\s*\w+\s*\]\s*\*", re.I)
RE_RAND_CODE = re.compile(r"randn|np\.random\.(normal|standard_normal|randn)|torch\.rand", re.I)
RE_RAND_TXT = re.compile(r"random(ly)?[- ](direction|vector|unit|control|columns?|units?)", re.I)
RE_REV_TXT = re.compile(r"(revers|opposite|flipp?ed|negat)\w*\s+(the\s+)?(sign|direction)|sign[- ]revers|"
                        r"reverse[d-]?\s*(sign|direction)", re.I)
RE_PLAN = re.compile(r"\bplan\b|\bnext steps?\b|\bstep \d\b", re.I)
RE_LOADW = re.compile(r"\bload(_original)?\(|safe_open|load_file")
# Steps 5-8: what the reasoning and final text talk about
RE_EARLIER = re.compile(r"earlier (intended )?edit|applied_edits|prior .{0,20}edit|existing .{0,20}edit|"
                        r"already (includes?|contains?|applied)|intended earlier", re.I)
RE_STORM = re.compile(r"</think>.{0,40}(storm|spam|cascade|restart|repeat|repetition|re-?emi)|"
                      r"(repeated|consecutive|multiple|nested|spurious|second)\s+`?</think>", re.I)
RE_NOBOX = re.compile(r"no[- ]box|without (a )?\\?boxed|never (emit|produce|reach)\w* .{0,10}boxed|missing .{0,10}boxed", re.I)
RE_KEEP = re.compile(r"(earlier|prior|intended|existing|given|starting).{0,80}(untouched|intact|left (alone|in place)|"
                     r"not (undone|reversed|reverted|touched))|on top of", re.I)
RE_BOOST_EARLIER = re.compile(r"(amplif|strengthen|scal|boost|increas)\w*.{0,30}(prior|earlier|existing|intended|inherited)"
                              r".{0,30}(edit|boost|rank)", re.I)


def load(path):
    return [json.loads(l) for l in open(path)]


def pj(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def strip_aborted_start(recs):
    """If turn numbers restart inside a log (an aborted start of the previous app followed by the real session),
    keep only the records after the last restart. Returns (records, number of message turns dropped)."""
    cut, prev = 0, 0
    for i, r in enumerate(recs):
        if "reasoning" in r:
            t = int(r["turn"])
            if t <= prev:
                cut = i
            prev = t
    dropped = sum(1 for r in recs[:cut] if "reasoning" in r)
    return recs[cut:], dropped


def split_of(ids, sp):
    c = Counter(sp.get(x, "unknown") for x in ids)
    top, n = c.most_common(1)[0]
    return top if n >= 0.9 * len(ids) else "mixed"


def analyse(path, name, sp, adv=None):
    recs, dropped = strip_aborted_start(load(path))
    msgs = [r for r in recs if "reasoning" in r]
    tools = [r for r in recs if "tool" in r]
    summ = next((r for r in reversed(recs) if "summary" in r), None)
    o = dict(name=name, dropped=dropped, adv=adv)
    o.update(EDITS.get(name, dict(kinds="?", target="?", alpha="?")))
    if summ is None:
        o.update(ok=False, reward=None, minutes=None, turns=len(msgs))
        return o
    s = summ["summary"]
    s = ast.literal_eval(s) if isinstance(s, str) else s
    final_text = summ.get("final_text") or ""
    t0 = int(msgs[0]["t"]) if msgs else 0
    tt = {int(r["turn"]): int(r["t"]) - t0 for r in msgs}
    last_t = (int(msgs[-1]["t"]) - t0) / 60 if msgs else 0
    o.update(ok=True, reward=s["reward"] * 100, ended=s["ended"], minutes=s["minutes"], work_min=round(last_t, 1),
             turns=s["turns"], fwd=s["forward_used"], py=s["python_used"], n_tensors=len(s.get("edited_tensors") or []),
             hit_cap=s.get("edited_hit_cap"), sampled=(s.get("tokens") or {}).get("sampled"))
    o["net_edit"] = o["kinds"] not in ("none", "failed")

    py = [t for t in tools if t["tool"] == "python"]
    fw = [t for t in tools if t["tool"] == "forward"]
    saves = [t for t in py if "save_edit(" in t["input"]]
    o["save_calls"] = len(saves)
    o["first_save_min"] = round(tt.get(int(saves[0]["turn"]), 0) / 60, 1) if saves else None
    o["last_save_min"] = round(tt.get(int(saves[-1]["turn"]), 0) / 60, 1) if saves else None
    last_save_turn = int(saves[-1]["turn"]) if saves else None

    tests, tf, failed = [], [], 0
    for t in fw:
        res = t["result"]
        if res.startswith("[request failed"):
            failed += 1
            continue
        r = pj(res) or {}
        inp = pj(t["input"]) or {}
        typ = r.get("type")
        if typ == "teacher_forced":
            tf.append(int(t["turn"]))
            continue
        if typ != "generate":
            continue
        probs = inp.get("problems") or []
        e, og, p = r.get("edited") or {}, r.get("original") or {}, r.get("paired_edited_minus_original") or {}
        split = split_of(probs, sp) if probs else "own prompts"
        tests.append(dict(turn=int(t["turn"]), min=round(tt.get(int(t["turn"]), 0) / 60, 1), model=inp.get("model"),
                          split=split, n_edit=e.get("n"), n_orig=og.get("n"), diff=p.get("diff_pp"), se=p.get("se_pp"),
                          acc_e=e.get("accuracy_pct"), acc_o=og.get("accuracy_pct"),
                          cap_e=e.get("cut_at_max_tokens_pct"), nobox_e=e.get("no_boxed_answer_pct")))
    et = [x for x in tests if x["model"] in ("both", "edited")]
    o["tests"] = tests
    o["n_tf"] = len(tf)
    o["n_gen"] = len(tests)
    o["n_edit_tests"] = len(et)
    o["n_paired"] = sum(x["model"] == "both" for x in tests)
    o["n_orig_only"] = sum(x["model"] == "original" for x in tests)
    o["test_sizes"] = [x["n_edit"] for x in et]
    o["confirm_tests"] = sum(x["split"].startswith("confirm") for x in et)
    o["select_tests"] = sum(x["split"] == "select" for x in et)
    o["first_test_min"] = et[0]["min"] if et else None
    o["untested"] = o["net_edit"] and not et
    o["after_last_save_turns"] = (o["turns"] - last_save_turn) if last_save_turn else None
    o["after_last_save_min"] = round(last_t - tt.get(last_save_turn, 0) / 60, 1) if last_save_turn else None
    o["tests_after_last_save"] = sum(1 for x in et if last_save_turn and x["turn"] > last_save_turn)
    o["max_diff"] = max((x["diff"] for x in et if x["diff"] is not None), default=None)
    o["failed_fw"] = failed
    o["tf_before_save"] = sum(1 for x in tf if saves and x < int(saves[0]["turn"]))
    o["weights_only_first_edit"] = bool(saves) and o["tf_before_save"] == 0

    code = [t["input"] for t in py]
    o["attr"] = sum(bool(RE_ATTR.search(c)) for c in code)
    o["dom"] = sum(bool(RE_DOM.search(c)) and "mean" in c for c in code)
    o["lens"] = sum(bool(RE_LENS.search(c)) for c in code)
    o["read_summary"] = any("SUMMARY.txt" in c for c in code)
    o["rand_code"] = any(RE_RAND_CODE.search(c) for c in code)
    o["rand_saved"] = any(RE_RAND_CODE.search(t["input"]) for t in saves)
    reasoning = "\n".join((r.get("reasoning") or "") for r in msgs)
    alltext = reasoning + "\n" + final_text
    o["rand_text"] = bool(RE_RAND_TXT.search(alltext))
    o["rev_text"] = bool(RE_REV_TXT.search(alltext))
    o["tracebacks"] = sum("Traceback" in t["result"] for t in py)
    o["repaired"] = sum(str(r.get("repaired")) == "True" for r in msgs)
    o["no_call"] = sum(1 for r in msgs if r.get("call") in (None, "None", "")
                       and not (r.get("final") or "").lstrip().startswith("FINAL"))
    o["len_finish"] = sum(r.get("finish") == "length" for r in msgs)
    comp = [int(r["completion_tokens"]) for r in msgs]
    o["comp_total"] = sum(comp)
    o["comp_per_turn"] = st.mean(comp) if comp else 0
    o["reason_chars_per_turn"] = st.mean(len(r.get("reasoning") or "") for r in msgs) if msgs else 0
    o["empty_reasoning"] = sum(1 for r in msgs if not (r.get("reasoning") or "").strip()) / max(1, len(msgs))
    o["plan_turns"] = sum(bool(RE_PLAN.search(r.get("reasoning") or "")) for r in msgs)
    o["final_chars"] = len(final_text)
    o["earlier_mentions"] = sum(bool(RE_EARLIER.search(r.get("reasoning") or "")) for r in msgs)
    o["storm"] = sum(bool(RE_STORM.search(r.get("reasoning") or "")) for r in msgs)
    o["nobox"] = sum(bool(RE_NOBOX.search(r.get("reasoning") or "")) for r in msgs)
    o["storm_final"] = bool(RE_STORM.search(final_text))
    o["nobox_final"] = bool(RE_NOBOX.search(final_text))
    o["keep_earlier_final"] = bool(RE_KEEP.search(final_text))
    o["boost_earlier"] = bool(RE_BOOST_EARLIER.search(alltext))
    ks, tg = o["kinds"].split("+"), o["target"].split("+")
    o["k_RA"], o["k_RR"], o["k_LD"], o["k_RW"] = "RA" in ks, "RR" in ks, "LD" in ks, "RW" in ks
    o["k_unit"] = any(k in ks for k in ("UZ", "US", "UF", "UW"))
    o["k_combined"] = len(ks) > 1
    o["t_think"] = "think" in tg
    o["t_waitside"] = any(t in tg for t in ("wait", "anti-wait", "loop"))
    # which edits were built in code that saved an edit: target tokens and forms
    sc = "\n".join(t["input"] for t in saves)
    o["saved_targets"] = sorted(k for k, pat in {
        "think": r"248069|</think>|think", "wait": r"13784|13428|[Ww]ait", "boxed": r"76367|boxed",
        "the/to": r"\b760\b|'The'|\"The\"|'To'"}.items() if re.search(pat, sc))
    o["save_think"], o["save_boxed"] = "think" in o["saved_targets"], "boxed" in o["saved_targets"]
    return o


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def f(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4:
        return None
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            for k in range(i, j + 1):
                r[idx[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else None


def fisher_p(a, n1, b, n2):
    """Two-sided Fisher exact p for a/n1 vs b/n2."""
    K, N = a + b, n1 + n2
    pr = lambda x: math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)  # noqa: E731
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, K - n2), min(K, n1) + 1) if pr(x) <= p0 + 1e-12))


def perm_p(a, b, n=20000, seed=0):
    """Two-sided permutation p-value for a difference in means."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b:
        return None
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng, hits = a + b, len(a), random.Random(seed), 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= obs - 1e-12:
            hits += 1
    return hits / n


FEATS = [  # (key, label) for per-step tables; booleans become shares
    ("work_min", "editor minutes (first to last turn)"), ("minutes", "session minutes (summary)"),
    ("turns", "turns"), ("fwd", "forward passes used"), ("py", "python calls"),
    ("n_tf", "teacher-forced passes"), ("n_gen", "generation passes"), ("n_paired", "paired ('both') tests"),
    ("n_edit_tests", "tests with an edited side"), ("confirm_tests", "of which on confirm_1/2"),
    ("first_save_min", "first save_edit (min)"), ("last_save_min", "last save_edit (min)"),
    ("first_test_min", "first edited test (min)"), ("after_last_save_min", "minutes after last save"),
    ("tests_after_last_save", "edited tests after last save"), ("save_calls", "save_edit calls"),
    ("attr", "python calls: attribution (regex)"), ("dom", "python calls: difference of means (regex)"),
    ("lens", "python calls: readout through E (regex)"), ("read_summary", "read SUMMARY.txt"),
    ("weights_only_first_edit", "first edit saved before any teacher-forced pass"),
    ("rand_text", "mentions a random-direction control"), ("rand_saved", "random vector in code that saved an edit"),
    ("rev_text", "mentions a reversed-sign control"),
    ("untested", "edit submitted with no edited test"), ("net_edit", "submitted a net edit"),
    ("comp_per_turn", "completion tokens per turn"), ("comp_total", "completion tokens per session"),
    ("reason_chars_per_turn", "reasoning characters per turn"), ("empty_reasoning", "share of turns with empty reasoning"),
    ("plan_turns", "turns whose reasoning says plan/next step"), ("tracebacks", "python tracebacks"),
    ("failed_fw", "failed forward requests"), ("repaired", "repaired tool calls"), ("no_call", "turns without a parsed call"),
    ("len_finish", "turns cut at the reply length limit"), ("final_chars", "FINAL SUBMISSION characters"),
    ("earlier_mentions", "turns mentioning the earlier edit (regex)"), ("storm", "turns mentioning repeated </think> (regex)"),
    ("nobox", "turns mentioning missing boxed answers (regex)"), ("storm_final", "FINAL text names repeated </think>"),
    ("nobox_final", "FINAL text names missing boxed answers"), ("keep_earlier_final", "FINAL text says earlier edit kept/edit on top"),
    ("boost_earlier", "considers strengthening the earlier edit (regex)"),
    ("save_think", "code that saved an edit names </think>"), ("save_boxed", "code that saved an edit names boxed"),
    ("k_RA", "edit: readout amplified"), ("k_RR", "edit: readout removed"), ("k_unit", "edit: unit columns"),
    ("k_LD", "edit: loop direction removed"), ("k_RW", "edit: rank-one write r(T) x b"), ("k_combined", "edit: 2+ kinds"),
    ("t_think", "target: </think> amplified"), ("t_waitside", "target: Wait / loop side"),
]


def agg(rows, key):
    v = [o.get(key) for o in rows if o.get("ok")]
    v = [float(x) if isinstance(x, bool) else x for x in v if x is not None]
    return (st.mean(v), len(v)) if v else (None, 0)


def kind_counts(rows):
    c = Counter()
    for o in rows:
        for k in o["kinds"].split("+"):
            c[k] += 1
    return c


def target_counts(rows):
    c = Counter()
    for o in rows:
        for k in o["target"].split("+"):
            c[k] += 1
    return c


def main():
    detail = "--detail" in sys.argv
    fetch("/data/math_practice/splits.json", SPLITS_PATH)
    sp = {}
    for k, ids in json.load(open(SPLITS_PATH)).items():
        for x in ids:
            sp[x] = k
    steps = [json.loads(l) for l in open(HERE / "steps.jsonl")]
    srec = {r["step"]: r for r in steps if r.get("event") == "step"}
    S = {}
    for s in range(8):
        advs = srec[s]["advantages"]
        ok_rids = [ro["rid"] for ro in srec[s]["rollouts"] if ro.get("ok")]
        amap = dict(zip(ok_rids, advs)) if len(advs) == len(ok_rids) else {}
        S[s + 1] = [analyse(HERE / f"cur4g16-s{s}-r{i}.jsonl", f"s{s}-r{i}", sp, amap.get(f"cur4g16-s{s}-r{i}"))
                    for i in range(16)]
    EV = {}
    for key, nm in EVAL_NAMES.items():
        for fn in ["steps.jsonl", "scores.json"] + [f"{nm}-s0-r{i}.jsonl" for i in range(4)]:
            fetch(f"/eval/{nm}/{fn}", EVAL_DIR / nm / fn)
        sc = json.load(open(EVAL_DIR / nm / "scores.json"))
        rows = [analyse(EVAL_DIR / nm / f"{nm}-s0-r{i}.jsonl", f"{key}-r{i}", sp) for i in range(4)]
        for o, x in zip(rows, sc):
            o["heldout"] = x.get("gain_pp")
        EV[key] = rows

    allrows = [o for s in S for o in S[s]]
    print(f"# cur4g16 behaviour\n\nleading turns dropped as an aborted start: "
          f"{[(o['name'], o['dropped']) for o in allrows if o['dropped']] or 'none (no turn-number restart in any log)'}")
    print(f"failed sessions: {[o['name'] for o in allrows if not o['ok']]}")

    # ---- per-step table
    print("\n## Per-step means (failed sessions excluded; booleans are shares)\n")
    hdr = "| measure | " + " | ".join(f"step {s}" for s in S) + " | 1-4 | 5-8 | untrained (eval) | step-4 editor (eval) |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    r14 = [o for s in (1, 2, 3, 4) for o in S[s]]
    r58 = [o for s in (5, 6, 7, 8) for o in S[s]]
    for key, lab in [("reward", "reward (pts)")] + FEATS:
        cells = [f(agg(S[s], key)[0]) for s in S] + [f(agg(r14, key)[0]), f(agg(r58, key)[0]),
                                                      f(agg(EV["untrained"], key)[0]), f(agg(EV["step4"], key)[0])]
        print(f"| {lab} | " + " | ".join(cells) + " |")

    # ---- noise reference: sd across sessions per step for a few measures
    print("\n## Session-to-session sd within a step (noise reference)\n")
    for key in ("work_min", "fwd", "n_tf", "n_edit_tests", "first_save_min", "comp_per_turn"):
        print(f"- {key}: " + ", ".join(f"s{s} {st.pstdev([o[key] for o in S[s] if o.get('ok') and o.get(key) is not None]):.1f}"
                                     for s in S))

    # ---- block comparisons with permutation p-values
    print("\n## Same-setup comparisons (permutation p, two-sided)\n")
    pairs = [("step 1 vs 2", S[1], S[2]), ("step 3 vs 4", S[3], S[4]), ("steps 5-6 vs 7-8", S[5] + S[6], S[7] + S[8]),
             ("steps 1-2 vs 3-4 (prompt changed: 128 cap)", S[1] + S[2], S[3] + S[4]),
             ("eval untrained vs step-4 editor", EV["untrained"], EV["step4"])]
    for key in ("work_min", "turns", "fwd", "py", "n_tf", "n_paired", "n_edit_tests", "first_save_min", "last_save_min",
                "after_last_save_min", "comp_per_turn", "comp_total", "reason_chars_per_turn", "attr", "dom", "tracebacks",
                "failed_fw", "n_tensors", "final_chars"):
        cells = []
        for lab, a, b in pairs:
            va = [float(o[key]) for o in a if o.get("ok") and o.get(key) is not None]
            vb = [float(o[key]) for o in b if o.get("ok") and o.get(key) is not None]
            p = perm_p(va, vb, n=5000)
            cells.append(f"{lab}: {f(mean(va))} -> {f(mean(vb))} (p {f(p, 2)})")
        print(f"- {key}: " + "; ".join(cells))

    # ---- testing discipline
    print("\n## Testing\n")
    for s in S:
        rows = [o for o in S[s] if o["ok"]]
        sizes = Counter(x for o in rows for x in o["test_sizes"])
        spl = Counter(x["split"] for o in rows for x in o["tests"] if x["model"] in ("both", "edited"))
        mods = Counter(x["model"] for o in rows for x in o["tests"])
        print(f"- step {s}: untested submissions {[o['name'] for o in rows if o['untested']]}; "
              f"no-confirm sessions with an edit {sum(1 for o in rows if o['net_edit'] and not o['confirm_tests'])}; "
              f"test sizes {dict(sizes.most_common(6))}; splits {dict(spl)}; models {dict(mods)}; "
              f"random-direction text {sum(o['rand_text'] for o in rows)}, random vector in saving code "
              f"{sum(o['rand_saved'] for o in rows)}, reversed-sign text {sum(o['rev_text'] for o in rows)}")
    for k, rows in EV.items():
        sizes = Counter(x for o in rows for x in o["test_sizes"])
        print(f"- eval {k}: test sizes {dict(sizes)}; paired {[o['n_paired'] for o in rows]}; "
              f"confirm tests {[o['confirm_tests'] for o in rows]}")

    # ---- edits
    print("\n## Edits\n")
    print("| step | kinds (count of sessions using each) | targets | tensors per edit (median, range) | distinct kind combos |")
    print("|---|---|---|---|---|")
    for s in S:
        rows = [o for o in S[s] if o["ok"]]
        ed = [o for o in rows if o["net_edit"]]
        nt = sorted(o["n_tensors"] for o in ed)
        print(f"| {s} | {dict(kind_counts(rows).most_common())} | {dict(target_counts(ed).most_common())} | "
              f"{st.median(nt) if nt else '-'} ({nt[0] if nt else '-'}-{nt[-1] if nt else '-'}) | "
              f"{len(set(o['kinds'] for o in ed))} |")
    for k, rows in EV.items():
        print(f"| eval {k} | {dict(kind_counts(rows))} | {dict(target_counts(rows))} | "
              f"{sorted(o['n_tensors'] for o in rows)} | {len(set(o['kinds'] for o in rows))} |")
    print("\nReward by kind family, steps 1-4 (a session counts once per kind it uses):")
    fam = {}
    for o in r14:
        for k in o["kinds"].split("+"):
            fam.setdefault(k, []).append(o["reward"])
    for k, v in sorted(fam.items(), key=lambda x: -st.mean(x[1])):
        print(f"  {k}: n={len(v)} mean {st.mean(v):.1f} median {st.median(v):.1f}")
    fam = {}
    for o in r14:
        fam.setdefault(o["target"], []).append(o["reward"])
    print("Reward by target, steps 1-4:")
    for k, v in sorted(fam.items(), key=lambda x: -st.mean(x[1])):
        print(f"  {k}: n={len(v)} mean {st.mean(v):.1f}")
    print("Single-kind sessions per step (RR, RA, UZ alone): " +
          "; ".join(f"s{s} " + str({k: sum(o['kinds'] == k for o in S[s]) for k in ('RR', 'RA', 'UZ', 'US', 'LD', 'RW')})
                    for s in S))
    print("Combined edits (2+ kinds) per step: " + str({s: sum('+' in o['kinds'] for o in S[s]) for s in S}))
    print("Sessions with u573 (L22 unit 573 scaled up): " +
          str({s: sum('u573' in o['alpha'] for o in S[s]) for s in S}))
    print("Wait-readout removal on all 48 writers: " +
          str({s: sum(o['kinds'].startswith('RR') and '48' in o['alpha'] and 'Wait' in o['alpha'] for o in S[s]) for s in S}))

    # ---- outcome links within steps
    print("\n## Within-step links to reward (steps 1-4: reward minus step mean, pooled n=64; Spearman)\n")
    for blk, rows_by_step in (("1-4", [S[s] for s in (1, 2, 3, 4)]), ("5-8", [S[s] for s in (5, 6, 7, 8)])):
        cen = []
        for rows in rows_by_step:
            ok = [o for o in rows if o["ok"]]
            m = st.mean(o["reward"] for o in ok)
            cen += [(o, o["reward"] - m) for o in ok]
        out = []
        for key, lab in FEATS + [("n_tensors", "tensors"), ("hit_cap", "edited hit-cap")]:
            if key in ("net_edit",) and blk == "1-4":
                continue
            x = [float(o.get(key)) if o.get(key) is not None else None for o, _ in cen]
            rho = spearman(x, [c for _, c in cen])
            if rho is not None:
                out.append((rho, lab))
        out.sort(key=lambda z: -abs(z[0]))
        print(f"- steps {blk} (n={len(cen)}): " + "; ".join(f"{lab} {rho:+.2f}" for rho, lab in out[:14]))
    for s in (1, 2, 3, 4):
        rows = [o for o in S[s] if o["ok"]]
        un = [o["reward"] for o in rows if o["untested"]]
        te = [o["reward"] for o in rows if not o["untested"]]
        print(f"- step {s}: untested {[round(x, 1) for x in un]} vs tested mean {mean(te):.1f}; "
              f"rho(fwd, reward) {f(spearman([o['fwd'] for o in rows], [o['reward'] for o in rows]), 2)}; "
              f"rho(n_tf, reward) {f(spearman([o['n_tf'] for o in rows], [o['reward'] for o in rows]), 2)}; "
              f"rho(work_min, reward) {f(spearman([o['work_min'] for o in rows], [o['reward'] for o in rows]), 2)}")

    # ---- direction of the training signal: for update k -> k+1, corr(feature, advantage) in step k vs change in mean
    print("\n## Did features correlated with advantage in step k become more common in step k+1?\n")
    upd = [(1, 2), (3, 4), (5, 6), (6, 7), (7, 8), (2, 3), (4, 5)]
    keys = ["work_min", "fwd", "n_tf", "n_paired", "n_edit_tests", "first_save_min", "attr", "dom", "comp_per_turn",
            "reason_chars_per_turn", "n_tensors", "net_edit", "tracebacks", "confirm_tests", "after_last_save_min",
            "repaired", "k_RA", "k_RR", "k_unit", "k_LD", "k_RW", "k_combined", "t_think", "t_waitside"]
    agree = Counter()
    for key in keys:
        cells = []
        for a, b in upd:
            ra = [o for o in S[a] if o["ok"] and o.get("adv") is not None and o.get(key) is not None]
            rho = spearman([float(o[key]) for o in ra], [o["adv"] for o in ra])
            ma, mb = agg(S[a], key)[0], agg(S[b], key)[0]
            d = (mb - ma) if (ma is not None and mb is not None) else None
            same = None
            if rho is not None and d is not None and abs(rho) >= 0.3 and (a, b) not in ((2, 3), (4, 5)):
                same = (rho > 0) == (d > 0)
                agree[same] += 1
            cells.append(f"{a}->{b}: rho {f(rho, 2)}, change {f(d, 2)}{'' if same is None else (' (with)' if same else ' (against)')}")
        print(f"- {key}: " + "; ".join(cells))
    print(f"\nWithin-setup updates where |rho| >= 0.3: change in the direction of the advantage {agree[True]}, against {agree[False]}")
    for s, r in srec.items():
        uc = r.get("update_check") or {}
        print(f"  update after step {s + 1}: KL {uc.get('kl_old_new_k1')}, corr(change in log-likelihood, advantage) "
              f"{uc.get('corr_change_advantage')}, entropy of the sampled step {r.get('entropy')}")

    # ---- kinds: did the high-advantage kinds become more common?
    print("\nKind of the top-4 sessions by reward per step: " +
          "; ".join(f"s{s} " + ", ".join(o['kinds'] + ':' + o['target'] for o in sorted([o for o in S[s] if o['ok']], key=lambda o: -o['reward'])[:4])
                    for s in S))

    # ---- proportions with Fisher exact p
    print("\n## Proportions (Fisher exact, two-sided)\n")
    def cnt(rows, key):
        ok = [o for o in rows if o["ok"]]
        return sum(bool(o.get(key)) for o in ok), len(ok)
    for lab, key, a, b in [("</think> amplified, step 1 vs step 4", "t_think", S[1], S[4]),
                           ("Wait/loop side, step 1 vs step 4", "t_waitside", S[1], S[4]),
                           ("untested submission, step 1 vs steps 2-4", "untested", S[1], S[2] + S[3] + S[4]),
                           ("rank-one write r(T) x b, steps 1-4 vs 5-8 (edits only)", "k_RW",
                            [o for o in r14 if o.get("net_edit")], [o for o in r58 if o.get("net_edit")]),
                           ("2+ kinds, steps 1-4 vs 5-8 (edits only)", "k_combined",
                            [o for o in r14 if o.get("net_edit")], [o for o in r58 if o.get("net_edit")]),
                           ("net edit, step 5-6 vs 7-8", "net_edit", S[5] + S[6], S[7] + S[8]),
                           ("any repaired call, step 1 vs step 2", "repaired", S[1], S[2])]:
        (x1, n1), (x2, n2) = cnt(a, key), cnt(b, key)
        print(f"- {lab}: {x1}/{n1} vs {x2}/{n2}, p {fisher_p(x1, n1, x2, n2):.2f}")

    # ---- steps 5-8
    print("\n## Steps 5-8\n")
    for s in (5, 6, 7, 8):
        rows = [o for o in S[s] if o["ok"]]
        ne = [o for o in rows if not o["net_edit"]]
        print(f"- step {s}: no net edit {len(ne)}/{len(rows)}; no-edit sessions: edited tests "
              f"{[o['n_edit_tests'] for o in ne]}, save_edit calls {[o['save_calls'] for o in ne]}, best paired diff seen "
              f"{[f(o['max_diff']) for o in ne]}, minutes {[o['work_min'] for o in ne]}; edit sessions: rewards "
              f"{[round(o['reward'], 1) for o in rows if o['net_edit']]}, edited tests {[o['n_edit_tests'] for o in rows if o['net_edit']]}")
        print(f"  mentions per session: earlier edit {[o['earlier_mentions'] for o in rows]}, repeated </think> "
              f"{[o['storm'] for o in rows]}, missing boxed {[o['nobox'] for o in rows]}")
        print(f"  targets in code that saved an edit: {Counter(t for o in rows for t in o['saved_targets'])}")
    ne_all = [o for s in (5, 6, 7, 8) for o in S[s] if o["ok"] and not o["net_edit"]]
    ed_all = [o for s in (5, 6, 7, 8) for o in S[s] if o["ok"] and o["net_edit"]]
    print(f"- no-edit vs edit sessions, steps 5-8: n {len(ne_all)} vs {len(ed_all)}; minutes {mean([o['work_min'] for o in ne_all]):.1f} vs "
          f"{mean([o['work_min'] for o in ed_all]):.1f}; forward {mean([o['fwd'] for o in ne_all]):.1f} vs {mean([o['fwd'] for o in ed_all]):.1f}; "
          f"edited tests {mean([o['n_edit_tests'] for o in ne_all]):.1f} vs {mean([o['n_edit_tests'] for o in ed_all]):.1f}; "
          f"sessions that ever saw a paired diff > +2 SE: "
          f"{sum(1 for o in ne_all if any(x['diff'] is not None and x['se'] and x['diff'] > 2 * x['se'] for x in o['tests']))} vs "
          f"{sum(1 for o in ed_all if any(x['diff'] is not None and x['se'] and x['diff'] > 2 * x['se'] for x in o['tests']))}")
    print(f"- edit sessions' reward, steps 5-8: {sorted(round(o['reward'], 1) for o in ed_all)}; mean {mean([o['reward'] for o in ed_all]):.2f}")
    print(f"- repaired tool calls per session, step 1: {[o['repaired'] for o in S[1]]}; eval untrained "
          f"{[o['repaired'] for o in EV['untrained']]}, step-4 editor {[o['repaired'] for o in EV['step4']]}")
    print(f"- reward-set reward, eval untrained {[round(o['reward'], 1) for o in EV['untrained']]} vs step-4 editor "
          f"{[round(o['reward'], 1) for o in EV['step4']]}")
    print(f"- rewards of the no-op submissions (tensors saved identical): "
          f"{[(o['name'], round(o['reward'], 1)) for o in ne_all if o['n_tensors']]}")

    # ---- eval detail
    print("\n## Held-out sessions\n")
    for k, rows in EV.items():
        for o in rows:
            print(f"- {o['name']}: reward {o['reward']:.1f}, held-out {o['heldout']}, {o['kinds']}:{o['target']} ({o['alpha']}), "
                  f"{o['work_min']} min, ended {o['ended']}, fwd {o['fwd']}, TF {o['n_tf']}, edited tests {o['n_edit_tests']} "
                  f"sizes {o['test_sizes']}, confirm {o['confirm_tests']}, first save {o['first_save_min']}, last save {o['last_save_min']}, "
                  f"after last save {o['after_last_save_min']} min, comp/turn {o['comp_per_turn']:.0f}")

    # ---- per-session table
    print("\n## Sessions\n")
    print("| session | reward | adv | min | turns | fwd/py | TF | edited tests (confirm) | sizes | 1st/last save (min) | after last save (min) | kinds | target | strength |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for o in allrows + EV["untrained"] + EV["step4"]:
        if not o["ok"]:
            print(f"| {o['name']} | failed | | | | | | | | | | | | |")
            continue
        sz = Counter(o["test_sizes"])
        print(f"| {o['name']} | {o['reward']:.1f} | {f(o['adv'], 2)} | {o['work_min']} | {o['turns']} | {o['fwd']}/{o['py']} | {o['n_tf']} | "
              f"{o['n_edit_tests']} ({o['confirm_tests']}) | {' '.join(f'{k}x{v}' for k, v in sz.items())} | "
              f"{f(o['first_save_min'])}/{f(o['last_save_min'])} | {f(o['after_last_save_min'])} | {o['kinds']} | {o['target']} | {o['alpha']} |")
    if detail:
        for o in allrows + EV["untrained"] + EV["step4"]:
            if o["ok"]:
                print(f"\n### {o['name']}")
                for x in o["tests"]:
                    print("   ", x)


if __name__ == "__main__":
    main()
