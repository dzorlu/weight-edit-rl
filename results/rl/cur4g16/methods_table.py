"""One row per cur4g16 session: diagnosis methods used and the edit made. Writes methods_by_session.csv and prints a summary.
Uses analyze_cur4g16.analyse (regex counts are approximate; edit kinds were labelled by hand)."""
import csv, json, sys, collections
from pathlib import Path
sys.argv = [sys.argv[0]]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_cur4g16 as A

A.fetch("/data/math_practice/splits.json", A.SPLITS_PATH)
sp = {x: k for k, ids in json.load(open(A.SPLITS_PATH)).items() for x in ids}
steps = {r["step"]: r for r in map(json.loads, open(HERE / "steps.jsonl")) if r.get("event") == "step"}
KIND = {"RA": "readout amplify", "RR": "readout remove", "UZ": "units zero", "US": "units scale", "UW": "unit writes token",
        "LD": "loop direction remove", "GW": "gated rank-1 write", "HZ": "head zero", "UF": "units flip", "RW": "gated rank-1 write", "H": "attention head"}
rows = []
for s in range(8):
    rew = {ro["rid"]: ro.get("reward") for ro in steps[s]["rollouts"]}
    for i in range(16):
        p = HERE / f"cur4g16-s{s}-r{i}.jsonl"
        if not p.exists():
            continue
        o = A.analyse(p, f"s{s}-r{i}", sp)
        diag = [n for n, on in (("teacher-forced passes", o.get("n_tf", 0) > 0), ("attribution", o.get("attr", 0) > 0),
                               ("difference of means", o.get("dom", 0) > 0), ("logit/readout lens", o.get("lens", 0) > 0),
                               ("weights only before first edit", o.get("weights_only_first_edit"))) if on]
        kinds = o.get("kinds", "")
        rows.append({"step": s + 1, "session": f"s{s}-r{i}", "reward": round(100 * (rew.get(f"cur4g16-s{s}-r{i}") or 0), 1),
                     "diagnosis": "; ".join(diag), "tf": o.get("n_tf", 0), "attr_calls": o.get("attr", 0), "dom_calls": o.get("dom", 0),
                     "lens_calls": o.get("lens", 0), "edit": " + ".join(KIND.get(k, k) for k in kinds.split("+")) if kinds else "",
                     "target": o.get("target", ""), "detail": o.get("alpha", ""), "edited_tests": o.get("n_edit_tests", 0)})
with open(HERE / "methods_by_session.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "sessions")
for s in range(1, 9):
    R = [r for r in rows if r["step"] == s]
    c = collections.Counter()
    for r in R:
        for k in r["edit"].split(" + "):
            c[k or "none"] += 1
    d = collections.Counter(x for r in R for x in r["diagnosis"].split("; ") if x)
    print(f"step {s}: diagnosis {dict(d)} | edits {dict(c.most_common())}")
