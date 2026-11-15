"""Why did one RL update not raise the reward? Compare hero3 step 1 (s0, the rollouts the update trained on)
with nolib1 step 2 (same updated checkpoint, prompt without the web/library section).

Reads the rollout logs and steps.jsonl only. Reuses analyse() from results/rl/hero3/analyze_steps.py.
Run from anywhere:  python3 results/rl/nolib1/analyze_step2.py [--detail]
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HERO = HERE.parent / "hero3"
sys.path.insert(0, str(HERO))
import analyze_steps as A  # noqa: E402

# nolib1 used the same ready-made splits as hero3 (same first problem ids).
SPLITS = A.HERO_SPLITS

# Final edits, annotated by hand from each FINAL SUBMISSION text (final_text in the summary line).
# family: readout = rank-one edit along a token readout direction r(T) = E[T](1+gamma) (Wait and/or </think>);
#         unit = down_proj column zero/scale/flip; loop = difference-of-means direction removal.
EDITS = {
    "s0-r0": dict(family="unit", edit="7 Wait-loop units of L23 down_proj = 0"),
    "s0-r1": dict(family="readout", edit="</think> specific (top-80, 5 PCs) +0.45 on 48 writers"),
    "s0-r2": dict(family="unit", edit="16 units = 0, 5 units x2-3"),
    "s0-r3": dict(family="unit", edit="18 units x5, 4 units = 0"),
    "s0-r4": dict(family="readout", edit="Wait specific (top-600) -1.0 + </think> specific +2.0 on 48 writers"),
    "s0-r5": dict(family="none", edit="no edit (no tool call in five replies)"),
    "s0-r6": dict(family="loop", edit="difference-of-means loop direction -1 on L23 o_proj"),
    "s0-r7": dict(family="readout", edit="plain Wait readout -1.0 on 48 writers + plain </think> +0.6 on L18-23 down_proj"),
    "n-r0": dict(family="readout", edit="plain Wait readout -1.0 on 48 writers + plain </think> +1.0 on L23 o_proj"),
    "n-r1": dict(family="unit", edit="6 Wait units x(-1): L16 u998, L20 u1111, L22 u194, L23 u36/u269/u346"),
    "n-r2": dict(family="readout", edit="plain Wait readout -1.0 on 48 writers"),
    "n-r3": dict(family="readout", edit="plain </think> readout +6 on L23 o_proj"),
    "n-r4": dict(family="unit", edit="L19 down_proj unit 31 x2"),
    "n-r5": dict(family="unit+loop", edit="8 Wait units (7 = 0, L16 u998 x-1.5) + loop direction -1 on L15-23 writers"),
    "n-r6": dict(family="readout", edit="plain </think> readout +6 on L23 o_proj"),
    "n-r7": dict(family="loop", edit="difference-of-means loop direction -1 on 48 writers"),
}


def turns_tokens(path):
    """Completion tokens per editor turn, and the turns whose python call saves an edit."""
    recs = A.load(path)
    comp = {int(r["turn"]): int(r["completion_tokens"]) for r in recs if "reasoning" in r}
    save = {int(r["turn"]) for r in recs if r.get("tool") == "python" and "save_edit(" in r["input"]}
    final = {int(r["turn"]) for r in recs if "reasoning" in r and (r.get("final") or "").lstrip().startswith("FINAL")}
    gen = {int(r["turn"]) for r in recs if r.get("tool") == "forward" and '"generate"' in r["input"]}
    return comp, save, final, gen


def main():
    detail = "--detail" in sys.argv
    s1 = [A.analyse(HERO / f"hero3-s0-r{i}.jsonl", f"s0-r{i}", SPLITS) for i in range(8)]
    s2 = [A.analyse(HERE / f"nolib1-s0-r{i}.jsonl", f"n-r{i}", SPLITS) for i in range(8)]
    for o in s1 + s2:
        o.update(EDITS[o["name"]])
        o["form"], o["dirs"], o["alpha"] = o["family"], "", o["edit"]
        o["test"] = None
    for label, rows in (("hero3 step 1 (s0)", s1), ("nolib1 step 2", s2)):
        print(f"\n## {label}\n")
        print(A.table(rows))
        A.group_stats(rows, label)

    # 1. recipe families
    print("\n## Recipe family: count, mean reward (pts)")
    for label, rows in (("step 1", s1), ("nolib1 step 2", s2)):
        fam = {}
        for o in rows:
            fam.setdefault(o["family"], []).append(round(o["reward"] * 100, 1))
        print(f"  {label}: " + "; ".join(f"{k} n={len(v)} mean {st.mean(v):.1f} {v}" for k, v in sorted(fam.items())))

    # Editor's own confirm estimate vs reward
    print("\n## Own confirm estimate (mean of final confirm_1/confirm_2 diffs) vs reward, pts")
    for rows in (s1, s2):
        gaps = []
        for o in rows:
            fc = [p["diff"] for p in o["final_confirm"].values() if p["diff"] is not None]
            est = st.mean(fc) if fc else None
            r = o["reward"] * 100
            if est is not None:
                gaps.append(est - r)
            print(f"  {o['name']}: confirm {[round(x, 1) for x in fc]} est {fmt(est)} reward {r:.1f}")
        print(f"  mean overstatement {st.mean(gaps):.1f} pts over {len(gaps)} rollouts")

    # 3. training signal of the update (hero3 step 1)
    steps1 = [json.loads(l) for l in open(HERO / "steps.jsonl")]
    steps2 = [json.loads(l) for l in open(HERE / "steps.jsonl")]
    u = next(r for r in steps1 if r.get("event") == "step" and r["step"] == 0)
    n2 = next(r for r in steps2 if r.get("event") == "step" and r["step"] == 0)
    h2 = next(r for r in steps1 if r.get("event") == "step" and r["step"] == 1)
    print("\n## Update trained on hero3 step 1")
    print(f"  advantages {u['advantages']}; sequences {u['training_sequences']}; training tokens {u['training_tokens']}; "
          f"action tokens {u['action_tokens']}; lr 2e-5")
    print(f"  -mean sampled logprob: step 1 {u['entropy']} ({u['action_tokens']} tok), nolib1 step 2 {n2['entropy']} "
          f"({n2['action_tokens']} tok), hero3 step 2 {h2['entropy']} ({h2['action_tokens']} tok)")
    sampled = [ro["tokens"]["sampled"] for ro in u["rollouts"]]
    adv = u["advantages"]
    pos = sum(t for t, a in zip(sampled, adv) if a > 0)
    neg = sum(t for t, a in zip(sampled, adv) if a < 0)
    print(f"  sampled tokens per rollout {sampled}")
    print(f"  tokens with positive advantage {pos} ({pos / sum(sampled):.1%}), negative {neg}; "
          f"sum(adv x tokens) = {sum(a * t for a, t in zip(adv, sampled)):.0f}; sum(|adv| x tokens) = {sum(abs(a) * t for a, t in zip(adv, sampled)):.0f}")
    tot_c = tot_save = tot_final = tot_gen = 0
    for i in range(8):
        comp, save, final, gen = turns_tokens(HERO / f"hero3-s0-r{i}.jsonl")
        tot_c += sum(comp.values())
        tot_save += sum(comp[t] for t in save if t in comp)
        tot_final += sum(comp[t] for t in final if t in comp)
        tot_gen += sum(comp[t] for t in gen if t in comp)
        print(f"  s0-r{i}: adv {adv[i]:+.2f}, completion tokens {sum(comp.values())}, in save_edit turns {sum(comp[t] for t in save if t in comp)} "
              f"({len(save)} turns), final reply {sum(comp[t] for t in final if t in comp)}")
    print(f"  all: completion tokens {tot_c}; save_edit turns {tot_save} ({tot_save / tot_c:.1%}); final replies {tot_final} ({tot_final / tot_c:.1%}); turns issuing a generate call {tot_gen} ({tot_gen / tot_c:.1%})")
    for a, t, i in sorted(zip(adv, sampled, range(8)), key=lambda x: -abs(x[0] * x[1])):
        print(f"  s0-r{i}: adv x tokens = {a * t:+.0f}")

    # Power: two groups of 8
    r1 = [o["reward"] * 100 for o in s1]
    r2 = [o["reward"] * 100 for o in s2]
    sd1, sd2 = st.stdev(r1), st.stdev(r2)
    se = math.sqrt(sd1 ** 2 / 8 + sd2 ** 2 / 8)
    print("\n## Detectable difference, 8 vs 8")
    print(f"  means {st.mean(r1):.1f} vs {st.mean(r2):.1f}; sample sd {sd1:.1f} / {sd2:.1f}; SE of difference {se:.1f}; "
          f"t = {(st.mean(r2) - st.mean(r1)) / se:.2f}")
    print(f"  difference detectable at 5% two-sided with 80% power ~ (2.14 + 0.87) x SE = {3.01 * se:.1f} pts "
          f"(t, 14 df); with only 3-pt scoring noise: {3.01 * 3 * math.sqrt(2 / 8):.1f} pts")
    for p, nm in ((st.mean(r1) / 100 + 0.074, "step1"), (st.mean(r2) / 100 + 0.074, "nolib1")):
        print(f"  scoring SE at edited accuracy {p:.3f}, 256 answers: {100 * math.sqrt(p * (1 - p) / 256):.1f} pts ({nm})")

    if detail:
        for o in s1 + s2:
            print(f"\n### {o['name']} {o['edit']}  tf_layers={o['tf_layers']}")
            for p in o["paired"]:
                print("   ", p)


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


if __name__ == "__main__":
    main()
