"""Benchmark an unedited model served on Tinker (default Inkling 2 Small): MATH-500, MMLU-Pro, MedXpertQA, SciBench.

    TINKER_API_KEY=... PYTHONPATH=. uvx --with openai --with datasets python scripts/baseline_sweep_tinker.py

Writes results/baselines/<tag>/. Stops sending requests after `USD_CAP` of API spend.
"""
import os, re, json, sys, threading, concurrent.futures as cf
from openai import OpenAI
from harness.grading import grade_math, grade_sci

MODEL = os.environ.get("SWEEP_MODEL", "thinkingmachines/Inkling-2-Small:peft:262144")
TAG = os.environ.get("SWEEP_TAG", "inkling2s")
USD_CAP, PRICE_IN, PRICE_OUT = 12.0, 0.30, 1.00
LIMIT = int(os.environ.get("SWEEP_LIMIT", "0"))
client = OpenAI(base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", api_key=os.environ["TINKER_API_KEY"], timeout=1800, max_retries=3)
spent, lock = [0.0], threading.Lock()


CAP = 16384
N = {"math500": 150, "mmlu_pro": 250, "medxpertqa": 250, "scibench": 250}


def build_items():
    import random, ast
    from datasets import load_dataset
    rng = random.Random(20261113)
    items = []

    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    for i in rng.sample(range(len(ds)), N["math500"]):
        r = ds[i]
        items.append(dict(bench="math500", id=r["unique_id"], gold=r["answer"], meta=r["subject"] + "/L" + str(r["level"]),
                          prompt=r["problem"] + "\n\nPut your final answer in \\boxed{}."))

    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    for i in rng.sample(range(len(ds)), N["mmlu_pro"]):
        r = ds[i]
        opts = r["options"] if isinstance(r["options"], list) else ast.literal_eval(r["options"])
        letters = "ABCDEFGHIJ"
        body = "\n".join(f"({letters[k]}) {o}" for k, o in enumerate(opts))
        items.append(dict(bench="mmlu_pro", id=str(r["question_id"]), gold=r["answer"], meta=r["category"],
                          prompt=f"{r['question']}\n{body}\n\nFinish with a line of the form 'Answer: X' where X is the letter of the correct option."))

    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text")["test"]
    for i in rng.sample(range(len(ds)), N["medxpertqa"]):
        r = ds[i]
        items.append(dict(bench="medxpertqa", id=r["id"], gold=r["label"], meta=r["question_type"],
                          prompt=f"{r['question']}\n\nFinish with a line of the form 'Answer: X' where X is the letter of the correct option."))

    ds = load_dataset("xw27/scibench")["train"]
    idx = [i for i in range(len(ds)) if str(ds[i]["answer_number"]).strip()]
    for i in rng.sample(idx, N["scibench"]):
        r = ds[i]
        unit = (r["unit"] or "").strip()
        items.append(dict(bench="scibench", id=r["source"] + ":" + r["problemid"].strip(), gold=str(r["answer_number"]).strip(), meta=r["source"],
                          prompt=f"{r['problem_text']}\n\nGive the numeric answer in units of {unit if unit else '(dimensionless)'}. Put only the number in \\boxed{{}}."))
    return items


def grade(bench, text, gold):
    if bench == "math500":
        ext, ok = grade_math(text, gold); return ext, ok, None
    if bench == "scibench":
        ext, ok, bare = grade_sci(text, gold); return ext, ok, bare
    m = re.findall(r"Answer:\s*\**\(?([A-J])\)?", text)
    return (m[-1] if m else None), bool(m and m[-1] == gold.strip()), None


def one(it):
    if spent[0] >= USD_CAP:
        return dict(it, skipped=True)
    try:
        r = client.chat.completions.create(model=MODEL, max_tokens=CAP, reasoning_effort="high", messages=[{"role": "user", "content": it["prompt"]}])
    except Exception as e:
        return dict(it, error=repr(e)[:300])
    ch = r.choices[0]
    with lock:
        spent[0] += (r.usage.prompt_tokens * PRICE_IN + r.usage.completion_tokens * PRICE_OUT) / 1e6
    content = ch.message.content or ""
    reasoning = getattr(ch.message, "reasoning_content", None) or ""
    ext, ok, bare = grade(it["bench"], content if content.strip() else reasoning, it["gold"])
    return dict(it, content=content, reasoning=reasoning[:20000], finish=ch.finish_reason, completion_tokens=r.usage.completion_tokens, extracted=ext, correct=bool(ok), bare_number=bare)


items = build_items()
if LIMIT:
    seen, keep = {}, []
    for it in items:
        if seen.setdefault(it["bench"], 0) < LIMIT:
            seen[it["bench"]] += 1; keep.append(it)
    items = keep
out = f"results/baselines/{TAG}"
os.makedirs(out, exist_ok=True)
with cf.ThreadPoolExecutor(int(os.environ.get("SWEEP_CONC", "16"))) as ex:
    res = list(ex.map(one, items))
summary = {}
for b in sorted({it["bench"] for it in items}):
    rs = [r for r in res if r["bench"] == b]; done = [r for r in rs if "correct" in r]
    with open(f"{out}/{b}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rs)
    if not done:
        summary[b] = {"n": 0, "skipped_or_error": len(rs), "first_error": next((r.get("error") for r in rs if r.get("error")), None)}; continue
    summary[b] = {"n": len(done), "skipped_or_error": len(rs) - len(done), "accuracy": round(100 * sum(r["correct"] for r in done) / len(done), 1),
                  "cap_hit_pct": round(100 * sum(r["finish"] == "length" for r in done) / len(done), 1),
                  "no_answer_pct": round(100 * sum(r["extracted"] is None for r in done) / len(done), 1),
                  "mean_tokens": round(sum(r["completion_tokens"] for r in done) / len(done))}
    if b == "scibench":
        bare = [r for r in done if r["bare_number"]]
        summary[b]["bare_number_pct"] = round(100 * len(bare) / len(done), 1)
summary["_run"] = {"model": MODEL, "cap": CAP, "reasoning_effort": "high", "usd": round(spent[0], 2)}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
