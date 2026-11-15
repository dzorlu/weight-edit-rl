"""RL loop: train Inkling 2 Small (on Tinker) to be a better weight editor of Qwen3.5-0.8B (on Modal).

One RL step: sample a group of rollouts with the current checkpoint, score each submitted edit,
take one policy-gradient step on Tinker, save a checkpoint.

    uvx modal run -m rl.loop::prep_reward_set          # once
    uvx modal run -m rl.loop::prep_practice_index      # once: index and ready-made splits of the practice problems
    uvx modal run -m rl.loop::prep_baseline            # once: the original weights' answers every rollout starts from
    uvx modal run -m rl.loop::prep_originals           # once: the original weights' answers on the splits, served from a cache
    uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32
    uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32 --rollouts-per-gpu 2   # two rollouts share each L40S
    uvx modal run --detach -m rl.loop::evaluate --name ev1 --checkpoint tinker://... --dataset gsm8k_test   # rollouts with no update, edits scored

Cost bounds: each rollout holds one L40S (or half of one with --rollouts-per-gpu 2) for at most `rollout_minutes`; the whole run stops at `hours`.
Outputs: /runs/rl/<exp>/ on the runs volume (steps.jsonl, one transcript per rollout).
"""
import modal
from harness.target import app, runs, base, base_src, SRC, heldout_vol, Target, PackedTarget, PY_BUDGET, GEN_CAP, last_boxed, heldout_eval
from harness.target import weights, offline, _Core, base_dirs, heldout_cache, heldout_rows, start_vllm, stop_proc, BASES_W, BASES_R, EVAL_SETS, prep_eval_set
from harness.prompts import PROMPT, GUIDE

MODEL = "thinkingmachines/Inkling-2-Small:peft:262144"  # 256K-context variant; a rollout is one long sequence
PRICE = dict(prefill=0.30, cached=0.03, sample=1.00, train=3.00)  # $ per million tokens
rl_image = modal.Image.debian_slim(python_version="3.12").pip_install("tinker-cookbook", "openai").add_local_python_source(*SRC)

TOOLS = [
    {"name": "python", "description": "Run code in the persistent Python kernel. If the last line is an expression, its value is shown.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}}, "required": ["code"]}},
    # Flat arguments: Inkling sometimes drops a closing brace when the request is nested three objects deep.
    {"name": "forward", "description": "One forward pass on the target model. type 'generate': the model answers problems. type 'teacher_forced': the model reads texts and returns log-probabilities and activations. A successful call uses one forward pass of the budget.",
     "parameters": {"type": "object", "properties": {
         "type": {"type": "string", "enum": ["generate", "teacher_forced"]},
         "model": {"type": "string", "enum": ["original", "edited", "both"]},
         "problems": {"type": "array", "items": {"type": "string"}, "description": "generate: practice ids"},
         "prompts": {"type": "array", "items": {"type": "string"}, "description": "generate: your own user messages, used verbatim"},
         "n": {"type": "integer", "description": "generate: samples per item"},
         "max_tokens": {"type": "integer", "description": "generate: at most 8192"},
         "texts": {"type": "array", "items": {"type": "string"}, "description": "teacher_forced: raw texts"},
         "chats": {"type": "array", "description": "teacher_forced: chats, each a list of {role, content} messages"},
         "enable_thinking": {"type": "boolean", "description": "teacher_forced with chats: use the thinking template"},
         "token_logprobs": {"type": "boolean"}, "topk": {"type": "integer", "description": "at most 50"},
         "hidden_states": {"type": "array", "items": {"type": "integer"}, "description": "layers 0..24"},
         "mlp_act": {"type": "array", "items": {"type": "integer"}, "description": "layers 0..23"}},
      "required": ["type", "model"]}},
]
RETURN_KEYS = ("token_logprobs", "topk", "hidden_states", "mlp_act")
WEB_TOOL = {"name": "web_search", "description": "Search the web (Exa). Returns titles, links and short excerpts. Use it for methods and background, for example interpretability techniques.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
WEB_READ_TOOL = {"name": "web_read", "description": "Fetch one web page or paper as text. The full text is saved to /library/<name>.md and the first part is returned.",
                 "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}
WEB_BUDGET = 20
LIBRARY = "/library"
# Pages that publish ready-made edits for this very model would let the editor copy an answer; they are kept out.
WEB_EXCLUDE = ["raw.githubusercontent.com/RikiyaT", "rikiyat.github.io", "github.com/RikiyaT", "huggingface.co/RikiyaT", "github.com/WSP-Scope", "arxiv.org/abs/2611.09382", "arxiv.org/pdf/2611.09382", "arxiv.org/html/2611.09382"]
WEB_BLOCK = ("weight-space programming", "weight space programming", "weight-space-programming", "2611.09382", "qwen3", "qwen35", "qwen 3", "qwen-3", "gpt-6 astra", "astra", "rikiyat", "lm_understand", "wsp-scope", "math weight editing")
WEB_TEXT = """

WEB AND LIBRARY. You have two more tools: `web_search` (parameter `query`) returns titles, links and short excerpts, and `web_read` (parameter `url`) fetches a page or paper, saves its text to /library/<name>.md and returns the first part. Together at most 20 calls. Use them to look up methods you want to apply, for example how an interpretability or model-editing technique is computed, and read the paper before relying on it. Results about this specific model or about published edits to it are filtered out. The python kernel itself has no internet access.
/library is shared: it already holds what earlier sessions saved, and what you add is kept for later sessions. Read /library/INDEX.md first. When you learn something reusable from a paper, write a short note file in /library (title, link, the method in a few lines, how you applied it and whether it worked here) and add one line for it to /library/INDEX.md. Keep the folder tidy: one file per paper or method, no scratch files, no duplicates. Files you only need for this session belong in /work."""

KNOWN_TEXT = """

WHAT IS ALREADY KNOWN ABOUT THIS MODEL. You do not need to spend a forward pass on a first diagnosis. /practice/baseline/original.jsonl holds the original weights' answers to {n} generations on practice problems (same format as the gen files: qid, sample, source, prompt, gold, text, extracted, correct, tokens, finish), and /practice/baseline/SUMMARY.txt has the counts:
{summary}
Do not use these baseline problems to confirm an edit.

SANDBOX NOTES.
- Practice ids look like "math/geometry/461" and "gsm8k/2754". /practice/index.tsv lists every problem (id, source, level, subject); /practice/problems.jsonl has the full text and reference solutions. Do not guess ids.
- /practice/splits.json holds ready-made disjoint lists of ids: "baseline" (already answered by the original weights, see above), "select" (256 problems for choosing between edits), "confirm_1" and "confirm_2" (256 each, never to be used for a choice), and "extra" (the rest). Use these; do not build your own.
- Tensor names: model.language_model.layers.<L>.mlp.{{gate_proj,up_proj,down_proj}}.weight for L = 0..23. Layers 3, 7, 11, 15, 19, 23 have full attention (…layers.<L>.self_attn.{{q_proj,k_proj,v_proj,o_proj}}.weight); the other layers have linear attention (…layers.<L>.linear_attn.…, see NAMES).
- load(name) returns a float32 torch tensor. save_edit(name, tensor) takes a torch tensor of the same shape; convert numpy arrays with torch.from_numpy(a).float().
- tok is a tokenizers.Tokenizer: tok.encode(text).ids, tok.decode(ids), tok.token_to_id(s).
- The original weights' answers to every problem in "select", "confirm_1" and "confirm_2" (2 samples each, 8192-token cap) are already in /practice/original/<split>.jsonl. Read them there; do not spend a forward pass on them. A generation with "model": "both" on these problems only has to run the edited weights, so it returns in about half the time. This holds for "n": 2 or 1 and "max_tokens": 8192.
- A generation with "model": "both" takes at most 256 answers per model; with one model at most 512. One split with "n": 1 is 256 answers; half a split with "n": 2 is 256 answers.
- Tool output is shown up to about 6,000 characters. Print summaries, and keep long results in files."""

FREE_REIN_OLD = "C. Write the edit. These forms all change a matrix that writes into the residual stream (MLP down_proj, attention output projection)."
FREE_REIN_NEW = ("C. Write the edit. The forms below are starting points that have worked on models like this one. You are not limited to them: any tensor inside the transformer blocks is in scope "
                 "(MLP gate_proj and up_proj, attention query, key, value and output projections, several layers at once, combinations, or a method of your own or from a paper you read), "
                 "as long as you test it. The listed forms change a matrix that writes into the residual stream (MLP down_proj, attention output projection).")
DIAG_OLD_START, DIAG_OLD_END = "1. Open with one large generation on the original weights", "2. Before any analysis"
DIAG_NEW = ("1. Start from the baseline answers in /practice/baseline (see WHAT IS ALREADY KNOWN): count how many are cut at the token limit, and find which words or phrases are far more frequent in the failed texts than in the completed ones.\n"
            "2. Use the splits in /practice/splits.json: choose between edits on \"select\", and confirm the final candidate on \"confirm_1\" and \"confirm_2\", which must not be used for any choice.\n")
SPLIT_OLD_END = "\n\nB. Locate the component."


def exa_search(query, key):
    import json, urllib.request
    body = {"query": str(query)[:500], "type": "auto", "numResults": 8, "contents": {"highlights": True}, "excludeDomains": WEB_EXCLUDE}
    req = urllib.request.Request("https://api.exa.ai/search", json.dumps(body).encode(), {"x-api-key": key, "Content-Type": "application/json", "User-Agent": "research-harness"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    out, dropped = [], 0
    for x in r.get("results", []):
        hl = " ... ".join(x.get("highlights") or [])[:900]
        blob = " ".join([x.get("title") or "", x.get("url") or "", hl]).lower()
        if any(b in blob for b in WEB_BLOCK):
            dropped += 1
            continue
        out.append(f"- {x.get('title')}\n  {x.get('url')}\n  {hl}")
    return ("\n".join(out[:5]) or "[no results]") + (f"\n[{dropped} results were filtered out]" if dropped else "")


def exa_read(url, key):
    """(title, text) of a page through Exa's contents endpoint; refuses pages the search filter would drop."""
    import json, urllib.request
    low = str(url).lower()
    if any(b in low for b in WEB_BLOCK) or any(x.lower() in low for x in WEB_EXCLUDE):
        return None, "[this page is filtered out]"
    body = {"urls": [str(url)[:2000]], "text": {"maxCharacters": 120000}}
    req = urllib.request.Request("https://api.exa.ai/contents", json.dumps(body).encode(), {"x-api-key": key, "Content-Type": "application/json", "User-Agent": "research-harness"})
    r = json.loads(urllib.request.urlopen(req, timeout=90).read())
    res = r.get("results") or []
    if not res or not (res[0].get("text") or "").strip():
        return None, "[the page could not be read]"
    title, text = res[0].get("title") or "", res[0]["text"]
    head = (title + " " + text[:3000]).lower()
    if any(b in head for b in WEB_BLOCK):
        return None, "[this page is filtered out]"
    return title, text


def forward_request(args):
    """The target's request JSON from the tool's flat arguments (a nested `request` or `return` is also accepted)."""
    import json
    req = args.get("request", args)
    if isinstance(req, str):
        try:
            req = json.loads(req)
        except Exception:
            return req
    req = dict(req)
    ret = dict(req.pop("return", None) or {})
    for k in RETURN_KEYS:
        if k in req:
            ret[k] = req.pop(k)
    if req.get("type") == "teacher_forced":
        req["return"] = ret
    return json.dumps(req)


def _lenient_json(js, tries=6):
    """Parse a JSON object that may be followed by other text and may lack up to `tries` closing brackets,
    braces or quotes (Inkling sometimes writes `}],` where `}]],` is meant). Returns the object or None."""
    import json
    dec = json.JSONDecoder(strict=False)
    for _ in range(tries + 1):
        try:
            return dec.raw_decode(js)[0]
        except json.JSONDecodeError as e:
            best = None
            # the missing bracket belongs at the error or just before one of the last commas before it
            # ({"chats": [[{...}], "type": ...} fails at the colon after "type")
            pos, p = [e.pos], e.pos
            for _ in range(3):
                p = js.rfind(",", 0, p)
                if p < 0:
                    break
                pos.append(p)
            for at in pos:
                for ins in ("]", "}", '"'):
                    cand = js[:at] + ins + js[at:]
                    try:
                        return dec.raw_decode(cand)[0]
                    except json.JSONDecodeError as e2:
                        if e2.pos > e.pos + 1 and (best is None or e2.pos > best[0]):
                            best = (e2.pos, cand)
            if best is None:
                return None
            js = best[1]
    return None


def repair_reply(raw):
    """Read a reply the renderer could not parse: (thinking, tool name, arguments) or None.

    Causes: tool-call JSON that is short of closing braces, and Inkling writing its own tool-call markers
    (<|message_model|>NAME<|content_invoke_tool_json|>{...}) as plain text, which the renderer does not parse.
    The thinking often quotes the marker while discussing it, so every occurrence is tried, last first."""
    import json, re
    think = "\n\n".join(t.strip() for t in re.findall(r"<\|content_thinking\|>(.*?)<\|end_message\|>", raw, re.S))
    raw = re.sub(r"<\|content_thinking\|>.*?<\|end_message\|>", "", raw, flags=re.S)  # markers quoted inside the thinking are not calls
    ms = list(re.finditer(r"<\|message_model\|>([\w.]+)<\|content_invoke_tool_json\|>(.*?)(?=<\|end_message\|>|<\|content_model_end_sampling\|>|<\|message_model\|>|$)", raw, re.S))
    if not ms:
        return None
    for m in reversed(ms):
        if m.group(1).split(".")[-1] not in ("python", "forward", "web_search", "web_read"):
            continue
        obj = _lenient_json(m.group(2).strip())
        for obj in ([obj] if obj is not None else []):
            if isinstance(obj, dict):
                args = obj.get("args", obj.get("arguments", obj))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                return think, obj.get("name") or m.group(1), args if isinstance(args, dict) else {}
    return None


TEST_SPLITS = ("select", "confirm_1", "confirm_2")  # the splits whose original answers are cached for each starting model


# In RL the reward should reflect the best edit a rollout found, so the editor always submits one.
SUBMIT_RULE_OLD = 'Submit an edit only if a paired comparison on at least 300 generations that were not used to choose it shows a gain larger than 2 standard errors. Otherwise call drop_edit on every edited tensor and submit the original weights; that is a valid result and better than submitting an edit the evidence does not support.'
SUBMIT_RULE_RL = 'Always finish with your best edit saved in /work/edited. Your score is the accuracy of that edit minus the accuracy of the original weights on unseen problems, so a harmful edit scores below zero and the original weights score zero. Do not drop a promising edit because you ran out of budget to confirm it: submit the candidate your evidence supports best, and say how strong that evidence is.'


BASE_TEXT = """

STARTING WEIGHTS. The weights you are given are Qwen3.5-0.8B with an earlier edit already applied. In this session "original weights" means these starting weights: load_original, the cached original answers in /practice and your score all refer to them. Your task is to improve on them."""


def editor_prompt(forward_budget, web=False, known=None, rebased=False):
    a, b = PROMPT.index("WHAT COUNTS."), PROMPT.index("WHAT YOU HAVE.")
    body = PROMPT[:a] + GUIDE + "\n\n" + PROMPT[b:]
    for old in ("a budget of 50 forward passes in total", "When all 50 forward passes are used"):
        assert old in body
        body = body.replace(old, old.replace("50", str(forward_budget)))
    for old, new in (
        ("Call the `forward` tool with the parameter `request` set to JSON of this form:", "Call the `forward` tool with these arguments, given directly (not wrapped in a `request` object):"),
        ("Call the `forward` tool with `request` set to:", "Call the `forward` tool with these arguments, given directly:"),
        ('"return": {"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers 0..24], "mlp_act": [layers 0..23]}}', '"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers 0..24], "mlp_act": [layers 0..23]}'),
    ):
        assert old in body, old
        body = body.replace(old, new)
    assert SUBMIT_RULE_OLD in body
    body = body.replace(SUBMIT_RULE_OLD, SUBMIT_RULE_RL)
    assert FREE_REIN_OLD in body
    body = body.replace(FREE_REIN_OLD, FREE_REIN_NEW)
    extra = ""
    if known:
        a, b = body.index(DIAG_OLD_START), body.index(SPLIT_OLD_END)  # replaces steps 1 and 2 of the guide
        body = body[:a] + DIAG_NEW.rstrip() + body[b:]
        extra += KNOWN_TEXT.format(n=known["n"], summary=known["summary"])
    if web:
        extra += WEB_TEXT
    if rebased:
        extra += BASE_TEXT
    b = body.index("HOW TO WORK.")
    return body[:b].rstrip() + extra + "\n\n" + body[b:]


@app.function(image=base_src, volumes={"/runs": runs, "/heldout": heldout_vol}, timeout=1800, cpu=4, memory=8192, retries=0)
def prep_reward_set(n: int = 256):
    """MATH training problems that are in neither the practice set nor MATH-500. The target never mounts this volume."""
    import json, os, random, re
    from datasets import load_dataset
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    used = {norm(json.loads(l)["problem"]) for l in open("/runs/data/math_practice/problems.jsonl")}
    used |= {norm(json.loads(l)["problem"]) for l in open("/heldout/math500.jsonl")}
    rows = []
    for s in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s)["train"]):
            ans = last_boxed(r["solution"])
            if ans is not None and norm(r["problem"]) not in used:
                rows.append({"qid": f"math/{s}/{i}", "problem": r["problem"], "answer": ans})
    random.Random(20261114).shuffle(rows)
    os.makedirs("/heldout/rl", exist_ok=True)
    with open("/heldout/rl/reward_math.jsonl", "w") as f:
        for r in rows[:n]:
            f.write(json.dumps(r) + "\n")
    heldout_vol.commit()
    print("reward set:", min(n, len(rows)), "of", len(rows), "unused MATH training problems")


BASELINE_CODE = r"""
import json, os, collections, shutil
rows = [json.loads(l) for l in open('/work/gen_1.jsonl')]
os.makedirs('/practice/baseline', exist_ok=True)
shutil.copy('/work/gen_1.jsonl', '/practice/baseline/original.jsonl')
pct = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 1)
cut = [r for r in rows if r['finish'] == 'length']; done = [r for r in rows if r['finish'] != 'length']
def opens(rs):
    c, toks = collections.Counter(), 0
    for r in rs:
        toks += r['tokens']
        for line in r['text'].split('\n'):
            w = line.strip().split(' ')[0].strip('*,.:')[:16]
            if w.isalpha(): c[w] += 1
    return c, max(1, toks)
cc, ct = opens(cut); dc, dt = opens(done)
rate = lambda c, t, w: 1000 * c[w] / t
top = sorted((w for w in cc if cc[w] >= 40), key=lambda w: -(rate(cc, ct, w) + 0.01) / (rate(dc, dt, w) + 0.01))[:8]
lines = [
 f"- {len(rows)} answers to {len({r['qid'] for r in rows})} practice problems: {pct([r['correct'] for r in rows])}% correct "
 f"({pct([r['correct'] for r in rows if r['source'] == 'math_train'])}% on MATH, {pct([r['correct'] for r in rows if r['source'] == 'gsm8k_train'])}% on GSM8K).",
 f"- {pct([r['finish'] == 'length' for r in rows])}% run to the 8,192-token limit and {pct([r['extracted'] is None for r in rows])}% have no boxed answer. Of the answers that finish, {pct([r['correct'] for r in done])}% are correct.",
 "- Words that open a line far more often in the cut-off answers than in the finished ones (per 1,000 tokens, cut-off vs finished): "
 + ", ".join(f"{w} {rate(cc, ct, w):.1f} vs {rate(dc, dt, w):.1f}" for w in top) + ".",
]
open('/practice/baseline/SUMMARY.txt', 'w').write("\n".join(lines) + "\n")
json.dump({"n": len(rows), "summary": "\n".join(lines)}, open('/practice/baseline/meta.json', 'w'))
print("\n".join(lines))
"""


@app.function(image=base_src, volumes={"/runs": runs}, timeout=900)
def prep_practice_index(n_math: int = 192, n_gsm: int = 64):
    """Once per target: a compact index of the practice problems and ready-made disjoint splits, so editors do not guess ids."""
    import json, random
    d = "/runs/data/math_practice"
    rows = [json.loads(l) for l in open(f"{d}/problems.jsonl")]
    with open(f"{d}/index.tsv", "w") as f:
        f.write("id\tsource\tlevel\tsubject\n")
        for r in rows:
            subject = r["id"].split("/")[1] if r["source"] == "math_train" else "word_problem"
            f.write(f"{r['id']}\t{r['source']}\t{r.get('level') or '-'}\t{subject}\n")
    m = [r["id"] for r in rows if r["source"] == "math_train"]; g = [r["id"] for r in rows if r["source"] == "gsm8k_train"]
    random.Random(7).shuffle(m); random.Random(7).shuffle(g)  # the same order prep_baseline uses
    take = lambda k: m[n_math + 192 * k: n_math + 192 * (k + 1)] + g[n_gsm + 64 * k: n_gsm + 64 * (k + 1)]
    splits = {"baseline": m[:n_math] + g[:n_gsm], "select": take(0), "confirm_1": take(1), "confirm_2": take(2)}
    used = set(sum(splits.values(), []))
    splits["extra"] = [i for i in m + g if i not in used]
    assert sum(len(v) for v in splits.values()) == len(set(sum(splits.values(), []))) == len(rows)
    json.dump(splits, open(f"{d}/splits.json", "w"))
    runs.commit()
    print({k: len(v) for k, v in splits.items()}, "| first ids", splits["select"][:2], splits["select"][-1])


@app.function(image=rl_image, volumes={"/runs": runs}, timeout=3600)
async def prep_baseline(n_math: int = 192, n_gsm: int = 64):
    """Once per target: the original weights' answers on practice problems, so rollouts need not regenerate them."""
    import json
    tgt = Target.with_options(scaledown_window=60)(rid="baseline")
    out = await tgt.py.remote.aio(
        "import json, random\nrows = [json.loads(l) for l in open('/practice/problems.jsonl')]\n"
        "m = [r['id'] for r in rows if r['source'] == 'math_train']; g = [r['id'] for r in rows if r['source'] == 'gsm8k_train']\n"
        f"random.Random(7).shuffle(m); random.Random(7).shuffle(g)\nprint(json.dumps(m[:{n_math}] + g[:{n_gsm}]))")
    ids = json.loads(out.strip().split("\n")[-1])
    ok, res = await tgt.forward.remote.aio(json.dumps({"type": "generate", "model": "original", "problems": ids, "n": 2, "max_tokens": GEN_CAP}))
    assert ok, res
    print(await tgt.py.remote.aio(BASELINE_CODE))
    await tgt.snapshot.remote.aio("rl/_baseline_work")  # commits the runs volume, where /practice lives


@app.function(image=rl_image, volumes={"/runs": runs}, timeout=3600)
async def prep_originals(splits: str = "select,confirm_1,confirm_2"):
    """Once per target: the original weights' answers on the ready-made splits. The forward tool serves them from this cache."""
    import asyncio, json
    runs.reload()
    sp = json.load(open("/runs/data/math_practice/splits.json"))

    async def one(name):
        tgt = Target.with_options(scaledown_window=60)(rid=f"originals-{name}")
        ok, res = await tgt.forward.remote.aio(json.dumps({"type": "generate", "model": "original", "problems": sp[name], "n": 2, "max_tokens": GEN_CAP}))
        assert ok, res
        print(await tgt.py.remote.aio(
            "import os, shutil, json\nos.makedirs('/practice/original', exist_ok=True)\n"
            f"shutil.copy('/work/gen_1.jsonl', '/practice/original/{name}.jsonl')\n"
            f"rows = [json.loads(l) for l in open('/work/gen_1.jsonl')]\nprint('{name}', len(rows), 'answers,', round(100 * sum(r['correct'] for r in rows) / len(rows), 1), '% correct')"))
        await tgt.snapshot.remote.aio(f"rl/_originals_work_{name}")  # commits the runs volume

    await asyncio.gather(*[one(n) for n in splits.split(",")])


# ----------------------------------------------------------------------------- rebasing: a new starting model
@app.function(image=offline, volumes={"/weights": weights, "/runs": runs}, timeout=40 * 60, cpu=8, memory=32768)
def make_base(tag: str, prev_tag: str, edit_run: str):
    """A starting model `tag`: the starting model `prev_tag` with the edit of /runs/<edit_run>/work/edited applied
    (each edited tensor replaces the one in the weights), plus a practice folder holding the problem files."""
    import json, os, shutil
    from safetensors.torch import load_file, save_file
    runs.reload()
    src, psrc = base_dirs(prev_tag)
    dst, pdst = base_dirs(tag)
    ed = f"/runs/{edit_run}/work/edited"
    edits = {f[:-len(".safetensors")]: load_file(os.path.join(ed, f))["t"] for f in sorted(os.listdir(ed)) if f.endswith(".safetensors")}
    assert edits, f"no edited tensors in {ed}"
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    done = set()
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f.endswith(".safetensors"):
            sd = load_file(p)
            for n, t in edits.items():
                if n in sd:
                    sd[n] = t.to(sd[n].dtype); done.add(n)
            save_file(sd, os.path.join(dst, f), metadata={"format": "pt"})
        elif os.path.isfile(os.path.realpath(p)):
            shutil.copy(os.path.realpath(p), os.path.join(dst, f))
    assert done == set(edits), f"tensors not found in the weights: {sorted(set(edits) - done)}"
    os.makedirs(pdst, exist_ok=True)
    for f in ("problems.jsonl", "index.tsv", "splits.json", "README.txt"):
        shutil.copy(os.path.join(psrc, f), os.path.join(pdst, f))
    json.dump({"tag": tag, "prev": prev_tag, "edit_run": edit_run, "tensors": sorted(edits)}, open(f"{BASES_R}/{tag}/meta.json", "w"), indent=1)
    weights.commit(); runs.commit()
    return sorted(edits)


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, timeout=100 * 60, memory=65536, cpu=8)
def gen_base(tag: str, part: int, reward_items_json: str = "", reward_n: int = 2, reward_items: int = 128):
    """The starting model's own answers, generated once and served from cache, as for the downloaded model.
    part 0: the baseline answers and their summary, the reward set, the select split. part 1: the two confirm splits.
    part 2: the test set (only needed for test scoring, so the RL loop does not wait for it)."""
    import json, os, threading, time
    runs.reload()
    if not reward_items_json:
        reward_items_json = json.dumps([json.loads(l) for l in open("/heldout/rl/reward_math.jsonl")][:reward_items])
    mdir, pdir = base_dirs(tag)
    proc = start_vllm(mdir, "original", 8001, 0.90, GEN_CAP + 3000)
    sp = json.load(open(f"{pdir}/splits.json"))
    practice = {json.loads(l)["id"]: json.loads(l) for l in open(f"{pdir}/problems.jsonl")}
    items = lambda ids: [{"qid": i, "source": practice[i]["source"], "answer": practice[i]["answer"],
                          "prompt": practice[i]["problem"] + "\n\nPut your final answer in \\boxed{}."} for i in ids]
    gen = lambda its, n: _Core._gen(None, "original", 8001, its, n, GEN_CAP)  # the forward tool's generation: same prompts, seeds, sampling
    write = lambda path, rows: (os.makedirs(os.path.dirname(path), exist_ok=True), open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows)))
    jobs, out, t = {}, {}, time.time()
    def baseline():
        rows = gen(items(sp["baseline"]), 2)
        write("/tmp/bw/gen_1.jsonl", rows)
        exec(BASELINE_CODE.replace("/work/gen_1.jsonl", "/tmp/bw/gen_1.jsonl").replace("/practice/baseline", f"{pdir}/baseline"), {})
        out["baseline"] = len(rows)
    def reward():
        its = [{"qid": "reward/" + it["qid"], "source": "reward", "answer": it["answer"], "prompt": it["problem"] + "\n\nPut your final answer in \\boxed{}."}
               for it in json.loads(reward_items_json)]
        rows = [{"qid": r["qid"], "sample": r["sample"], "correct": bool(r["correct"]), "tokens": r["tokens"], "finish": r["finish"]} for r in gen(its, reward_n)]
        json.dump(rows, open(f"{BASES_R}/{tag}/reward_base_{len(its)}x{reward_n}.json", "w"))
        out["reward"] = round(sum(r["correct"] for r in rows) / len(rows), 4)
    def test():
        its = [json.loads(l) for l in open("/heldout/math500.jsonl")]
        rows = heldout_rows(8001, "original", its, 2)
        write(heldout_cache(tag), rows)
        out["test"] = round(sum(bool(r["correct"]) for r in rows) / len(rows), 4)
    def split(name):
        def f():
            rows = gen(items(sp[name]), 2)
            write(f"{pdir}/original/{name}.jsonl", rows)
            out[name] = round(sum(r["correct"] for r in rows) / len(rows), 4)
        return f
    if part == 0:
        jobs = {"baseline": baseline, "reward": reward, "select": split("select")}
    elif part == 1:
        jobs = {n: split(n) for n in ("confirm_1", "confirm_2")}
    else:
        jobs = {"test": test}
    ths = [threading.Thread(target=f, name=n) for n, f in jobs.items()]
    [th.start() for th in ths]; [th.join() for th in ths]
    stop_proc(proc)
    missing = set(jobs) - set(out)
    assert not missing, f"failed: {missing}"
    runs.commit()
    out["seconds"] = round(time.time() - t)
    return out


@app.function(image=rl_image, volumes={"/runs": runs, "/heldout": heldout_vol}, secrets=[modal.Secret.from_name("tinker"), modal.Secret.from_name("exa")],
              timeout=int(11.5 * 3600), nonpreemptible=True, cpu=4, memory=16384)
async def rl_run(exp: str, steps: int = 2, group_size: int = 4, forward_budget: int = 32, rollout_minutes: int = 120, hours: float = 10.0,
                 reward_items: int = 128, reward_n: int = 2, effort: float = 0.9, lr: float = 2e-5, max_turns: int = 400, start_from: str = "", web: bool = False, library: str = "",
                 rollouts_per_gpu: int = 1, rebase_every: int = 0, update_check: bool = True, start_base: str = "", force_rewards: str = "",
                 mode: str = "rl", eval_dataset: str = "math500", eval_limit: int = 0, eval_parallel: int = 4,
                 advantage: str = "grpo", adv_scale: float = 0.10):
    """start_base: begin with this starting model (a tag made by rebasing) instead of the downloaded one.
    mode "eval" (rl.loop::evaluate): one group of rollouts from `start_from` (a state or sampler path; "" is the untrained
    editor) with no training update, written to /runs/eval/<exp>/; every submitted edit is then scored on `eval_dataset`.
    rebase_every k: after every k steps, the best-rewarded submitted edit of those k steps is applied to the starting model.
    update_check: after each update, score the trained batch under the new weights (KL, change per rollout).
    force_rewards: tests only, comma-separated rewards that replace the measured ones so a short run trains."""
    import asyncio, re, json, os, time
    import tinker, torch
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer, get_text_content, ToolCall
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Transition, Trajectory
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    if rollouts_per_gpu not in (1, 2):
        raise ValueError("rollouts_per_gpu must be 1 or 2")
    evaluating = mode == "eval"
    if evaluating:
        steps, rebase_every = 1, 0
    root = "eval" if evaluating else "rl"
    d = f"/runs/{root}/{exp}"
    if os.path.exists(f"{d}/steps.jsonl"):
        raise RuntimeError(f"{d} already has a run; a restart would train on top of it. Use a new --exp.")
    os.makedirs(d, exist_ok=True)
    T0 = time.time()
    run_deadline = T0 + hours * 3600
    base_name = MODEL.split(":")[0]
    tokenizer = get_tokenizer(base_name)
    renderer = get_renderer(model_info.get_recommended_renderer_name(base_name), tokenizer)
    stop = renderer.get_stop_sequences()
    svc = tinker.ServiceClient()
    if evaluating and "/sampler_weights/" in start_from:
        tc = None  # sampling only: the sampler checkpoint is used as it is
    elif start_from:
        tc = await asyncio.to_thread(lambda: svc.create_training_client_from_state(start_from).result())
    else:
        tc = await asyncio.to_thread(lambda: svc.create_lora_training_client(base_model=MODEL, rank=32).result())
    items = [json.loads(l) for l in open("/heldout/rl/reward_math.jsonl")][:reward_items]
    items_json = json.dumps(items)
    pdir_of = lambda tag: f"{BASES_R}/{tag}/practice" if tag else "/runs/data/math_practice"
    bm = f"{pdir_of('')}/baseline/meta.json"
    known = json.load(open(bm)) if os.path.exists(bm) else None
    prompt_text = editor_prompt(forward_budget, web, known)
    tools = TOOLS + ([WEB_TOOL, WEB_READ_TOOL] if web else [])
    lib_dir = f"{d}/_library" if evaluating else f"/runs/rl/_library/{library or exp}"  # the shared library: papers and notes that persist across rollouts and steps
    os.makedirs(lib_dir, exist_ok=True)
    if not os.path.exists(f"{lib_dir}/INDEX.md"):
        open(f"{lib_dir}/INDEX.md", "w").write("# Library index\n\nOne line per file: file name, what it is, whether it helped.\n")

    def lib_files():
        return {n: open(os.path.join(lib_dir, n), errors="replace").read() for n in sorted(os.listdir(lib_dir)) if n.endswith((".md", ".txt"))}

    def lib_merge(files):
        """Fold what a rollout left in /library back into the shared copy: new files are added, the index gains new lines."""
        for n, text in (files or {}).items():
            n = os.path.basename(n); path = os.path.join(lib_dir, n)
            if n == "INDEX.md":
                have = open(path).read().splitlines()
                add = [l for l in text.splitlines() if l.strip() and l not in have]
                if add:
                    open(path, "a").write("\n".join(add) + "\n")
            elif not os.path.exists(path) or len(text) > os.path.getsize(path):
                open(path, "w").write(text)
    json.dump({"exp": exp, "model": MODEL, "steps": steps, "group_size": group_size, "forward_budget": forward_budget, "rollout_minutes": rollout_minutes,
               "reward_items": len(items), "reward_n": reward_n, "effort": effort, "lr": lr, "advantage": advantage, "adv_scale": adv_scale, "web": web, "library": lib_dir, "rollouts_per_gpu": rollouts_per_gpu,
               "rebase_every": rebase_every, "start_base": start_base, "update_check": update_check, "start_from": start_from,
               "baseline_given": bool(known), "mode": mode, "eval_dataset": eval_dataset if evaluating else None, "eval_limit": eval_limit,
               "prompt": prompt_text}, open(f"{d}/config.json", "w"), indent=1)

    def slog(rec):
        with open(f"{d}/steps.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        runs.commit()
        print("STEP", json.dumps(rec)[:1500])

    async def load_base(tag):
        """What the rollouts of one starting model share: its prompt (baseline summary) and its answers on the reward set
        (the reward is edited minus these)."""
        runs.commit()  # keep this run's own writes, then see the starting model's files written by make_base / gen_base
        runs.reload()
        t = time.time()
        pdir = pdir_of(tag)
        kn = json.load(open(f"{pdir}/baseline/meta.json")) if os.path.exists(f"{pdir}/baseline/meta.json") else None
        if tag:
            rows = json.load(open(f"{BASES_R}/{tag}/reward_base_{len(items)}x{reward_n}.json"))
        else:
            base_cache = f"/runs/rl/_reward_base_{len(items)}x{reward_n}.json"  # the original weights' answers on the reward set never change
            if os.path.exists(base_cache):
                rows = json.load(open(base_cache))
            else:
                rows = (await Target.with_options(scaledown_window=120)(rid=f"{exp}-base").reward.remote.aio(items_json, reward_n, GEN_CAP, True))["original"]
                json.dump(rows, open(base_cache, "w"))
        ok = {(r["qid"], r["sample"]): r["correct"] for r in rows}
        B = {"tag": tag, "prompt": editor_prompt(forward_budget, web, kn, rebased=bool(tag)), "ok": ok, "acc": sum(ok.values()) / len(ok)}
        slog({"event": "reward_base", "base": tag, "answers": len(ok), "accuracy": round(B["acc"], 4),
              "hit_cap": round(sum(r["finish"] == "length" for r in rows) / len(rows), 4), "seconds": round(time.time() - t)})
        return B

    B = await load_base(start_base)
    cache_call = None
    if evaluating and not eval_limit and not os.path.exists(heldout_cache(start_base, eval_dataset)):
        # the starting model's answers on the evaluation set, generated once while the rollouts run
        cache_call = await heldout_eval.spawn.aio(f"eval/{exp}/_original", 2, 0, True, start_base, eval_dataset)

    async def rollout(sampler, rid, gid, B):
        base_ok, prompt_text = B["ok"], B["prompt"]
        t0 = time.time()
        deadline = min(t0 + rollout_minutes * 60, run_deadline)
        if rollouts_per_gpu > 1:  # one container per gid; every call names the rollout
            obj = PackedTarget.with_options(scaledown_window=300)(gid=gid, base=B["tag"])
            call = lambda m, *a: getattr(obj, m).remote.aio(rid, *a)
        else:
            obj = Target.with_options(scaledown_window=300)(rid=rid, base=B["tag"])
            call = lambda m, *a: getattr(obj, m).remote.aio(*a)
        n_repaired = 0
        messages = renderer.create_conversation_prefix_with_tools(tools, system_prompt="")  # Inkling's own tool-call format
        messages.append({"role": "user", "content": prompt_text})
        transitions, used_f, used_py, used_web, nags, turn, cutoffs = [], 0, 0, 0, 0, 0, 0
        tok = dict(prefill=0, cached=0, sampled=0)
        asked_final, ended, final_text = False, "turn limit", None
        ctx = 0
        if web:
            await call("put_files", LIBRARY, json.dumps(lib_files()))
        boot = await call("boot_id")
        restarts = 0
        log = open(f"{d}/{rid}.jsonl", "a")
        w = lambda rec: (log.write(json.dumps(rec) + "\n"), log.flush())
        def status():
            parts = [f"turn {turn}", f"forward passes left: {max(0, forward_budget - used_f)} of {forward_budget}", f"python runs left: {max(0, PY_BUDGET - used_py)}"]
            if web:
                parts.append(f"web calls left: {max(0, WEB_BUDGET - used_web)}")
            parts += [f"minutes left: about {max(0, int((deadline - time.time()) / 60))}", f"context left: about {max(0, 235000 - ctx) // 1000}k tokens"]
            return "\n\n[" + "; ".join(parts) + "]"
        must_finish = lambda: used_f >= forward_budget or time.time() > deadline
        try:
            while turn < max_turns:
                turn += 1
                ob = renderer.build_generation_prompt(messages, effort=effort)
                if ob.length > 235000:
                    ended = "context limit"; break
                params = tinker.SamplingParams(max_tokens=min(8000, 258000 - ob.length), stop=stop)
                resp = await asyncio.to_thread(lambda: sampler.sample(prompt=ob, num_samples=1, sampling_params=params).result())
                s = resp.sequences[0]
                msg, _ = renderer.parse_response(s.tokens)
                transitions.append(Transition(ob=ob, ac=TokensWithLogprobs(tokens=list(s.tokens), maybe_logprobs=list(s.logprobs)), reward=0.0, episode_done=False))
                hit = int(getattr(resp, "prompt_cache_hit_tokens", 0) or 0)
                tok["prefill"] += ob.length - hit; tok["cached"] += hit; tok["sampled"] += len(s.tokens)
                ctx = ob.length + len(s.tokens)
                text = get_text_content(msg) or ""
                content = msg.get("content")
                thinking = "\n\n".join(str(p.get("thinking") or p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "thinking") if isinstance(content, list) else ""
                tcs, bad = msg.get("tool_calls") or [], msg.get("unparsed_tool_calls") or []
                cut = str(s.stop_reason) != "stop"
                repaired = False
                if not tcs and not cut:
                    fix = repair_reply(tokenizer.decode(list(s.tokens)))
                    if fix:  # rebuild a well-formed message, so the history the editor sees is clean
                        thinking, nm, fargs = fix
                        tcs = [ToolCall(function=ToolCall.FunctionBody(name=nm, arguments=json.dumps(fargs)), id=f"functions.{nm}:0")]
                        msg = {"role": "assistant", "content": [{"type": "thinking", "thinking": thinking}] if thinking else "", "tool_calls": tcs}
                        text, bad, repaired = "", [], True
                        n_repaired += 1
                w({"turn": turn, "t": round(time.time() - t0), "prompt_tokens": ob.length, "completion_tokens": len(s.tokens), "finish": str(s.stop_reason),
                   "reasoning": "\n\n".join(x for x in (thinking, text if tcs else "") if x), "final": "" if tcs else text, "call": tcs[0].function.name if tcs else None, "repaired": repaired})
                messages.append(msg)
                if cut:  # the reply ran to the length limit without ending its turn
                    cutoffs += 1
                    if cutoffs >= 3:
                        ended = "three replies cut off at the length limit"; break
                    messages.append({"role": "user", "content": "[Your reply was cut off at the length limit. Reply again, briefly, with one tool call.]" + status()})
                    continue
                cutoffs = 0
                if not tcs:
                    if asked_final or must_finish() or "FINAL SUBMISSION" in text:
                        final_text = text
                        ended = "editor finished" if not must_finish() else ("budget used" if used_f >= forward_budget else "time limit")
                        break
                    nags += 1
                    if nags >= 5:
                        final_text, ended = text, "no tool call in five replies"; break
                    note = ("[The tool call could not be read: " + "; ".join(str(getattr(x, "error", ""))[:200] for x in bad) + ". Send it again.]") if bad else \
                        "[No tool call was found. The session is still running. Call python or forward, or finish with a reply that starts with FINAL SUBMISSION.]"
                    messages.append({"role": "user", "content": note + status()})
                    continue
                nags = 0
                tc = tcs[0]
                kind = tc.function.name.split(".")[-1]
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                if kind == "python":
                    body = str(args.get("code", ""))
                    if used_py >= PY_BUDGET:
                        res = "[python budget exhausted]"
                    else:
                        used_py += 1
                        res = await call("py", body)
                elif kind == "forward":
                    body = forward_request(args)
                    if must_finish():
                        res = "[no forward passes left]"
                    else:
                        ok, res = await call("forward", body)
                        if ok:
                            used_f += 1
                        else:
                            res = "[request failed; it does not count against the budget]\n" + res
                elif kind == "web_read" and web:
                    body = str(args.get("url", ""))
                    if used_web >= WEB_BUDGET:
                        res = "[no web calls left]"
                    else:
                        used_web += 1
                        try:
                            title, text = await asyncio.to_thread(exa_read, body, os.environ["EXA_API_KEY"])
                            if title is None:
                                res = text
                            else:
                                slug = re.sub(r"[^a-z0-9]+", "-", (title or body).lower()).strip("-")[:60] or "page"
                                await call("put_files", LIBRARY, json.dumps({slug + ".md": f"# {title}\n\nSource: {body}\n\n{text}"}))
                                res = f"[saved to /library/{slug}.md, {len(text)} characters]\n\n{title}\n\n{text[:4000]}"
                        except Exception as e:
                            res = "[the page could not be read: " + repr(e)[:200] + "]"
                elif kind == "web_search" and web:
                    body = str(args.get("query", ""))
                    if used_web >= WEB_BUDGET:
                        res = "[no web searches left]"
                    else:
                        used_web += 1
                        try:
                            res = await asyncio.to_thread(exa_search, body, os.environ["EXA_API_KEY"])
                        except Exception as e:
                            res = "[the search failed: " + repr(e)[:200] + "]"
                else:
                    body, res = tc.function.arguments[:2000], f"[unknown tool {kind!r}; the tools are python and forward]"
                now = await call("boot_id")
                if now != boot:  # the target machine was replaced (for example after running out of memory): /work and the kernel are gone
                    boot, restarts = now, restarts + 1
                    res = (res + "\n\n[The target machine restarted after this call, probably because it ran out of memory. /work was reset and all python variables are lost, "
                           "including any saved edit: save your edit again. Keep large arrays out of memory: load activations for a few layers at a time.]")
                    if web:
                        await call("put_files", LIBRARY, json.dumps(lib_files()))
                w({"turn": turn, "tool": kind, "input": body[:20000], "result": res[:20000], "used_f": used_f, "used_py": used_py, "target_restarts": restarts})
                if turn % 8 == 0:
                    await runs.commit.aio()  # so a running step can be watched
                if len(res) > 6000:
                    res = res[:6000] + "\n[output cut at 6,000 characters]"
                extra = ""
                if must_finish() and not asked_final:
                    asked_final = True
                    extra = "\n\n[The budget or the time limit is reached. You may still run a few python calls to set /work/edited to your best edit. Then reply without a tool call, starting with FINAL SUBMISSION: what you changed, the evidence, and why you expect it to help.]"
                messages.append({"role": "tool", "tool_call_id": tc.id or "call_1", "name": tc.function.name, "content": res + status() + extra})
                for other in tcs[1:]:
                    messages.append({"role": "tool", "tool_call_id": other.id or "call_x", "name": other.function.name, "content": "[not run: only the first tool call of a reply is run]"})
                if asked_final and time.time() > deadline + 15 * 60:
                    ended = "time limit (hard)"; break
            # reward: the submitted edit against the cached original answers, paired by problem and sample
            rw = await call("reward", items_json, reward_n, GEN_CAP, False)
            if rw["edited"] is None:
                reward, acc_e, cap_e = 0.0, None, None
            else:
                ok_e = {(r["qid"], r["sample"]): r["correct"] for r in rw["edited"]}
                acc_e = sum(ok_e.values()) / len(ok_e)
                reward = sum(float(ok_e[k]) - float(base_ok[k]) for k in ok_e) / len(ok_e)
                cap_e = sum(r["finish"] == "length" for r in rw["edited"]) / len(rw["edited"])
            lib_out = {}
            if web:
                try:
                    lib_out = await call("get_files", LIBRARY)
                except Exception as e:
                    w({"library_error": repr(e)[:300]})
            try:
                await call("snapshot", f"{root}/{exp}/work/{rid}")
            except Exception as e:
                w({"snapshot_error": repr(e)[:300]})
            usd = (tok["prefill"] * PRICE["prefill"] + tok["cached"] * PRICE["cached"] + tok["sampled"] * PRICE["sample"]) / 1e6
            info = {"rid": rid, "ok": True, "reward": round(reward, 4), "edited_accuracy": acc_e, "edited_hit_cap": cap_e, "edited_tensors": rw["edited_tensors"], "ended": ended,
                    "turns": turn, "forward_used": used_f, "python_used": used_py, "web_used": used_web, "target_restarts": restarts, "minutes": round((time.time() - t0) / 60, 1), "tokens": tok, "sampling_usd": round(usd, 3),
                    "base": B["tag"], "repaired_turns": n_repaired}
            info["library_files"] = sorted(lib_out)
            w({"summary": info, "final_text": final_text})
            return Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty()), info, lib_out
        except Exception as e:
            w({"error": repr(e)[:1000]})
            return None, {"rid": rid, "ok": False, "error": repr(e)[:400], "turns": turn, "minutes": round((time.time() - t0) / 60, 1)}, {}
        finally:
            log.close()

    adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sampler0, recent, n_base = None, [], 0  # recent: (step, rollout info) since the last rebase

    def get_sampler(path):
        s = svc.create_sampling_client(model_path=path)
        return s if hasattr(s, "compute_logprobs") else s.result()

    async def check_update(new_sampler, batch, step):
        """Score the batch just trained on under the new weights. Old = the sampler's log-probabilities, which are the
        pre-update policy's (on-policy, one update per batch). Also against the step-0 policy (KL from the start)."""
        def score(sc):
            futs = [sc.compute_logprobs(tinker.ModelInput.from_ints(b["tokens"] + [b["target"][-1]])) for b in batch]
            return [torch.tensor([x if x is not None else float("nan") for x in f.result()[1:]]) for f in futs]
        new = await asyncio.to_thread(score, new_sampler)
        ref = await asyncio.to_thread(score, sampler0) if (sampler0 is not None and step > 0) else None
        lo, ln, adv_all, per = [], [], [], {}
        for i, b in enumerate(batch):
            m = b["mask"] > 0
            o, n = b["logprobs"][m], new[i][m]
            lo.append(o); ln.append(n); adv_all.append(b["advantages"][m])
            per.setdefault(b["rid"], [0.0, b["a_final"]])[0] += float((n - o).sum())
        lo, ln, adv_all = torch.cat(lo), torch.cat(ln), torch.cat(adv_all)
        dl = ln - lo  # log pi_new - log pi_old on tokens sampled from pi_old
        out = {"tokens": int(dl.numel()), "kl_old_new_k1": round(float((-dl).mean()), 6), "kl_old_new_k3": round(float((dl.exp() - 1 - dl).mean()), 6),
               "mean_logprob_change": round(float(dl.mean()), 6), "objective_change": round(float((adv_all * dl).sum()), 3),
               "per_rollout": {r: {"total_change": round(v[0], 3), "advantage": round(v[1], 3)} for r, v in per.items()}}
        xs, ys = [v[1] for v in per.values()], [v[0] for v in per.values()]
        if len(xs) > 2:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
            out["corr_change_advantage"] = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3) if sx > 0 and sy > 0 else None
        if ref is not None:
            l0 = torch.cat([ref[i][b["mask"] > 0] for i, b in enumerate(batch)])
            d0 = l0 - lo  # log pi_0 - log pi_old, tokens from pi_old: KL(pi_old || pi_0)
            out.update(kl_old_step0_k1=round(float((-d0).mean()), 6), kl_old_step0_k3=round(float((d0.exp() - 1 - d0).mean()), 6))
        return out

    for step in range(steps):
        if time.time() > run_deadline - 20 * 60:
            slog({"event": "stopped", "reason": "run time limit", "step": step}); break
        t = time.time()
        sampler = await asyncio.to_thread(lambda: get_sampler(start_from) if tc is None else tc.save_weights_and_get_sampling_client().result())
        if sampler0 is None:
            sampler0 = sampler
        out = await asyncio.gather(*[rollout(sampler, f"{exp}-s{step}-r{i}", f"{exp}-s{step}-g{i // rollouts_per_gpu}", B) for i in range(group_size)])
        for _, _, lib_out in out:
            lib_merge(lib_out)
        good = [(tr, info) for tr, info, _ in out if info["ok"]]
        if force_rewards:
            for (_, info), r in zip(good, [float(x) for x in force_rewards.split(",")]):
                info["reward"], info["forced_reward"] = r, True
        rewards = [info["reward"] for _, info in good]
        recent += [(step, info) for _, info in good]
        rec = {"event": "step", "step": step, "base": B["tag"], "rollouts": [info for _, info, _ in out], "rewards": rewards, "library": sorted(os.listdir(lib_dir)),
               "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None, "rollout_minutes": round((time.time() - t) / 60, 1)}
        if evaluating:  # no training update: score the submitted edits and stop
            rec.update(trained=False, reason="evaluation")
            slog(rec)
            if cache_call is not None:
                try:
                    await cache_call.get.aio()
                except Exception as e:
                    slog({"event": "original_cache", "failed": repr(e)[:500]})
            infos = {f"{root}/{exp}/work/{info['rid']}": info for _, info, _ in out}
            summary = await score_edits(f"eval/{exp}", [p for p, i in infos.items() if i["ok"]], eval_dataset, start_base, eval_limit, eval_parallel,
                                        {p: i["reward"] for p, i in infos.items() if i["ok"]},
                                        extra={"checkpoint": start_from or "untrained", "failed_rollouts": [i["rid"] for i in infos.values() if not i["ok"]]})
            slog({"event": "done", "summary": summary, "minutes_total": round((time.time() - T0) / 60, 1)})
            return summary
        if len(set(rewards)) >= 2:
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if advantage == "mean":  # Dr. GRPO (arXiv 2503.20783): no division by the group's spread, which inflates near-identical groups;
                # a fixed scale (0.10 = the typical group spread on Qwen as downloaded) keeps the update size of groups with real spread
                adv = [(r - mean) / adv_scale for r in rewards]
            else:
                adv = [(r - mean) / (std + 1e-6) for r in rewards]  # GRPO: centred and scaled within the group
            t1, n_tok, futs, batch = time.time(), 0, [], []
            for (tr, info), a in zip(good, adv):
                for dm in trajectory_to_data(tr, float(a), include_provenance=False):
                    mask = dm.loss_fn_inputs.pop("mask").to_torch()  # importance_sampling takes target_tokens, logprobs, advantages
                    batch.append({"rid": info["rid"], "a_final": a, "tokens": dm.model_input.to_ints(), "target": dm.loss_fn_inputs["target_tokens"].to_torch().tolist(),
                                  "mask": mask, "logprobs": dm.loss_fn_inputs["logprobs"].to_torch(), "advantages": dm.loss_fn_inputs["advantages"].to_torch()})
                    n_tok += dm.model_input.length
                    futs.append(tc.forward_backward([dm], loss_fn="importance_sampling"))
            opt = tc.optim_step(adam)
            results = await asyncio.to_thread(lambda: [f.result() for f in futs])
            optim = await asyncio.to_thread(lambda: opt.result())
            metrics = [r.metrics for r in results]
            # Sampler against trainer on the same tokens (as in tinker_cookbook.rl.metrics.compute_kl_sample_train)
            masks = [b["mask"] > 0 for b in batch]; samp = [b["logprobs"] for b in batch]
            diffs = torch.cat([sp[mk] - r.loss_fn_outputs[0]["logprobs"].to_torch()[mk] for sp, mk, r in zip(samp, masks, results)])
            sp_all = torch.cat([sp[mk] for sp, mk in zip(samp, masks)])
            rec.update(kl_sample_train_v1=round(diffs.mean().item(), 5), kl_sample_train_v2=round(0.5 * (diffs ** 2).mean().item(), 5),
                       logprob_diff_max=round(diffs.abs().max().item(), 3), frac_tokens_diff_over_0p2=round((diffs.abs() > 0.2).float().mean().item(), 5),
                       entropy=round(-sp_all.mean().item(), 4), action_tokens=int(sp_all.numel()),
                       optim_metrics={k: round(float(v), 5) for k, v in (getattr(optim, "metrics", None) or {}).items()})
            rec.update(trained=True, lr=lr, advantages=[round(a, 3) for a in adv], training_sequences=len(futs), training_tokens=n_tok,
                       training_usd=round(n_tok * PRICE["train"] / 1e6, 2), loss_sum=round(sum(m.get("loss:sum", 0.0) for m in metrics), 3), training_seconds=round(time.time() - t1))
            os.makedirs(f"{d}/datums", exist_ok=True)  # the exact training data of this step
            torch.save([{k: (torch.tensor(v, dtype=torch.int32) if isinstance(v, list) else v) for k, v in b.items()} for b in batch], f"{d}/datums/step{step}.pt")
        else:
            batch = []
            rec.update(trained=False, reason="all rewards equal or fewer than two rollouts finished; no gradient")
        ck = await asyncio.to_thread(lambda: tc.save_weights_for_sampler(f"{exp}-step{step + 1}", ttl_seconds=7 * 24 * 3600).result())
        st = await asyncio.to_thread(lambda: tc.save_state(f"{exp}-step{step + 1}", ttl_seconds=2 * 24 * 3600).result())
        rec.update(sampler_checkpoint=ck.path, state_checkpoint=st.path, minutes_total=round((time.time() - T0) / 60, 1))
        if update_check and batch:
            try:
                t2 = time.time()
                rec["update_check"] = await check_update(await asyncio.to_thread(get_sampler, ck.path), batch, step)
                rec["update_check"]["seconds"] = round(time.time() - t2)
            except Exception as e:
                rec["update_check"] = {"error": repr(e)[:500]}
        slog(rec)
        # A new starting model: the best-rewarded edit of the last `rebase_every` steps, applied to the current one.
        if rebase_every and (step + 1) % rebase_every == 0 and step + 1 < steps:
            cands = [(info["reward"], s, info) for s, info in recent if info.get("edited_tensors") and info["reward"] > 0]
            recent = []
            if not cands:
                slog({"event": "rebase", "skipped": "no submitted edit with a positive reward", "step": step})
                continue
            r_best, s_best, info = max(cands, key=lambda x: x[0])
            n_base += 1
            tag, t3 = f"{exp}-k{n_base}", time.time()
            try:
                tensors = await make_base.remote.aio(tag, B["tag"], f"rl/{exp}/work/{info['rid']}")
                parts = await asyncio.gather(gen_base.remote.aio(tag, 0, items_json, reward_n), gen_base.remote.aio(tag, 1, items_json, reward_n))
                await gen_base.spawn.aio(tag, 2, items_json, reward_n)  # the test set's cache, for test scoring; not waited for
                prev = B
                B = await load_base(tag)
                slog({"event": "rebase", "step": step, "base": tag, "previous": prev["tag"], "from_rollout": info["rid"], "its_reward": r_best,
                      "tensors": len(tensors), "new_base_reward_set_accuracy": round(B["acc"], 4), "previous_base_accuracy": round(prev["acc"], 4),
                      "generated": parts, "minutes": round((time.time() - t3) / 60, 1)})
            except Exception as e:
                n_base -= 1
                slog({"event": "rebase", "step": step, "failed": repr(e)[:800], "kept_base": B["tag"]})
    slog({"event": "done", "minutes_total": round((time.time() - T0) / 60, 1)})
    return True


@app.local_entrypoint()
def rl(exp: str, steps: int = 2, group_size: int = 4, forward_budget: int = 32, rollout_minutes: int = 120, hours: float = 10.0,
       reward_items: int = 128, effort: float = 0.9, lr: float = 2e-5, start_from: str = "", web: bool = False, library: str = "", rollouts_per_gpu: int = 1,
       rebase_every: int = 0, update_check: bool = True, start_base: str = "", force_rewards: str = "", advantage: str = "grpo", adv_scale: float = 0.10):
    """advantage: "grpo" = (reward - group mean) / group spread; "mean" = (reward - group mean) / adv_scale (Dr. GRPO)."""
    call = rl_run.spawn(exp, steps, group_size, forward_budget, rollout_minutes, hours, reward_items, 2, effort, lr, 400, start_from, web, library, rollouts_per_gpu,
                        rebase_every, update_check, start_base, force_rewards, advantage=advantage, adv_scale=adv_scale)
    print("SPAWNED", call.object_id)


@app.function(image=rl_image, volumes={"/runs": runs}, timeout=int(12 * 3600))
async def test_step_edits(exp: str, steps: str = "0", parallel: int = 2):
    """Score each rollout's submitted edit on the test set (MATH-500), which the RL loop never uses.
    Only the edited side is generated; the original side comes from one cached set of answers."""
    import asyncio, json, os, shutil
    runs.reload()
    cached = "/runs/heldout_original_cached.jsonl"
    if not os.path.exists(cached):
        shutil.copy("/runs/inkling2s-qwen08-math-methods-1/heldout/original.jsonl", cached)
        await runs.commit.aio()
    sem = asyncio.Semaphore(parallel)
    # each edit is compared with the starting model it was made on (rebasing changes it between steps)
    base_of = {}
    if os.path.exists(f"/runs/rl/{exp}/steps.jsonl"):
        for l in open(f"/runs/rl/{exp}/steps.jsonl"):
            r = json.loads(l)
            if r.get("event") == "step":
                base_of[r["step"]] = r.get("base", "")

    async def one(rid):
        path = f"rl/{exp}/work/{rid}"
        if os.path.exists(f"/runs/{path}/heldout.json"):
            return rid, json.load(open(f"/runs/{path}/heldout.json"))
        async with sem:
            return rid, await heldout_eval.remote.aio(path, 2, 0, True, base_of.get(int(rid.rsplit("-s", 1)[1].split("-")[0]), ""))

    rids = sorted(d for d in os.listdir(f"/runs/rl/{exp}/work") if any(d.startswith(f"{exp}-s{s}-") for s in steps.split(",")))
    res = await asyncio.gather(*[one(r) for r in rids])
    out = {rid: {"original": h["original"]["accuracy_pct"], "edited": (h.get("edited") or {}).get("accuracy_pct"),
                 "paired": h.get("paired_edited_minus_original"), "base": h.get("base", "")} for rid, h in res}
    runs.reload()
    p = f"/runs/rl/{exp}/test.json"
    allres = json.load(open(p)) if os.path.exists(p) else {}
    allres.update(out)
    json.dump(allres, open(p, "w"), indent=1)
    await runs.commit.aio()
    print(json.dumps(out, indent=1))


@app.local_entrypoint()
def rebase(tag: str, edit_run: str, prev: str = ""):
    """By hand: a starting model `tag` = starting model `prev` + the edit in /runs/<edit_run>/work/edited, with its caches."""
    import time
    t = time.time()
    print("tensors", len(make_base.remote(tag, prev, edit_run)), round(time.time() - t), "s")
    calls = [gen_base.spawn(tag, k) for k in (0, 1, 2)]
    print([c.get() for c in calls], round(time.time() - t), "s")


@app.local_entrypoint()
def test(exp: str, steps: str = "0", parallel: int = 2):
    print("SPAWNED", test_step_edits.spawn(exp, steps, parallel).object_id)


# ----------------------------------------------------------------------------- evaluating an editor checkpoint
async def score_edits(out, paths, dataset, base_tag, limit=0, parallel=4, rewards=None, extra=None):
    """Score the edit in /runs/<path>/work/edited of each path on `dataset` against starting model `base_tag`, paired
    by problem (heldout_eval, with the starting model's answers cached per (base, dataset)). A path without an edit scores 0.
    Writes /runs/<out>/scores.json (one row per path) and /runs/<out>/summary.json (mean and standard error of the
    paired gain across paths), and returns the summary."""
    import asyncio, json, os, time
    t0 = time.time()
    await runs.reload.aio()
    cached = heldout_cache(base_tag, dataset)
    if not limit and not os.path.exists(cached):  # once, before the edits, so they share one set of original answers
        await heldout_eval.remote.aio(f"{out}/_original", 2, 0, True, base_tag, dataset)
        await runs.reload.aio()
    sem = asyncio.Semaphore(parallel)
    has_edit = lambda p: os.path.isdir(f"/runs/{p}/work/edited") and any(f.endswith(".safetensors") for f in os.listdir(f"/runs/{p}/work/edited"))

    async def one(p):
        row = {"run": p, "reward_pp": round(100 * rewards[p], 2) if rewards and rewards.get(p) is not None else None}
        if not has_edit(p):
            return {**row, "edit": False, "gain_pp": 0.0}
        async with sem:
            try:
                h = await heldout_eval.remote.aio(p, 2, limit, True, base_tag, dataset)
            except Exception as e:
                return {**row, "edit": True, "error": repr(e)[:300]}
        pr = h["paired_edited_minus_original"]
        return {**row, "edit": True, "original_pct": h["original"]["accuracy_pct"], "edited_pct": h["edited"]["accuracy_pct"],
                "gain_pp": pr["diff_pp"], "se_pp": pr["se_pp"], "wrong_to_right": pr["wrong_to_right"], "right_to_wrong": pr["right_to_wrong"],
                "edited_cut_pct": h["edited"]["cut_at_max_tokens_pct"], "edited_tensors": h["edited_tensors"]}

    rows = await asyncio.gather(*[one(p) for p in paths])
    g = [r["gain_pp"] for r in rows if "gain_pp" in r]
    rw = [r["reward_pp"] for r in rows if r["reward_pp"] is not None]
    k = len(g)
    mean = sum(g) / k if k else None
    se = (sum((x - mean) ** 2 for x in g) / (k - 1)) ** 0.5 / k ** 0.5 if k > 1 else None
    summary = {"dataset": dataset, "base": base_tag or "original", "problems": limit or "all", "answers_per_problem": 2,
               "scored": k, "with_edit": sum(r.get("edit", False) for r in rows), "errors": [r["run"] for r in rows if "error" in r],
               "gain_mean_pp": round(mean, 2) if mean is not None else None, "gain_se_pp": round(se, 2) if se is not None else None,
               "gains_pp": g, "reward_mean_pp": round(sum(rw) / len(rw), 2) if rw else None, "minutes": round((time.time() - t0) / 60, 1), **(extra or {})}
    os.makedirs(f"/runs/{out}", exist_ok=True)
    json.dump(rows, open(f"/runs/{out}/scores.json", "w"), indent=1)
    json.dump(summary, open(f"/runs/{out}/summary.json", "w"), indent=1)
    await runs.commit.aio()
    print("SUMMARY", json.dumps(summary))
    return summary


@app.function(image=rl_image, volumes={"/runs": runs, "/heldout": heldout_vol}, timeout=int(8 * 3600))
async def eval_score(name: str, edits: str, dataset: str = "math500", base_tag: str = "", limit: int = 0, parallel: int = 4):
    """Score existing edits (comma-separated run paths under /runs, e.g. rl/nolib1/work/nolib1-s1-r0) into /runs/eval/<name>/;
    their reward-set rewards are read from the RL run's steps.jsonl when there is one."""
    import json, os
    await runs.reload.aio()
    paths = [p.strip().strip("/") for p in edits.split(",") if p.strip()]
    rewards = {}
    for p in paths:
        sj = f"/runs/{p.split('/work/')[0]}/steps.jsonl"
        if "/work/" in p and os.path.exists(sj):
            for l in open(sj):
                r = json.loads(l)
                for i in r.get("rollouts") or []:
                    if i.get("ok") and i["rid"] == p.rsplit("/", 1)[1]:
                        rewards[p] = i["reward"]
    return await score_edits(f"eval/{name}", paths, dataset, base_tag, limit, parallel, rewards, extra={"edits": paths})


@app.local_entrypoint()
def evaluate(name: str, checkpoint: str = "", base: str = "original", n: int = 8, dataset: str = "math500", limit: int = 0,
             forward_budget: int = 32, rollout_minutes: int = 120, rollouts_per_gpu: int = 2, reward_items: int = 128, parallel: int = 4, edits: str = ""):
    """Evaluate an editor checkpoint: n rollouts with no training update (same prompt and settings as RL rollouts; no notes,
    no web), then every submitted edit scored on `dataset` against starting model `base`. Outputs in /runs/eval/<name>/.
    checkpoint: a Tinker state or sampler path (tinker://...), "" for the untrained editor (a fresh LoRA).
    base: "original" (Qwen3.5-0.8B as downloaded) or a starting-model tag under /weights/bases/.
    limit: score on the first `limit` problems only (tests). edits: skip the rollouts and score these run paths instead."""
    if dataset not in EVAL_SETS:
        raise SystemExit(f"dataset must be one of {sorted(EVAL_SETS)}")
    tag = "" if base == "original" else base
    print("evaluation set:", dataset, prep_eval_set.remote(dataset), "problems")
    if edits:
        call = eval_score.spawn(name, edits, dataset, tag, limit, parallel)
    else:
        call = rl_run.spawn(name, steps=1, group_size=n, forward_budget=forward_budget, rollout_minutes=rollout_minutes, hours=rollout_minutes / 60 + 0.5,
                            reward_items=reward_items, start_from=checkpoint, web=False, rollouts_per_gpu=rollouts_per_gpu, update_check=False,
                            start_base=tag, mode="eval", eval_dataset=dataset, eval_limit=limit, eval_parallel=parallel)
    print("SPAWNED", call.object_id, "-> /runs/eval/" + name + "/summary.json")
