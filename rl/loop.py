"""RL loop: train Inkling 2 Small (on Tinker) to be a better weight editor of a target model (on Modal; Qwen3.5-0.8B by
default, `--target` picks another key of harness.target.TARGETS).

One RL step: sample a group of rollouts with the current checkpoint, score each submitted edit,
take one policy-gradient step on Tinker, save a checkpoint.

    uvx modal run -m rl.loop::prep_reward_set          # once
    uvx modal run -m rl.loop::prep_practice_index      # once: index and ready-made splits of the practice problems
    uvx modal run -m rl.loop::prep_baseline            # once: the original weights' answers every rollout starts from
    uvx modal run -m rl.loop::prep_originals           # once: the original weights' answers on the splits, served from a cache
    uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32
    uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32 --rollouts-per-gpu 2   # two rollouts share each L40S
    uvx modal run --detach -m rl.loop::evaluate --name ev1 --checkpoint tinker://... --dataset gsm8k_test   # rollouts with no update, edits scored
    uvx modal run --detach -m rl.loop::prep_target --target qwen3.5-2b   # another target: download, practice files, cached answers
    uvx modal run --detach -m rl.loop::sweep --target qwen3.5-2b         # its accuracy and hit-cap rate on 200 MATH train problems

Cost bounds: each rollout holds one L40S (or half of one with --rollouts-per-gpu 2) for at most `rollout_minutes`; the whole run stops at `hours`.
Outputs: /runs/rl/<exp>/ on the runs volume (steps.jsonl, one transcript per rollout).
"""
import modal
from harness.target import app, runs, base, base_src, SRC, heldout_vol, Target, PackedTarget, PY_BUDGET, GEN_CAP, last_boxed, heldout_eval, ban_eval
from harness.target import weights, offline, _Core, base_dirs, heldout_cache, heldout_rows, start_vllm, stop_proc, EVAL_SETS, prep_eval_set
from harness.target import TARGETS, DEFAULT_TARGET, target_arch, target_paths, target_dir, prep, summarize, TASKS, _check_task
from harness.prompts import make_prompt, make_guide, tensor_names_text, layer_ranges, QWEN35_08B

MODEL = "thinkingmachines/Inkling-2-Small:peft:262144"  # 256K-context variant; a rollout is one long sequence
PRICE = dict(prefill=0.30, cached=0.03, sample=1.00, train=3.00)  # $ per million tokens
rl_image = modal.Image.debian_slim(python_version="3.12").pip_install("tinker-cookbook", "openai").add_local_python_source(*SRC)

TOOLS_TEMPLATE = [
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
         "hidden_states": {"type": "array", "items": {"type": "integer"}, "description": "layers <<HS>>"},
         "mlp_act": {"type": "array", "items": {"type": "integer"}, "description": "layers <<MLP>>"}},
      "required": ["type", "model"]}},
]


def tools_for(a):
    """The tool definitions with the target's layer ranges."""
    import json
    hs, mlp = layer_ranges(a)
    return json.loads(json.dumps(TOOLS_TEMPLATE).replace("<<HS>>", hs).replace("<<MLP>>", mlp))


TOOLS = tools_for(QWEN35_08B)
RETURN_KEYS = ("token_logprobs", "topk", "hidden_states", "mlp_act")
GRAD_PARAMS = {
    "objective": {"type": "string", "enum": ["logprob", "loss"], "description": "gradient: 'logprob' = summed log-prob of target_tokens at positions; 'loss' = mean next-token NLL (attributions for lowering it)"},
    "target_tokens": {"type": "array", "items": {"type": "string"}, "description": "gradient: single-token strings, e.g. [\" Wait\", \"Wait\"]"},
    "positions": {"description": "gradient: \"all\" or a list of token positions (position p predicts token p+1)"},
    "topk_units": {"type": "integer", "description": "gradient: MLP units listed per layer, at most 50"}}


def tools_with_grad(tools):
    """The tool list with the forward tool's 'gradient' request type added."""
    import copy
    tools = copy.deepcopy(tools)
    for t in tools:
        if t["name"] == "forward":
            t["description"] += " type 'gradient': one backward pass for diagnosis; returns attributions (activation x gradient) of a target to every layer, MLP unit and attention head. Weights are not changed."
            t["parameters"]["properties"]["type"]["enum"].append("gradient")
            t["parameters"]["properties"].update(GRAD_PARAMS)
    return tools


GRAD_TEXT = """

GRADIENT ATTRIBUTION. The forward tool also accepts type "gradient": one backward pass on up to 8 texts or chats of up to 4,096 tokens each, on the original or edited model. Arguments: "objective": "logprob" with "target_tokens" (single-token strings) and "positions" ("all" or a list; position p predicts token p+1), or "loss" (mean next-token loss); "topk_units" (at most 50). It returns, for that target, the attribution (sum over positions of activation x gradient, the first-order effect of removing a component) of each layer's residual stream, the top MLP units of every layer (down_proj columns), every attention head (full-attention o_proj and linear-attention out_proj blocks) and a score per tensor; the full lists are saved to /work/fwd_<k>/gradient.json. It counts as one forward pass. Gradients are for diagnosis only: no optimizer runs and the weights do not change. Any edit must still be written by hand in Python and saved with save_edit."""
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
- {ids} /practice/index.tsv lists every problem (id, source, level, subject); /practice/problems.jsonl has the full text and reference solutions. Do not guess ids.
- /practice/splits.json holds ready-made disjoint lists of ids: "baseline" (already answered by the original weights, see above), "select" (256 problems for choosing between edits), "confirm_1" and "confirm_2" (256 each, never to be used for a choice), and "extra" (the rest). Use these; do not build your own.
- Tensor names: {tensors}
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


def test_too_big(body, cap):
    """Reason to refuse a generation request with more than `cap` answers per model (cap 0: no limit), else ""."""
    import json
    if not cap:
        return ""
    try:
        req = json.loads(body)
    except Exception:
        return ""
    if not isinstance(req, dict) or req.get("type") != "generate":
        return ""
    n_items = len(req.get("problems") or []) + len(req.get("prompts") or [])
    try:
        n = int(req.get("n", 1))
    except Exception:
        n = 1
    if n_items * n > cap:
        return (f"this test asks for {n_items * n} answers per model ({n_items} problems x n={n}); the limit is {cap} answers per model. "
                f"Use fewer problems (e.g. half of a split with \"n\": 1, or a quarter with \"n\": 2).")
    return ""


def cap_tests(prompt, cap):
    """Rewrite the prompt's statements about test sizes for a limit of `cap` answers per model."""
    if not cap:
        return prompt
    old = 'A generation with "model": "both" takes at most 256 answers per model; with one model at most 512. One split with "n": 1 is 256 answers; half a split with "n": 2 is 256 answers.'
    new = (f'A generation takes at most {cap} answers per model (with "model": "both", {cap} from each). Half a split with "n": 1 is {cap} answers; '
           f'a quarter of a split with "n": 2 is {cap} answers. To confirm a final edit on more answers, run several such tests on different parts of confirm_1 and confirm_2.')
    assert old in prompt, "test-size sentence not found in the prompt"
    prompt = prompt.replace(old, new)
    for a, b in (("the model answers up to 512 generations (problems x samples)", f"the model answers up to {cap} generations (problems x samples) per model"),
                 ('(items x n <= 512; with "both" <= 256 per model)', f"(items x n <= {cap} per model)"),
                 ('on at least 256 generations.', f"on up to {cap} generations per model, repeated on another part of the split when a difference needs confirming.")):
        prompt = prompt.replace(a, b)
    return prompt


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

STARTING WEIGHTS. The weights you are given are {name} with an earlier edit already applied. In this session "original weights" means these starting weights: load_original, the cached original answers in /practice and your score all refer to them. Your task is to improve on them."""

APPLIED_TEXT = """

STARTING WEIGHTS. The weights you are given are {name} with {n} earlier edit{s} already applied, listed below (oldest first). In this session "original weights" means these starting weights: load_original, the cached original answers in /practice and your score all refer to them. Your task is to improve on them.
The earlier edits are intended and are part of the starting weights. Do not undo, reverse or "repair" them: for example, columns or units that are zero because an earlier edit zeroed them on purpose are not damage. Build on top of them.
{items}"""


def applied_text(name, applied):
    """The STARTING WEIGHTS section for a starting model that holds the edits in `applied` (applied_edits.json)."""
    items = []
    for i, a in enumerate(applied, 1):
        items.append(f"{i}. Tensors changed: {a['change']}\n   The editor's own description: {a['description']}")
    return APPLIED_TEXT.format(name=name, n=len(applied), s="s" if len(applied) != 1 else "", items="\n".join(items))


MATH_MEASURED = "HOW IT WILL BE MEASURED. The 500 problems of MATH-500, 2 samples per problem, separately for original and edited weights."
MATH_PRACTICE = "2. Practice problems: 3,000 from the MATH training set and 1,500 from the GSM8K training set, each with a reference solution. They are disjoint from the held-out test after normalized text matching."
MATH_IDS = 'Practice ids look like "math/geometry/461" and "gsm8k/2754".'


def editor_prompt(forward_budget, web=False, known=None, rebased=False, arch=None, task="math", n_practice=None, applied=None):
    """The RL editor's prompt; every architecture statement comes from `arch` (target_arch; default Qwen3.5-0.8B).
    task: a key of TASKS; its texts replace the MATH-500 / MATH and GSM8K statements (n_practice: its practice problem count)."""
    arch = arch or QWEN35_08B
    T = TASKS[_check_task(task)]
    PROMPT, GUIDE = make_prompt(arch), make_guide(arch)
    for old, new in ((MATH_MEASURED, T["measured"]), (MATH_PRACTICE, T["practice"])):
        assert old in PROMPT, old
        if new:
            PROMPT = PROMPT.replace(old, new.replace("<<N>>", f"{n_practice:,}" if n_practice else "about 3,700"))
    hs, mlp = layer_ranges(arch)
    a, b = PROMPT.index("WHAT COUNTS."), PROMPT.index("WHAT YOU HAVE.")
    body = PROMPT[:a] + GUIDE + "\n\n" + PROMPT[b:]
    for old in ("a budget of 50 forward passes in total", "When all 50 forward passes are used"):
        assert old in body
        body = body.replace(old, old.replace("50", str(forward_budget)))
    for old, new in (
        ("Call the `forward` tool with the parameter `request` set to JSON of this form:", "Call the `forward` tool with these arguments, given directly (not wrapped in a `request` object):"),
        ("Call the `forward` tool with `request` set to:", "Call the `forward` tool with these arguments, given directly:"),
        (f'"return": {{"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers {hs}], "mlp_act": [layers {mlp}]}}}}',
         f'"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers {hs}], "mlp_act": [layers {mlp}]}}'),
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
        extra += KNOWN_TEXT.format(n=known["n"], summary=known["summary"], tensors=tensor_names_text(arch), ids=T["ids"] or MATH_IDS)
    if web:
        extra += WEB_TEXT
    if applied:
        extra += applied_text(arch["name"], applied)
    elif rebased:
        extra += BASE_TEXT.format(name=arch["name"])
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
LABEL = {"math_train": "MATH", "gsm8k_train": "GSM8K", "cn_k12": "NuminaMath cn_k12"}
SRCS = ["math_train", "gsm8k_train"] if {r['source'] for r in rows} & {"math_train", "gsm8k_train"} else sorted({r['source'] for r in rows})
rate = lambda c, t, w: 1000 * c[w] / t
top = sorted((w for w in cc if cc[w] >= 40), key=lambda w: -(rate(cc, ct, w) + 0.01) / (rate(dc, dt, w) + 0.01))[:8]
lines = [
 f"- {len(rows)} answers to {len({r['qid'] for r in rows})} practice problems: {pct([r['correct'] for r in rows])}% correct "
 "(" + ", ".join(f"{pct([r['correct'] for r in rows if r['source'] == s])}% on {LABEL.get(s, s)}" for s in SRCS) + ").",
 f"- {pct([r['finish'] == 'length' for r in rows])}% run to the 8,192-token limit and {pct([r['extracted'] is None for r in rows])}% have no boxed answer. Of the answers that finish, {pct([r['correct'] for r in done])}% are correct.",
 "- Words that open a line far more often in the cut-off answers than in the finished ones (per 1,000 tokens, cut-off vs finished): "
 + ", ".join(f"{w} {rate(cc, ct, w):.1f} vs {rate(dc, dt, w):.1f}" for w in top) + ".",
]
open('/practice/baseline/SUMMARY.txt', 'w').write("\n".join(lines) + "\n")
json.dump({"n": len(rows), "summary": "\n".join(lines)}, open('/practice/baseline/meta.json', 'w'))
print("\n".join(lines))
"""


@app.function(image=base_src, volumes={"/runs": runs}, timeout=900)
def prep_practice_index(n_math: int = 192, n_gsm: int = 64, target: str = DEFAULT_TARGET):
    """Once: a compact index of the practice problems and ready-made disjoint splits, so editors do not guess ids.
    (Other targets get copies of the default target's files from `prep`, so all targets share the splits.)"""
    import json, random
    d = target_paths(target)["practice"]
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
async def prep_baseline(n_math: int = 192, n_gsm: int = 64, target: str = DEFAULT_TARGET):
    """Once per target: the original weights' answers on practice problems, so rollouts need not regenerate them.
    (prep_target does the same through gen_base, on one GPU at 0.90.)"""
    import json
    tgt = Target.with_options(scaledown_window=60)(rid="baseline", target=target)
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
async def prep_originals(splits: str = "select,confirm_1,confirm_2", target: str = DEFAULT_TARGET):
    """Once per target: the original weights' answers on the ready-made splits. The forward tool serves them from this cache."""
    import asyncio, json
    runs.reload()
    sp = json.load(open(f"{target_paths(target)['practice']}/splits.json"))

    async def one(name):
        tgt = Target.with_options(scaledown_window=60)(rid=f"originals-{name}", target=target)
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
def make_base(tag: str, prev_tag: str, edit_run: str, target: str = DEFAULT_TARGET, task: str = "math"):
    """A starting model `tag`: the starting model `prev_tag` with the edit of /runs/<edit_run>/work/edited applied
    (each edited tensor replaces the one in the weights), plus a practice folder holding the problem files."""
    import json, os, shutil
    from safetensors.torch import load_file, save_file
    runs.reload()
    src, psrc = base_dirs(prev_tag, target, task)
    dst, pdst = base_dirs(tag, target, task)
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
    br = target_paths(target, task)["bases_r"]
    json.dump({"tag": tag, "prev": prev_tag, "edit_run": edit_run, "tensors": sorted(edits), "target": target},
              open(f"{br}/{tag}/meta.json", "w"), indent=1)
    prev_applied = applied_chain(prev_tag, target, task) if prev_tag else []
    json.dump(prev_applied + [describe_edit(edit_run, edits, src)], open(f"{br}/{tag}/applied_edits.json", "w"), indent=1)
    weights.commit(); runs.commit()
    return sorted(edits)


def _final_change_text(edit_run, limit=500):
    """The 'What I changed' part of the FINAL SUBMISSION of the rollout behind /runs/<edit_run> (rl/<exp>/work/<rid>)."""
    import json, os, re
    root, _, rid = edit_run.rpartition("/work/")
    log = f"/runs/{root}/{rid}.jsonl"
    text = ""
    if os.path.exists(log):
        for line in open(log):
            r = json.loads(line)
            if "summary" in r and r.get("final_text"):
                text = r["final_text"]
    text = re.sub(r"^\s*FINAL SUBMISSION\s*", "", text or "")
    m = re.search(r"what i changed[^\n:]*[:\n]+(.*?)(?:\n\s*\n\s*\**\s*(?:why|evidence|how|what i expect|diagnosis|results?)\b|\Z)", text, re.I | re.S)
    out = (m.group(1) if m else text).strip()
    out = re.sub(r"\s+", " ", out.replace("**", ""))
    return (out[:limit] + " ...") if len(out) > limit else (out or "(no description in the log)")


def describe_edit(edit_run, edits, prev_dir):
    """One applied edit: which tensors changed and how (measured against the previous starting model), plus the editor's words."""
    import os, torch
    from safetensors import safe_open
    prev = {}
    for f in os.listdir(prev_dir):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(prev_dir, f), "pt") as fh:
                for n in edits:
                    if n in fh.keys():
                        prev[n] = fh.get_tensor(n)
    parts = []
    for n in sorted(edits):
        short = n.replace("model.language_model.", "")
        if n not in prev or prev[n].shape != edits[n].shape or edits[n].dim() != 2:
            parts.append(f"{short} (replaced)"); continue
        new = edits[n].to(prev[n].dtype).float()  # as stored in the starting model
        prev[n] = prev[n].float()
        d = new - prev[n]
        cols = (d.abs().sum(0) > 0).nonzero().flatten().tolist()
        rows = (d.abs().sum(1) > 0).nonzero().flatten().tolist()
        zeroed = [c for c in cols if new[:, c].abs().sum() == 0 and prev[n][:, c].abs().sum() > 0]
        rel = (d.norm() / prev[n].norm()).item()
        if zeroed and len(cols) == len(zeroed):
            parts.append(f"{short}: column{'s' if len(zeroed) > 1 else ''} {', '.join(map(str, zeroed[:20]))} set to zero")
        elif len(cols) <= 20 and len(cols) < new.shape[1]:
            parts.append(f"{short}: column{'s' if len(cols) > 1 else ''} {', '.join(map(str, cols))} changed" + (f" ({', '.join(map(str, zeroed))} set to zero)" if zeroed else ""))
        elif len(rows) <= 20 and len(rows) < new.shape[0]:
            parts.append(f"{short}: row{'s' if len(rows) > 1 else ''} {', '.join(map(str, rows))} changed")
        else:
            try:  # effective rank: directions holding 95% of the change (bf16 rounding adds a small full-rank floor)
                sv2 = torch.linalg.svdvals(d) ** 2
                rank = int((torch.cumsum(sv2, 0) / sv2.sum() < 0.95).sum().item()) + 1
            except Exception:
                rank = -1
            parts.append(f"{short}: rank-{rank} change across the whole matrix ({rel:.1%} of its norm)" if rank > 0 else f"{short}: changed ({rel:.1%} of its norm)")
    if len(parts) > 6:  # many matrices with the same kind of change: one line per kind
        import re
        kinds = {}
        for p in parts:
            k = re.sub(r" \(\d+(\.\d+)?% of its norm\)", "", p.split(": ", 1)[-1])
            kinds.setdefault(k, []).append(p.split(":")[0])
        if len(kinds) < len(parts) / 2:  # only when it shortens the list
            parts = [(f"{len(v)} matrices ({', '.join(v) if len(v) <= 3 else v[0] + ', ..., ' + v[-1]})" if len(v) > 1 else v[0]) + f": {k}" for k, v in kinds.items()]
    return {"edit_run": edit_run, "tensors": sorted(edits), "change": "; ".join(parts), "description": _final_change_text(edit_run)}


def applied_chain(tag, target=DEFAULT_TARGET, task="math"):
    """applied_edits.json of a starting model; for a model made before that file existed, rebuilt from its meta.json chain."""
    import json, os
    from safetensors.torch import load_file
    br = target_paths(target, task)["bases_r"]
    f = f"{br}/{tag}/applied_edits.json"
    if os.path.exists(f):
        return json.load(open(f))
    meta = json.load(open(f"{br}/{tag}/meta.json"))
    out = applied_chain(meta["prev"], target, task) if meta["prev"] else []
    ed = f"/runs/{meta['edit_run']}/work/edited"
    edits = {g[:-len(".safetensors")]: load_file(os.path.join(ed, g))["t"] for g in sorted(os.listdir(ed)) if g.endswith(".safetensors")}
    out = out + [describe_edit(meta["edit_run"], edits, base_dirs(meta["prev"], target, task)[0])]
    json.dump(out, open(f, "w"), indent=1)
    return out


@app.function(image=offline, volumes={"/weights": weights, "/runs": runs}, timeout=30 * 60, cpu=8, memory=32768)
def backfill_applied(tags: str, target: str = DEFAULT_TARGET, task: str = "math"):
    """Write applied_edits.json for starting models made before it existed (comma-separated tags); returns the prompt section."""
    runs.reload()
    out = {}
    for tag in [t for t in tags.split(",") if t]:
        out[tag] = applied_text(target_arch(target)["name"], applied_chain(tag, target, task))
    runs.commit()
    return out


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, timeout=100 * 60, memory=65536, cpu=8)
def gen_base(tag: str, part: int, reward_items_json: str = "", reward_n: int = 2, reward_items: int = 128, target: str = DEFAULT_TARGET, limit: int = 0,
             task: str = "math", test_limit: int = 0):
    """The starting model's own answers, generated once and served from cache, as for the downloaded model.
    tag "" is the target as downloaded (prep_target). part 0: the baseline answers and their summary, the reward set,
    the select split. part 1: the two confirm splits. part 2: the test set (only needed for test scoring, so the RL loop
    does not wait for it). part 3: the test set and gsm8k_test. part 4: parts 0, 1 and 2 on one GPU. part 5: part 4 without the
    confirm splits (a quick prep; the forward tool then generates their original side when asked).
    test_limit k: only the first k problems of the test set (a cache that serves evaluations with --limit <= k).
    task (a key of TASKS) other than "math": its practice folder, reward set and held-out set (EVAL_SETS[eval_set]) in place of MATH-500.
    limit k (tests): only the first k problems of each list, written under /runs/data/targets/<target>/_limit_test/, never to a cache."""
    import json, os, threading, time
    runs.reload()
    if not reward_items_json:
        reward_items_json = json.dumps([json.loads(l) for l in open(TASKS[task]["reward_file"])][:reward_items])
    P = target_paths(target, task)
    mdir, pdir = base_dirs(tag, target, task)
    test_set = TASKS[task]["eval_set"]
    out_pdir = pdir
    reward_path = f"{P['bases_r']}/{tag}/reward_base_{{}}x{reward_n}.json" if tag else P["reward_base"].format(f"{{}}x{reward_n}")
    cache_of = lambda ds: heldout_cache(tag, ds, target)
    cut = lambda xs: xs[:limit] if limit else xs
    if limit:  # a test: same code, outputs kept away from every cache
        tdir = f"{P['data']}/_limit_test/{tag or 'original'}"
        out_pdir, reward_path = f"{tdir}/practice", f"{tdir}/reward_base_{{}}x{reward_n}.json"
        cache_of = lambda ds: f"{tdir}/eval_{ds}_original.jsonl"
    proc = start_vllm(mdir, "original", 8001, 0.90, GEN_CAP + 3000)
    sp = json.load(open(f"{pdir}/splits.json"))
    practice = {json.loads(l)["id"]: json.loads(l) for l in open(f"{pdir}/problems.jsonl")}
    items = lambda ids: [{"qid": i, "source": practice[i]["source"], "answer": practice[i]["answer"],
                          "prompt": practice[i]["problem"] + "\n\nPut your final answer in \\boxed{}."} for i in cut(ids)]
    gen = lambda its, n: _Core._gen(None, "original", 8001, its, n, GEN_CAP)  # the forward tool's generation: same prompts, seeds, sampling
    write = lambda path, rows: (os.makedirs(os.path.dirname(path), exist_ok=True), open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows)))
    jobs, out, t = {}, {}, time.time()
    def baseline():
        rows = gen(items(sp["baseline"]), 2)
        write("/tmp/bw/gen_1.jsonl", rows)
        os.makedirs(f"{out_pdir}/baseline", exist_ok=True)
        exec(BASELINE_CODE.replace("/work/gen_1.jsonl", "/tmp/bw/gen_1.jsonl").replace("/practice/baseline", f"{out_pdir}/baseline"), {})
        out["baseline"] = len(rows)
    def reward():
        its = [{"qid": "reward/" + it["qid"], "source": "reward", "answer": it["answer"], "prompt": it["problem"] + "\n\nPut your final answer in \\boxed{}."}
               for it in cut(json.loads(reward_items_json))]
        rows = [{"qid": r["qid"], "sample": r["sample"], "correct": bool(r["correct"]), "tokens": r["tokens"], "finish": r["finish"]} for r in gen(its, reward_n)]
        rp = reward_path.format(len(json.loads(reward_items_json)))  # named by the full reward set, as rl_run looks it up
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        json.dump(rows, open(rp, "w"))
        out["reward"] = round(sum(r["correct"] for r in rows) / len(rows), 4)
    def test(ds):
        def f():
            its = cut([json.loads(l) for l in open(EVAL_SETS[ds]["file"])])
            its = its[:test_limit] if test_limit else its
            rows = heldout_rows(8001, "original", its, 2, EVAL_SETS[ds]["grade"])
            write(cache_of(ds), rows)
            out[ds] = round(sum(bool(r["correct"]) for r in rows) / len(rows), 4)
        return f
    def split(name):
        def f():
            rows = gen(items(sp[name]), 2)
            write(f"{out_pdir}/original/{name}.jsonl", rows)
            out[name] = round(sum(r["correct"] for r in rows) / len(rows), 4)
        return f
    if part == 0:
        jobs = {"baseline": baseline, "reward": reward, "select": split("select")}
    elif part == 1:
        jobs = {n: split(n) for n in ("confirm_1", "confirm_2")}
    elif part == 2:
        jobs = {test_set: test(test_set)}
    elif part in (4, 5):
        jobs = {"baseline": baseline, "reward": reward, **{n: split(n) for n in ("select", "confirm_1", "confirm_2")[:1 if part == 5 else 3]}, test_set: test(test_set)}
    else:
        jobs = {ds: test(ds) for ds in ("math500", "gsm8k_test")}
    ths = [threading.Thread(target=f, name=n) for n, f in jobs.items()]
    [th.start() for th in ths]; [th.join() for th in ths]
    stop_proc(proc)
    missing = set(jobs) - set(out)
    assert not missing, f"failed: {missing}"
    runs.commit()
    out["seconds"] = round(time.time() - t)
    return out


@app.function(image=rl_image, volumes={"/runs": runs, "/heldout": heldout_vol, "/weights": weights}, secrets=[modal.Secret.from_name("tinker"), modal.Secret.from_name("exa")],
              timeout=int(11.5 * 3600), nonpreemptible=True, cpu=4, memory=16384)
async def rl_run(exp: str, steps: int = 2, group_size: int = 4, forward_budget: int = 32, rollout_minutes: int = 120, hours: float = 10.0,
                 reward_items: int = 128, reward_n: int = 2, effort: float = 0.9, lr: float = 2e-5, max_turns: int = 400, start_from: str = "", web: bool = False, library: str = "",
                 rollouts_per_gpu: int = 1, rebase_every: int = 0, update_check: bool = True, start_base: str = "", force_rewards: str = "",
                 mode: str = "rl", eval_dataset: str = "math500", eval_limit: int = 0, eval_parallel: int = 4,
                 advantage: str = "grpo", adv_scale: float = 0.10, target: str = DEFAULT_TARGET, task: str = "math",
                 max_test_answers: int = 0, step_offset: int = 0, grad_tool: bool = False):
    """grad_tool: offer the forward tool's 'gradient' request type (attribution from one backward pass; diagnosis only).
    target: a key of harness.target.TARGETS; the prompt, tools, caches and starting models are that target's.
    task: a key of harness.target.TASKS; the practice folder, reward set and prompt wording are that task's.
    start_base: begin with this starting model (a tag made by rebasing) instead of the downloaded one.
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
    if os.path.exists(f"{d}/steps.jsonl") and not step_offset:
        raise RuntimeError(f"{d} already has a run; a restart would train on top of it. Use a new --exp, or --step-offset to continue it.")
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
    _check_task(task)
    items = [json.loads(l) for l in open(TASKS[task]["reward_file"])][:reward_items]
    items_json = json.dumps(items)
    arch = target_arch(target)  # from the target's config.json and safetensors headers
    TP = target_paths(target, task)
    pdir_of = lambda tag: f"{TP['bases_r']}/{tag}/practice" if tag else TP["practice"]
    bm = f"{pdir_of('')}/baseline/meta.json"
    known = json.load(open(bm)) if os.path.exists(bm) else None
    n_practice = sum(1 for _ in open(f"{pdir_of('')}/problems.jsonl"))
    prompt_text = cap_tests(editor_prompt(forward_budget, web, known, arch=arch, task=task, n_practice=n_practice), max_test_answers) + (GRAD_TEXT if grad_tool else "")
    tools = (tools_with_grad(tools_for(arch)) if grad_tool else tools_for(arch)) + ([WEB_TOOL, WEB_READ_TOOL] if web else [])
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
    json.dump({"exp": exp, "model": MODEL, "target": target, "task": task, "arch": {k: v for k, v in arch.items() if k != "dir"}, "steps": steps, "group_size": group_size, "forward_budget": forward_budget, "rollout_minutes": rollout_minutes,
               "reward_items": len(items), "reward_n": reward_n, "effort": effort, "lr": lr, "advantage": advantage, "adv_scale": adv_scale, "web": web, "library": lib_dir, "rollouts_per_gpu": rollouts_per_gpu,
               "rebase_every": rebase_every, "start_base": start_base, "update_check": update_check, "start_from": start_from,
               "baseline_given": bool(known), "mode": mode, "eval_dataset": eval_dataset if evaluating else None, "eval_limit": eval_limit,
               "max_test_answers": max_test_answers, "step_offset": step_offset, "grad_tool": grad_tool,
               "prompt": prompt_text}, open(f"{d}/config.json" if not step_offset else f"{d}/config_from_step{step_offset + 1}.json", "w"), indent=1)

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
            rows = json.load(open(f"{TP['bases_r']}/{tag}/reward_base_{len(items)}x{reward_n}.json"))
        else:
            base_cache = TP["reward_base"].format(f"{len(items)}x{reward_n}")  # the original weights' answers on the reward set never change
            if os.path.exists(base_cache):
                rows = json.load(open(base_cache))
            else:
                rows = (await Target.with_options(scaledown_window=120)(rid=f"{exp}-base", target=target, task=task).reward.remote.aio(items_json, reward_n, GEN_CAP, True))["original"]
                json.dump(rows, open(base_cache, "w"))
        ok = {(r["qid"], r["sample"]): r["correct"] for r in rows}
        ap_file = f"{TP['bases_r']}/{tag}/applied_edits.json" if tag else ""
        applied = json.load(open(ap_file)) if ap_file and os.path.exists(ap_file) else None
        B = {"tag": tag, "prompt": cap_tests(editor_prompt(forward_budget, web, kn, rebased=bool(tag), arch=arch, task=task, n_practice=n_practice, applied=applied), max_test_answers) + (GRAD_TEXT if grad_tool else ""),
             "ok": ok, "acc": sum(ok.values()) / len(ok)}
        slog({"event": "reward_base", "base": tag, "answers": len(ok), "accuracy": round(B["acc"], 4),
              "hit_cap": round(sum(r["finish"] == "length" for r in rows) / len(rows), 4), "seconds": round(time.time() - t)})
        return B

    B = await load_base(start_base)
    cache_call = None
    if evaluating and not eval_limit and not os.path.exists(heldout_cache(start_base, eval_dataset, target)):
        # the starting model's answers on the evaluation set, generated once while the rollouts run
        cache_call = await heldout_eval.spawn.aio(f"eval/{exp}/_original", 2, 0, True, start_base, eval_dataset, target)

    async def rollout(sampler, rid, gid, B):
        base_ok, prompt_text = B["ok"], B["prompt"]
        t0 = time.time()
        deadline = min(t0 + rollout_minutes * 60, run_deadline)
        if rollouts_per_gpu > 1:  # one container per gid; every call names the rollout
            obj = PackedTarget.with_options(scaledown_window=300)(gid=gid, base=B["tag"], target=target, task=task)
            call = lambda m, *a: getattr(obj, m).remote.aio(rid, *a)
        else:
            obj = Target.with_options(scaledown_window=300)(rid=rid, base=B["tag"], target=target, task=task)
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
                    too_big = test_too_big(body, max_test_answers)
                    if not grad_tool:
                        try:
                            if json.loads(body).get("type") == "gradient":
                                too_big = "type must be 'generate' or 'teacher_forced'"
                        except Exception:
                            pass
                    if must_finish():
                        res = "[no forward passes left]"
                    elif too_big:
                        res = "[request failed; it does not count against the budget]\n" + too_big
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
            # Save the work folder (with the submitted edit) first, whatever ended the session: a target replaced after the
            # reward call left an empty folder once, and held-out scoring then found no edit.
            snap_dir = f"/runs/{root}/{exp}/work/{rid}/work/edited"
            try:
                await call("snapshot", f"{root}/{exp}/work/{rid}")
            except Exception as e:
                w({"snapshot_error": repr(e)[:300]})
            # reward: the submitted edit against the cached original answers, paired by problem and sample
            rw = await call("reward", items_json, reward_n, GEN_CAP, False)
            try:
                runs.reload()
                saved = sorted(f[:-len(".safetensors")] for f in os.listdir(snap_dir) if f.endswith(".safetensors")) if os.path.isdir(snap_dir) else []
                if sorted(rw["edited_tensors"] or []) != saved:
                    w({"snapshot_mismatch": {"scored": len(rw["edited_tensors"] or []), "saved": len(saved)}})
            except Exception as e:
                w({"snapshot_check_error": repr(e)[:300]})
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
        ref = await asyncio.to_thread(score, sampler0) if (sampler0 is not None and step > 0 and not step_offset) else None  # a continued run has no step-0 sampler
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

    if step_offset:  # continuing a run: rollouts since the last starting-model update count for the next one
        for l in open(f"{d}/steps.jsonl"):
            r = json.loads(l)
            if r.get("event") == "step" and r["step"] < step_offset and (not rebase_every or r["step"] >= step_offset - step_offset % rebase_every):
                recent += [(r["step"], info) for info in r.get("rollouts", []) if info.get("ok")]
    for step in range(step_offset, step_offset + steps):
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
                                        extra={"checkpoint": start_from or "untrained", "failed_rollouts": [i["rid"] for i in infos.values() if not i["ok"]]}, target=target)
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
        if rebase_every and (step + 1) % rebase_every == 0 and step + 1 < step_offset + steps:
            cands = [(info["reward"], s, info) for s, info in recent if info.get("edited_tensors") and info["reward"] > 0]
            recent = []
            if not cands:
                slog({"event": "rebase", "skipped": "no submitted edit with a positive reward", "step": step})
                continue
            r_best, s_best, info = max(cands, key=lambda x: x[0])
            n_base += 1
            tag, t3 = f"{exp}-k{n_base}", time.time()
            try:
                tensors = await make_base.remote.aio(tag, B["tag"], f"rl/{exp}/work/{info['rid']}", target, task)
                parts = await asyncio.gather(gen_base.remote.aio(tag, 0, items_json, reward_n, target=target, task=task), gen_base.remote.aio(tag, 1, items_json, reward_n, target=target, task=task))
                await gen_base.spawn.aio(tag, 2, items_json, reward_n, target=target, task=task)  # the test set's cache, for test scoring; not waited for
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
       rebase_every: int = 0, update_check: bool = True, start_base: str = "", force_rewards: str = "", advantage: str = "grpo", adv_scale: float = 0.10,
       target: str = DEFAULT_TARGET, task: str = "math", max_test_answers: int = 0, step_offset: int = 0):
    """advantage: "grpo" = (reward - group mean) / group spread; "mean" = (reward - group mean) / adv_scale (Dr. GRPO).
    target: a key of harness.target.TARGETS (run prep_target for it first)."""
    if target not in TARGETS:
        raise SystemExit(f"target must be one of {sorted(TARGETS)}")
    call = rl_run.spawn(exp, steps, group_size, forward_budget, rollout_minutes, hours, reward_items, 2, effort, lr, 400, start_from, web, library, rollouts_per_gpu,
                        rebase_every, update_check, start_base, force_rewards, advantage=advantage, adv_scale=adv_scale, target=target, task=task,
                        max_test_answers=max_test_answers, step_offset=step_offset)
    print("SPAWNED", call.object_id)


@app.function(image=rl_image, volumes={"/runs": runs}, timeout=int(12 * 3600))
async def test_step_edits(exp: str, steps: str = "0", parallel: int = 2):
    """Score each rollout's submitted edit on the test set (MATH-500), which the RL loop never uses.
    Only the edited side is generated; the original side comes from one cached set of answers."""
    import asyncio, json, os, shutil
    runs.reload()
    cf = f"/runs/rl/{exp}/config.json"
    target = (json.load(open(cf)).get("target") if os.path.exists(cf) else None) or DEFAULT_TARGET  # runs before the registry are the default target
    cached = heldout_cache("", "math500", target)
    if not os.path.exists(cached) and target == DEFAULT_TARGET:
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
            return rid, await heldout_eval.remote.aio(path, 2, 0, True, base_of.get(int(rid.rsplit("-s", 1)[1].split("-")[0]), ""), "math500", target)

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
def rebase(tag: str, edit_run: str, prev: str = "", target: str = DEFAULT_TARGET):
    """By hand: a starting model `tag` = starting model `prev` + the edit in /runs/<edit_run>/work/edited, with its caches."""
    import time
    t = time.time()
    print("tensors", len(make_base.remote(tag, prev, edit_run, target)), round(time.time() - t), "s")
    calls = [gen_base.spawn(tag, k, target=target) for k in (0, 1, 2)]
    print([c.get() for c in calls], round(time.time() - t), "s")


@app.local_entrypoint()
def test(exp: str, steps: str = "0", parallel: int = 2):
    print("SPAWNED", test_step_edits.spawn(exp, steps, parallel).object_id)


# ----------------------------------------------------------------------------- evaluating an editor checkpoint
async def score_edits(out, paths, dataset, base_tag, limit=0, parallel=4, rewards=None, extra=None, target=DEFAULT_TARGET):
    """Score the edit in /runs/<path>/work/edited of each path on `dataset` against starting model `base_tag`, paired
    by problem (heldout_eval, with the starting model's answers cached per (base, dataset)). A path without an edit scores 0.
    Writes /runs/<out>/scores.json (one row per path) and /runs/<out>/summary.json (mean and standard error of the
    paired gain across paths), and returns the summary."""
    import asyncio, json, os, time
    t0 = time.time()
    await runs.reload.aio()
    cached = heldout_cache(base_tag, dataset, target)
    if not limit and not os.path.exists(cached):  # once, before the edits, so they share one set of original answers
        await heldout_eval.remote.aio(f"{out}/_original", 2, 0, True, base_tag, dataset, target)
        await runs.reload.aio()
    sem = asyncio.Semaphore(parallel)
    has_edit = lambda p: os.path.isdir(f"/runs/{p}/work/edited") and any(f.endswith(".safetensors") for f in os.listdir(f"/runs/{p}/work/edited"))

    async def one(p):
        row = {"run": p, "reward_pp": round(100 * rewards[p], 2) if rewards and rewards.get(p) is not None else None}
        if not has_edit(p):
            return {**row, "edit": False, "gain_pp": 0.0}
        async with sem:
            try:
                h = await heldout_eval.remote.aio(p, 2, limit, True, base_tag, dataset, target)
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
    summary = {"dataset": dataset, "target": target, "base": base_tag or "original", "problems": limit or "all", "answers_per_problem": 2,
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
async def eval_score(name: str, edits: str, dataset: str = "math500", base_tag: str = "", limit: int = 0, parallel: int = 4, target: str = DEFAULT_TARGET):
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
    return await score_edits(f"eval/{name}", paths, dataset, base_tag, limit, parallel, rewards, extra={"edits": paths}, target=target)


@app.local_entrypoint()
def evaluate(name: str, checkpoint: str = "", base: str = "original", n: int = 8, dataset: str = "math500", limit: int = 0,
             forward_budget: int = 32, rollout_minutes: int = 120, rollouts_per_gpu: int = 2, reward_items: int = 128, parallel: int = 4, edits: str = "",
             target: str = DEFAULT_TARGET, task: str = "math", grad_tool: bool = False):
    """Evaluate an editor checkpoint: n rollouts with no training update (same prompt and settings as RL rollouts; no notes,
    no web), then every submitted edit scored on `dataset` against starting model `base`. Outputs in /runs/eval/<name>/.
    checkpoint: a Tinker state or sampler path (tinker://...), "" for the untrained editor (a fresh LoRA).
    base: "original" (the target as downloaded) or a starting-model tag of that target (under /weights/bases/, or /weights/bases/<target>/).
    target: a key of harness.target.TARGETS (default qwen3.5-0.8b).
    task: a key of harness.target.TASKS (default math): the practice folder, reward set and prompt wording of the rollouts
    (run rl.loop::prep_task first); --dataset picks the set the edits are scored on (a task's own: TASKS[task]["eval_set"]).
    limit: score on the first `limit` problems only (tests). edits: skip the rollouts and score these run paths instead."""
    if dataset not in EVAL_SETS:
        raise SystemExit(f"dataset must be one of {sorted(EVAL_SETS)}")
    if target not in TARGETS:
        raise SystemExit(f"target must be one of {sorted(TARGETS)}")
    if task not in TASKS:
        raise SystemExit(f"task must be one of {sorted(TASKS)}")
    tag = "" if base == "original" else base
    print("evaluation set:", dataset, prep_eval_set.remote(dataset), "problems")
    if edits:
        call = eval_score.spawn(name, edits, dataset, tag, limit, parallel, target)
    else:
        call = rl_run.spawn(name, steps=1, group_size=n, forward_budget=forward_budget, rollout_minutes=rollout_minutes, hours=rollout_minutes / 60 + 0.5,
                            reward_items=reward_items, start_from=checkpoint, web=False, rollouts_per_gpu=rollouts_per_gpu, update_check=False,
                            start_base=tag, mode="eval", eval_dataset=dataset, eval_limit=limit, eval_parallel=parallel, target=target, task=task,
                            grad_tool=grad_tool)
    print("SPAWNED", call.object_id, "-> /runs/eval/" + name + "/summary.json")


# ----------------------------------------------------------------------------- scoring a starting model
@app.function(image=rl_image, volumes={"/runs": runs, "/heldout": heldout_vol}, timeout=4 * 3600)
async def base_vs_original(base_tag: str, dataset: str = "math500", limit: int = 0, target: str = DEFAULT_TARGET):
    """A starting model (a rebase tag) against Qwen as downloaded on an evaluation set, paired by problem and sample.
    Both sides come from heldout_eval with no edit: each model's answers are cached per (model, dataset) and generated
    only when missing (with `limit`, the cache is used when complete and nothing is written to it).
    Writes /runs/eval/base-<tag>-<dataset>.json (with `limit`: ...-limit<k>.json)."""
    import asyncio, json, os
    from harness.target import paired_stats, summarize
    name = ("" if target == DEFAULT_TARGET else f"{target}-") + f"base-{base_tag}-{dataset}" + (f"-limit{limit}" if limit else "")
    sides = {"base": base_tag, "original": ""}
    paths = {k: f"eval/{name}/_{k}" for k in sides}
    await runs.reload.aio()
    items = [json.loads(l) for l in open(EVAL_SETS[dataset]["file"])]
    want = {(it["qid"], s) for it in (items[:limit] if limit else items) for s in range(2)}
    sub = "heldout" if dataset == "math500" else f"heldout_{dataset}"

    async def answers(k, tag):
        """From the cache when it covers the wanted problems (no GPU), else generated by heldout_eval."""
        c = heldout_cache(tag, dataset, target)
        if os.path.exists(c):
            rows = [r for r in map(json.loads, open(c)) if (r["qid"], r["sample"]) in want]
            if len(rows) == len(want):
                return rows, True
        await heldout_eval.remote.aio(paths[k], 2, limit, True, tag, dataset, target)
        await runs.reload.aio()
        return [json.loads(l) for l in open(f"/runs/{paths[k]}/{sub}/original.jsonl")], False

    got = dict(zip(sides, await asyncio.gather(*[answers(k, t) for k, t in sides.items()])))
    rows = {k: v[0] for k, v in got.items()}
    pr = paired_stats(rows["original"], rows["base"])  # base minus original
    out = {"base": base_tag, "target": target, "dataset": dataset, "problems": limit or "all", "answers_per_problem": 2,
           "original": summarize(rows["original"], GEN_CAP), "base_model": summarize(rows["base"], GEN_CAP),
           "original_pct": None, "base_pct": None, "gain_pp": pr["diff_pp"], "se_pp": pr["se_pp"], "questions": pr["questions"],
           "wrong_to_right": pr["wrong_to_right"], "right_to_wrong": pr["right_to_wrong"],
           "from_cache": {k: v[1] for k, v in got.items()}}
    out["original_pct"], out["base_pct"] = out["original"]["accuracy_pct"], out["base_model"]["accuracy_pct"]
    json.dump(out, open(f"/runs/eval/{name}.json", "w"), indent=1)
    await runs.commit.aio()
    print("BASE", json.dumps({k: v for k, v in out.items() if k not in ("original", "base_model")}))
    return out


@app.local_entrypoint()
def score_base(base: str, dataset: str = "math500", limit: int = 0, target: str = DEFAULT_TARGET):
    """Score a starting model (a rebase tag) against the target as downloaded on `dataset`, paired by problem.
    Output: /runs/eval/base-<tag>-<dataset>.json (another target: /runs/eval/<target>-base-<tag>-<dataset>.json)."""
    if dataset not in EVAL_SETS:
        raise SystemExit(f"dataset must be one of {sorted(EVAL_SETS)}")
    if target not in TARGETS:
        raise SystemExit(f"target must be one of {sorted(TARGETS)}")
    print("evaluation set:", dataset, prep_eval_set.remote(dataset), "problems")
    call = base_vs_original.spawn(base, dataset, limit, target)
    name = ("" if target == DEFAULT_TARGET else f"{target}-") + f"base-{base}-{dataset}" + (f"-limit{limit}" if limit else "")
    print("SPAWNED", call.object_id, f"-> /runs/eval/{name}.json")


# ----------------------------------------------------------------------------- preparing another target
@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, timeout=90 * 60, memory=32768, cpu=8)
def baseline_sweep(target: str = DEFAULT_TARGET, n_problems: int = 200, n: int = 1, cap: int = GEN_CAP, task: str = "math"):
    """The target as downloaded on the first `n_problems` MATH training problems of the practice "extra" split (the same
    problems for every target, never in the baseline or the select/confirm splits), with the forward tool's prompt, seeds
    and sampling. Reports accuracy, the share cut at `cap`, and how often "Wait" appears in cut-off and finished answers.
    Writes /runs/data/targets/<target>/sweep_math<k>_n<n>_cap<cap>.json and the answers as .jsonl.
    task other than "math": the first `n_problems` of that task's "extra" split, written to the task's folder (sweep_<task><k>_...)."""
    import json, os, re, time
    runs.reload()
    P = target_paths(target, task)
    pdir = P["practice"]
    sp = json.load(open(f"{pdir}/splits.json"))
    practice = {json.loads(l)["id"]: json.loads(l) for l in open(f"{pdir}/problems.jsonl")}
    ids = [i for i in sp["extra"] if task != "math" or practice[i]["source"] == "math_train"][:n_problems]
    items = [{"qid": i, "source": practice[i]["source"], "answer": practice[i]["answer"], "prompt": practice[i]["problem"] + "\n\nPut your final answer in \\boxed{}."} for i in ids]
    t = time.time()
    proc = start_vllm(target_dir(target), "original", 8001, 0.90, cap + 3000)
    rows = _Core._gen(None, "original", 8001, items, n, cap)
    stop_proc(proc)
    waits = lambda rs: sum(len(re.findall(r"\bWait\b", r["text"])) for r in rs)
    toks = lambda rs: max(1, sum(r["tokens"] for r in rs))
    cut = [r for r in rows if r["finish"] == "length"]; done = [r for r in rows if r["finish"] != "length"]
    out = {"target": target, "task": task, "problems": len(ids), "samples_per_problem": n, "cap": cap, **summarize(rows, cap),
           "accuracy_of_finished_pct": round(100 * sum(bool(r["correct"]) for r in done) / len(done), 2) if done else None,
           "wait_per_1k_tokens": {"cut": round(1000 * waits(cut) / toks(cut), 2) if cut else None, "finished": round(1000 * waits(done) / toks(done), 2) if done else None},
           "wait_per_answer": {"cut": round(waits(cut) / len(cut), 1) if cut else None, "finished": round(waits(done) / len(done), 1) if done else None},
           "seconds": round(time.time() - t)}
    f = f"{P['data']}/sweep_{'math' if task == 'math' else task}{len(ids)}_n{n}_cap{cap}"
    os.makedirs(P["data"], exist_ok=True)
    json.dump(out, open(f + ".json", "w"), indent=1)
    with open(f + ".jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    runs.commit()
    print("SWEEP", json.dumps(out))
    return out


@app.local_entrypoint()
def sweep(target: str = DEFAULT_TARGET, n_problems: int = 200, n: int = 1, cap: int = GEN_CAP, task: str = "math"):
    """Quick baseline of a target as downloaded: accuracy and hit-cap rate on MATH training problems (1 GPU).
    Another target is downloaded first (prep, CPU only), so this can run before prep_target.
    task other than "math": its practice problems (built first by prep_task_data if missing, CPU only)."""
    if target not in TARGETS:
        raise SystemExit(f"target must be one of {sorted(TARGETS)}")
    if task not in TASKS:
        raise SystemExit(f"task must be one of {sorted(TASKS)}")
    if target != DEFAULT_TARGET:
        prep.remote(target)
    if task != "math":
        print("task data", prep_task_data.remote(task, target))
    call = baseline_sweep.spawn(target, n_problems, n, cap, task)
    print("SPAWNED", call.object_id, f"-> {target_paths(target, task)['data']}/sweep_{task}{n_problems}_n{n}_cap{cap}.json")


@app.local_entrypoint()
def prep_target(target: str, limit: int = 0):
    """Everything a new target needs before rollouts, evaluate or rebasing, in that target's own folders:
    download (prep), the practice files copied from the default target (same splits), then on 3 GPUs (gen_base with tag ""):
    the baseline answers (512 generations) and SUMMARY.txt, the reward-set answers, the select split (part 0); the confirm
    splits (part 1); MATH-500 and gsm8k_test (part 3). --limit k: a test on the first k problems of each list, written
    under /runs/data/targets/<target>/_limit_test/ instead of the caches."""
    if target not in TARGETS or target == DEFAULT_TARGET:
        raise SystemExit(f"target must be one of {sorted(set(TARGETS) - {DEFAULT_TARGET})} (the default target is prepared by harness.target --step prep and the prep_* functions)")
    print("arch", prep.remote(target))
    print("evaluation sets", {ds: prep_eval_set.remote(ds) for ds in ("math500", "gsm8k_test")})
    calls = {k: gen_base.spawn("", k, target=target, limit=limit) for k in (0, 1, 3)}
    for k, c in calls.items():
        print("part", k, c.object_id)
    for k, c in calls.items():
        print("part", k, "done", c.get())


# ----------------------------------------------------------------------------- another task (practice, reward and held-out problems)
def _numina_ok(r):
    """A NuminaMath-CoT row the math grader can score: one \\boxed{} in the solution holding a short single answer,
    not multi-part, not a proof, not a multiple-choice letter. Returns the answer or None."""
    import re
    p, sol = r["problem"], r["solution"]
    if sol.count("\\boxed") != 1:
        return None
    a = last_boxed(sol)
    if a is None:
        return None
    a = a.strip()
    if not a or len(a) > 40 or not a.isascii() or re.search(r"[\u3000-\u9fff\uff01-\uff5e]", p):  # Chinese text or full-width punctuation left in
        return None
    if re.fullmatch(r"\(?[A-Ea-e]\)?", re.sub(r"\\text\{([^{}]*)\}|\\textbf\{([^{}]*)\}", r"\1\2", a).strip()):
        return None  # multiple choice: the answer is a letter
    if re.search(r"\\begin|\\\\|;|\\text\{[^{}]*[a-z]{4}|\b(and|or)\b", a):
        return None  # several answers, or words
    if re.search(r"(?<![\w)])\(\s*(1|I)\s*\).*(?<![\w)])\(\s*(2|II)\s*\)|\b(prove|Prove|show that|Show that)\b|[\u2160-\u2169\u2460-\u2469]", p, re.S):
        return None  # multi-part or proof
    if re.search(r"(^|\n|\s)A[:.\uff1a]\s*.*(\n|\s)B[:.\uff1a]", p, re.S):
        return None  # options listed: multiple choice
    return a


@app.function(image=base_src, volumes={"/runs": runs, "/heldout": heldout_vol}, timeout=3600, cpu=4, memory=16384)
def prep_task_data(task: str, target: str = DEFAULT_TARGET, n_total: int = 4500, n_heldout: int = 500, n_reward: int = 256, seed: int = 20261115):
    """Once per task: a fixed-seed sample of the task's problems (streamed, shuffled with a seed, filtered by _numina_ok until
    n_total rows), split into the held-out set (EVAL_SETS[eval_set], first n_heldout), the reward set (next n_reward; RL uses
    the first 128) and the practice pool (the rest), all disjoint by normalized problem text. Per target: the practice folder
    (problems.jsonl, index.tsv, splits.json with baseline/select/confirm_1/confirm_2 of 256 each and extra, README.txt).
    Skips what exists. Returns the counts."""
    import json, os, random, re
    T = TASKS[_check_task(task)]
    assert task != "math", "the math task is built by harness.target --step prep and the prep_* functions"
    heldout_vol.reload(); runs.reload()
    hd = os.path.dirname(T["reward_file"])
    pool_f = f"{hd}/practice_pool.jsonl"
    held_f = EVAL_SETS[T["eval_set"]]["file"]
    if not all(os.path.exists(f) for f in (pool_f, held_f, T["reward_file"])):
        from datasets import load_dataset
        norm = lambda x: re.sub(r"\s+", " ", x.strip().lower())
        ds = load_dataset(T["hf"], split="train", streaming=True).shuffle(seed=seed, buffer_size=20000)
        rows, seen, scanned = [], set(), 0
        for r in ds:
            scanned += 1
            if r.get("source") != T["source"]:
                continue
            a = _numina_ok(r)
            k = norm(r["problem"])
            if a is None or k in seen:
                continue
            seen.add(k)
            rows.append({"problem": r["problem"], "answer": a, "solution": r["solution"]})
            if len(rows) >= n_total:
                break
        print("scanned", scanned, "kept", len(rows))
        assert len(rows) == n_total, len(rows)
        os.makedirs(hd, exist_ok=True)
        wr = lambda f, xs: open(f, "w").write("".join(json.dumps(x) + "\n" for x in xs))
        wr(held_f, [{"qid": f"{task}_test/{i}", "problem": r["problem"], "answer": r["answer"], "source": task} for i, r in enumerate(rows[:n_heldout])])
        wr(T["reward_file"], [{"qid": f"{task}_reward/{i}", "problem": r["problem"], "answer": r["answer"]} for i, r in enumerate(rows[n_heldout:n_heldout + n_reward])])
        wr(pool_f, [{"id": f"{T['source']}/{i}", "source": T["source"], "level": "", "problem": r["problem"], "answer": r["answer"], "solution": r["solution"]}
                    for i, r in enumerate(rows[n_heldout + n_reward:])])
        json.dump({"task": task, "hf": T["hf"], "source": T["source"], "seed": seed, "buffer_size": 20000, "scanned": scanned, "n_total": n_total,
                   "heldout": n_heldout, "reward": n_reward, "practice": n_total - n_heldout - n_reward}, open(f"{hd}/meta.json", "w"), indent=1)
        heldout_vol.commit()
    pool = [json.loads(l) for l in open(pool_f)]
    d = target_paths(target, task)["practice"]
    if not os.path.exists(f"{d}/splits.json"):
        os.makedirs(d, exist_ok=True)
        open(f"{d}/problems.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in pool))
        with open(f"{d}/index.tsv", "w") as f:
            f.write("id\tsource\tlevel\tsubject\n")
            for r in pool:
                f.write(f"{r['id']}\t{r['source']}\t-\t{T['source']}\n")
        ids = [r["id"] for r in pool]
        random.Random(7).shuffle(ids)
        splits = {k: ids[256 * j: 256 * (j + 1)] for j, k in enumerate(("baseline", "select", "confirm_1", "confirm_2"))}
        splits["extra"] = ids[1024:]
        json.dump(splits, open(f"{d}/splits.json", "w"))
        open(f"{d}/README.txt", "w").write(
            f"problems.jsonl: one JSON per line with id, source ({T['source']}), level (empty), problem, answer, solution (reference solution).\n"
            f"{len(pool)} problems from the {T['source']} part of {T['hf']}, each with a single final answer. None of them appears in the held-out test.\n")
        runs.commit()
    sp = json.load(open(f"{d}/splits.json"))
    return {"practice": len(pool), **{k: len(v) for k, v in sp.items()}, "reward": sum(1 for _ in open(T["reward_file"])),
            "heldout": sum(1 for _ in open(held_f))}


@app.local_entrypoint()
def prep_task(task: str, target: str = DEFAULT_TARGET, gpus: int = 1, limit: int = 0, quick: bool = False, reward_items: int = 128, heldout: int = 0, gpu: str = "L40S"):
    """Everything a (target, task) pair needs before rollouts or evaluate --task: the sampled problems (prep_task_data,
    CPU), then the target's own answers: baseline (512 generations) and SUMMARY.txt, the reward set, select, confirm_1,
    confirm_2 and the task's held-out set. --gpus 1: all on one L40S (gen_base part 4); --gpus 3: parts 0, 1, 2 on three.
    --limit k: a test on the first k problems of each list, written under the task folder's _limit_test/.
    --gpu: the GPU type of the generation jobs (default L40S). --quick: one GPU, no confirm splits. --reward-items k: the reward set's size (evaluate and rl then
    need the same --reward-items). --heldout k: the original answers on only the first k held-out problems (evaluate --limit <= k)."""
    if task not in TASKS or task == "math":
        raise SystemExit(f"task must be one of {sorted(set(TASKS) - {'math'})}")
    if target not in TARGETS:
        raise SystemExit(f"target must be one of {sorted(TARGETS)}")
    if target != DEFAULT_TARGET:
        prep.remote(target)
    print("task data", prep_task_data.remote(task, target))
    parts = (5,) if quick else (4,) if gpus <= 1 else (0, 1, 2)
    fn = gen_base.with_options(gpu=gpu)
    calls = {k: fn.spawn("", k, reward_items=reward_items, target=target, limit=limit, task=task, test_limit=heldout) for k in parts}
    for k, c in calls.items():
        print("part", k, c.object_id)
    for k, c in calls.items():
        print("part", k, "done", c.get())

@app.function(image=rl_image, secrets=[modal.Secret.from_name("tinker")], timeout=3600)
def resave_checkpoint(state_path: str, name: str):
    """Load a Tinker training state and save it again, state and sampler weights, with no expiry."""
    import tinker
    svc = tinker.ServiceClient()
    tc = svc.create_training_client_from_state(state_path).result()
    st = tc.save_state(name).result()
    sp = tc.save_weights_for_sampler(name).result()
    return {"state": st.path, "sampler": sp.path}


@app.function(image=rl_image, secrets=[modal.Secret.from_name("tinker")], timeout=600)
def find_checkpoints(substr: str):
    """Paths of this account's Tinker checkpoints whose name contains `substr`."""
    import tinker
    rc = tinker.ServiceClient().create_rest_client()
    out = []
    for c in rc.list_user_checkpoints().result().checkpoints:
        p = getattr(c, "tinker_path", None) or getattr(c, "path", None) or str(c)
        if substr in p:
            out.append((p, str(getattr(c, "expires_at", None))))
    return out


@app.local_entrypoint()
def find(substr: str):
    for row in find_checkpoints.remote(substr):
        print(*row)


@app.local_entrypoint()
def keep(state_path: str, name: str):
    """Keep a checkpoint: re-save a training state and its sampler weights without an expiry date."""
    print(resave_checkpoint.remote(state_path, name))


@app.local_entrypoint()
def ban(configs: str, dataset: str = "math500", limit: int = 0):
    """Decode-time ban baselines. configs: JSON list of {"label", "base", "ban": [token strings]}; one GPU, one config at a time.
    Example: --configs '[{"label": "orig-wait", "base": "", "ban": [" Wait", "Wait"]}]'"""
    for r in ban_eval.remote(configs, dataset, 2, limit):
        b, p = r["banned"], r["paired_vs_original"]
        print(r["label"], "acc", b["accuracy_pct"], "cut", b["cut_at_max_tokens_pct"], "tokens", b["mean_tokens"],
              "vs original", p["diff_pp"], "+/-", p["se_pp"], "w2r", p["wrong_to_right"], "r2w", p["right_to_wrong"])
