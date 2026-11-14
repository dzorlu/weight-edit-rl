"""An editing search with an editor served on an OpenAI-compatible endpoint (default: Inkling 2 Small on Tinker).

Tool calls use a text protocol (one fenced block per reply), so the endpoint does not need function calling.
Needs a Modal secret named `tinker` holding TINKER_API_KEY.

    uvx modal run --detach -m editors.tinker_openai::oai --run-id inkling-1

Cost bounds: the target on one L40S for at most `hours`; API spend stops at `api_cap_usd`.
"""
import modal
from harness.target import app, runs, base, SRC, Target, FORWARD_BUDGET, PY_BUDGET, heldout_eval
from harness.prompts import PROMPT, GUIDE

oai_image = base.pip_install("openai").add_local_python_source(*SRC)

TINKER_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
INKLING = "thinkingmachines/Inkling-2-Small:peft:262144"  # Inkling 2 Small (594B total, 17B active), 256K context
PRICE = {"thinkingmachines/Inkling-2-Small:peft:262144": (0.30, 1.00)}  # $ per million input, output tokens
DEFAULT_PRICE = (1.00, 4.00)  # assumed upper bound when the model's price is not listed above, so the API cap still binds

PROTOCOL = """

HOW TO CALL THE TOOLS IN THIS SESSION. There is no function-calling interface. To call a tool, end your reply with exactly one fenced block:
```python
<code for the python tool>
```
or
```forward
<the JSON request for the forward tool>
```
Only the last such block in a reply is run, and its result is sent back to you as the next message. A reply with no such block that starts with FINAL SUBMISSION ends the session."""

METHODS = """

METHODS THAT WORKED FOR ANOTHER EDITOR ON THIS KIND OF TASK. These are procedures only. Which tokens, layers and units matter must come from your own measurements in this run.
1. Open with one large diagnostic generation on the original weights (256 to 512 generations in a single forward pass). Read the failures: how many are cut at the token limit, and which words or phrases are far more frequent in the cut-off texts than in the completed ones.
2. Before any analysis, split the practice problems into fixed disjoint sets and save them (for example /work/splits.json): one for diagnosis, one for choosing between edits, and one or two confirmation sets that are never used for any choice.
3. To find what produces an unwanted next token T: take the model's own failed generations, send them through a teacher-forced pass that returns mlp_act and hidden_states for many layers, and look at the positions right before T is generated. The final score of T is dot(r(T), h_final) / rms(h_final), where r(T) = E[T] * (1 + gamma), E is the embedding matrix and gamma the final norm weight (check the norm formula against the config). The MLP of layer L adds down_proj_L @ act_L to the residual stream, so unit j of layer L contributes act_L[j] * dot(r(T), down_proj_L[:, j]) / rms(h_final) to that score. Compute this for every unit of every layer, rank them, and repeat at several positions. Attention outputs can be split the same way per layer from the hidden states. Then look at when the top units are active over all positions, to learn what they respond to.
4. The same decomposition works for a wanted token that is too weak (for example the token that ends the thinking section): compare, per component, how much is written along its readout direction at positions where it should come next against ordinary positions. A readout direction specific to one token can be made by removing what it shares with the frequent tokens.
5. Edit forms that were useful: multiplying one down_proj column by a factor (try 0, -1, -2), and a rank-one gain W + alpha * outer(d, d @ W) along a unit readout direction d on a chosen set of writer matrices (MLP down_proj and attention output projections), with alpha swept.
6. Every candidate is tested causally with a paired generation ("model": "both") on at least 256 generations. Sweep the strength for a dose-response. For a combined edit, also measure each part alone. Check side effects in the generated texts (for example thinking that closes after very few tokens).
7. Confirm the final candidate on the untouched confirmation sets, keep /work/notes.md current, and copy each good edit to its own folder.
8. Forward passes are scarce: use few of them with large batches, and never spend one on fewer than 64 generations."""


def parse_block(text):
    import re
    m = list(re.finditer(r"```(python|forward)[^\n]*\n(.*?)```", text, re.S))
    return (m[-1].group(1), m[-1].group(2).strip()) if m else None


@app.function(image=oai_image, volumes={"/runs": runs}, secrets=[modal.Secret.from_name("tinker")], timeout=int(7.5 * 3600), nonpreemptible=True)
def run_search_oai(run_id: str, model: str = INKLING, base_url: str = TINKER_URL, key_env: str = "TINKER_API_KEY", effort: str = "xhigh",
                   forward_budget: int = FORWARD_BUDGET, hours: float = 4.0, max_turns: int = 420, api_cap_usd: float = 15.0, restart_at: int = 180000, methods: bool = True, guide: bool = False):
    import json, os, time
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=os.environ[key_env], timeout=1800.0, max_retries=3)
    t0 = time.time()
    deadline = t0 + hours * 3600
    d = f"/runs/{run_id}"
    if os.path.exists(f"{d}/transcript.jsonl"):
        raise RuntimeError(f"{d}/transcript.jsonl exists: this run was already started (a restart after an interruption would begin a second search on top of it). Use a new run id.")
    os.makedirs(d, exist_ok=True)
    tgt = Target()
    if guide:  # one consolidated section replaces WHAT COUNTS and the methods list
        a, b = PROMPT.index("WHAT COUNTS."), PROMPT.index("WHAT YOU HAVE.")
        body = PROMPT[:a] + GUIDE + "\n\n" + PROMPT[b:]
    else:
        body = PROMPT + (METHODS if methods else "")
    first = {"role": "user", "content": body + PROTOCOL}
    messages = [first]
    used_f = used_py = turn = nags = restarts = fails = 0
    spent = 0.0
    final_text, ended, asked_final = None, "turn limit", False
    tlog = open(f"{d}/transcript.jsonl", "a")
    pin, pout = PRICE.get(model, DEFAULT_PRICE)

    def log(rec):
        tlog.write(json.dumps(rec) + "\n"); tlog.flush()

    def status():
        left = max(0, int((deadline - time.time()) / 60))
        return f"\n\n[forward passes used: {used_f}/{forward_budget}; python runs used: {used_py}/{PY_BUDGET}; about {left} minutes left]"

    def must_finish():
        return used_f >= forward_budget or time.time() > deadline or spent >= api_cap_usd

    while turn < max_turns:
        turn += 1
        try:
            r = client.chat.completions.create(model=model, messages=messages, max_tokens=16000, reasoning_effort=effort)
            fails = 0
        except Exception as e:
            log({"turn": turn, "error": repr(e)[:800]})
            fails += 1
            if fails >= 6:
                ended = "editor request kept failing"; break
            time.sleep(60); continue
        ch = r.choices[0]
        text = ch.message.content or ""
        reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None) or ""
        ptoks, ctoks = r.usage.prompt_tokens, r.usage.completion_tokens
        spent += (ptoks * pin + ctoks * pout) / 1e6
        call = parse_block(text)
        log({"turn": turn, "t": round(time.time() - t0), "prompt_tokens": ptoks, "completion_tokens": ctoks, "finish": ch.finish_reason,
             "reasoning": "\n\n".join(x for x in (reasoning, text if call else "") if x), "final": "" if call else text,
             "call": call[0] if call else None, "api_usd": round(spent, 2)})
        messages.append({"role": "assistant", "content": text or "(empty reply)"})
        if call is None:
            if asked_final or must_finish() or "FINAL SUBMISSION" in text:
                final_text = text
                ended = "editor finished" if not must_finish() else ("budget used" if used_f >= forward_budget else "time or API cap")
                break
            nags += 1
            if nags >= 5:
                final_text, ended = text, "no tool call in five replies"; break
            note = "[Your reply was cut off at the length limit before a tool block. Reply again, more briefly.]" if ch.finish_reason == "length" else \
                "[No ```python or ```forward block was found. The session is still running. End your reply with one such block, or finish with a reply that starts with FINAL SUBMISSION.]"
            messages.append({"role": "user", "content": note + status()})
            continue
        nags = 0
        kind, body = call
        if kind == "python":
            if used_py >= PY_BUDGET:
                res = "[python budget exhausted]"
            else:
                used_py += 1
                res = tgt.py.remote(body)
        else:
            if used_f >= forward_budget or time.time() > deadline:
                res = "[no forward passes left]"
            else:
                ok, res = tgt.forward.remote(body)
                if ok:
                    used_f += 1
                else:
                    res = "[request failed; it does not count against the budget]\n" + res
        log({"turn": turn, "tool": kind, "input": body[:20000], "result": res[:20000], "used_f": used_f, "used_py": used_py})
        extra = ""
        if must_finish() and not asked_final:
            asked_final = True
            extra = "\n\n[The budget or the time limit is reached. You may still run a few python blocks to set /work/edited to your final choice (drop_edit everything to submit the original weights). Then reply without a block, starting with FINAL SUBMISSION: what you changed, the evidence, and why you expect it to help.]"
        messages.append({"role": "user", "content": f"[result of the {kind} tool]\n" + res + status() + extra})
        if asked_final and time.time() > deadline + 20 * 60:
            ended = "time limit (hard)"; break
        if ptoks > restart_at:
            restarts += 1
            notes = tgt.py.remote("import os\nprint(open('/work/notes.md').read()[-8000:] if os.path.exists('/work/notes.md') else 'no notes file yet')")
            messages = [first, {"role": "assistant", "content": "(earlier turns removed)"},
                        {"role": "user", "content": "[The conversation was restarted because it grew too long. Kernel variables and files in /work are intact. Contents of /work/notes.md:]\n" + notes + status()}] + messages[-4:]
        if turn % 10 == 0:
            runs.commit()
    tlog.close()
    runs.commit()
    files = tgt.snapshot.remote(run_id)
    summary = {"run_id": run_id, "editor": model, "effort": effort, "ended": ended, "turns": turn, "forward_used": used_f, "python_used": used_py,
               "restarts": restarts, "seconds": round(time.time() - t0), "methods_in_prompt": methods, "guide_in_prompt": guide, "api_usd": round(spent, 2), "final_text": final_text, "work_files": files[:200]}
    with open(f"{d}/search_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    runs.commit()
    print("SEARCH DONE", json.dumps({k: v for k, v in summary.items() if k not in ("final_text", "work_files")}))
    return summary


@app.function(image=oai_image, volumes={"/runs": runs}, timeout=int(9.5 * 3600))
def pipeline_oai(run_id: str, model: str, effort: str, forward_budget: int, hours: float, api_cap_usd: float, methods: bool = True, guide: bool = False):
    s = run_search_oai.remote(run_id, model, TINKER_URL, "TINKER_API_KEY", effort, forward_budget, hours, 420, api_cap_usd, 180000, methods, guide)
    h = heldout_eval.remote(run_id)
    return {"search": {k: v for k, v in s.items() if k != "work_files"}, "heldout": h}


@app.local_entrypoint()
def oai(run_id: str = "inkling2s-qwen08-math-1", model: str = INKLING, effort: str = "xhigh",
        forward_budget: int = FORWARD_BUDGET, hours: float = 4.0, api_cap_usd: float = 15.0, methods: bool = True, guide: bool = False):
    call = pipeline_oai.spawn(run_id, model, effort, forward_budget, hours, api_cap_usd, methods, guide)
    print("SPAWNED", call.object_id)
