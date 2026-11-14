"""An editing search with a Claude model as the editor (Messages API).

Needs a Modal secret named `anthropic` holding ANTHROPIC_API_KEY.

    uvx modal run --detach -m editors.claude::api --run-id opus-1

Cost bounds: the target on one L40S for at most `hours`; API spend stops at `api_cap_usd`.
"""
import modal
from harness.target import app, runs, base, SRC, Target, FORWARD_BUDGET, PY_BUDGET, heldout_eval
from harness.prompts import PROMPT

api_image = base.pip_install("anthropic").add_local_python_source(*SRC)

# $ per million tokens: input, 1-hour cache write (2x input), cache read, output
PRICE = {"claude-opus-5-5": (4.0, 8.0, 0.20, 20.0)}

API_TOOLS = [
    {"name": "python", "description": "Run code in the persistent Python kernel. If the last line is an expression, its value is shown.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}}, "required": ["code"]}},
    {"name": "forward", "description": "One forward pass on the target model (generation or teacher-forced). A successful call uses one of the 50 forward passes.",
     "input_schema": {"type": "object", "properties": {"request": {"type": "object", "description": "The JSON request, as described in the task"}}, "required": ["request"]}},
]


@app.function(image=api_image, volumes={"/runs": runs}, secrets=[modal.Secret.from_name("anthropic")], timeout=int(7.5 * 3600), nonpreemptible=True)
def run_search_api(run_id: str, model: str = "claude-opus-5-5", effort: str = "xhigh", forward_budget: int = FORWARD_BUDGET,
                   hours: float = 6.0, max_turns: int = 420, api_cap_usd: float = 40.0):
    import json, os, time
    import anthropic
    client = anthropic.Anthropic(timeout=1200.0, max_retries=4)
    t0 = time.time()
    deadline = t0 + hours * 3600
    d = f"/runs/{run_id}"
    if os.path.exists(f"{d}/transcript.jsonl"):
        raise RuntimeError(f"{d}/transcript.jsonl exists: this run was already started (a restart after an interruption would begin a second search on top of it). Use a new run id.")
    os.makedirs(d, exist_ok=True)
    tgt = Target()
    messages = [{"role": "user", "content": PROMPT}]
    used_f = used_py = turn = nags = 0
    spent = 0.0
    final_text, ended, asked_final = None, "turn limit", False
    tlog = open(f"{d}/transcript.jsonl", "a")

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
            r = client.messages.create(
                model=model, max_tokens=16000, messages=messages, tools=API_TOOLS,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": effort},
                cache_control={"type": "ephemeral", "ttl": "1h"},
            )
        except anthropic.RateLimitError as e:
            log({"turn": turn, "error": "rate limit: " + repr(e)[:300]}); time.sleep(120); continue
        except anthropic.APIStatusError as e:
            log({"turn": turn, "error": repr(e)[:800]})
            if e.status_code >= 500:
                time.sleep(60); continue
            ended = f"API error {e.status_code}"; break
        except anthropic.APIConnectionError as e:
            log({"turn": turn, "error": repr(e)[:300]}); time.sleep(60); continue
        u = r.usage
        pin, pw, pr, pout = PRICE[model]
        cw, cr = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
        spent += (u.input_tokens * pin + cw * pw + cr * pr + u.output_tokens * pout) / 1e6
        reasoning = "\n\n".join(b.thinking for b in r.content if b.type == "thinking" and b.thinking)
        text = "\n\n".join(b.text for b in r.content if b.type == "text")
        calls = [b for b in r.content if b.type == "tool_use"]
        log({"turn": turn, "t": round(time.time() - t0), "prompt_tokens": u.input_tokens + cw + cr, "completion_tokens": u.output_tokens,
             "finish": r.stop_reason, "reasoning": "\n\n".join(x for x in (reasoning, text if calls else "") if x),
             "final": "" if calls else text, "call": calls[0].name if calls else None, "served_by": r.model, "api_usd": round(spent, 2)})
        if r.stop_reason == "refusal":
            ended = "refusal: " + str(getattr(r.stop_details, "category", None)); break
        messages.append({"role": "assistant", "content": r.content})
        if not calls:
            if asked_final or must_finish() or "FINAL SUBMISSION" in text:
                final_text = text
                ended = "editor finished" if not must_finish() else ("budget used" if used_f >= forward_budget else "time or API cap")
                break
            nags += 1
            if nags >= 4:
                final_text, ended = text, "no tool call in four replies"; break
            note = "[Your reply was cut off at the length limit. Reply again, more briefly.]" if r.stop_reason == "max_tokens" else \
                "[No tool call was found. The session is still running. Call python or forward, or finish with a reply that starts with FINAL SUBMISSION.]"
            messages.append({"role": "user", "content": note + status()})
            continue
        nags = 0
        results = []
        for c in calls:
            inp = c.input if isinstance(c.input, dict) else {}
            if r.stop_reason == "max_tokens":
                kind, body, res = c.name, json.dumps(inp)[:2000], "[the call was cut off at the length limit and was not run; send it again, shorter]"
            elif c.name == "python":
                body = inp.get("code", "")
                if used_py >= PY_BUDGET:
                    res = "[python budget exhausted]"
                else:
                    used_py += 1
                    res = tgt.py.remote(body)
                kind = "python"
            else:
                req = inp.get("request", inp)
                body = req if isinstance(req, str) else json.dumps(req)
                if used_f >= forward_budget or time.time() > deadline:
                    res = "[no forward passes left]"
                else:
                    ok, res = tgt.forward.remote(body)
                    if ok:
                        used_f += 1
                    else:
                        res = "[request failed; it does not count against the budget]\n" + res
                kind = "forward"
            log({"turn": turn, "tool": kind, "input": body[:20000], "result": res[:20000], "used_f": used_f, "used_py": used_py})
            extra = ""
            if must_finish() and not asked_final:
                asked_final = True
                extra = "\n\n[The budget or the time limit is reached. You may still run a few python calls to set /work/edited to your final choice (drop_edit everything to submit the original weights). Then reply without a tool call, starting with FINAL SUBMISSION: what you changed, the evidence, and why you expect it to help.]"
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": res + status() + extra})
        messages.append({"role": "user", "content": results})
        if asked_final and time.time() > deadline + 20 * 60:
            ended = "time limit (hard)"; break
        if turn % 10 == 0:
            runs.commit()
    tlog.close()
    runs.commit()
    files = tgt.snapshot.remote(run_id)
    summary = {"run_id": run_id, "editor": model, "effort": effort, "ended": ended, "turns": turn, "forward_used": used_f, "python_used": used_py,
               "restarts": 0, "seconds": round(time.time() - t0), "api_usd": round(spent, 2), "final_text": final_text, "work_files": files[:200]}
    with open(f"{d}/search_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    runs.commit()
    print("SEARCH DONE", json.dumps({k: v for k, v in summary.items() if k not in ("final_text", "work_files")}))
    return summary


@app.function(image=api_image, volumes={"/runs": runs}, timeout=int(9.5 * 3600))
def pipeline_api(run_id: str, model: str, effort: str, forward_budget: int, hours: float, api_cap_usd: float):
    s = run_search_api.remote(run_id, model, effort, forward_budget, hours, 420, api_cap_usd)
    h = heldout_eval.remote(run_id)
    return {"search": {k: v for k, v in s.items() if k != "work_files"}, "heldout": h}


@app.local_entrypoint()
def api(run_id: str = "opus55-qwen08-math-1", model: str = "claude-opus-5-5", effort: str = "xhigh",
         forward_budget: int = FORWARD_BUDGET, hours: float = 6.0, api_cap_usd: float = 40.0):
    call = pipeline_api.spawn(run_id, model, effort, forward_budget, hours, api_cap_usd)
    print("SPAWNED", call.object_id)
