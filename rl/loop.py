"""RL loop: train Inkling 2 Small (on Tinker) to be a better weight editor of Qwen3.5-0.8B (on Modal).

One RL step: sample a group of rollouts with the current checkpoint, score each submitted edit,
take one policy-gradient step on Tinker, save a checkpoint.

    uvx modal run -m rl.loop::prep_reward_set          # once
    uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32

Cost bounds: each rollout holds one L40S for at most `rollout_minutes`; the whole run stops at `hours`.
Outputs: /runs/rl/<exp>/ on the runs volume (steps.jsonl, one transcript per rollout).
"""
import modal
from harness.target import app, runs, base, base_src, SRC, heldout_vol, Target, PY_BUDGET, GEN_CAP, last_boxed
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
WEB_BUDGET = 20
# Pages that publish ready-made edits for this very model would let the editor copy an answer; they are kept out.
WEB_EXCLUDE = ["raw.githubusercontent.com/RikiyaT", "rikiyat.github.io", "github.com/RikiyaT", "huggingface.co/RikiyaT", "github.com/WSP-Scope", "arxiv.org/abs/2611.09382", "arxiv.org/pdf/2611.09382", "arxiv.org/html/2611.09382"]
WEB_BLOCK = ("weight-space programming", "weight space programming", "weight-space-programming", "2611.09382", "qwen3", "qwen35", "qwen 3", "qwen-3", "gpt-6 astra", "astra", "rikiyat", "lm_understand", "wsp-scope", "math weight editing")
WEB_TEXT = """

WEB SEARCH. You also have a `web_search` tool with the parameter `query` (at most 20 searches). It returns titles, links and short excerpts. It is for methods and background, such as how an interpretability technique is computed. Results about this specific model or about published edits to it are filtered out. The python kernel itself still has no internet access."""


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


def repair_reply(raw):
    """Read a reply the renderer could not parse: (thinking, tool name, arguments) or None.

    The usual cause is tool-call JSON that is short of closing braces."""
    import json, re
    m = re.search(r"<\|message_model\|>([\w.]+)<\|content_invoke_tool_json\|>(.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>|$)", raw, re.S)
    if not m:
        return None
    think = "\n\n".join(t.strip() for t in re.findall(r"<\|content_thinking\|>(.*?)<\|end_message\|>", raw, re.S))
    js = m.group(2).strip()
    for extra in ("", "}", "}}", "}}}", "]}", "]}}", '"}', '"}}'):
        try:
            obj = json.loads(js + extra)
        except Exception:
            continue
        if isinstance(obj, dict):
            args = obj.get("args", obj.get("arguments", obj))
            return think, obj.get("name") or m.group(1), args if isinstance(args, dict) else {}
    return None


def editor_prompt(forward_budget, web=False):
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
    if web:
        b = body.index("HOW TO WORK.")
        body = body[:b].rstrip() + WEB_TEXT + "\n\n" + body[b:]
    return body


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


@app.function(image=rl_image, volumes={"/runs": runs, "/heldout": heldout_vol}, secrets=[modal.Secret.from_name("tinker"), modal.Secret.from_name("exa")],
              timeout=int(11.5 * 3600), nonpreemptible=True, cpu=4, memory=16384)
async def rl_run(exp: str, steps: int = 2, group_size: int = 4, forward_budget: int = 32, rollout_minutes: int = 120, hours: float = 10.0,
                 reward_items: int = 128, reward_n: int = 2, effort: float = 0.9, lr: float = 2e-5, max_turns: int = 400, start_from: str = "", web: bool = False):
    import asyncio, json, os, time
    import tinker, torch
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer, get_text_content, ToolCall
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Transition, Trajectory
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    d = f"/runs/rl/{exp}"
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
    if start_from:
        tc = await asyncio.to_thread(lambda: svc.create_training_client_from_state(start_from).result())
    else:
        tc = await asyncio.to_thread(lambda: svc.create_lora_training_client(base_model=MODEL, rank=32).result())
    items = [json.loads(l) for l in open("/heldout/rl/reward_math.jsonl")][:reward_items]
    items_json = json.dumps(items)
    prompt_text = editor_prompt(forward_budget, web)
    tools = TOOLS + ([WEB_TOOL] if web else [])
    json.dump({"exp": exp, "model": MODEL, "steps": steps, "group_size": group_size, "forward_budget": forward_budget, "rollout_minutes": rollout_minutes,
               "reward_items": len(items), "reward_n": reward_n, "effort": effort, "lr": lr, "web": web, "prompt": prompt_text}, open(f"{d}/config.json", "w"), indent=1)

    def slog(rec):
        with open(f"{d}/steps.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        runs.commit()
        print("STEP", json.dumps(rec)[:1500])

    # The original weights' answers on the reward set are the same for every rollout: generate them once.
    t = time.time()
    base_rows = (await Target.with_options(scaledown_window=120)(rid=f"{exp}-base").reward.remote.aio(items_json, reward_n, GEN_CAP, True))["original"]
    base_ok = {(r["qid"], r["sample"]): r["correct"] for r in base_rows}
    base_acc = sum(base_ok.values()) / len(base_ok)
    slog({"event": "reward_base", "answers": len(base_ok), "accuracy": round(base_acc, 4), "hit_cap": round(sum(r["finish"] == "length" for r in base_rows) / len(base_rows), 4), "seconds": round(time.time() - t)})

    async def rollout(sampler, rid):
        t0 = time.time()
        deadline = min(t0 + rollout_minutes * 60, run_deadline)
        tgt = Target.with_options(scaledown_window=300)(rid=rid)
        messages = renderer.create_conversation_prefix_with_tools(tools, system_prompt="")  # Inkling's own tool-call format
        messages.append({"role": "user", "content": prompt_text})
        transitions, used_f, used_py, used_web, nags, turn, cutoffs = [], 0, 0, 0, 0, 0, 0
        tok = dict(prefill=0, cached=0, sampled=0)
        asked_final, ended, final_text = False, "turn limit", None
        log = open(f"{d}/{rid}.jsonl", "a")
        w = lambda rec: (log.write(json.dumps(rec) + "\n"), log.flush())
        status = lambda: f"\n\n[forward passes used: {used_f}/{forward_budget}; python runs used: {used_py}/{PY_BUDGET}; about {max(0, int((deadline - time.time()) / 60))} minutes left]"
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
                        res = await tgt.py.remote.aio(body)
                elif kind == "forward":
                    body = forward_request(args)
                    if must_finish():
                        res = "[no forward passes left]"
                    else:
                        ok, res = await tgt.forward.remote.aio(body)
                        if ok:
                            used_f += 1
                        else:
                            res = "[request failed; it does not count against the budget]\n" + res
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
                w({"turn": turn, "tool": kind, "input": body[:20000], "result": res[:20000], "used_f": used_f, "used_py": used_py})
                extra = ""
                if must_finish() and not asked_final:
                    asked_final = True
                    extra = "\n\n[The budget or the time limit is reached. You may still run a few python calls to set /work/edited to your final choice (drop_edit everything to submit the original weights). Then reply without a tool call, starting with FINAL SUBMISSION: what you changed, the evidence, and why you expect it to help.]"
                messages.append({"role": "tool", "tool_call_id": tc.id or "call_1", "name": tc.function.name, "content": res + status() + extra})
                for other in tcs[1:]:
                    messages.append({"role": "tool", "tool_call_id": other.id or "call_x", "name": other.function.name, "content": "[not run: only the first tool call of a reply is run]"})
                if asked_final and time.time() > deadline + 15 * 60:
                    ended = "time limit (hard)"; break
            # reward: the submitted edit against the cached original answers, paired by problem and sample
            rw = await tgt.reward.remote.aio(items_json, reward_n, GEN_CAP, False)
            if rw["edited"] is None:
                reward, acc_e, cap_e = 0.0, None, None
            else:
                ok_e = {(r["qid"], r["sample"]): r["correct"] for r in rw["edited"]}
                acc_e = sum(ok_e.values()) / len(ok_e)
                reward = sum(float(ok_e[k]) - float(base_ok[k]) for k in ok_e) / len(ok_e)
                cap_e = sum(r["finish"] == "length" for r in rw["edited"]) / len(rw["edited"])
            try:
                await tgt.snapshot.remote.aio(f"rl/{exp}/work/{rid}")
            except Exception as e:
                w({"snapshot_error": repr(e)[:300]})
            usd = (tok["prefill"] * PRICE["prefill"] + tok["cached"] * PRICE["cached"] + tok["sampled"] * PRICE["sample"]) / 1e6
            info = {"rid": rid, "ok": True, "reward": round(reward, 4), "edited_accuracy": acc_e, "edited_hit_cap": cap_e, "edited_tensors": rw["edited_tensors"], "ended": ended,
                    "turns": turn, "forward_used": used_f, "python_used": used_py, "web_used": used_web, "minutes": round((time.time() - t0) / 60, 1), "tokens": tok, "sampling_usd": round(usd, 3)}
            w({"summary": info, "final_text": final_text})
            return Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty()), info
        except Exception as e:
            w({"error": repr(e)[:1000]})
            return None, {"rid": rid, "ok": False, "error": repr(e)[:400], "turns": turn, "minutes": round((time.time() - t0) / 60, 1)}
        finally:
            log.close()

    adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    for step in range(steps):
        if time.time() > run_deadline - 20 * 60:
            slog({"event": "stopped", "reason": "run time limit", "step": step}); break
        t = time.time()
        sampler = await asyncio.to_thread(lambda: tc.save_weights_and_get_sampling_client().result())
        out = await asyncio.gather(*[rollout(sampler, f"{exp}-s{step}-r{i}") for i in range(group_size)])
        good = [(tr, info) for tr, info in out if info["ok"]]
        rewards = [info["reward"] for _, info in good]
        rec = {"event": "step", "step": step, "rollouts": [info for _, info in out], "rewards": rewards,
               "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None, "rollout_minutes": round((time.time() - t) / 60, 1)}
        if len(set(rewards)) >= 2:
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            adv = [(r - mean) / (std + 1e-6) for r in rewards]  # centred and scaled within the group
            t1, n_tok, futs = time.time(), 0, []
            for (tr, _), a in zip(good, adv):
                for dm in trajectory_to_data(tr, float(a), include_provenance=False):
                    dm.loss_fn_inputs.pop("mask", None)  # importance_sampling takes target_tokens, logprobs, advantages
                    n_tok += dm.model_input.length
                    futs.append(tc.forward_backward([dm], loss_fn="importance_sampling"))
            opt = tc.optim_step(adam)
            metrics = await asyncio.to_thread(lambda: [f.result().metrics for f in futs])
            await asyncio.to_thread(lambda: opt.result())
            rec.update(trained=True, advantages=[round(a, 3) for a in adv], training_sequences=len(futs), training_tokens=n_tok,
                       training_usd=round(n_tok * PRICE["train"] / 1e6, 2), loss_sum=round(sum(m.get("loss:sum", 0.0) for m in metrics), 3), training_seconds=round(time.time() - t1))
        else:
            rec.update(trained=False, reason="all rewards equal or fewer than two rollouts finished; no gradient")
        ck = await asyncio.to_thread(lambda: tc.save_weights_for_sampler(f"{exp}-step{step + 1}").result())
        st = await asyncio.to_thread(lambda: tc.save_state(f"{exp}-step{step + 1}").result())
        rec.update(sampler_checkpoint=ck.path, state_checkpoint=st.path, minutes_total=round((time.time() - T0) / 60, 1))
        slog(rec)
    slog({"event": "done", "minutes_total": round((time.time() - T0) / 60, 1)})
    return True


@app.local_entrypoint()
def rl(exp: str, steps: int = 2, group_size: int = 4, forward_budget: int = 32, rollout_minutes: int = 120, hours: float = 10.0,
       reward_items: int = 128, effort: float = 0.9, lr: float = 2e-5, start_from: str = "", web: bool = False):
    call = rl_run.spawn(exp, steps, group_size, forward_budget, rollout_minutes, hours, reward_items, 2, effort, lr, 400, start_from, web)
    print("SPAWNED", call.object_id)
