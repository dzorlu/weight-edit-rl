"""Did hero3's first RL update move the editor in the direction of the advantages, and by how much?

Rebuilds each hero3 step-1 rollout as rl/loop.py rendered it (same renderer, tool declaration, prompt,
status lines, effort 0.9), then scores the editor's sampled tokens with Tinker's compute_logprobs under
the pre-update policy (a fresh rank-32 LoRA, made as rl/loop.py made it) and under hero3-step1.
No sampling, no training.

    uvx --python 3.12 --with tinker-cookbook --with openai python scripts/update_check.py --check   # local: rebuild only, compare lengths with the logs
    uvx modal run scripts/update_check.py::main                    # Modal (CPU): rebuild and score, writes results/rl/hero3/update_check.json
    uvx modal run scripts/update_check.py::main --variant nolib    # also score with nolib1's prompt and tools (no web section)
    uvx modal run scripts/update_check.py::noise_check             # repeat scoring and a second fresh LoRA (noise floor)
    uvx modal run scripts/update_check.py::sample_check            # compute_logprobs against the sampler's own log-probabilities
    uvx modal run scripts/update_check.py::info                    # Tinker's record of hero3's training run and checkpoints
"""
import ast, json, os, re, sys

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERO = os.path.join(REPO, "results/rl/hero3")
NOLIB = os.path.join(REPO, "results/rl/nolib1")
MODEL = "thinkingmachines/Inkling-2-Small:peft:262144"
POST = "tinker://2ca0a9ef-7226-5b53-a9d0-673f54ba489d:train:0/sampler_weights/hero3-step1"
DEBUG = False
EFFORT, FORWARD_BUDGET, PY_BUDGET, WEB_BUDGET, ROLLOUT_S = 0.9, 32, 300, 20, 120 * 60

# The same image definition as rl/loop.py's rl_image, so Modal reuses the cached build (same package versions).
image = modal.Image.debian_slim(python_version="3.12").pip_install("tinker-cookbook", "openai")
app = modal.App("weight-edit-update-check")


def loop_defs():
    """TOOLS, WEB_TOOL, WEB_READ_TOOL and forward_request's RETURN_KEYS, taken from rl/loop.py's source."""
    src = open(os.path.join(REPO, "rl/loop.py")).read()
    ns = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("TOOLS_TEMPLATE", "WEB_TOOL", "WEB_READ_TOOL", "RETURN_KEYS") for t in node.targets):
            exec(compile(ast.Module([node], []), "loop.py", "exec"), ns)
    # the tools as the hero3 and nolib1 runs had them (target Qwen3.5-0.8B: 24 layers)
    ns["TOOLS"] = json.loads(json.dumps(ns["TOOLS_TEMPLATE"]).replace("<<HS>>", "0..24").replace("<<MLP>>", "0..23"))
    return ns


def load_inputs():
    d = loop_defs()
    hero_prompt = json.load(open(f"{HERO}/config.json"))["prompt"]
    nolib_prompt = json.load(open(f"{NOLIB}/config.json"))["prompt"]
    steps = [json.loads(l) for l in open(f"{HERO}/steps.jsonl")]
    s0 = next(r for r in steps if r.get("event") == "step" and r["step"] == 0)
    rolls = {f"r{i}": [json.loads(l) for l in open(f"{HERO}/hero3-s0-r{i}.jsonl")] for i in range(8)}
    return dict(tools=d["TOOLS"], web_tools=[d["WEB_TOOL"], d["WEB_READ_TOOL"]], hero_prompt=hero_prompt, nolib_prompt=nolib_prompt,
                advantages=s0["advantages"], rewards=s0["rewards"], entropy=s0["entropy"], action_tokens=s0["action_tokens"], rolls=rolls)


def unforward(inp):
    """Invert rl/loop.py forward_request: the tool's flat arguments from the logged request JSON."""
    try:
        obj = json.loads(inp)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    obj.update(obj.pop("return", None) or {})
    return obj


def rebuild(recs, prompt_text, tools, web):
    """Replay one rollout's message history as rl/loop.py built it.

    Returns the final rendered sequence and, per editor turn, the position of its sampled tokens in that
    sequence (TMLv0 renders each message independently, so every turn's prompt is a prefix of the next)."""
    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer, ToolCall
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    base = MODEL.split(":")[0]
    tokenizer = get_tokenizer(base)
    renderer = get_renderer(model_info.get_recommended_renderer_name(base), tokenizer)
    render = lambda m: renderer.build_generation_prompt(m, effort=EFFORT).to_ints()

    tt = tokenizer.tml_tokenizer
    names = sorted(tt.special_tokens(), key=len, reverse=True)
    pat = re.compile("(" + "|".join(re.escape(f"<|{n}|>") for n in names) + ")")
    senc = lambda text: [t for piece in pat.split(text) if piece for t in ([tt.encode_special(piece[2:-2])] if pat.fullmatch(piece) else tt.encode_ordinary(piece))]
    kind_of = lambda s: s["call"].split(".")[-1] if s["call"] else None
    def unsorted(tr):
        o = unforward(tr["input"]) if tr else None
        if not o or "args" in o:
            return False
        own = [k for k in o if k not in ("token_logprobs", "topk", "hidden_states", "mlp_act")]  # forward_request moved these to the end
        return own != sorted(own)
    samples = [r for r in recs if "prompt_tokens" in r]
    tools_by_turn = {r["turn"]: r for r in recs if "tool" in r}
    messages = renderer.create_conversation_prefix_with_tools(tools, system_prompt="")
    messages.append({"role": "user", "content": prompt_text})
    used_f = used_py = used_web = 0
    turns, notes = [], []
    for j, s in enumerate(samples):
        k = s["turn"]
        ob = render(messages)
        # The assistant message as rl/loop.py appended it (parsed, or rebuilt by repair_reply)
        tr = tools_by_turn.get(k)
        if s["call"]:
            name = s["call"]
            kind = name.split(".")[-1]
            if kind == "python":
                args = {"code": tr["input"]}
            elif kind == "forward":
                args = unforward(tr["input"]) or {}
            elif kind == "web_search":
                args = {"query": tr["input"]}
            elif kind == "web_read":
                args = {"url": tr["input"]}
            else:
                args = {}
            tc = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)), id=f"functions.{kind}:0")
            # The log joins thinking and a text part with a blank line; try thinking only, then a split at the last blank line.
            options = [[{"type": "thinking", "thinking": s["reasoning"]}] if s["reasoning"] else ""]
            if "\n\n" in s["reasoning"]:
                a, b = s["reasoning"].rsplit("\n\n", 1)
                options.append([{"type": "thinking", "thinking": a}, {"type": "text", "text": b}])
            msgs = [{"role": "assistant", "content": c, "tool_calls": [tc]} for c in options]
        else:
            content = ([{"type": "thinking", "thinking": s["reasoning"]}] if s["reasoning"] else []) + ([{"type": "text", "text": s["final"]}] if s["final"] else [])
            msgs = [{"role": "assistant", "content": content}]
        for msg in msgs:
            full = render(messages + [msg])
            if len(full) - len(ob) == s["completion_tokens"]:
                break
        else:
            msg, full = msgs[0], render(messages + [msgs[0]])
        messages.append(msg)
        assert full[:len(ob)] == ob
        act = full[len(ob):]
        # Tokens the editor actually sampled, where they differ from how its reply is rendered back into the history
        raw, how = act, "rendered"
        if len(act) != s["completion_tokens"] or (kind_of(s) == "forward" and unsorted(tr)):
            txt = tokenizer.decode(act)
            cands = []
            if kind_of(s) == "forward" and unsorted(tr):  # the editor's own key order (tool-call JSON is re-rendered with sorted keys)
                o = unforward(tr["input"])
                srt = json.dumps({"name": s["call"], "args": dict(sorted(o.items()))}, separators=(",", ":"), ensure_ascii=False)
                if srt in txt:
                    cands.append(("key order", senc(txt.replace(srt, json.dumps({"name": s["call"], "args": o}, separators=(",", ":"), ensure_ascii=False)))))
            if not s["call"] and "<|" in s["final"]:  # malformed reply: the history holds its decoded text
                th = "<|message_model|><|content_thinking|>" + s["reasoning"] + "<|end_message|>" if s["reasoning"] else ""
                cands += [("special tokens", senc(th + s["final"])), ("special tokens", senc(th + "<|message_model|><|content_text|>" + s["final"])),
                          ("special tokens", senc(th + s["final"] + "<|end_message|><|content_model_end_sampling|>"))]
            if DEBUG:
                print("    candidates turn", k, [(nm, len(c) - s["completion_tokens"]) for nm, c in cands])
            for nm, c in cands:
                if len(c) == s["completion_tokens"]:
                    raw, how = c, nm
                    break
            else:
                how = "rendered (length differs)" if len(act) != s["completion_tokens"] else "rendered (key order unknown)"
        kind = s["call"].split(".")[-1] if s["call"] else None
        turns.append(dict(turn=k, start=len(ob), n=len(raw), nr=len(act), raw=None if raw is act else raw, how=how, logged_prompt=s["prompt_tokens"], logged_n=s["completion_tokens"],
                          kind=kind, gen=bool(kind == "forward" and '"generate"' in tr["input"]),
                          save=bool(kind == "python" and "save_edit(" in tr["input"]),
                          final=(s.get("final") or "").lstrip().startswith("FINAL"), repaired=s["repaired"]))
        # what followed the reply (rl/loop.py lines 431-523)
        ctx = s["prompt_tokens"] + s["completion_tokens"]
        nxt = samples[j + 1] if j + 1 < len(samples) else None
        if nxt is None:
            break
        t_status = nxt["t"] - nxt["completion_tokens"] / 40.0  # not logged: time of the status line, estimated
        def status():
            parts = [f"turn {k}", f"forward passes left: {max(0, FORWARD_BUDGET - used_f)} of {FORWARD_BUDGET}", f"python runs left: {max(0, PY_BUDGET - used_py)}"]
            if web:
                parts.append(f"web calls left: {max(0, WEB_BUDGET - used_web)}")
            parts += [f"minutes left: about {max(0, int((ROLLOUT_S - t_status) / 60))}", f"context left: about {max(0, 235000 - ctx) // 1000}k tokens"]
            return "\n\n[" + "; ".join(parts) + "]"
        if s["finish"] != "stop":
            messages.append({"role": "user", "content": "[Your reply was cut off at the length limit. Reply again, briefly, with one tool call.]" + status()})
            continue
        if not s["call"]:
            messages.append({"role": "user", "content": "[No tool call was found. The session is still running. Call python or forward, or finish with a reply that starts with FINAL SUBMISSION.]" + status()})
            continue
        used_f, used_py = tr["used_f"], tr["used_py"]
        if tr["tool"] in ("web_search", "web_read"):
            used_web += 1
        res = tr["result"]
        if len(res) > 6000:
            res = res[:6000] + "\n[output cut at 6,000 characters]"
        messages.append({"role": "tool", "tool_call_id": f"functions.{tr['tool']}:0", "name": s["call"], "content": res + status()})
    return dict(turns=turns, messages=messages, render=render)


def rebuild_all(inp, variant):
    web = variant == "hero"
    prompt = inp["hero_prompt"] if web else inp["nolib_prompt"]
    tools = inp["tools"] + (inp["web_tools"] if web else [])
    out = {}
    for rid, recs in inp["rolls"].items():
        r = rebuild(recs, prompt, tools, web)
        last = r["turns"][-1]
        # The final sequence: every turn's prompt plus the last reply. Its prefix of length start+n holds each turn's reply.
        seq = r["render"](r["messages"])
        assert len(seq) >= last["start"] + last["nr"]
        out[rid] = dict(seq=seq[:last["start"] + last["nr"]], turns=r["turns"])
    return out


def length_report(built):
    for rid, b in built.items():
        T = b["turns"]
        dp = [t["start"] - t["logged_prompt"] for t in T]
        dn = [t["n"] - t["logged_n"] for t in T]
        print(rid, "turns", len(T), "| prompt length off in", sum(x != 0 for x in dp), "turns, max |off|", max(map(abs, dp)),
              "| reply length off in", sum(x != 0 for x in dn), "turns, sum off", sum(dn), "| reply tokens", sum(t["n"] for t in T), "logged", sum(t["logged_n"] for t in T))
        for t, a, c in zip(T, dp, dn):
            if c != 0 or a != 0 or t["how"] != "rendered":
                print("   turn", t["turn"], t["kind"], "prompt off", a, "reply off", c, t["how"], "repaired" if t["repaired"] else "")


@app.function(image=image, secrets=[modal.Secret.from_name("tinker")], cpu=4, memory=16384, timeout=3 * 3600)
def score(inp: dict, variants: list):
    import importlib.metadata as md
    import tinker
    print("versions: tinker-cookbook", md.version("tinker-cookbook"), "tinker", md.version("tinker"), "tml-renderers", md.version("tml-renderers"))
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(base_model=MODEL, rank=32).result()  # a fresh LoRA, as rl/loop.py made hero3's
    pre = tc.save_weights_and_get_sampling_client().result()
    post = svc.create_sampling_client(model_path=POST)
    if not hasattr(post, "compute_logprobs"):
        post = post.result()
    out = {}
    for variant in variants:
        built = rebuild_all(inp, variant)
        length_report(built)
        # One sequence per rollout holds every reply that was rendered back as sampled; a reply whose sampled tokens
        # differ from the history (malformed reply, key order) is scored on its own: that turn's prompt + sampled tokens.
        futs = {}
        for rid, b in built.items():
            seqs = {"main": b["seq"]}
            for t in b["turns"]:
                if t["raw"] is not None:
                    seqs[t["turn"]] = b["seq"][:t["start"]] + t["raw"]
            futs[rid] = {}
            for key, q in seqs.items():
                kw = {} if key == "main" else {"prompt_logprobs_last_n": len(q) - [t for t in b["turns"] if t["turn"] == key][0]["start"]}
                futs[rid][key] = (pre.compute_logprobs(tinker.ModelInput.from_ints(q), **kw), post.compute_logprobs(tinker.ModelInput.from_ints(q), **kw))
        res = {}
        for rid, fs in futs.items():
            lps = {key: (fp.result(), fq.result()) for key, (fp, fq) in fs.items()}
            turns = []
            for t in built[rid]["turns"]:
                lp_pre, lp_post = lps[t["turn"] if t["raw"] is not None else "main"]
                off = t["start"] if len(lp_pre) >= t["start"] + t["n"] else len(lp_pre) - t["n"]  # prompt_logprobs_last_n may return only the tail
                a, b2 = lp_pre[off:off + t["n"]], lp_post[off:off + t["n"]]
                assert len(a) == t["n"] and None not in a
                turns.append({**{k: v for k, v in t.items() if k != "raw"}, "pre": float(sum(a)), "post": float(sum(b2)),
                              "pre_tok": [round(x, 4) for x in a], "post_tok": [round(x, 4) for x in b2]})
            res[rid] = dict(seq_len=len(built[rid]["seq"]), turns=turns)
            print(variant, rid, "tokens", sum(t["n"] for t in turns), "pre", round(sum(t["pre"] for t in turns), 1), "post", round(sum(t["post"] for t in turns), 1))
        out[variant] = res
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("tinker")], cpu=4, memory=16384, timeout=3 * 3600)
def noise(inp: dict, rids: list):
    """Noise floor: the same sequences scored again under the pre-update policy, and under a second fresh LoRA."""
    import tinker
    svc = tinker.ServiceClient()
    pre = svc.create_lora_training_client(base_model=MODEL, rank=32).result().save_weights_and_get_sampling_client().result()
    pre2 = svc.create_lora_training_client(base_model=MODEL, rank=32).result().save_weights_and_get_sampling_client().result()
    built = rebuild_all(dict(inp, rolls={r: inp["rolls"][r] for r in rids}), "hero")
    out = {}
    for rid, b in built.items():
        mi = tinker.ModelInput.from_ints(b["seq"])
        fa, fb, fc = pre.compute_logprobs(mi), pre.compute_logprobs(mi), pre2.compute_logprobs(mi)
        la, lb, lc = fa.result(), fb.result(), fc.result()
        ts = [t for t in b["turns"] if t["raw"] is None]
        pick = lambda lp: [x for t in ts for x in lp[t["start"]:t["start"] + t["n"]]]
        out[rid] = dict(a=pick(la), b=pick(lb), c=pick(lc))
        print(rid, len(out[rid]["a"]), "sum a", round(sum(out[rid]["a"]), 2), "b-a", round(sum(out[rid]["b"]) - sum(out[rid]["a"]), 2), "c-a", round(sum(out[rid]["c"]) - sum(out[rid]["a"]), 2))
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("tinker")], cpu=4, memory=16384, timeout=3600)
def sample_vs_score(inp: dict):
    """Does compute_logprobs reproduce the sampler's own log-probabilities? Sample from rollout r6's prompts at turn 1, its middle and its end, then score."""
    import tinker
    svc = tinker.ServiceClient()
    pre = svc.create_lora_training_client(base_model=MODEL, rank=32).result().save_weights_and_get_sampling_client().result()
    b = rebuild_all(dict(inp, rolls={"r6": inp["rolls"]["r6"]}), "hero")["r6"]
    rows = []
    for t in [b["turns"][i] for i in (0, len(b["turns"]) // 2, len(b["turns"]) - 2)]:
        ob = b["seq"][:t["start"]]
        r = pre.sample(prompt=tinker.ModelInput.from_ints(ob), num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=400)).result()
        sq = r.sequences[0]
        lp = pre.compute_logprobs(tinker.ModelInput.from_ints(ob + list(sq.tokens)), prompt_logprobs_last_n=len(sq.tokens)).result()
        lp = lp[-len(sq.tokens):]
        rows.append((len(ob), len(sq.tokens), sum(sq.logprobs), sum(lp)))
        print("prompt", len(ob), "sampled", len(sq.tokens), "sum logprob at sampling", round(sum(sq.logprobs), 3), "compute_logprobs", round(sum(lp), 3),
              "max |diff|", round(max(abs(a - c) for a, c in zip(sq.logprobs, lp)), 4))
    return rows


@app.function(image=image, secrets=[modal.Secret.from_name("tinker")], cpu=1, timeout=600)
def run_info():
    """What Tinker records about hero3's training run (read-only)."""
    import tinker
    rc = tinker.ServiceClient().create_rest_client()
    rid = POST.split("//")[1].split("/")[0]
    tr = rc.get_training_run(rid).result()
    print("training run:", {k: v for k, v in (tr.model_dump() if hasattr(tr, "model_dump") else vars(tr)).items() if "key" not in k.lower()})
    cks = rc.list_checkpoints(rid).result()
    for c in (getattr(cks, "checkpoints", None) or cks):
        d = c.model_dump() if hasattr(c, "model_dump") else vars(c)
        print("checkpoint:", {k: d[k] for k in d if k in ("checkpoint_id", "checkpoint_type", "time", "tinker_path", "size_bytes")})


@app.local_entrypoint()
def info():
    run_info.remote()


@app.local_entrypoint()
def sample_check():
    sample_vs_score.remote(load_inputs())


@app.local_entrypoint()
def noise_check(rids: str = "r4,r6"):
    out = noise.remote(load_inputs(), rids.split(","))
    json.dump(out, open(f"{HERO}/update_check_noise.json", "w"))


@app.local_entrypoint()
def main(variant: str = "hero"):
    inp = load_inputs()
    variants = ["hero"] + (["nolib"] if variant == "nolib" else [])
    path = f"{HERO}/update_check.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    out = score.remote(inp, [v for v in variants if v not in prev])
    prev.update(out)
    prev["meta"] = dict(advantages=inp["advantages"], rewards=inp["rewards"], entropy=inp["entropy"], action_tokens=inp["action_tokens"])
    json.dump(prev, open(path, "w"))
    summarize(prev)


def summarize(res):
    import statistics as st
    meta = res["meta"]; adv = meta["advantages"]
    for variant in [v for v in ("hero", "nolib") if v in res]:
        print(f"\n== {variant} prompt ==")
        rows = []
        for i in range(8):
            T = res[variant][f"r{i}"]["turns"]
            def agg(sel):
                ts = [t for t in T if sel(t)]
                n = sum(t["n"] for t in ts); d = sum(t["post"] - t["pre"] for t in ts)
                return n, d, (d / n if n else float("nan")), sum(t["pre"] for t in ts)
            a = agg(lambda t: True); s = agg(lambda t: t["save"]); f = agg(lambda t: t["final"]); g = agg(lambda t: t["gen"])
            rows.append((i, adv[i], a, s, f, g))
            print(f"r{i} adv {adv[i]:+.3f} | all n={a[0]} nll_pre={-a[3]/a[0]:.4f} total {a[1]:+.2f} per-tok {a[2]:+.5f} | save n={s[0]} {s[1]:+.2f} ({s[2]:+.5f}) "
                  f"| final n={f[0]} {f[1]:+.2f} ({f[2]:+.5f}) | gen n={g[0]} {g[1]:+.2f} ({g[2]:+.5f})")
        N = sum(r[2][0] for r in rows)
        print("all tokens", N, "mean NLL pre", round(-sum(r[2][3] for r in rows) / N, 4), "post", round(-(sum(r[2][3] + r[2][1] for r in rows)) / N, 4), "(logged entropy", meta["entropy"], "over", meta["action_tokens"], ")")
        for nm, idx in (("total", 1), ("per-token", 2)):
            xs = [r[1] for r in rows]; ys = [r[2][idx] for r in rows]
            print(f"corr(adv, {nm} change) = {st.correlation(xs, ys):+.3f}")
        print("adv-weighted sum of total change:", round(sum(r[1] * r[2][1] for r in rows), 2))


if __name__ == "__main__" and "--check" in sys.argv:
    inp = load_inputs()
    for v in (["hero", "nolib"] if "--nolib" in sys.argv else ["hero"]):
        print("==", v); length_report(rebuild_all(inp, v))
