"""A one-pass hypernetwork that writes a LoRA for each question, trained through the frozen target's loss.

For a question, the frozen target (Qwen3.5-0.8B) is run on the question text; the hypernetwork reads the activations going into
each layer and writes a rank-R LoRA on that layer's mlp.down_proj. Training: apply the LoRA, compute the cross-entropy of one of
the target's own correct, finished answers to that question, and backpropagate through the frozen target into the hypernetwork
(the "SFT" scheme of Text-to-LoRA, arXiv 2506.06105, Eq. 5; the per-layer design of Doc-to-LoRA, arXiv 2602.15902, Eqs. 5-8).

    HyperNet(
      proj    : Linear(D -> d) + LayerNorm(d)            shared across layers
      queries : Parameter[R, d]                          R learned queries, shared
      mlp     : Linear(d -> 4d) -> SiLU -> Linear(4d -> d)   shared, residual; then LayerNorm(d) and a 1/d scale
      head    : Parameter[L, d, K + M]                   one Linear(d -> K + M) per target layer
    )
    activations X [L, N, D] -> H [L, N, d] -> pooled P [L, R, d] -> O [L, R, K + M] -> A [L, R, K], B [L, M, R]
    layer l: down_proj(x) + (x @ A[l].T) @ B[l].T

mode "static": the pooled vectors are a learned Parameter[L, R, d] and the question is ignored (a fixed LoRA of the same shape,
trained on the same data and steps). Scoring "junk": the hypernetwork is given another question's activations.

  modal run --detach -m harness.hyper::answers                       # the target's own answers to practice problems
  modal run --detach -m harness.hyper::train --run-id hyper1         # then --mode static --run-id static1
  modal run --detach -m harness.hyper::score --run-id hyper1 --ckpt 3 --cond matched
"""
import modal

from harness.target import (app, offline, weights, runs, heldout_vol, GEN_CAP, DEFAULT_TARGET, EVAL_SETS, base_dirs,
                            start_vllm, stop_proc, chat, grade_math)

RANK = 8
WIDTH = 512
MAX_TOKENS = 6000   # training sequences (question + answer) longer than this are left out; also the cap when collecting answers
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)


def data_dir(target=DEFAULT_TARGET):
    return f"/runs/data/hyper/{target}"


def _prompt(problem):
    return problem + "\n\nPut your final answer in \\boxed{}."


# ----------------------------------------------------------------------------- training data
@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, block_network=True, timeout=8 * 60 * 60,
              memory=65536, cpu=8)
def gen_answers(target: str = DEFAULT_TARGET, n: int = 4, limit: int = 0):
    """The target's own answers to the practice problems outside the select and confirm splits (n samples each, cap
    MAX_TOKENS, the harness's sampling and seeds). All rows are saved; training uses the correct, finished ones."""
    import concurrent.futures as cf, json, os, time, zlib
    mdir, pdir = base_dirs("", target, "math")
    splits = json.load(open(f"{pdir}/splits.json"))
    held = set(splits["select"]) | set(splits["confirm_1"]) | set(splits["confirm_2"])
    probs = [json.loads(l) for l in open(f"{pdir}/problems.jsonl")]
    items = [p for p in probs if p["id"] not in held]
    if limit:
        items = items[:limit]
    proc = start_vllm(mdir, "original", 8001, 0.90, GEN_CAP + 3000)

    def one(job):
        p, k = job
        seed = zlib.crc32(f"{p['id']}|{k}".encode()) & 0x7FFFFFFF
        try:
            r = chat(8001, "original", [{"role": "user", "content": _prompt(p["problem"])}], MAX_TOKENS, seed=seed,
                     chat_template_kwargs={"enable_thinking": True}, **SAMPLING)
            ch = r["choices"][0]
            rs = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
            text = (("<think>" + rs + "</think>") if rs else "") + (ch["message"].get("content") or "")
            fin, toks = ch["finish_reason"], r["usage"]["completion_tokens"]
        except Exception as e:
            text, fin, toks = f"[request failed: {repr(e)[:200]}]", "error", 0
        return {"qid": p["id"], "sample": k, "prompt": _prompt(p["problem"]), "gold": p["answer"], "text": text,
                "correct": bool(grade_math(text, p["answer"])[1]), "tokens": toks, "finish": fin}
    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(256) as ex:
            rows = list(ex.map(one, [(p, k) for p in items for k in range(n)]))
    finally:
        stop_proc(proc)
    d = data_dir(target)
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/answers.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    good = [r for r in rows if r["correct"] and r["finish"] == "stop"]
    out = {"target": target, "problems": len(items), "answers": len(rows), "correct_finished": len(good),
           "problems_with_a_correct_answer": len({r["qid"] for r in good}), "tokens": sum(r["tokens"] for r in rows),
           "seconds": round(time.time() - t0)}
    json.dump(out, open(f"{d}/answers_meta.json", "w"), indent=1)
    runs.commit()
    print("ANSWERS", json.dumps(out))
    return out


# ----------------------------------------------------------------------------- the hypernetwork and the hooked target
def _build(mdir, mode):
    """The frozen target with a per-example LoRA hook on every mlp.down_proj, and the hypernetwork (float32, trainable)."""
    import math, torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(mdir)
    hf = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16).to("cuda").eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    cfg = hf.config.text_config if hasattr(hf.config, "text_config") else hf.config
    L, D = cfg.num_hidden_layers, cfg.hidden_size
    mods = dict(hf.named_modules())
    downs = []
    for l in range(L):
        name = [n for n in mods if n.endswith(f"layers.{l}.mlp.down_proj") and "vis" not in n.lower()]
        assert len(name) == 1, (l, name)
        downs.append(mods[name[0]])
    M, K = downs[0].weight.shape

    class HyperNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = WIDTH
            self.mode = mode
            self.proj, self.norm = nn.Linear(D, d), nn.LayerNorm(d)
            self.queries = nn.Parameter(torch.randn(RANK, d) / math.sqrt(d))
            self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
            self.static = nn.Parameter(torch.randn(L, RANK, d)) if mode == "static" else None
            self.out_norm = nn.LayerNorm(d)
            head = torch.zeros(L, d, K + M)          # the B part starts at zero, so the LoRA starts as no change
            # The head sees P / d (P has unit RMS after out_norm), so one optimizer step of size lr moves an entry of A or B
            # by about lr, as it would in a directly trained LoRA; A entries start at about 1/sqrt(K), as in a LoRA's A.
            head[:, :, :K] = torch.randn(L, d, K) * math.sqrt(d / K)
            self.head = nn.Parameter(head)

        def forward(self, X):
            """X: [L, N, D] float32 activations going into each layer. Returns A [L, R, K], B [L, M, R]."""
            if self.static is not None:
                P = self.static
            else:
                H = self.norm(self.proj(X))                                             # [L, N, d]
                att = torch.softmax(self.queries @ H.transpose(1, 2) / math.sqrt(WIDTH), -1)   # [L, R, N]
                P = att @ H                                                             # [L, R, d]
            P = self.out_norm(P + self.mlp(P)) / WIDTH
            O = torch.einsum("lrd,ldk->lrk", P, self.head)                              # [L, R, K + M]
            return O[:, :, :K], O[:, :, K:].transpose(1, 2)

    hyper = HyperNet().to("cuda")
    lora = {"A": None, "B": None}   # A [b, L, R, K], B [b, L, M, R] for the current batch, or None for no LoRA

    def make_hook(l):
        def hook(m, inp, out):
            if lora["A"] is None:
                return out
            x = inp[0]                                                    # [b, T, K]
            a, b = lora["A"][:, l].to(x.dtype), lora["B"][:, l].to(x.dtype)
            return out + torch.einsum("btr,bmr->btm", torch.einsum("btk,brk->btr", x, a), b)
        return hook
    for l, m in enumerate(downs):
        m.register_forward_hook(make_hook(l))

    def acts(ids_list):
        """Activations going into each layer for each question: a list of [L, N_i, D] float32 tensors (no gradient, no LoRA)."""
        prev = (lora["A"], lora["B"])
        lora["A"] = lora["B"] = None
        out = []
        with torch.no_grad():
            for ids in ids_list:
                o = hf.model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True)
                out.append(torch.stack([o.hidden_states[l][0] for l in range(L)]).float())
        lora["A"], lora["B"] = prev
        return out
    return tok, hf, hyper, lora, acts, dict(L=L, D=D, M=M, K=K)


def _prefix(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)


def _train_rows(tok, target, extra_cached=True):
    """(qid, question ids, answer ids) for the target's correct, finished answers."""
    import glob, json, os
    mdir, pdir = base_dirs("", target, "math")
    splits = json.load(open(f"{pdir}/splits.json"))
    held = set(splits["select"]) | set(splits["confirm_1"]) | set(splits["confirm_2"])
    files = [f"{data_dir(target)}/answers.jsonl"] + (glob.glob(f"{pdir}/original/*.jsonl") + glob.glob(f"{pdir}/baseline/original.jsonl") if extra_cached else [])
    seen, out = set(), []
    for f in files:
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            key = (r["qid"], r["sample"])
            if key in seen or r["qid"] in held or not r.get("correct") or r.get("finish") != "stop":
                continue
            seen.add(key)
            prefix, text = _prefix(tok, r["prompt"]), r["text"]
            if prefix.rstrip().endswith("<think>") and text.startswith("<think>"):
                text = text[len("<think>"):].lstrip("\n")
            if "</think>" in text:  # the server strips the whitespace around the reasoning; the chat template has it
                rs, content = text.split("</think>", 1)
                text = rs.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
            q = tok(prefix, add_special_tokens=False)["input_ids"]
            a = tok(text + "<|im_end|>", add_special_tokens=False)["input_ids"]
            if len(q) + len(a) <= MAX_TOKENS:
                out.append((r["qid"], q, a))
    return out


def _nll(hf, lora, hyper_out, q, a):
    """Mean next-token loss of answer `a` after question `q`, with the LoRA (A [L,R,K], B [L,M,R]) applied; differentiable in A, B.
    Leaves the LoRA set (see below); the next call or acts() replaces it."""
    import torch
    from torch.utils.checkpoint import checkpoint
    A, B = hyper_out
    lora["A"], lora["B"] = A[None], B[None]
    x = torch.tensor([q + a]).cuda()
    h = hf.model(input_ids=x).last_hidden_state[0]
    pos = torch.arange(len(q) - 1, x.shape[1] - 1).cuda()   # position p predicts token p + 1

    def head_nll(hc, nxt):  # the output head on 256 positions at a time: full-vocabulary logits of a whole text take many GB
        return -torch.log_softmax(hf.lm_head(hc).float(), -1).gather(-1, nxt[:, None]).sum()
    loss = h.new_zeros((), dtype=torch.float32)
    for c0 in range(0, len(pos), 256):
        loss = loss + checkpoint(head_nll, h[pos[c0:c0 + 256]], x[0, pos[c0:c0 + 256] + 1], use_reentrant=False)
    # The LoRA stays set until the caller has run backward(): the checkpointed layers are recomputed then and must see it.
    return loss / len(pos)


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, block_network=True, timeout=10 * 60 * 60,
              memory=65536, cpu=8)
def train_hyper(run_id: str, mode: str = "hyper", epochs: int = 3, lr: float = 1e-4, batch: int = 4, target: str = DEFAULT_TARGET,
                max_rows: int = 0, val_rows: int = 64):
    """Train the hypernetwork (mode "hyper") or the fixed LoRA of the same shape (mode "static") through the frozen target.
    A checkpoint is saved after every epoch; the loss on held-out finished answers (problems not trained on) is logged."""
    import json, math, os, random, time, torch
    assert mode in ("hyper", "static")
    mdir, _ = base_dirs("", target, "math")
    tok, hf, hyper, lora, acts, dims = _build(mdir, mode)
    rows = _train_rows(tok, target)
    rng = random.Random(0)
    qids = sorted({r[0] for r in rows}); rng.shuffle(qids)
    val_q = set(qids[:max(1, len(qids) // 16)])
    val = [r for r in rows if r[0] in val_q][:val_rows]
    tr = [r for r in rows if r[0] not in val_q]
    if max_rows:
        tr = tr[:max_rows]
    n_par = sum(p.numel() for p in hyper.parameters())
    print("HYPER", json.dumps({"mode": mode, **dims, "rank": RANK, "width": WIDTH, "params": n_par, "train_rows": len(tr),
                               "train_problems": len({r[0] for r in tr}), "val_rows": len(val)}))
    assert len(tr) >= 8, "too few training answers"
    try:  # recompute each layer in the backward pass instead of storing all activations
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.train()
        ckpt_on = True
    except Exception as e:
        ckpt_on = False
        print("no layer checkpointing:", repr(e)[:200])
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.0)
    steps_total = epochs * math.ceil(len(tr) / batch)
    d = f"/runs/hyper/{run_id}"
    os.makedirs(d, exist_ok=True)
    log = open(f"{d}/log.jsonl", "a")

    def out_for(q):
        X = None if mode == "static" else acts([q])[0]
        return hyper(X)

    def val_loss(with_lora=True):
        tot = 0.0
        with torch.no_grad():
            for _, q, a in val:
                A, B = out_for(q)
                if not with_lora:
                    A, B = A * 0, B * 0
                tot += float(_nll(hf, lora, (A, B), q, a))
        return tot / len(val)
    base_val = val_loss(False)
    print("VAL without LoRA", round(base_val, 4))
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        order = list(range(len(tr))); rng.shuffle(order)
        run = 0.0
        for b0 in range(0, len(order), batch):
            for i in order[b0:b0 + batch]:
                _, q, a = tr[i]
                loss = _nll(hf, lora, out_for(q), q, a)
                (loss / batch).backward()
                run += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps_total)))
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"{mode} ep {ep} step {step}/{steps_total} train loss {run / (25 * batch):.4f} {round(time.time() - t0)}s")
                run = 0.0
        v = val_loss()
        with torch.no_grad():
            A, B = out_for(tr[0][1])
            dn = float(torch.einsum("lmr,lrk->lmk", B, A).norm())
        rec = {"epoch": ep, "step": step, "val_loss": round(v, 4), "val_loss_without_lora": round(base_val, 4),
               "delta_w_norm_example": round(dn, 4), "seconds": round(time.time() - t0), "layer_checkpointing": ckpt_on}
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"state": hyper.state_dict(), "mode": mode, "dims": dims, "rank": RANK, "width": WIDTH}, f"{d}/ckpt_e{ep}.pt")
        runs.commit()
        print("EPOCH", json.dumps(rec))
    return {"run_id": run_id, "mode": mode, "params": n_par, "train_rows": len(tr), "steps": step}


# ----------------------------------------------------------------------------- scoring with one LoRA per question
@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, block_network=True,
              timeout=10 * 60 * 60, memory=65536, cpu=8)
def score_hyper(run_id: str = "", ckpt: int = 0, cond: str = "matched", dataset: str = "math500", limit: int = 250, n: int = 2,
                split: str = "", batch: int = 48, max_new: int = GEN_CAP, target: str = DEFAULT_TARGET, tag: str = ""):
    """Generate answers with the hypernetwork's LoRA for each question and grade them.
    cond: "matched" (the question's own activations), "junk" (another question's activations), "none" (no LoRA: the target
    as downloaded, through the same generation code). split: score practice problems of that split instead of `dataset`."""
    import json, os, time, torch
    from transformers import LogitsProcessor, LogitsProcessorList
    assert cond in ("matched", "junk", "none")
    mdir, pdir = base_dirs("", target, "math")
    mode = "hyper"
    if cond != "none":
        ck = torch.load(f"/runs/hyper/{run_id}/ckpt_e{ckpt}.pt", map_location="cpu")
        mode = ck["mode"]
    tok, hf, hyper, lora, acts, dims = _build(mdir, mode)
    if cond != "none":
        hyper.load_state_dict(ck["state"])
    hyper.eval()
    if split:
        ids = json.load(open(f"{pdir}/splits.json"))[split]
        P = {}
        for line in open(f"{pdir}/problems.jsonl"):
            r = json.loads(line); P[r["id"]] = r
        items = [{"qid": i, "problem": P[i]["problem"], "answer": P[i]["answer"]} for i in ids if i in P]
    else:
        items = [json.loads(l) for l in open(EVAL_SETS[dataset]["file"])]
    if limit:
        items = items[:limit]
    jobs = [(it, k) for it in items for k in range(n)]
    q_ids = {it["qid"]: tok(_prefix(tok, _prompt(it["problem"])), add_special_tokens=False)["input_ids"] for it in items}
    AB = {}
    if cond != "none":
        with torch.no_grad():
            for j, it in enumerate(items):
                src = items[(j + len(items) // 2) % len(items)] if cond == "junk" else it   # junk: a question half the set away
                X = None if mode == "static" else acts([q_ids[src["qid"]]])[0]
                A, B = hyper(X)
                AB[it["qid"]] = (A.to(torch.bfloat16).cpu(), B.to(torch.bfloat16).cpu())
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    class Presence(LogitsProcessor):   # vLLM's presence penalty: a fixed amount off every token already generated in this answer
        def __init__(self, n_prompt, penalty):
            self.n_prompt, self.penalty = n_prompt, penalty

        def __call__(self, input_ids, scores):
            gen = input_ids[:, self.n_prompt:]
            if gen.shape[1]:
                seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
                scores = scores - self.penalty * seen
            return scores
    order = sorted(range(len(jobs)), key=lambda i: len(q_ids[jobs[i][0]["qid"]]))
    rows, t0 = [None] * len(jobs), time.time()
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        qs = [q_ids[jobs[i][0]["qid"]] for i in idx]
        n_p = max(len(q) for q in qs)
        x = torch.full((len(idx), n_p), pad, dtype=torch.long); am = torch.zeros((len(idx), n_p), dtype=torch.long)
        for r, q in enumerate(qs):   # left padding, so every prompt ends at the same position
            x[r, n_p - len(q):] = torch.tensor(q); am[r, n_p - len(q):] = 1
        if cond != "none":
            lora["A"] = torch.stack([AB[jobs[i][0]["qid"]][0] for i in idx]).cuda()
            lora["B"] = torch.stack([AB[jobs[i][0]["qid"]][1] for i in idx]).cuda()
        torch.manual_seed(1000 + b0)
        with torch.no_grad():
            out = hf.generate(input_ids=x.cuda(), attention_mask=am.cuda(), do_sample=True, temperature=SAMPLING["temperature"],
                              top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"], max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad,
                              logits_processor=LogitsProcessorList([Presence(n_p, SAMPLING["presence_penalty"])]))
        lora["A"] = lora["B"] = None
        for r, i in enumerate(idx):
            g = out[r, n_p:].tolist()
            done = eos in g
            g = g[:g.index(eos)] if done else g
            text = "<think>" + tok.decode(g, skip_special_tokens=False)
            it, k = jobs[i]
            ext, ok = grade_math(text, it["answer"])
            rows[i] = {"qid": it["qid"], "sample": k, "text": text, "extracted": ext, "correct": bool(ok), "tokens": len(g),
                       "finish": "stop" if done else "length"}
        print(f"scored {min(b0 + batch, len(order))}/{len(order)} in {round(time.time() - t0)}s")
    acc = 100 * sum(r["correct"] for r in rows) / len(rows)
    res = {"run_id": run_id, "ckpt": ckpt, "mode": mode if cond != "none" else "none", "cond": cond, "dataset": split or dataset,
           "problems": len(items), "n": n, "max_new": max_new, "accuracy_pct": round(acc, 2),
           "cut_at_cap_pct": round(100 * sum(r["finish"] == "length" for r in rows) / len(rows), 2),
           "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)), "seconds": round(time.time() - t0)}
    name = tag or f"{split or dataset}_{cond}" + (f"_e{ckpt}" if cond != "none" else "") + (f"_limit{limit}" if limit else "")
    d = f"/runs/hyper/{run_id or '_original'}"
    os.makedirs(f"{d}/scores", exist_ok=True)
    json.dump(res, open(f"{d}/scores/{name}.json", "w"), indent=1)
    with open(f"{d}/scores/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    runs.commit()
    print("SCORE", json.dumps(res))
    return res


# ----------------------------------------------------------------------------- entrypoints
@app.local_entrypoint()
def answers(target: str = DEFAULT_TARGET, n: int = 4, limit: int = 0):
    print("SPAWNED", gen_answers.spawn(target, n, limit).object_id)


@app.local_entrypoint()
def train(run_id: str, mode: str = "hyper", epochs: int = 3, lr: float = 1e-4, target: str = DEFAULT_TARGET, max_rows: int = 0):
    print("SPAWNED", train_hyper.spawn(run_id, mode, epochs, lr, 4, target, max_rows).object_id)


@app.local_entrypoint()
def score(run_id: str = "", ckpt: int = 0, cond: str = "matched", dataset: str = "math500", limit: int = 250, n: int = 2, split: str = "",
          max_new: int = GEN_CAP, target: str = DEFAULT_TARGET, tag: str = ""):
    print("SPAWNED", score_hyper.spawn(run_id, ckpt, cond, dataset, limit, n, split, 48, max_new, target, tag).object_id)


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, block_network=True,
              timeout=60 * 60, memory=65536, cpu=8)
def hyper_smoke(target: str = DEFAULT_TARGET):
    """Small end-to-end check on cached answers: shapes, a few training steps that lower the loss, a gradient that reaches the
    hypernetwork, the LoRA starting as no change, and generation with one LoRA per question (speed measured)."""
    import json, time, torch
    mdir, _ = base_dirs("", target, "math")
    tok, hf, hyper, lora, acts, dims = _build(mdir, "hyper")
    rows = _train_rows(tok, target)[:6]
    print("rows", len(rows), [(len(q), len(a)) for _, q, a in rows], dims, "params", sum(p.numel() for p in hyper.parameters()))
    _, q, a = rows[0]
    X = acts([q])[0]
    A, B = hyper(X)
    print("X", tuple(X.shape), "A", tuple(A.shape), "B", tuple(B.shape), "B norm at init", float(B.norm()))
    with torch.no_grad():
        l0 = float(_nll(hf, lora, (A * 0, B * 0), q, a)); l1 = float(_nll(hf, lora, (A, B), q, a))
    print("loss without LoRA", round(l0, 4), "with the initial LoRA", round(l1, 4))
    try:
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); hf.train()
        print("layer checkpointing on")
    except Exception as e:
        print("no layer checkpointing", repr(e)[:200])
    opt = torch.optim.AdamW(hyper.parameters(), lr=1e-4)
    t0 = time.time()
    for s in range(24):
        _, q, a = rows[s % len(rows)]
        loss = _nll(hf, lora, hyper(acts([q])[0]), q, a)
        loss.backward()
        gn = {n: round(float(p.grad.norm()), 6) for n, p in hyper.named_parameters() if p.grad is not None and n in ("head", "queries", "proj.weight")}
        torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        print("step", s, "loss", round(float(loss), 4), "grad norms", gn, "peak GB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), round(time.time() - t0), "s")
    hf.eval()
    try:
        hf.gradient_checkpointing_disable()
    except Exception:
        pass
    torch.save({"state": hyper.state_dict(), "mode": "hyper", "dims": dims, "rank": RANK, "width": WIDTH}, "/tmp/smoke.pt")
    import os
    os.makedirs("/runs/hyper/_smoke", exist_ok=True)
    torch.save({"state": hyper.state_dict(), "mode": "hyper", "dims": dims, "rank": RANK, "width": WIDTH}, "/runs/hyper/_smoke/ckpt_e1.pt")
    runs.commit()
    for cond in ("none", "matched", "junk"):
        r = score_hyper.local("_smoke", 1, cond, "math500", 8, 2, "", 16, 512, target, f"smoke_{cond}")
        print("GEN", cond, json.dumps(r))


@app.local_entrypoint()
def smoke(target: str = DEFAULT_TARGET):
    hyper_smoke.remote(target)
