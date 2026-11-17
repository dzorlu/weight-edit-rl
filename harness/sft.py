"""Plain fine-tuning baseline for the gradient ablation: no editor.

The target is fine-tuned on its own correct, finished answers to practice problems (rejection sampling: cached answers plus 2,048 new ones), on the same tensors an
editor may change (the transformer blocks of the language model). A small grid of learning rate x steps is scored on part of
the `select` split, the split the editor chooses edits on; the best checkpoint is saved as an edit in
/runs/<run_id>/work/edited, so harness.target.heldout_eval scores it on MATH-500 exactly like an editor's edit.

  modal run --detach -m harness.sft::sft --run-id sft-baseline
"""
import modal

from harness.target import (app, offline, weights, runs, GEN_CAP, DEFAULT_TARGET, base_dirs, start_vllm, stop_proc, chat,
                            grade_math, heldout_eval)

LRS = (1e-6, 3e-6, 1e-5)
STEPS = (16, 48, 144)   # optimizer steps at which a checkpoint is scored; each step is BATCH sequences
BATCH = 4
MAX_TOKENS = 6000       # training sequences longer than this are left out
GEN_PROBLEMS = 1024     # practice problems (2 samples each) answered by the original model to collect training answers
SELECT_N = 128          # problems of the select split (1 sample each) that score a checkpoint


def _in_scope(name):
    low = name.lower()
    return ".layers." in low and not any(w in low for w in ("embed_tokens", "lm_head", "visual", "vision"))


def _answers(model_dir, name, items, n):
    """Answers of a model directory to practice items, with the harness's generation protocol and seeds."""
    import concurrent.futures as cf, time, zlib
    proc = start_vllm(model_dir, name, 8001, 0.90, GEN_CAP + 3000)

    def one(job):
        it, k = job
        seed = zlib.crc32(f"{it['qid']}|{k}".encode()) & 0x7FFFFFFF
        try:
            r = chat(8001, name, [{"role": "user", "content": it["prompt"]}], GEN_CAP, seed=seed, temperature=0.6, top_p=0.95,
                     top_k=20, presence_penalty=1.5, chat_template_kwargs={"enable_thinking": True})
            ch = r["choices"][0]
            rs = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
            text = (("<think>" + rs + "</think>") if rs else "") + (ch["message"].get("content") or "")
            fin, toks = ch["finish_reason"], r["usage"]["completion_tokens"]
        except Exception as e:
            text, fin, toks = f"[request failed: {repr(e)[:200]}]", "error", 0
        return {"qid": it["qid"], "sample": k, "prompt": it["prompt"], "text": text, "correct": grade_math(text, it["answer"])[1], "tokens": toks, "finish": fin}
    try:
        with cf.ThreadPoolExecutor(256) as ex:
            return list(ex.map(one, [(it, k) for it in items for k in range(n)]))
    finally:
        stop_proc(proc)
        time.sleep(10)


def _score(model_dir, name, items):
    """Accuracy of a model directory on practice items (1 sample each)."""
    res = _answers(model_dir, name, items, 1)
    n = len(res)
    return {"accuracy": round(100 * sum(r["correct"] for r in res) / n, 2), "cut_at_cap": round(100 * sum(r["finish"] == "length" for r in res) / n, 2),
            "mean_tokens": round(sum(r["tokens"] for r in res) / n), "n": n}


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, block_network=True, timeout=5 * 60 * 60,
              memory=65536, cpu=8)
def sft_baseline(run_id: str = "sft-baseline", target: str = DEFAULT_TARGET, heldout_limit: int = 250):
    import glob, json, os, random, shutil, torch
    from safetensors.torch import load_file, save_file
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    mdir, pdir = base_dirs("", target, "math")
    splits = json.load(open(f"{pdir}/splits.json"))
    held = set(splits["select"]) | set(splits["confirm_1"]) | set(splits["confirm_2"])
    probs = {}
    for line in open(f"{pdir}/problems.jsonl"):
        r = json.loads(line)
        probs[r["id"]] = r
    # training data: the original model's correct, finished answers that already exist, outside select and confirm
    seen, data = set(), []
    for f in glob.glob(f"{pdir}/original/*.jsonl") + glob.glob(f"{pdir}/baseline/original.jsonl"):
        for line in open(f):
            r = json.loads(line)
            key = (r["qid"], r["sample"])
            if key in seen or r["qid"] in held or r["qid"] not in probs or not r.get("correct") or r.get("finish") != "stop":
                continue
            seen.add(key)
            data.append(r)
    # ... and new ones: GEN_PROBLEMS more practice problems x 2 samples from the original model, kept when correct and finished
    rng0 = random.Random(1)
    pool = sorted(i for i in probs if i not in held and i not in {r["qid"] for r in data})
    rng0.shuffle(pool)
    gen_items = [{"qid": i, "prompt": probs[i]["problem"] + "\n\nPut your final answer in \\boxed{}.", "answer": probs[i]["answer"]} for i in pool[:GEN_PROBLEMS]]
    new_rows = _answers(mdir, "original", gen_items, 2)
    data += [r for r in new_rows if r["correct"] and r["finish"] == "stop"]
    print("generated", len(new_rows), "answers; correct and finished", sum(1 for r in new_rows if r["correct"] and r["finish"] == "stop"))
    tok = AutoTokenizer.from_pretrained(mdir)
    seqs = []
    for r in data:
        prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = r["text"]
        if prefix.rstrip().endswith("<think>") and text.startswith("<think>"):
            text = text[len("<think>"):].lstrip("\n")
        if "</think>" in text:  # the server strips the whitespace around the reasoning; the chat template has it
            rs, content = text.split("</think>", 1)
            text = rs.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        a_ids = tok(text + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) <= MAX_TOKENS:
            seqs.append((p_ids, a_ids))
    print("SFT data: correct finished answers", len(data), "kept", len(seqs), "problems", len({r["qid"] for r in data}),
          "mean answer tokens", round(sum(len(a) for _, a in seqs) / max(1, len(seqs))))
    if seqs:
        print("example prefix end:", repr(tok.decode(seqs[0][0][-12:])), "| answer start:", repr(tok.decode(seqs[0][1][:24])), "| answer end:", repr(tok.decode(seqs[0][1][-12:])))
    assert len(seqs) >= 16, "too few training answers"
    sel_ids = [i for i in splits["select"] if i in probs][:SELECT_N]
    items = [{"qid": i, "prompt": probs[i]["problem"] + "\n\nPut your final answer in \\boxed{}.", "answer": probs[i]["answer"]} for i in sel_ids]

    def write_model(dst, changed):
        shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        for f in os.listdir(mdir):
            src = os.path.join(mdir, f)
            if f.endswith(".safetensors"):
                sd = load_file(src)
                for n, t in changed.items():
                    if n in sd:
                        sd[n] = t.to(sd[n].dtype)
                save_file(sd, os.path.join(dst, f), metadata={"format": "pt"})
            elif os.path.isfile(src):
                shutil.copy(os.path.realpath(src), os.path.join(dst, f))

    ckpts = {}
    for lr in LRS:
        model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.float32).to("cuda")
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception as e:
            print("no layer checkpointing:", repr(e)[:200])
        model.train()
        params = {n: p for n, p in model.named_parameters() if _in_scope(n)}
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params.values():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
        def head_nll(hc, nxt):
            return -torch.log_softmax(model.lm_head(hc.float()).float(), -1).gather(-1, nxt[:, None]).sum()
        rng = random.Random(0)
        order, step = [], 0
        while step < max(STEPS):
            tot = 0.0
            for _ in range(BATCH):
                if not order:
                    order = list(range(len(seqs))); rng.shuffle(order)
                p_ids, a_ids = seqs[order.pop()]
                x = torch.tensor([p_ids + a_ids]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.model(input_ids=x).last_hidden_state[0]
                # the output head on 256 answer positions at a time: full-vocabulary logits of a whole text take many GB
                pos = torch.arange(len(p_ids) - 1, x.shape[1] - 1).cuda()  # position p predicts token p + 1
                loss = h.new_zeros((), dtype=torch.float32)
                for c0 in range(0, len(pos), 256):
                    loss = loss + checkpoint(head_nll, h[pos[c0:c0 + 256]], x[0, pos[c0:c0 + 256] + 1], use_reentrant=False)
                loss = loss / len(pos)
                (loss / BATCH).backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 4)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 8 == 0 or step in STEPS:
                print(f"lr {lr:g} step {step} loss {tot / BATCH:.4f}")
            if step in STEPS:
                tag = f"lr{lr:g}_s{step}"
                ckpts[tag] = {"lr": lr, "steps": step, "train_loss": round(tot / BATCH, 4), "dir": f"/tmp/sft/{tag}"}
                write_model(ckpts[tag]["dir"], {n: p.detach().cpu() for n, p in params.items()})
        del model, opt, params
        torch.cuda.empty_cache()
    res = {"run_id": run_id, "target": target, "train_answers": len(seqs), "batch": BATCH, "select_problems": len(items),
           "original": _score(mdir, "original", items), "grid": {}}
    print("SELECT original", res["original"])
    for tag, c in ckpts.items():
        c.update(_score(c["dir"], "edited", items))
        res["grid"][tag] = {k: v for k, v in c.items() if k != "dir"}
        print("SELECT", tag, res["grid"][tag])
    best = max(ckpts, key=lambda t: ckpts[t]["accuracy"])
    res["best"] = best
    ed = f"/runs/{run_id}/work/edited"
    shutil.rmtree(ed, ignore_errors=True); os.makedirs(ed)
    orig = {}
    for f in os.listdir(mdir):
        if f.endswith(".safetensors"):
            orig.update(load_file(os.path.join(mdir, f)))
    new = {}
    for f in os.listdir(ckpts[best]["dir"]):
        if f.endswith(".safetensors"):
            new.update(load_file(os.path.join(ckpts[best]["dir"], f)))
    n_changed = 0
    for n, t in new.items():
        if _in_scope(n) and not torch.equal(t, orig[n]):
            save_file({"t": t}, f"{ed}/{n}.safetensors")
            n_changed += 1
    res["changed_tensors"] = n_changed
    json.dump(res, open(f"/runs/{run_id}/sft.json", "w"), indent=1)
    runs.commit()
    print("SFT", json.dumps(res))
    call = heldout_eval.spawn(run_id, 2, heldout_limit, True)
    print("HELDOUT SPAWNED", call.object_id)
    return res


@app.local_entrypoint()
def sft(run_id: str = "sft-baseline", target: str = DEFAULT_TARGET, heldout_limit: int = 250):
    print("SPAWNED", sft_baseline.spawn(run_id, target, heldout_limit).object_id)
