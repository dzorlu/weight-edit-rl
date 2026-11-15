"""Test of --rollouts-per-gpu 2: one rollout on Target, then two rollouts on one PackedTarget, same steps, timed.

    uvx modal run -m scripts.packing_test::packtest         # both phases, one GPU at a time
    uvx modal run -m scripts.packing_test::packtest --phase pair
    uvx modal run -m scripts.packing_test::packstress   # worst case: both rollouts generating 256 answers per model at once, long all-layer teacher-forced passes, a restart during the other's generation, rewards at the same time

Steps per rollout: python tool, save_edit, paired generation ("both") on 16 practice problems that have no cached
original answers (8192-token cap), a teacher-forced pass on the edited weights, a second save_edit and an edited
generation (restarts the edited server), reward scoring (128 problems x 2), snapshot, isolation checks.
"""
import modal
from harness.target import app, runs, heldout_vol, weights, Target, PackedTarget, SRC

drv_image = modal.Image.debian_slim(python_version="3.12").pip_install("tokenizers").add_local_python_source(*SRC, "scripts")
EDITS = {"A": (23, 507), "B": (20, 100), "solo": (23, 507)}


async def sequence(call, tag, ids, chats, items_json):
    import json, time
    L, col = EDITS[tag]
    name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    t, out = {}, {}

    async def step(key, *a):
        t0 = time.time()
        r = await call(*a)
        t[key] = round(time.time() - t0, 1)
        return r

    boot = await step("boot_id", "boot_id")
    out["iso"] = await step("py_first", "py", f"import os; open('/work/marker_{tag}', 'w').write('{tag}'); "
                            "print('work', sorted(os.listdir('/work')), 'slots', os.listdir('/slots') if os.path.exists('/slots') else None, "
                            "'pids', len([p for p in os.listdir('/proc') if p.isdigit()]))\n"
                            "import numpy as np\ntry:\n    x = np.empty(40 * 2 ** 30, np.uint8); del x; print('no memory limit')\nexcept MemoryError:\n    print('memory limit ok')")
    out["edit1"] = await step("py_edit1", "py", f"n = '{name}'; W = load(n); W[:, {col}] *= 0.5; print(save_edit(n, W)); edited()")
    ok, res = await step("gen_both_16", "forward", json.dumps({"type": "generate", "model": "both", "problems": ids, "n": 1, "max_tokens": 8192}))
    out["gen_both"] = (ok, json.loads(res) if ok else res)
    ok, res = await step("teacher_forced", "forward", json.dumps({"type": "teacher_forced", "model": "edited", "chats": chats, "enable_thinking": True,
                                                                "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 12, 24], "mlp_act": [L]}}))
    out["tf"] = (ok, {k: v for k, v in json.loads(res).items() if k in ("dir", "texts", "max_tokens_in_a_text", "mean_token_logprob_first_texts")} if ok else res)
    out["edit2"] = await step("py_edit2", "py", f"W[:, {col}] = 0; print(save_edit(n, W))")
    ok, res = await step("gen_edited_16", "forward", json.dumps({"type": "generate", "model": "edited", "problems": ids, "n": 1, "max_tokens": 8192}))
    out["gen_edited"] = (ok, json.loads(res) if ok else res)
    rw = await step("reward_128x2", "reward", items_json, 2, 8192, False)
    out["reward"] = {"edited_tensors": rw["edited_tensors"], "answers": len(rw["edited"] or []),
                     "accuracy": round(sum(r["correct"] for r in rw["edited"]) / len(rw["edited"]), 4) if rw["edited"] else None}
    out["snapshot"] = await step("snapshot", "snapshot", f"rl/_packtest/{tag}")
    out["iso_end"] = await step("py_last", "py", "import os; print(sorted(os.listdir('/work')), edited(), sorted(os.listdir('/work/fwd_2')))")
    out["boot_unchanged"] = (await call("boot_id")) == boot
    out["boot"] = boot
    t["total"] = round(sum(v for k, v in t.items()), 1)
    return {"tag": tag, "seconds": t, "out": out}


@app.function(image=drv_image, volumes={"/runs": runs, "/heldout": heldout_vol}, timeout=3 * 3600)
async def drive(phase: str = "both"):
    import asyncio, json, time
    sp = json.load(open("/runs/data/math_practice/splits.json"))
    prob = {json.loads(l)["id"]: json.loads(l) for l in open("/runs/data/math_practice/problems.jsonl")}
    ids = sp["extra"][:16]  # not in the original-answer cache: the original server has to generate
    chats = [[{"role": "user", "content": prob[i]["problem"]}, {"role": "assistant", "content": prob[i]["solution"]}] for i in sp["extra"][16:32]]
    items_json = json.dumps([json.loads(l) for l in open("/heldout/rl/reward_math.jsonl")][:128])
    stamp = str(int(time.time()))
    res = {}
    if phase in ("both", "solo"):
        obj = Target.with_options(scaledown_window=10)(rid=f"packtest-solo-{stamp}")
        call = lambda m, *a: getattr(obj, m).remote.aio(*a)
        t0 = time.time()
        await call("boot_id")
        res["solo_boot_s"] = round(time.time() - t0, 1)
        res["solo"] = await sequence(call, "solo", ids, chats, items_json)
        print("SOLO", json.dumps(res["solo"]["seconds"]), flush=True)
        await asyncio.sleep(60)  # let the solo container scale down before the pair container takes a GPU
    if phase in ("both", "pair"):
        obj = PackedTarget.with_options(scaledown_window=10)(gid=f"packtest-pair-{stamp}")
        peak, done = {"mib": 0, "samples": []}, asyncio.Event()

        async def watch():
            while not done.is_set():
                try:
                    st = await obj.stats.remote.aio()
                    used = int(st["gpu_memory"].split(",")[0].split()[0])
                    peak["mib"] = max(peak["mib"], used); peak["samples"].append((round(time.time() - t0), used, st["original_server"], {k: v["edited_server"] for k, v in st["slots"].items()}))
                except Exception as e:
                    peak["samples"].append(repr(e)[:200])
                try:
                    await asyncio.wait_for(done.wait(), 20)
                except asyncio.TimeoutError:
                    pass

        t0 = time.time()
        await obj.stats.remote.aio()
        res["pair_boot_s"] = round(time.time() - t0, 1)
        w = asyncio.create_task(watch())
        mk = lambda rid: (lambda m, *a: getattr(obj, m).remote.aio(rid, *a))
        a, b = await asyncio.gather(sequence(mk(f"A-{stamp}"), "A", ids, chats, items_json), sequence(mk(f"B-{stamp}"), "B", ids, chats, items_json))
        res["pair_wall_s"] = round(time.time() - t0 - res["pair_boot_s"], 1)
        done.set(); await w
        res["A"], res["B"], res["gpu_peak_mib"], res["gpu_samples"] = a, b, peak["mib"], peak["samples"]
        print("PAIR", json.dumps(a["seconds"]), json.dumps(b["seconds"]), "peak MiB", peak["mib"], flush=True)
    import os
    os.makedirs("/runs/rl/_packtest", exist_ok=True)
    json.dump(res, open(f"/runs/rl/_packtest/result_{stamp}.json", "w"), indent=1, default=str)
    await runs.commit.aio()
    return res


@app.local_entrypoint()
def packtest(phase: str = "both", out: str = "/tmp/packtest_result.json"):
    import json
    res = drive.remote(phase)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:20000])


async def stress_seq(call, tag, t0, ev, long_text, gen_ids, items_json, barrier, delay=0):
    """Worst case for one rollout of a pair; every step logged with start and end times."""
    import asyncio, json, time
    L, col = EDITS[tag]
    name = f"model.language_model.layers.{L}.mlp.down_proj.weight"

    async def step(key, *a):
        a0 = time.time()
        try:
            r = await call(*a)
        except Exception as e:
            r = ("EXC", repr(e)[:500])
        rec = {"rollout": tag, "step": key, "start": round(a0 - t0), "end": round(time.time() - t0)}
        if isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[0], bool):
            rec["ok"] = r[0]
            rec["info"] = (r[1][:400] if not r[0] else {k: v for k, v in json.loads(r[1]).items() if k in ("original_from_cache", "original", "edited", "texts", "max_tokens_in_a_text", "files")})
        elif isinstance(r, str):
            rec["info"] = r[:400]
        elif isinstance(r, dict):
            rec["info"] = {"edited_tensors": r.get("edited_tensors"), "answers": len(r.get("edited") or []),
                           "accuracy": round(sum(x["correct"] for x in r["edited"]) / len(r["edited"]), 4) if r.get("edited") else None}
        else:
            rec["info"] = str(r)[:400]
        ev.append(rec); print("EV", json.dumps(rec, default=str)[:600], flush=True)
        return r

    tf_long = json.dumps({"type": "teacher_forced", "model": "edited", "texts": [long_text],
                          "return": {"token_logprobs": True, "topk": 20, "hidden_states": list(range(25)), "mlp_act": list(range(24))}})
    both = json.dumps({"type": "generate", "model": "both", "problems": gen_ids, "n": 2, "max_tokens": 8192})
    edited = json.dumps({"type": "generate", "model": "edited", "problems": gen_ids, "n": 2, "max_tokens": 8192})
    boot = await call("boot_id")
    await step("edit1", "py", f"n = '{name}'; W = load(n); W[:, {col}] *= 0.5; print(save_edit(n, W))")
    both = json.dumps({"type": "generate", "model": "both", "problems": gen_ids, "n": 2, "max_tokens": 4096})
    if tag == "A":  # 256 answers per model from the start, against the shared original server and A's edited server
        await step("both_256", "forward", both)
    else:  # start B's edited server, then restart it with a new edit while A is generating, then generate 256 per model
        await step("edited_256_short", "forward", json.dumps({"type": "generate", "model": "edited", "problems": gen_ids, "n": 2, "max_tokens": 1024}))
        await step("edit2", "py", f"W[:, {col}] = 0; print(save_edit(n, W))")
        await step("both_256_after_restart", "forward", both)
    await step("py_check", "py", "import os; print(sorted(os.listdir('/work')), edited(), os.listdir('/slots'))")
    await barrier.wait()
    await step("reward_128x2", "reward", items_json, 2, 2048, False)
    return {"boot_unchanged": (await call("boot_id")) == boot}


@app.function(image=drv_image, volumes={"/runs": runs, "/heldout": heldout_vol, "/weights": weights}, timeout=4 * 3600)
async def stress(mps: int = 0):
    import asyncio, glob, json, os, time
    from tokenizers import Tokenizer
    sp = json.load(open("/runs/data/math_practice/splits.json"))
    prob = {json.loads(l)["id"]: json.loads(l) for l in open("/runs/data/math_practice/problems.jsonl")}
    tok = Tokenizer.from_file(sorted(glob.glob("/weights/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/tokenizer.json"))[0])
    long_text = ""
    for i in sp["extra"][400:]:
        nxt = long_text + prob[i]["problem"] + "\n" + prob[i]["solution"] + "\n\n"
        if len(tok.encode(nxt).ids) > 8000:
            break
        long_text = nxt
    items_json = json.dumps([json.loads(l) for l in open("/heldout/rl/reward_math.jsonl")][:128])
    stamp = str(int(time.time()))
    obj = PackedTarget.with_options(scaledown_window=10)(gid=f"packstress-{stamp}", mps=mps)
    ev, mem, done = [], [], asyncio.Event()
    t0 = time.time()
    await obj.stats.remote.aio()
    boot_s = round(time.time() - t0)
    t0 = time.time()

    async def watch():
        while not done.is_set():
            try:
                st = await obj.stats.remote.aio()
                mem.append((round(time.time() - t0), int(st["gpu_memory"].split(",")[0].split()[0]), st["forward_model_peak_mib"], st["original_server"],
                            {k[:1]: v["edited_server"] for k, v in st["slots"].items()}))
            except Exception as e:
                mem.append((round(time.time() - t0), repr(e)[:200]))
            try:
                await asyncio.wait_for(done.wait(), 5)
            except asyncio.TimeoutError:
                pass

    w = asyncio.create_task(watch())
    barrier = asyncio.Barrier(2)
    mk = lambda rid: (lambda m, *a: getattr(obj, m).remote.aio(rid, *a))
    ra, rb = await asyncio.gather(stress_seq(mk(f"A-{stamp}"), "A", t0, ev, long_text, sp["extra"][100:228], items_json, barrier),
                                  stress_seq(mk(f"B-{stamp}"), "B", t0, ev, long_text, sp["extra"][228:356], items_json, barrier))
    done.set(); await w
    nums = [m for m in mem if len(m) == 5]
    res = {"mps": mps, "boot_s": boot_s, "long_text_tokens": len(tok.encode(long_text).ids), "events": ev, "A": ra, "B": rb,
           "gpu_peak_mib": max((m[1] for m in nums), default=None), "forward_model_peak_mib": max((m[2] for m in nums), default=None),
           "stats_errors": [m for m in mem if len(m) != 5], "mem": mem, "wall_s": round(time.time() - t0)}
    os.makedirs("/runs/rl/_packtest", exist_ok=True)
    json.dump(res, open(f"/runs/rl/_packtest/stress_{stamp}.json", "w"), indent=1, default=str)
    await runs.commit.aio()
    print("STRESS peak MiB", res["gpu_peak_mib"], "forward model peak MiB", res["forward_model_peak_mib"], "wall", res["wall_s"], flush=True)
    return res


@app.local_entrypoint()
def packstress(mps: int = 0, out: str = "/tmp/packstress_result.json"):
    import json
    res = stress.remote(mps)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "mem"}, indent=1, default=str)[:20000])
