"""The environment: the target model (Qwen3.5-0.8B by default; see TARGETS) on Modal with the two tools an editor uses, plus the scoring.

- `Target` (one L40S per search or rollout): a Python sandbox over the weight files, the forward tool
  (generation and teacher-forced passes), a reward method, and a snapshot of the work folder.
- `PackedTarget`: the same tools for two rollouts on one L40S (each method takes the rollout id first).
- `heldout_eval`: MATH-500 (or another set in EVAL_SETS), original weights against the submitted edit, paired by problem.
- `prep`: downloads the target and builds the practice and held-out sets (another target: downloads it and copies the practice files).
- `TARGETS`, `target_arch`, `target_paths`: the models that can be the target, their architecture read from config.json and
  the safetensors headers, and where each target's practice files and cached answers live.
- `paper_edit`: writes the published Math edit of arXiv 2611.09382 as a control.

    uvx modal run -m harness.target --step prep [--target qwen3.5-2b]
    uvx modal run -m harness.target --step arch --target qwen3.5-2b
    uvx modal run -m harness.target --step smoke
    uvx modal run --detach -m harness.target --step heldout --run-id <run>
"""
import re

import modal

# The models that can be the target. Only the Hugging Face id and the GPU shares are set here; layer count, sizes, attention
# layout and tensor names are read from the downloaded config.json and safetensors headers (target_arch).
# util: Target (one rollout per L40S): GPU share of the edited and original vLLM servers; tf_tokens: tokens per teacher-forced batch.
# pack_util / pack_tf_tokens: the same for PackedTarget (two rollouts per L40S: two edited servers, one shared original server,
# the teacher-forced model capped at forward_model). The 0.8B values were measured (results/packing_test.md).
# 2B (4.5 GB of bf16 weights against 1.7 GB; the same KV cache per token, since both have 6 full-attention layers with
# 2 KV heads of width 256): vLLM servers keep about 5 GB for the KV cache at 0.25, about 2 GB at 0.16; the teacher-forced
# model gets 0.22 (10 GB: 4.5 GB of weights, activations of 4,096-token batches at twice the hidden size). Not yet measured.
TARGETS = {
    "qwen3.5-0.8b": {"hf": "Qwen/Qwen3.5-0.8B", "util": {"edited": 0.40, "original": 0.40}, "tf_tokens": 40000,
                     "pack_util": {"edited": 0.30, "original": 0.15, "forward_model": 0.13}, "pack_tf_tokens": 8192},
    "qwen3.5-2b": {"hf": "Qwen/Qwen3.5-2B", "util": {"edited": 0.33, "original": 0.33}, "tf_tokens": 16000,
                   "pack_util": {"edited": 0.25, "original": 0.16, "forward_model": 0.22}, "pack_tf_tokens": 4096},
    # pretrained only (no post-training): can the editor improve a model before instruction tuning? Same architecture as the 0.8B.
    "qwen3.5-0.8b-base": {"hf": "Qwen/Qwen3.5-0.8B-Base", "util": {"edited": 0.40, "original": 0.40}, "tf_tokens": 40000,
                          "pack_util": {"edited": 0.30, "original": 0.15, "forward_model": 0.13}, "pack_tf_tokens": 8192},
}
DEFAULT_TARGET = "qwen3.5-0.8b"
TARGET = TARGETS[DEFAULT_TARGET]["hf"]
GEN_CAP = 8192
FORWARD_BUDGET = 50
PY_BUDGET = 300
SRC = ("harness", "editors", "rl")  # local packages every container needs

weights = modal.Volume.from_name("glimmer-weights", create_if_missing=True)
runs = modal.Volume.from_name("glimmer-runs", create_if_missing=True)
heldout_vol = modal.Volume.from_name("glimmer-heldout", create_if_missing=True)

base = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm", "huggingface_hub", "openai", "datasets", "safetensors", "numpy", "scipy",
                 "scikit-learn", "pandas", "sympy", "networkx", "tokenizers")
    .env({"HF_HOME": "/weights/hf", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
)
offline = base.env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}).add_local_python_source(*SRC)
base_src = base.add_local_python_source(*SRC)
app = modal.App("weight-edit")



# ----------------------------------------------------------------------------- grading
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def _strip_text(a):
    b = re.sub(r"\\(?:text|mbox|mathrm|textbf)\{[^{}]*\}", "", a)
    if re.sub(r"[\s\\,;!.]", "", b):
        return b
    return re.sub(r"\\(?:text|mbox|mathrm|textbf)\{([^{}]*)\}", r"\1", a)


def norm_math(a):
    a = a.strip()
    a = a.replace("\\displaystyle", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = _strip_text(a)
    for x in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad", "^\\circ", "^{\\circ}", "\\%", "%", "\\$", "$", "\n", " ", "~"]:
        a = a.replace(x, "")
    a = a.rstrip(".")
    if "=" in a and "\\begin" not in a:
        lhs, rhs = a.rsplit("=", 1)
        if rhs and not re.search(r"[<>]", lhs):
            a = rhs
    a = re.sub(r"^[a-zA-Z]\\in", "", a)
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac(\d)\{([^{}]+)\}", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", a)
    a = re.sub(r"\^\{(\w)\}", r"^\1", a)
    a = re.sub(r"_\{(\w)\}", r"_\1", a)
    a = re.sub(r"\\begin\{pmatrix\}", "[", a); a = re.sub(r"\\end\{pmatrix\}", "]", a)
    a = re.sub(r"\\\\\[\d+pt\]", r"\\\\", a)
    a = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", a) if ("[" in a) else a
    m = re.fullmatch(r"\(?([A-Ja-j])\)?", a)
    if m:
        return m.group(1).upper()
    if a.startswith("\\{") and a.endswith("\\}"):
        a = a[2:-2]
    return a


def to_num(a):
    a = a.replace(",", "")
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m and int(m.group(3)) != 0:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", a)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(a)
    except Exception:
        return None


def math_equal(pred, gold):
    x, g = norm_math(pred), norm_math(gold)
    if x == g:
        return True
    fx, fg = to_num(x), to_num(g)
    if fx is not None and fg is not None:
        return abs(fx - fg) < 1e-9
    if "(" not in x and "[" not in x and "(" not in g and "[" not in g and "," in x and "," in g:
        return sorted(x.split(",")) == sorted(g.split(","))
    return False


def grade_math(text, gold):
    """Returns (extracted, correct). Uses only the text after the last </think> when present."""
    tail = text.split("</think>")[-1] if "</think>" in text else text
    b = last_boxed(tail)
    if b is None:
        return None, False
    return b, math_equal(b, gold)


def paired_stats(rows_o, rows_e):
    """Paired per-question difference edited - original, clustered by question."""
    import collections, math
    by = collections.defaultdict(lambda: [[], []])
    for r in rows_o:
        by[r["qid"]][0].append(int(r["correct"]))
    for r in rows_e:
        by[r["qid"]][1].append(int(r["correct"]))
    d = [sum(e) / len(e) - sum(o) / len(o) for o, e in by.values() if o and e]
    Q = len(d)
    mean = sum(d) / Q if Q else 0.0
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (Q * (Q - 1))) if Q > 1 else float("nan")
    key = lambda r: (r["qid"], r["sample"])
    eo = {key(r): r["correct"] for r in rows_o}
    w2r = sum(1 for r in rows_e if key(r) in eo and (not eo[key(r)]) and r["correct"])
    r2w = sum(1 for r in rows_e if key(r) in eo and eo[key(r)] and (not r["correct"]))
    return {"diff_pp": round(100 * mean, 2), "se_pp": round(100 * se, 2), "questions": Q, "wrong_to_right": w2r, "right_to_wrong": r2w}


def summarize(rows, cap):
    import collections
    n = len(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("source", "custom")].append(r)
    graded = [r for r in rows if r["correct"] is not None]
    out = {
        "n": n,
        "accuracy_pct": round(100 * sum(r["correct"] for r in graded) / len(graded), 2) if graded else None,
        "correct": sum(bool(r["correct"]) for r in graded), "graded": len(graded),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / n) if n else 0,
        "cut_at_max_tokens_pct": round(100 * sum(r["finish"] == "length" for r in rows) / n, 2) if n else 0,
        "no_boxed_answer_pct": round(100 * sum(r["extracted"] is None for r in graded) / len(graded), 2) if graded else None,
        "by_source": {s: f"{sum(bool(r['correct']) for r in v)}/{len(v)}" for s, v in by.items() if v and v[0]["correct"] is not None},
    }
    return out


# ----------------------------------------------------------------------------- vLLM helper
def start_vllm(model, name, port, util, max_len, extra=(), log=None):
    import subprocess, time, urllib.request
    logf = log or f"/tmp/vllm_{name}.log"
    log = open(logf, "w")
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--served-model-name", name, "--port", str(port), "--host", "127.0.0.1",
         "--max-model-len", str(max_len), "--gpu-memory-utilization", str(util),
         "--limit-mm-per-prompt", '{"image": 0, "video": 0}', *extra],
        stdout=log, stderr=log)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            tail = open(logf).read()[-3000:]
            raise RuntimeError(f"vllm {name} exited with code {proc.returncode}\n{tail}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(3)
        if time.time() - t0 > 25 * 60:
            proc.kill()
            raise RuntimeError(f"vllm {name} did not come up in 25 minutes")


def stop_proc(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


def chat(port, name, messages, max_tokens, seed=None, timeout=3000, **kw):
    import json, urllib.request
    body = {"model": name, "messages": messages, "max_tokens": max_tokens, **kw}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _check_target(target):
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; known: {sorted(TARGETS)}")
    return target


def target_dir(target=DEFAULT_TARGET):
    """The downloaded weights of a target: the snapshot the Hugging Face cache on the weights volume points to
    (what snapshot_download(..., local_files_only=True) returns). `prep --target` downloads them."""
    root = f"/weights/hf/hub/models--{TARGETS[_check_target(target)]['hf'].replace('/', '--')}"
    return f"{root}/snapshots/{open(f'{root}/refs/main').read().strip()}"


def safetensors_names(mdir):
    """Tensor names of every .safetensors file in a folder, read from the file headers (no weights are loaded)."""
    import json, os, struct
    names = []
    for f in sorted(os.listdir(mdir)):
        if f.endswith(".safetensors"):
            with open(os.path.join(mdir, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                names += [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
    return names


def arch_from(cfg, names, target="", hf=""):
    """The architecture facts the prompt and tools state, from a config.json dict and the list of tensor names.
    Raises if the config and the tensor names disagree (for example on which layers have full attention)."""
    import re, collections
    tc = cfg.get("text_config", cfg)
    L = tc["num_hidden_layers"]
    types = tc.get("layer_types") or ["full_attention"] * L
    assert len(types) == L, "layer_types does not match num_hidden_layers"
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    linear = [i for i, t in enumerate(types) if t == "linear_attention"]
    assert len(full) + len(linear) == L, f"unknown layer types {sorted(set(types))}"
    # the transformer blocks: the prefix whose layers 0..L-1 all have an MLP down projection (mtp.layers.* is a separate head)
    pre = collections.Counter(m.group(1) for n in names if (m := re.fullmatch(r"(.+)\.layers\.(\d+)\.mlp\.down_proj\.weight", n)))
    prefix = [p for p, c in pre.items() if c == L]
    assert len(prefix) == 1, f"cannot find the {L} transformer blocks among {dict(pre)}"
    prefix = prefix[0]
    sub = lambda l: [n[len(f"{prefix}.layers.{l}."):] for n in names if n.startswith(f"{prefix}.layers.{l}.")]
    order = lambda xs, pref: sorted(xs, key=lambda x: (pref.index(x) if x in pref else len(pref), x))
    mods = lambda l: {x.split(".")[0] for x in sub(l)} - {"mlp", "input_layernorm", "post_attention_layernorm"}
    mlp = order({x.split(".")[1] for x in sub(0) if x.startswith("mlp.") and x.endswith("_proj.weight")}, ["gate_proj", "up_proj", "down_proj"])
    attn_mod = {m for l in full for m in mods(l)}
    lin_mod = {m for l in linear for m in mods(l)}
    assert len(attn_mod) == 1 and (not linear or len(lin_mod) == 1) and not (attn_mod & lin_mod), \
        f"attention modules do not follow layer_types: full {attn_mod}, linear {lin_mod}"
    attn_mod = attn_mod.pop()
    attn_proj = order({x.split(".")[1] for x in sub(full[0]) if x.startswith(attn_mod + ".") and x.endswith("_proj.weight")}, ["q_proj", "k_proj", "v_proj", "o_proj"])
    tied = bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    lm_head = [n for n in names if n.endswith("lm_head.weight")]
    assert tied != bool(lm_head), f"tie_word_embeddings={tied} but lm_head tensors {lm_head}"
    k = tc.get("full_attention_interval")
    return {"target": target, "hf": hf, "name": hf.split("/")[-1] if hf else target,
            "layers": L, "hidden": tc["hidden_size"], "mlp": tc["intermediate_size"], "vocab": tc.get("vocab_size"),
            "full_attn": full, "linear_attn": linear, "full_every": k if k and full == [i for i in range(L) if (i + 1) % k == 0] else None,
            "heads": tc["num_attention_heads"], "kv_heads": tc.get("num_key_value_heads"), "head_dim": tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"],
            "tied": tied, "prefix": prefix, "mlp_proj": mlp, "attn_mod": attn_mod, "attn_proj": attn_proj,
            "linear_mod": lin_mod.pop() if linear else None, "vision": "vision_config" in cfg}


def target_arch(target=DEFAULT_TARGET, mdir=""):
    """arch_from for a downloaded target (needs the weights volume at /weights)."""
    import json, os
    mdir = mdir or target_dir(target)
    a = arch_from(json.load(open(os.path.join(mdir, "config.json"))), safetensors_names(mdir), target, TARGETS[target]["hf"])
    a["dir"] = mdir
    return a


# The task: which problems the editor practises on and is rewarded on, and which held-out set measures it. "math" (the default)
# is the original setup (MATH + GSM8K practice, MATH reward set, MATH-500). Every other task has its own practice folder,
# reward set and held-out set (EVAL_SETS[eval_set]), under a per-(target, task) folder, built once by rl.loop::prep_task.
# Prompt texts (rl.loop.editor_prompt): `measured` replaces the MATH-500 sentence, `practice` the MATH/GSM8K practice sentence,
# `ids` the example practice ids; None keeps the math wording.
TASKS = {
    "math": {"reward_file": "/heldout/rl/reward_math.jsonl", "eval_set": "math500", "measured": None, "practice": None, "ids": None},
    "numina_cn_k12": {
        "reward_file": "/heldout/tasks/numina_cn_k12/reward.jsonl", "eval_set": "numina_cn_k12", "hf": "AI-MO/NuminaMath-CoT", "source": "cn_k12",
        "measured": ("HOW IT WILL BE MEASURED. 500 held-out problems from the same source as the practice problems (Chinese K-12 school mathematics, "
                     "the cn_k12 part of NuminaMath-CoT, each with a single final answer), 2 samples per problem, separately for original and edited weights."),
        "practice": ("2. Practice problems: <<N>> problems from the cn_k12 part of NuminaMath-CoT (Chinese K-12 school mathematics in English: "
                     "algebra, functions, sequences, geometry, probability, calculus), each with a single final answer and a reference solution. "
                     "They are disjoint from the held-out test."),
        "ids": 'Practice ids look like "cn_k12/17" and "cn_k12/2954".'},
}


def _check_task(task):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return task


def target_paths(target=DEFAULT_TARGET, task="math"):
    """Where a target's practice folder, cached answers and starting models live. The default target's files predate
    the registry and stay where they were; every other target has its own folders, so answers are never mixed.
    A task other than "math" has its own folder per target: /runs/data/targets/<target>/tasks/<task>/."""
    if _check_task(task) != "math":
        r = f"/runs/data/targets/{_check_target(target)}/tasks/{task}"
        return {"practice": f"{r}/practice", "eval_cache": f"{r}/eval_cache", "math500_cache": f"{r}/eval_cache/math500_original.jsonl",
                "reward_base": r + "/reward_base_{}.json", "bases_w": target_paths(target)["bases_w"], "bases_r": f"{r}/bases", "data": r}
    if _check_target(target) == DEFAULT_TARGET:
        return {"practice": "/runs/data/math_practice", "eval_cache": "/runs/data/eval_cache", "math500_cache": "/runs/heldout_original_cached.jsonl",
                "reward_base": "/runs/rl/_reward_base_{}.json", "bases_w": "/weights/bases", "bases_r": "/runs/data/bases", "data": f"/runs/data/targets/{target}"}
    r = f"/runs/data/targets/{target}"
    return {"practice": f"{r}/practice", "eval_cache": f"{r}/eval_cache", "math500_cache": f"{r}/eval_cache/math500_original.jsonl",
            "reward_base": r + "/reward_base_{}.json", "bases_w": f"/weights/bases/{target}", "bases_r": f"{r}/bases", "data": r}


BASES_W, BASES_R = "/weights/bases", "/runs/data/bases"  # the default target's starting models made by rebasing: weights, and their practice folder and caches


def base_dirs(tag="", target=DEFAULT_TARGET, task="math"):
    """(model dir, practice dir) of a starting model; tag "" is the target as downloaded."""
    P = target_paths(target, task)
    if not tag:
        return target_dir(target), P["practice"]
    return f"{P['bases_w']}/{tag}", f"{P['bases_r']}/{tag}/practice"


# ----------------------------------------------------------------------------- evaluation sets
def _build_gsm8k_test(n=500, seed=20261115):
    """500 problems of GSM8K's test split, a fixed random sample (the practice set uses only GSM8K's train split)."""
    import random
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main")["test"]):
        sol, ans = r["answer"].rsplit("####", 1)
        rows.append({"qid": f"gsm8k_test/{i}", "problem": r["question"], "answer": ans.strip().replace(",", ""), "source": "gsm8k_test"})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _build_aime2024(copies=4):
    """AIME 2024 (30 problems), each listed 4 times under distinct qids so that 2 samples per row give 8 answers per problem
    (240 answers); 60 answers would be too few to resolve a gain."""
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset("HuggingFaceH4/aime_2024")["train"]):
        for k in range(copies):
            rows.append({"qid": f"aime2024/{i}#{k}", "problem": r["problem"], "answer": str(r["answer"]).strip(), "source": "aime2024"})
    return rows


# Sets an edit can be scored on (heldout_eval, rl.loop::evaluate). To add one: a file of {qid, problem, answer, source} rows,
# a function that builds it (run once by prep_eval_set; None if `prep` writes it) and a grader (text, gold) -> (extracted, correct).
# The starting model's answers on each set are cached per starting model (heldout_cache).
EVAL_SETS = {
    "math500": {"file": "/heldout/math500.jsonl", "build": None, "grade": grade_math},
    "gsm8k_test": {"file": "/heldout/gsm8k_test.jsonl", "build": _build_gsm8k_test, "grade": grade_math},
    "aime2024": {"file": "/heldout/aime2024.jsonl", "build": _build_aime2024, "grade": grade_math},
    # a task's held-out set (written by rl.loop::prep_task); its cached answers live in that task's folder
    "numina_cn_k12": {"file": "/heldout/tasks/numina_cn_k12/heldout.jsonl", "build": None, "grade": grade_math, "task": "numina_cn_k12"},
}


def heldout_cache(tag="", dataset="math500", target=DEFAULT_TARGET):
    """The starting model's answers on an evaluation set, generated once (2 per problem, same seeds as heldout_eval)."""
    P = target_paths(target, EVAL_SETS[dataset].get("task", "math"))
    if dataset == "math500":
        return f"{P['bases_r']}/{tag}/heldout_original.jsonl" if tag else P["math500_cache"]
    return f"{P['bases_r']}/{tag}/eval_{dataset}_original.jsonl" if tag else f"{P['eval_cache']}/{dataset}_original.jsonl"


@app.function(image=base_src, volumes={"/heldout": heldout_vol}, timeout=1800, cpu=2, memory=8192)
def prep_eval_set(dataset: str):
    """Write an evaluation set's problem file once; returns its number of problems."""
    import json, os
    spec = EVAL_SETS[dataset]
    heldout_vol.reload()
    if not os.path.exists(spec["file"]):
        if spec["build"] is None:
            raise RuntimeError(f"{spec['file']} is missing; run `uvx modal run -m harness.target --step prep`" + (f" / `rl.loop::prep_task --task {spec['task']}`" if spec.get("task") else ""))
        rows = spec["build"]()
        with open(spec["file"], "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        heldout_vol.commit()
    return sum(1 for _ in open(spec["file"]))


def heldout_rows(port, name, items, n, grade=grade_math, logit_bias=None):
    """Answers to evaluation problems, as heldout_eval generates them (seeded by problem and sample).
    logit_bias: {token id: bias}; -100 bans a token at decoding (thinking included), with no weight change."""
    import concurrent.futures as cf, zlib
    jobs = [(it, s) for it in items for s in range(n)]

    def one(job):
        it, s = job
        seed = zlib.crc32(f"heldout|{it['qid']}|{s}".encode()) & 0x7FFFFFFF
        try:
            r = chat(port, name, [{"role": "user", "content": it["problem"] + "\n\nPut your final answer in \\boxed{}."}], GEN_CAP, seed=seed,
                     temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, chat_template_kwargs={"enable_thinking": True},
                     **({"logit_bias": {str(k): v for k, v in logit_bias.items()}} if logit_bias else {}))
            ch = r["choices"][0]
            rs = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
            text = (("<think>" + rs + "</think>") if rs else "") + (ch["message"].get("content") or "")
            fin, toks = ch["finish_reason"], r["usage"]["completion_tokens"]
        except Exception as e:
            text, fin, toks = f"[request failed: {repr(e)[:200]}]", "error", 0
        ext, ok = grade(text, it["answer"])
        return {"qid": it["qid"], "sample": s, "source": it.get("source", "math500"), "text": text, "extracted": ext, "correct": ok, "tokens": toks, "finish": fin}

    with cf.ThreadPoolExecutor(256) as ex:
        return list(ex.map(one, jobs))


def ban_ids(tok, words):
    """Token ids for each string in `words` (each must be one token)."""
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{w!r} is not a single token: {ids}")
        out[w] = ids[0]
    return out


@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, timeout=6 * 3600, memory=32768)
def ban_eval(configs_json: str, dataset: str = "math500", n: int = 2, limit: int = 0, target: str = DEFAULT_TARGET):
    """Decode-time ban baselines: for each config {"label", "base", "ban": [token strings]}, generate the evaluation set
    with those tokens banned (logit bias -100) on starting model `base` ("" = the target as downloaded), one config at a
    time on one GPU, and compare with the target's cached original answers (same seeds). Results: /runs/eval/ban/<label>.json."""
    import json, os, time
    from transformers import AutoTokenizer
    runs.reload()
    spec = EVAL_SETS[dataset]
    items = [json.loads(l) for l in open(spec["file"])]
    if limit:
        items = items[:limit]
    want = {(it["qid"], s) for it in items for s in range(n)}
    orig = [r for r in map(json.loads, open(heldout_cache("", dataset, target))) if (r["qid"], r["sample"]) in want]
    results = []
    for cfg in json.loads(configs_json):
        mdir = base_dirs(cfg.get("base", ""), target)[0]
        tok = AutoTokenizer.from_pretrained(mdir)
        ids = ban_ids(tok, cfg["ban"])
        print(cfg["label"], "banned token ids", ids)
        proc = start_vllm(mdir, "banned", 8001, 0.90, GEN_CAP + 3000)
        t = time.time()
        rows = heldout_rows(8001, "banned", items, n, spec["grade"], logit_bias={i: -100 for i in ids.values()})
        stop_proc(proc); time.sleep(10)
        out = {"label": cfg["label"], "base": cfg.get("base", ""), "ban": ids, "dataset": dataset, "problems": len(items), "n_per_problem": n,
               "cap": GEN_CAP, "seconds": round(time.time() - t), "banned": summarize(rows, GEN_CAP), "original": summarize(orig, GEN_CAP),
               "paired_vs_original": paired_stats(orig, rows),
               "banned_token_in_output": sum(any(w in r["text"] for w in ids) for r in rows)}
        os.makedirs("/runs/eval/ban", exist_ok=True)
        with open(f"/runs/eval/ban/{cfg['label']}.json", "w") as f:
            json.dump(out, f, indent=1)
        with open(f"/runs/eval/ban/{cfg['label']}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        runs.commit()
        print("BAN", json.dumps(out))
        results.append(out)
    return results


# ----------------------------------------------------------------------------- prep
@app.function(image=base_src, volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, timeout=3600, cpu=4, memory=16384)
def prep(target: str = DEFAULT_TARGET):
    """The default target: download it, write MATH-500 and the practice set. Another target: download it and copy the
    default target's practice files (problems, index, splits), so every target is diagnosed and confirmed on the same problems."""
    import json, os, random, re, shutil
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    p = snapshot_download(TARGETS[_check_target(target)]["hf"])
    print("target at", p, os.listdir(p))
    if target != DEFAULT_TARGET:
        weights.commit()
        src, dst = target_paths(DEFAULT_TARGET)["practice"], target_paths(target)["practice"]
        os.makedirs(dst, exist_ok=True)
        for f in ("problems.jsonl", "index.tsv", "splits.json", "README.txt"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        runs.commit()
        a = target_arch(target, p)
        print("practice copied to", dst, "| arch", json.dumps({k: v for k, v in a.items() if k != "dir"}))
        return a
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    m500 = load_dataset("HuggingFaceH4/MATH-500")["test"]
    held = {norm(r["problem"]) for r in m500}
    os.makedirs("/heldout", exist_ok=True)
    with open("/heldout/math500.jsonl", "w") as f:
        for r in m500:
            f.write(json.dumps({"qid": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "source": "math500"}) + "\n")
    rng = random.Random(20261113)
    rows = []
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    math_rows = []
    for s in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", s)["train"]
        for i, r in enumerate(ds):
            ans = last_boxed(r["solution"])
            if ans is None or norm(r["problem"]) in held:
                continue
            math_rows.append({"id": f"math/{s}/{i}", "source": "math_train", "level": r.get("level", ""), "problem": r["problem"], "answer": ans, "solution": r["solution"]})
    rng.shuffle(math_rows)
    rows += math_rows[:3000]
    g = load_dataset("openai/gsm8k", "main")["train"]
    gs = []
    for i, r in enumerate(g):
        if "####" not in r["answer"]:
            continue
        sol, ans = r["answer"].rsplit("####", 1)
        gs.append({"id": f"gsm8k/{i}", "source": "gsm8k_train", "level": "", "problem": r["question"], "answer": ans.strip().replace(",", ""), "solution": sol.strip()})
    rng.shuffle(gs)
    rows += gs[:1500]
    os.makedirs("/runs/data/math_practice", exist_ok=True)
    with open("/runs/data/math_practice/problems.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open("/runs/data/math_practice/README.txt", "w") as f:
        f.write("problems.jsonl: one JSON per line with id, source (math_train or gsm8k_train), level, problem, answer, solution (reference solution).\n"
                f"{len(rows)} problems. None of them appears in the held-out test after normalized text matching.\n")
    runs.commit(); heldout_vol.commit(); weights.commit()
    print("practice", len(rows), "heldout", len(m500))


# ----------------------------------------------------------------------------- sandbox child
KERNEL_SRC = r'''
import sys, json, io, os, ast, traceback, contextlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_grad_enabled(False)
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tokenizers import Tokenizer
MODEL_DIR, WORK = sys.argv[1], sys.argv[2]
_ST = [os.path.join(MODEL_DIR, f) for f in sorted(os.listdir(MODEL_DIR)) if f.endswith(".safetensors")]
_WHERE = {}
for _p in _ST:
    with safe_open(_p, "pt") as _f:
        for _k in _f.keys():
            _WHERE[_k] = _p
NAMES = sorted(_WHERE)
_EDIT = os.path.join(WORK, "edited")
os.makedirs(_EDIT, exist_ok=True)
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
def _fn(name): return os.path.join(_EDIT, name + ".safetensors")
def load_original(name):
    with safe_open(_WHERE[name], "pt") as f:
        return f.get_tensor(name).to(torch.float32)
def load(name):
    if os.path.exists(_fn(name)):
        return load_file(_fn(name))["t"].to(torch.float32)
    return load_original(name)
def edited():
    return sorted(f[:-len(".safetensors")] for f in os.listdir(_EDIT) if f.endswith(".safetensors"))
def drop_edit(name):
    if os.path.exists(_fn(name)): os.remove(_fn(name))
def save_edit(name, tensor):
    if name not in _WHERE: raise ValueError("unknown tensor name")
    low = name.lower()
    if ".layers." not in low or "embed_tokens" in low or "lm_head" in low or "visual" in low or "vision" in low:
        raise ValueError("refused: only transformer-block tensors of the language model can be edited")
    o = load_original(name)
    t = torch.as_tensor(tensor).to(torch.float32)
    if t.shape != o.shape: raise ValueError(f"shape {tuple(t.shape)} does not match {tuple(o.shape)}")
    if not torch.isfinite(t).all(): raise ValueError("refused: non-finite values")
    den = float((o * o).sum())
    if den > 0:
        c = float((t * o).sum()) / den
        if abs(c - 1) > 1e-6 and torch.allclose(t, c * o, rtol=1e-4, atol=1e-7):
            raise ValueError("refused: the edit only multiplies the whole tensor by a constant")
    tb = t.to(torch.bfloat16)
    save_file({"t": tb}, _fn(name))
    changed = int((tb != o.to(torch.bfloat16)).sum())
    return f"saved {name}: {changed} BF16 entries differ from the original"
_NS = {"np": np, "torch": torch, "NAMES": NAMES, "load": load, "load_original": load_original,
       "save_edit": save_edit, "drop_edit": drop_edit, "edited": edited, "tok": tok, "__name__": "__main__"}
os.chdir(WORK)
sys.__stdout__.write("READY\n"); sys.__stdout__.flush()
for line in sys.stdin:
    code = json.loads(line)["code"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), _NS)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), _NS)
                if v is not None: print(repr(v))
        except BaseException:
            tb = traceback.format_exc().splitlines()
            print("\n".join(l for l in tb if "kernel_child" not in l))
    out = buf.getvalue().encode("ascii", "replace").decode()
    sys.__stdout__.write(json.dumps({"out": out}) + "\n"); sys.__stdout__.flush()
'''


# ----------------------------------------------------------------------------- Target
SLOTS = "/slots"  # PackedTarget: one folder per rollout, hidden from every kernel
PACK_UTIL = TARGETS[DEFAULT_TARGET]["pack_util"]  # PackedTarget: GPU share of each rollout's edited vLLM server, the shared original server, the teacher-forced model (per target: TARGETS)
KERNEL_MEM = 36 * 2 ** 30  # PackedTarget: address-space limit of each python kernel, so one rollout cannot take the container's memory from the other


class _Slot:
    """One rollout's state: its /work and /library, its python kernel, its edited vLLM server, its restart id."""

    def __init__(self, work, lib, port, ckpt, isolate=False):
        import threading, uuid
        self.work, self.lib, self.port, self.ckpt, self.isolate = work, lib, port, ckpt, isolate
        self.kernel, self.k = None, 0
        self.edited_proc, self.edited_hash = None, None
        self.boot = uuid.uuid4().hex  # changes if the container is replaced, which resets /work
        self.lock = threading.RLock()

    def real(self, path):
        """The container path of a path as the editor sees it: /work and /library belong to the rollout."""
        for v, r in (("/work", self.work), ("/library", self.lib)):
            if path == v or path.startswith(v + "/"):
                return r + path[len(v):]
        return path


class _Core:
    """The tools of one rollout (a `_Slot`) on a container that holds the original weights."""
    # edited_util, original_util, tf_tokens (tokens per teacher-forced batch): set by _setup from TARGETS

    def _setup(self, packed=False):
        import os, json, glob, threading
        # `target` (a modal.parameter on the classes) is a key of TARGETS. `base` names a starting model made by rl.loop's
        # rebasing: earlier edits applied to the weights, with its own cached original answers. "" is the target as downloaded.
        self.tgt = _check_target(getattr(self, "target", "") or DEFAULT_TARGET)
        spec = TARGETS[self.tgt]
        u = spec["pack_util"] if packed else spec["util"]
        self.edited_util, self.original_util = u["edited"], u["original"]
        self.tf_tokens = spec["pack_tf_tokens"] if packed else spec["tf_tokens"]
        self.task = _check_task(getattr(self, "task", "") or "math")
        self.mdir, self.pdir = base_dirs(getattr(self, "base", ""), self.tgt, self.task)
        os.makedirs("/work", exist_ok=True)
        for link, dst in [("/model", self.mdir), ("/practice", self.pdir)]:
            if not os.path.exists(link):
                os.symlink(dst, link)
        open("/tmp/kernel_child.py", "w").write(KERNEL_SRC)
        self.gpu_lock = threading.Lock()  # vLLM start-ups and teacher-forced passes: one at a time
        self.commit_lock = threading.Lock()
        self.orig_lock = threading.Lock()
        self.orig = None
        self.practice = {}
        for line in open(f"{self.pdir}/problems.jsonl"):
            r = json.loads(line)
            self.practice[r["id"]] = r
        # Answers of the original weights that were generated before (same prompt, seed and token cap): served instead of regenerated.
        self.ocache = {}
        for f in glob.glob(f"{self.pdir}/original/*.jsonl") + glob.glob(f"{self.pdir}/baseline/original.jsonl"):
            for line in open(f):
                r = json.loads(line)
                self.ocache[(r["qid"], r["sample"])] = r

    def _load_hf(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        self.tok = AutoTokenizer.from_pretrained(self.mdir)
        self.hf = AutoModelForImageTextToText.from_pretrained(self.mdir, dtype=torch.bfloat16).to("cuda").eval()
        self.sd_keys = set(self.hf.state_dict().keys())
        self.hf_state = "original"
        self.hf_backup = {}

    def _ensure_original(self):
        with self.orig_lock:
            if self.orig is None:
                with self.gpu_lock:
                    self.orig = start_vllm(self.mdir, "original", 8001, self.original_util, GEN_CAP + 3000)

    def _start_kernel(self, s):
        import subprocess, sys
        if s.kernel is not None:
            s.kernel.kill()
        cmd = [sys.executable, "-u", "/tmp/kernel_child.py", "/model", "/work"]
        if s.isolate:  # own mount and pid namespace: its /work and /library are the rollout's folders, and /slots is empty
            setup = f"mount --bind {s.work} /work && mount --bind {s.lib} /library && mount -t tmpfs none {SLOTS} && ulimit -v {KERNEL_MEM // 1024}"
            cmd = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--mount-proc",
                   "sh", "-c", setup + ' && exec "$@"', "sh", *cmd]
        s.kernel = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert s.kernel.stdout.readline().strip() == "READY"

    def _py(self, s, code):
        import json, threading
        res = {}

        def rd():
            try:
                s.kernel.stdin.write(json.dumps({"code": code}) + "\n"); s.kernel.stdin.flush()
                res["line"] = s.kernel.stdout.readline()
            except Exception as e:
                res["err"] = repr(e)

        th = threading.Thread(target=rd, daemon=True); th.start(); th.join(900)
        if th.is_alive() or not res.get("line"):
            self._start_kernel(s)
            return "[kernel restarted: the code exceeded 900 seconds or the memory limit, or the kernel crashed. All variables are lost. Files in /work remain.]"
        out = json.loads(res["line"])["out"]
        if len(out) > 10000:
            out = out[:10000] + "\n[output cut at 10,000 characters]"
        return out if out.strip() else "[no output]"

    # ---- edits
    def _edits(self, s):
        import os, hashlib
        from safetensors.torch import load_file
        d, out, h = s.real("/work/edited"), {}, hashlib.sha256()
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".safetensors"):
                    name = f[:-len(".safetensors")]
                    out[name] = load_file(os.path.join(d, f))["t"]
                    h.update(name.encode()); h.update(out[name].float().numpy().tobytes())
        return out, (h.hexdigest() if out else None)

    def _ensure_edited_server(self, s):
        import os, shutil
        from safetensors.torch import load_file, save_file
        edits, hsh = self._edits(s)
        if hsh is None:
            return None
        if hsh == s.edited_hash and s.edited_proc is not None and s.edited_proc.poll() is None:
            return hsh
        ed = s.ckpt
        shutil.rmtree(ed, ignore_errors=True); os.makedirs(ed)
        for f in os.listdir(self.mdir):
            src = os.path.join(self.mdir, f)
            if f.endswith(".safetensors"):
                sd = load_file(src)
                for n, t in edits.items():
                    if n in sd:
                        sd[n] = t.to(sd[n].dtype)
                save_file(sd, os.path.join(ed, f), metadata={"format": "pt"})
            else:
                shutil.copy(os.path.realpath(src), os.path.join(ed, f))
        with self.gpu_lock:
            stop_proc(s.edited_proc)
            s.edited_proc = start_vllm(ed, "edited", s.port, self.edited_util, GEN_CAP + 3000, log=f"/tmp/vllm_edited_{s.port}.log" if s.isolate else None)
        s.edited_hash = hsh
        return hsh

    def _set_hf(self, which, s):
        sd = self.hf.state_dict()
        for n, t in self.hf_backup.items():
            sd[n].copy_(t)
        self.hf_backup = {}
        if which == "edited":
            edits, _ = self._edits(s)
            for n, t in edits.items():
                if n not in sd:
                    raise ValueError(f"edited tensor {n} not found in the forward model")
                self.hf_backup[n] = sd[n].clone()
                sd[n].copy_(t.to(sd[n].dtype).to(sd[n].device))
        self.hf_state = which

    # ---- generation
    def _gen(self, name, port, items, n, max_tokens):
        import concurrent.futures as cf, zlib
        jobs = [(it, s) for it in items for s in range(n)]

        def one(job):
            it, s = job
            seed = zlib.crc32(f"{it['qid']}|{s}".encode()) & 0x7FFFFFFF
            try:
                r = chat(port, name, [{"role": "user", "content": it["prompt"]}], max_tokens, seed=seed,
                         temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                         chat_template_kwargs={"enable_thinking": True})
                ch = r["choices"][0]
                rs = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
                text = (("<think>" + rs + "</think>") if rs else "") + (ch["message"].get("content") or "")
                fin, toks = ch["finish_reason"], r["usage"]["completion_tokens"]
            except Exception as e:
                text, fin, toks = f"[request failed: {repr(e)[:200]}]", "error", 0
            if it.get("answer") is not None:
                ext, ok = grade_math(text, it["answer"])
            else:
                ext, ok = last_boxed(text), None
            return {"qid": it["qid"], "sample": s, "source": it.get("source", "custom"), "prompt": it["prompt"], "gold": it.get("answer"),
                    "text": text, "extracted": ext, "correct": ok, "tokens": toks, "finish": fin}

        with cf.ThreadPoolExecutor(min(len(jobs), 256)) as ex:
            return list(ex.map(one, jobs))

    def _generate(self, req, s):
        import json
        which = req.get("model", "original")
        if which not in ("original", "edited", "both"):
            return False, "model must be original, edited or both"
        n = int(req.get("n", 1))
        max_tokens = int(req.get("max_tokens", GEN_CAP))
        if not (1 <= max_tokens <= GEN_CAP):
            return False, f"max_tokens must be between 1 and {GEN_CAP}"
        items = []
        for pid in req.get("problems", []) or []:
            if pid not in self.practice:
                return False, f"unknown practice id {pid!r}"
            p = self.practice[pid]
            items.append({"qid": pid, "source": p["source"], "answer": p["answer"],
                          "prompt": p["problem"] + "\n\nPut your final answer in \\boxed{}."})
        for j, pr in enumerate(req.get("prompts", []) or []):
            items.append({"qid": f"custom/{j}", "source": "custom", "answer": None, "prompt": str(pr)})
        if not items:
            return False, "no problems or prompts given"
        lim = 256 if which == "both" else 512
        if len(items) * n > lim:
            return False, f"items x n = {len(items) * n} exceeds {lim}"
        if which in ("edited", "both") and self._edits(s)[1] is None:
            return False, "there are no edits; use model 'original' or save an edit first"
        k = s.k + 1
        out = {"request": k, "type": "generate", "max_tokens": max_tokens}
        rows_o = rows_e = None
        if which in ("edited", "both"):
            try:
                self._ensure_edited_server(s)
            except Exception as e:
                return False, "the edited weights could not be loaded for generation: " + repr(e)[-600:]
        if which in ("original", "both"):
            hit = [self.ocache.get((it["qid"], smp)) for it in items for smp in range(n)] if max_tokens == GEN_CAP else [None]
            if all(hit):
                rows_o = [dict(r) for r in hit]
                out["original_from_cache"] = True
            else:
                self._ensure_original()
                rows_o = self._gen("original", 8001, items, n, max_tokens)
        if which in ("edited", "both"):
            rows_e = self._gen("edited", s.port, items, n, max_tokens)
        for tag, rows in (("original", rows_o), ("edited", rows_e)):
            if rows is None:
                continue
            fn = f"/work/gen_{k}.jsonl" if which != "both" else f"/work/gen_{k}_{tag}.jsonl"
            with open(s.real(fn), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            out[tag] = dict(summarize(rows, max_tokens), file=fn)
        if which == "both" and any(r["correct"] is not None for r in rows_o):
            go = [r for r in rows_o if r["correct"] is not None]; ge = [r for r in rows_e if r["correct"] is not None]
            out["paired_edited_minus_original"] = paired_stats(go, ge)
        s.k = k
        return True, json.dumps(out, indent=1)

    # ---- teacher forcing
    def _teacher_forced(self, req, s):
        import json, os, numpy as np, torch
        which = req.get("model", "original")
        if which not in ("original", "edited"):
            return False, "model must be original or edited"
        ret = req.get("return", {}) or {}
        if "chats" in req:
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False,
                                                  enable_thinking=bool(req.get("enable_thinking", False))) for c in req["chats"]]
        else:
            texts = [str(t) for t in req.get("texts", [])]
        if not texts:
            return False, "no texts or chats given"
        if len(texts) > 256:
            return False, "at most 256 texts per request"
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        L = max(len(x) for x in ids)
        tot = sum(len(x) for x in ids)
        if L > 8192:
            return False, f"a text has {L} tokens; the limit is 8192 per text"
        if tot > 262144:
            return False, f"{tot} tokens in total; the limit is 262144"
        hs_layers = [int(x) for x in ret.get("hidden_states", []) or []]
        mlp_layers = [int(x) for x in ret.get("mlp_act", []) or []]
        topk = int(ret.get("topk", 0) or 0)
        if topk > 50:
            return False, "topk is at most 50"
        cfg = self.hf.config.text_config if hasattr(self.hf.config, "text_config") else self.hf.config
        H, M, NL = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
        if any(l < 0 or l > NL for l in hs_layers) or any(l < 0 or l >= NL for l in mlp_layers):
            return False, f"layer index out of range (hidden_states 0..{NL}, mlp_act 0..{NL - 1})"
        est = len(texts) * L * 2 * (len(hs_layers) * H + len(mlp_layers) * M)
        if est > 3 * 2 ** 30:
            return False, f"requested activations would take {est / 2 ** 30:.1f} GiB; the limit is 3 GiB per request. Ask for fewer texts, layers or shorter texts."
        if ret.get("attn_layers"):
            return False, "attention maps are not available in this setup"
        if which == "edited" and self._edits(s)[1] is None:
            return False, "there are no edits; use model 'original' or save an edit first"
        mods = dict(self.hf.named_modules())
        mlp_mods = {}
        for l in mlp_layers:
            name = [n for n in mods if n.endswith(f"layers.{l}.mlp.down_proj") and "vis" not in n.lower()]
            if len(name) != 1:
                return False, f"could not locate the MLP down projection of layer {l}"
            mlp_mods[l] = mods[name[0]]
        k = s.k + 1
        d = f"/work/fwd_{k}"
        os.makedirs(s.real(d), exist_ok=True)
        N = len(texts)
        lp = np.full((N, L), np.nan, dtype=np.float32)
        tk_i = np.zeros((N, L, topk), dtype=np.int32) if topk else None
        tk_l = np.zeros((N, L, topk), dtype=np.float32) if topk else None
        hs = {l: np.zeros((N, L, H), dtype=np.float16) for l in hs_layers}
        ml = {l: np.zeros((N, L, M), dtype=np.float16) for l in mlp_layers}
        cap, hooks, oom = {}, [], False
        order = sorted(range(N), key=lambda i: -len(ids[i]))
        bs = max(1, min(32, self.tf_tokens // max(L, 1)))
        with self.gpu_lock:  # the forward model is shared: its weights and hooks belong to one request at a time
            try:
                self._set_hf(which, s)
            except Exception as e:
                self._set_hf("original", s)
                return False, repr(e)[:600]
            for l, m in mlp_mods.items():
                hooks.append(m.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, i[0].detach())))
            try:
                with torch.no_grad():
                    for b0 in range(0, N, bs):
                        idx = order[b0:b0 + bs]
                        Lb = max(len(ids[i]) for i in idx)
                        x = torch.full((len(idx), Lb), self.tok.pad_token_id or 0, dtype=torch.long)
                        am = torch.zeros((len(idx), Lb), dtype=torch.long)
                        for r, i in enumerate(idx):
                            x[r, :len(ids[i])] = torch.tensor(ids[i]); am[r, :len(ids[i])] = 1
                        # The language model, then the output head on 1,024 positions at a time: full-vocabulary
                        # log-probabilities of a whole batch at once take tens of GB.
                        o = self.hf.model(input_ids=x.cuda(), attention_mask=am.cuda(), output_hidden_states=bool(hs_layers))
                        for r, i in enumerate(idx):
                            n_i = len(ids[i])
                            for c0 in range(0, n_i, 1024):
                                c1 = min(n_i, c0 + 1024)
                                logp = torch.log_softmax(self.hf.lm_head(o.last_hidden_state[r, c0:c1]).float(), -1)
                                j1 = min(c1, n_i - 1)  # position j predicts token j + 1
                                if j1 > c0:
                                    tgt = torch.tensor(ids[i][c0 + 1:j1 + 1]).cuda()
                                    lp[i, c0 + 1:j1 + 1] = logp[:j1 - c0].gather(-1, tgt[:, None])[:, 0].cpu().numpy()
                                if topk:
                                    v, ix = logp.topk(topk, -1)
                                    tk_i[i, c0:c1] = ix.cpu().numpy(); tk_l[i, c0:c1] = v.cpu().numpy()
                                del logp
                            for l in hs_layers:
                                hs[l][i, :n_i] = o.hidden_states[l][r, :n_i].float().cpu().numpy().astype(np.float16)
                            for l in mlp_layers:
                                ml[l][i, :n_i] = cap[l][r, :n_i].float().cpu().numpy().astype(np.float16)
                        del o
            except torch.cuda.OutOfMemoryError:
                oom = True
            finally:
                for h in hooks:
                    h.remove()
                cap.clear()
                self._set_hf("original", s)
                torch.cuda.empty_cache()
        if oom:
            return False, ("the teacher-forced pass ran out of GPU memory (it does not count). Ask for fewer or shorter texts, "
                           "or fewer hidden_states / mlp_act layers per request.")
        dr = s.real(d)
        files = {}
        if ret.get("token_logprobs", True):
            np.save(f"{dr}/token_logprobs.npy", lp); files["token_logprobs.npy"] = "[text, position] log-probability of the token at that position given the tokens before it (NaN at position 0 and in padding)"
        if topk:
            np.save(f"{dr}/topk_ids.npy", tk_i); np.save(f"{dr}/topk_logprobs.npy", tk_l)
            files["topk_ids.npy / topk_logprobs.npy"] = "[text, position, k] top-k next-token predictions made at that position"
        for l in hs_layers:
            np.save(f"{dr}/hidden_L{l}.npy", hs[l]); files[f"hidden_L{l}.npy"] = "[text, position, hidden] residual stream (0 = embeddings, l = after layer l), float16"
        for l in mlp_layers:
            np.save(f"{dr}/mlp_act_L{l}.npy", ml[l]); files[f"mlp_act_L{l}.npy"] = "[text, position, intermediate] input to the layer's mlp.down_proj, float16"
        json.dump({"model": which, "lengths": [len(x) for x in ids], "token_ids": ids, "files": files}, open(f"{dr}/meta.json", "w"))
        s.k = k
        mean_lp = [float(np.nanmean(lp[i, :len(ids[i])])) if len(ids[i]) > 1 else None for i in range(N)]
        return True, json.dumps({"request": k, "type": "teacher_forced", "model": which, "texts": N, "max_tokens_in_a_text": L, "dir": d,
                                 "files": files, "mean_token_logprob_first_texts": [round(x, 4) if x is not None else None for x in mean_lp[:16]],
                                 "note": "arrays are right-padded to the longest text; meta.json has token_ids and lengths"}, indent=1)

    def _forward(self, s, req_json):
        import json, traceback
        try:
            req = json.loads(req_json)
        except Exception as e:
            return False, f"the forward block is not valid JSON: {e}"
        try:
            if req.get("type") == "generate":
                return self._generate(req, s)
            if req.get("type") == "teacher_forced":
                return self._teacher_forced(req, s)
            return False, "type must be 'generate' or 'teacher_forced'"
        except Exception:
            return False, "the request failed with an internal error (it does not count):\n" + traceback.format_exc()[-1500:]

    def _reward(self, s, items_json, n, max_tokens, need_original):
        import json
        items = [{"qid": "reward/" + it["qid"], "source": "reward", "answer": it["answer"],
                  "prompt": it["problem"] + "\n\nPut your final answer in \\boxed{}."} for it in json.loads(items_json)]
        slim = lambda rows: [{"qid": r["qid"], "sample": r["sample"], "correct": bool(r["correct"]), "tokens": r["tokens"], "finish": r["finish"]} for r in rows]
        out = {"original": None, "edited": None, "edited_tensors": sorted(self._edits(s)[0])}
        if need_original:
            self._ensure_original()
            out["original"] = slim(self._gen("original", 8001, items, n, max_tokens))
        if self._edits(s)[1] is not None:
            self._ensure_edited_server(s)
            out["edited"] = slim(self._gen("edited", s.port, items, n, max_tokens))
        return out

    def _put_files(self, s, root, files_json):
        import json, os
        files = json.loads(files_json)
        root = s.real(root)
        for name, text in files.items():
            path = os.path.join(root, os.path.basename(name))
            os.makedirs(root, exist_ok=True)
            open(path, "w").write(text)
        return len(files)

    def _get_files(self, s, root, max_bytes):
        import os
        out, total = {}, 0
        root = s.real(root)
        if os.path.isdir(root):
            names = sorted((n for n in os.listdir(root) if n.endswith((".md", ".txt"))), key=lambda n: -os.path.getmtime(os.path.join(root, n)))
            for n in names:
                t = open(os.path.join(root, n), errors="replace").read()[:120000]
                if total + len(t) > max_bytes:
                    continue
                out[n] = t; total += len(t)
        return out

    def _snapshot(self, s, run_id):
        import os, shutil
        dst = f"/runs/{run_id}/work"
        shutil.rmtree(dst, ignore_errors=True)
        def ig(d, names):
            return [n for n in names if n.endswith(".npy") and os.path.getsize(os.path.join(d, n)) > 50 * 2 ** 20]
        shutil.copytree(s.work, dst, ignore=ig)
        with self.commit_lock:
            runs.commit()
        return sorted(os.listdir(dst))


@app.cls(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, block_network=True,
         timeout=150 * 60, scaledown_window=20 * 60, max_containers=1, memory=98304, cpu=8)
class Target(_Core):
    """One rollout per L40S: original vLLM server (8001) and edited server (8002) at the target's `util` share each (0.8B: 0.40)."""
    rid: str = modal.parameter(default="")  # one container per value, so each rollout gets its own target
    base: str = modal.parameter(default="")  # starting model (see base_dirs); "" is the downloaded model
    target: str = modal.parameter(default=DEFAULT_TARGET)  # a key of TARGETS
    task: str = modal.parameter(default="math")  # a key of TASKS: which practice folder is /practice

    @modal.enter()
    def boot(self):
        self._setup()
        self.s = _Slot("/work", "/library", 8002, "/tmp/edited_ckpt")
        self._start_kernel(self.s)
        self.orig = start_vllm(self.mdir, "original", 8001, self.original_util, GEN_CAP + 3000)
        self._load_hf()
        print("BOOT ok; tensors", len(self.sd_keys))

    @modal.method()
    def py(self, code: str) -> str:
        return self._py(self.s, code)

    @modal.method()
    def forward(self, req_json: str):
        return self._forward(self.s, req_json)

    @modal.method()
    def reward(self, items_json: str, n: int = 2, max_tokens: int = GEN_CAP, need_original: bool = False):
        """Answers to the reward problems with the original and/or the current edited weights.

        Called after the editor has finished, so nothing here can reach it. Nothing is written to /work."""
        return self._reward(self.s, items_json, n, max_tokens, need_original)

    @modal.method()
    def boot_id(self) -> str:
        return self.s.boot

    @modal.method()
    def put_files(self, root: str, files_json: str):
        """Write text files under `root` (used to give a rollout the shared library)."""
        return self._put_files(self.s, root, files_json)

    @modal.method()
    def get_files(self, root: str, max_bytes: int = 400000):
        """Text files under `root` (flat), newest first, up to `max_bytes` in total."""
        return self._get_files(self.s, root, max_bytes)

    @modal.method()
    def snapshot(self, run_id: str):
        """Copy /work (without large activation arrays) to the runs volume."""
        return self._snapshot(self.s, run_id)


@app.cls(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs}, block_network=True,
         timeout=150 * 60, scaledown_window=20 * 60, max_containers=1, memory=98304, cpu=8)
@modal.concurrent(max_inputs=8)
class PackedTarget(_Core):
    """Two rollouts per L40S. Every method takes the rollout id first.

    Per rollout: /work and /library (its python kernel runs in its own mount and pid namespace, sees only its own
    folders there, and has an address-space limit), an edited vLLM server on its own port, a restart id, a request counter.
    Shared: the original vLLM server, started only when an original answer is not cached, and the teacher-forced
    model (original weights; a pass swaps in the rollout's edits and hooks while it holds the GPU lock)."""
    gid: str = modal.parameter(default="")  # one container per value; the rollouts that share it pass their own rid
    mps: int = modal.parameter(default=0)  # 1: run the GPU processes under CUDA MPS, so their kernels can overlap instead of taking turns
    base: str = modal.parameter(default="")  # starting model (see base_dirs); "" is the downloaded model
    target: str = modal.parameter(default=DEFAULT_TARGET)  # a key of TARGETS; GPU shares from its pack_util
    task: str = modal.parameter(default="math")  # a key of TASKS: which practice folder is /practice

    @modal.enter()
    def boot(self):
        import os, threading, subprocess
        self.mps_on = False
        if self.mps:  # before any process touches the GPU; every child inherits the pipe directory
            os.makedirs("/tmp/mps", exist_ok=True); os.makedirs("/tmp/mpslog", exist_ok=True)
            os.environ["CUDA_MPS_PIPE_DIRECTORY"], os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/mps", "/tmp/mpslog"
            self.mps_on = subprocess.run(["nvidia-cuda-mps-control", "-d"]).returncode == 0
        self._setup(packed=True)
        os.makedirs("/library", exist_ok=True); os.makedirs(SLOTS, exist_ok=True)
        self.slots, self.slots_lock = {}, threading.Lock()
        self._load_hf()
        import torch
        torch.cuda.set_per_process_memory_fraction(TARGETS[self.tgt]["pack_util"]["forward_model"])  # past this a teacher-forced pass gets an out-of-memory error, never a vLLM server
        print("BOOT ok (packed); tensors", len(self.sd_keys))

    def _slot(self, rid):
        import os
        with self.slots_lock:
            if rid not in self.slots:
                i = len(self.slots)
                s = _Slot(f"{SLOTS}/{i}/work", f"{SLOTS}/{i}/library", 8002 + i, f"{SLOTS}/{i}/ckpt", isolate=True)
                os.makedirs(s.work, exist_ok=True); os.makedirs(s.lib, exist_ok=True)
                self._start_kernel(s)
                self.slots[rid] = s
            return self.slots[rid]

    @modal.method()
    def py(self, rid: str, code: str) -> str:
        s = self._slot(rid)
        with s.lock:
            return self._py(s, code)

    @modal.method()
    def forward(self, rid: str, req_json: str):
        s = self._slot(rid)
        with s.lock:
            return self._forward(s, req_json)

    @modal.method()
    def reward(self, rid: str, items_json: str, n: int = 2, max_tokens: int = GEN_CAP, need_original: bool = False):
        s = self._slot(rid)
        with s.lock:
            return self._reward(s, items_json, n, max_tokens, need_original)

    @modal.method()
    def boot_id(self, rid: str) -> str:
        return self._slot(rid).boot

    @modal.method()
    def put_files(self, rid: str, root: str, files_json: str):
        s = self._slot(rid)
        with s.lock:
            return self._put_files(s, root, files_json)

    @modal.method()
    def get_files(self, rid: str, root: str, max_bytes: int = 400000):
        s = self._slot(rid)
        with s.lock:
            return self._get_files(s, root, max_bytes)

    @modal.method()
    def snapshot(self, rid: str, run_id: str):
        s = self._slot(rid)
        with s.lock:
            return self._snapshot(s, run_id)

    @modal.method()
    def stats(self):
        """GPU memory in use and the rollouts on this container (for tests)."""
        import subprocess
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        import torch
        return {"gpu_memory": smi, "mps": self.mps_on, "original_server": self.orig is not None, "forward_model_peak_mib": torch.cuda.max_memory_reserved() // 2 ** 20,
                "slots": {rid: {"port": s.port, "edited_server": s.edited_proc is not None and s.edited_proc.poll() is None, "requests": s.k} for rid, s in self.slots.items()}}


# ----------------------------------------------------------------------------- held-out evaluation
@app.function(image=offline, gpu="L40S", volumes={"/weights": weights, "/runs": runs, "/heldout": heldout_vol}, timeout=90 * 60, memory=32768)
def heldout_eval(run_id: str, n: int = 2, limit: int = 0, use_cached_original: bool = False, base_tag: str = "", dataset: str = "math500", target: str = DEFAULT_TARGET):
    """The edit in /runs/<run_id>/work/edited against its starting model on an evaluation set (EVAL_SETS), paired by problem.
    `base_tag`: the edit was made on that starting model (rl.loop rebasing); it is compared with that model.
    `limit`: only the first `limit` problems. `target`: a key of TARGETS (base_tag is one of its starting models).
    Results: /runs/<run_id>/heldout.json (MATH-500) or heldout_<dataset>.json."""
    import json, os, shutil, concurrent.futures as cf, zlib, time
    from safetensors.torch import load_file, save_file
    runs.reload()
    mdir = base_dirs(base_tag, target)[0]
    spec = EVAL_SETS[dataset]
    items = [json.loads(l) for l in open(spec["file"])]
    if limit:
        items = items[:limit]
    ed_dir = f"/runs/{run_id}/work/edited"
    edits = {}
    if os.path.isdir(ed_dir):
        for f in sorted(os.listdir(ed_dir)):
            if f.endswith(".safetensors"):
                edits[f[:-len(".safetensors")]] = load_file(os.path.join(ed_dir, f))["t"]

    def run(model_path, name):
        proc = start_vllm(model_path, name, 8001, 0.90, GEN_CAP + 3000)
        t = time.time()
        rows = heldout_rows(8001, name, items, n, spec["grade"])
        print(name, "generation seconds", round(time.time() - t), "tokens", sum(r["tokens"] for r in rows))
        stop_proc(proc)
        time.sleep(10)
        return rows

    out = {"run_id": run_id, "dataset": dataset, "n_per_problem": n, "problems": len(items), "cap": GEN_CAP, "edited_tensors": sorted(edits), "base": base_tag, "target": target}
    cached = heldout_cache(base_tag, dataset, target)  # one fixed set of the starting model's answers, so every edit is compared with the same baseline
    rows_o = None
    if use_cached_original and os.path.exists(cached):  # with a limit, the cached answers to those problems
        want = {(it["qid"], s) for it in items for s in range(n)}
        rows_o = [r for r in map(json.loads, open(cached)) if (r["qid"], r["sample"]) in want]
        if len(rows_o) == len(want):
            out["original_from_cache"] = True
        else:
            rows_o = None
    if rows_o is None:
        rows_o = run(mdir, "original")
        if use_cached_original and not limit and n == 2 and not os.path.exists(cached):
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            with open(cached, "w") as f:
                for r in rows_o:
                    f.write(json.dumps(r) + "\n")
    out["original"] = summarize(rows_o, GEN_CAP)
    tag = "heldout" if dataset == "math500" else f"heldout_{dataset}"
    os.makedirs(f"/runs/{run_id}/{tag}", exist_ok=True)
    with open(f"/runs/{run_id}/{tag}/original.jsonl", "w") as f:
        for r in rows_o:
            f.write(json.dumps(r) + "\n")
    if edits:
        ed = "/tmp/edited_ckpt"
        shutil.rmtree(ed, ignore_errors=True); os.makedirs(ed)
        changed = {}
        for f in os.listdir(mdir):
            src = os.path.join(mdir, f)
            if f.endswith(".safetensors"):
                sd = load_file(src)
                for name, t in edits.items():
                    if name in sd:
                        changed[name] = int((t.to(sd[name].dtype) != sd[name]).sum())
                        sd[name] = t.to(sd[name].dtype)
                save_file(sd, os.path.join(ed, f), metadata={"format": "pt"})
            else:
                shutil.copy(os.path.realpath(src), os.path.join(ed, f))
        out["changed_bf16_entries"] = changed
        rows_e = run(ed, "edited")
        out["edited"] = summarize(rows_e, GEN_CAP)
        out["paired_edited_minus_original"] = paired_stats(rows_o, rows_e)
        with open(f"/runs/{run_id}/{tag}/edited.jsonl", "w") as f:
            for r in rows_e:
                f.write(json.dumps(r) + "\n")
    else:
        out["note"] = "the editor submitted the original weights"
    with open(f"/runs/{run_id}/{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    runs.commit()
    print("HELDOUT", json.dumps(out))
    return out


@app.function(image=offline, volumes={"/weights": weights, "/runs": runs}, timeout=20 * 60, memory=16384)
def paper_edit(run_id: str = "control-paper-edit"):
    """Control: write the Math (Primary) edit of arXiv 2611.09382 (Eq. 3) as if an editor had submitted it."""
    import os, torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    mdir = target_dir()
    sd = {}
    for f in os.listdir(mdir):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(mdir, f)))
    tk = AutoTokenizer.from_pretrained(mdir)
    E = sd["model.language_model.embed_tokens.weight"].float()
    g = sd["model.language_model.norm.weight"].float()
    A = ["Wait", "However", "But", "Maybe", "Perhaps"]
    B = ["Therefore", "Thus", "So", "Now", "The", "First", "Next", "Let"]
    first = lambda w: tk(w, add_special_tokens=False)["input_ids"][0]
    d = (E[[first(w) for w in A]].mean(0) - E[[first(w) for w in B]].mean(0)) * (1 + g)
    u = d / d.norm()
    name = "model.language_model.layers.23.mlp.down_proj.weight"
    W = sd[name].float().clone()
    print("norm gain mean", float(g.mean()), "col 507 norm", float(W[:, 507].norm()), "W shape", tuple(W.shape))
    W[:, 507] -= u
    out = f"/runs/{run_id}/work/edited"
    os.makedirs(out, exist_ok=True)
    save_file({"t": W.to(torch.bfloat16)}, os.path.join(out, name + ".safetensors"))
    runs.commit()
    return int((W.to(torch.bfloat16) != sd[name]).sum())




@app.function(image=offline, volumes={"/runs": runs}, timeout=40 * 60)
def smoke_target():
    """Cheap test of Target: sandbox, an edit, a paired generation, a teacher-forced pass."""
    import json
    t = Target()
    print(t.py.remote("print(len(NAMES)); print([n for n in NAMES if 'layers.23.' in n]); print(open('/practice/README.txt').read()); import json; ids=[json.loads(l)['id'] for l in open('/practice/problems.jsonl')][:6]; ids"))
    print(t.py.remote("n=[x for x in NAMES if x.endswith('layers.23.mlp.down_proj.weight')][0]; W=load(n); print(n, W.shape, W.dtype); W2=W.clone(); W2[:,507]*=0.5; save_edit(n,W2)"))
    print(t.py.remote("try:\n    save_edit(n, load_original(n)*2)\nexcept Exception as e:\n    print('refused ok:', e)\nedited()"))
    ids = json.loads(t.py.remote("import json; print(json.dumps([json.loads(l)['id'] for l in open('/practice/problems.jsonl')][:8]))"))
    ok, res = t.forward.remote(json.dumps({"type": "generate", "model": "both", "problems": ids, "n": 1, "max_tokens": 2048}))
    print("GEN", ok, res)
    ok, res = t.forward.remote(json.dumps({"type": "teacher_forced", "model": "edited", "chats": [[{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "Wait, let me think. 2+2=4. \\boxed{4}"}]],
                                           "enable_thinking": True, "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 23, 24], "mlp_act": [23]}}))
    print("TF", ok, res)
    print(t.py.remote("import numpy as np, os; print(os.listdir('/work')); print(os.listdir('/work/fwd_2')); a=np.load('/work/fwd_2/mlp_act_L23.npy'); print(a.shape, float(np.abs(a).mean())); r=[__import__('json').loads(l) for l in open('/work/gen_1_original.jsonl')]; print([(x['correct'], x['tokens'], x['finish']) for x in r]); print(r[0]['text'][-300:])"))
    print(t.snapshot.remote("smoke"))



@app.function(image=offline, volumes={"/weights": weights}, timeout=600)
def show_arch(target: str = DEFAULT_TARGET):
    """A downloaded target's architecture as the prompt states it, and the prompt's model sentence and tensor note."""
    from harness.prompts import model_sentence, tensor_names_text
    a = target_arch(target)
    return {"arch": a, "model_sentence": model_sentence(a), "tensor_names": tensor_names_text(a)}


@app.local_entrypoint()
def main(step: str = "prep", run_id: str = "", target: str = DEFAULT_TARGET):
    if step == "prep":
        print(prep.remote(target))
    elif step == "arch":
        import json
        print(json.dumps(show_arch.remote(target), indent=1))
    elif step == "smoke":
        smoke_target.remote()
    elif step == "heldout":
        print("SPAWNED", heldout_eval.spawn(run_id).object_id)  # spawn: a blocking call is cancelled when the local client exits
    elif step == "control":
        print("changed bf16 entries", paper_edit.remote("control-paper-edit"))
        print("SPAWNED", heldout_eval.spawn("control-paper-edit").object_id)
    else:
        raise SystemExit("step must be prep, arch, smoke, heldout or control")
