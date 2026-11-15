# weight-edit-rl

Train a language model to improve another model by editing its weights directly.

An **editor** model gets a small **target** model's weight files, a Python sandbox, and a limited number of forward passes on the target. It has no gradients. It reads activations, decides which weights to change, writes the change in code, and tests it. This repo is the harness for that, plus a reinforcement-learning loop that trains the editor on the gain its edits produce.

Built for the [AI Research Hackathon](https://luma.com/sht6n8du), San Francisco, 14 to 15 November 2026. The idea of editing weights this way comes from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL) and [arXiv 2611.09382](https://arxiv.org/abs/2611.09382).

## Results so far

Target: Qwen3.5-0.8B. Task: math. One search per row. Test set MATH-500, 1,000 answers per arm, run once after each search. The editor works on train problems (MATH and GSM8K training sets) and checks its final edit on validation problems it set aside.

| Editor | Original | Edited | Change |
|---|---|---|---|
| Claude Opus 5.5 | 7.6% | 34.8% | +27.2 |
| The published edit from 2611.09382, applied as a check | 7.1% | 28.9% | +21.8 |
| Inkling 2 Small, first search | 6.6% | 20.2% | +13.6 |
| Inkling 2 Small, second search | 6.6% | 11.8% | +5.2 |

Every edit tried so far, sorted by kind: [a map of weight edits](https://tinkerings.dev/posts/weight-space-programming/). Every search and rollout, turn by turn: [trajectories](https://tinkerings.dev/posts/weight-space-programming/traces/).

Qwen3.5-0.8B fails at math in one way: it writes "Wait..." and rechecks until it runs out of tokens. Every gain above comes from stopping that. Transcripts and scores are in `results/searches/`.

### Evaluation (15 November 2026)

Full tables: [`results/eval/SUMMARY.md`](results/eval/SUMMARY.md). Held-out sets only; small samples (2 to 4 editing sessions per editor), so a proof of concept.

- **The edited model gets much better, and it transfers.** Qwen3.5-0.8B with the editor's best edits stacked: MATH-500 6.6% → 42.5%, GSM8K test 12% → 58%, NuminaMath (new dataset) 8.2% → 44.6%. Nearly every change is wrong→right.
- **RL has not yet made the editor better.** Untrained vs trained editor on the same model: 0.8B MATH-500 +11.8 vs +13.9 (mean gain per edit); Qwen3.5-2B +19.3 vs +7.8; NuminaMath +16.1 vs −0.6. The RL updates did move the editor toward its higher-reward sessions (correlation 0.86 to 0.97 between each session's change in likelihood and its advantage), but after a handful of updates that does not show on held-out data.
- **One edit is most of the gain.** The best single edit from the untrained editor already reaches 39.4% on MATH-500 and 59.8% on GSM8K test.

## How it works

```
                              TINKER
        ┌──────────────────────────────────────────────┐
        │  Inkling 2 Small, the editor being trained   │
        │    sample a reply  ·  train  ·  checkpoints  │
        └──────▲───────────────────────────┬───────────┘
               │ sample                    │ train on scored rollouts,
               │                           │ once per RL step
 ═════════════ │ ═════════ MODAL ═════════ │ ══════════════════════════
               │                           │
        ┌──────┴───────────────────────────▼───────────┐
        │  RL LOOP (CPU)                               │
        │    runs a group of rollouts at once          │
        │    passes each tool call on, scores the edit │
        └──────┬───────────────────────────┬───────────┘
               │ python tool               │ forward tool
               ▼                           ▼
     ┌────────────────────┐     ┌──────────────────────────────┐
     │ SANDBOX (CPU)      │     │ TARGET (one L40S per rollout)│
     │  read weights      │     │  Qwen3.5-0.8B                │
     │  analyse           │────▶│  original and edited weights │
     │  save an edit      │edit │  generate · read text        │
     └────────────────────┘     └──────────────────────────────┘
```

- **Rollout.** One editing search on a fresh copy of the target, ending with the editor's best edit. The target always starts from its original weights; only the editor changes between RL steps.
- **Problem sets.** Train: MATH and GSM8K training problems, with a ready-made index and disjoint splits for diagnosing and choosing (`select`) and for checking the final edit (validation: `confirm_1`, `confirm_2`). Reward set: 128 hidden MATH training problems × 2 samples that score each rollout; this is the only set that feeds training. Test: MATH-500 (500 × 2 answers), from MATH's test split, used only to measure: every submitted edit, from standalone searches and from each RL step, is scored on it after the fact. The editor never sees the reward or test set.
- **Given at the start.** The original weights' answers on 512 train generations with a summary of the failure, and their answers on every split. The forward tool reuses these, so a paired test only generates the edited side.
- **`python` tool.** A CPU sandbox with the target's weights on disk: load a tensor, analyse it, save an edited tensor. No internet, no gradients.
- **`forward` tool.** Runs the target on a GPU. Either it answers train problems (original weights, edited weights, or both side by side), or it reads supplied text and returns log-probabilities and activations. The number of calls is limited.
- **Edits.** Any tensor inside the transformer blocks. So far every edit changed an output matrix: one or a few columns, or a rank-one term across every column.
- **Web and library (optional).** `web_search` and `web_read` let the editor look up methods and read papers. What it saves in `/library` is kept for later rollouts. Pages that publish an edit for this exact model are filtered out.
- **Countdown.** After every tool result the editor sees what is left: forward passes, Python runs, web calls, minutes and context.
- **Reward.** Accuracy with the submitted edit minus accuracy with the original weights, on the reward set, 256 paired answers. A harmful edit scores below zero.
- **RL step.** A group of 8 rollouts from the current checkpoint, rewards centred and scaled within the group, one policy-gradient step on Tinker (importance-sampling loss, LoRA rank 32), a new checkpoint.

## Run it

Needs a [Modal](https://modal.com) account and these Modal secrets: `tinker` (`TINKER_API_KEY`), `anthropic` (`ANTHROPIC_API_KEY`, only for the Claude editor), `exa` (`EXA_API_KEY`, only for the optional web-search tool). Run everything from the repo root.

```bash
# once: download the target, build the practice, reward and held-out sets
uvx modal run -m harness.target --step prep
uvx modal run -m rl.loop::prep_reward_set

# one search with Inkling 2 Small as the editor, then the held-out test
uvx modal run --detach -m editors.tinker_openai::oai --run-id inkling-1

# once: an index and ready-made splits of the practice problems, and the original
# weights' answers on them, which every rollout starts from and the forward tool reuses
uvx modal run -m rl.loop::prep_practice_index
uvx modal run -m rl.loop::prep_baseline
uvx modal run -m rl.loop::prep_originals

# the RL loop: 8 steps of 8 rollouts, 32 forward passes per rollout; --web adds web search and a shared paper library
uvx modal run --detach -m rl.loop::rl --exp run1 --steps 8 --group-size 8 --forward-budget 32 --web
./rl/watch.sh run1 <modal-app-id>

# resume from a checkpoint, 2 rollouts per GPU, new starting model every 2 steps, LoRA learning rate
uvx modal run --detach -m rl.loop::rl --exp rebase2 --steps 5 --group-size 8 --forward-budget 32 --rollouts-per-gpu 2 \
  --rebase-every 2 --lr 2e-4 --start-from tinker://<run>:train:0/weights/<checkpoint>

# score one finished step's submitted edits on the test set (MATH-500); k counts the loop's steps from 0
uvx modal run --detach -m rl.loop::test --exp rebase2 --steps <k>

# evaluate an editor checkpoint: 8 rollouts with no training update (RL settings: 32 forward passes, 120 minutes,
# 2 per GPU, no notes, no web), each submitted edit scored on --dataset (math500 or gsm8k_test) against --base
# ("original" or a starting-model tag); --checkpoint "" is the untrained editor. Output: /runs/eval/<name>/summary.json
uvx modal run --detach -m rl.loop::evaluate --name nolib1-s3-gsm8k --checkpoint tinker://<run>:train:0/sampler_weights/<checkpoint> \
  --base original --n 8 --dataset gsm8k_test
```

```bash
# score a starting model itself (a rebase tag) against Qwen as downloaded on --dataset, paired by problem;
# each model's answers come from its per-(model, set) cache and are generated only when missing.
# Output: /runs/eval/base-<tag>-<dataset>.json (accuracies, paired gain, SE, flips)
uvx modal run --detach -m rl.loop::score_base --base rebase2-k2 --dataset math500
```

Another target model. The target is a key of `TARGETS` in `harness/target.py` (`qwen3.5-0.8b`, the default, or `qwen3.5-2b`). Its layer count, sizes, attention layout and tensor names are read from its config.json and weight files, and the editor's prompt and tool descriptions are built from them. Each target has its own practice folder, cached answers and starting models under `/runs/data/targets/<target>/` (the default target's files stay where they were).

```bash
# accuracy and hit-cap rate of the target as downloaded on 200 MATH train problems at the 8,192-token cap (1 GPU; downloads it first)
uvx modal run --detach -m rl.loop::sweep --target qwen3.5-2b
# download, practice files (same splits as the default target), baseline (512 answers + SUMMARY.txt), the select/confirm splits,
# the reward set, MATH-500 and gsm8k_test: 3 GPUs; --limit 2 is a dry run that writes only to /runs/data/targets/<target>/_limit_test/
uvx modal run --detach -m rl.loop::prep_target --target qwen3.5-2b
# then --target on rl, evaluate, rebase and score_base, e.g.
uvx modal run --detach -m rl.loop::evaluate --target qwen3.5-2b --name 2b-untrained-math500 --checkpoint "" --base original --n 8 --dataset math500
```

Another task. `--task` (a key of `TASKS` in `harness/target.py`; default `math`, the setup above) gives the rollouts another practice pool, reward set and held-out set, and the prompt's task statements. `numina_cn_k12`: a fixed-seed sample of 4,500 problems from the cn_k12 part of AI-MO/NuminaMath-CoT with one short final answer in the reference solution's `\boxed{}` (multi-part, proof and multiple-choice problems dropped): 500 held out (`EVAL_SETS["numina_cn_k12"]`), 256 for the reward set (RL uses 128), 3,744 for practice. Its files live under `/runs/data/targets/<target>/tasks/<task>/` and `/heldout/tasks/<task>/`.

```bash
uvx modal run --detach -m rl.loop::sweep --task numina_cn_k12            # samples the problems (CPU), then 200 of them at the 8,192 cap (1 GPU)
uvx modal run --detach -m rl.loop::prep_task --task numina_cn_k12        # baseline + SUMMARY.txt, reward set, select/confirm, held-out answers (1 GPU)
uvx modal run --detach -m rl.loop::evaluate --task numina_cn_k12 --dataset numina_cn_k12 --name <name> --checkpoint "" --base original --n 8
```

`evaluate --edits rl/<exp>/work/<rid>,...` scores existing edits without new rollouts. Tested end to end at small scale on 14 November 2026: one rollout from `nolib1-step2` with a 6-minute cap submitted an edit (reward-set gain +34.0) that was then scored on 5 GSM8K test problems.

Measured cost of one full-length search: about 100 minutes on one L40S (about $3) plus under $1 of Tinker.

## What is where

```
harness/
  target.py        the target on Modal: sandbox, forward tool, reward, held-out test
  prompts.py       what the editor is told
  grading.py       answer graders for the baseline sweeps
editors/
  claude.py        a search with a Claude model as the editor
  tinker_openai.py a search with a Tinker-served model as the editor
rl/
  loop.py          the RL loop
  watch.sh         mirror a run's files and stop it on a time limit
scripts/
  baseline_sweep_tinker.py   benchmark an unedited Tinker model
  tinker_check.py            check the Tinker calls the RL loop depends on
viewer/
  build.py, template.html    a page for stepping through any trajectory
  edit_figures.py            draws the before/after matrix figures for each kind of edit
results/
  searches/        per search: transcript, summary, held-out result
  rl/              per RL run: steps and rollouts
  baselines/       unedited-model scores
```

To read a trajectory turn by turn:

```bash
python3 viewer/build.py && python3 -m http.server -d viewer/site
```

Changes to the prompt, tools and setup, with the reason for each, are in [CHANGELOG.md](CHANGELOG.md).

## Differences from arXiv 2611.09382

Answers are capped at 8,192 tokens instead of 32,768. The held-out test uses 2 samples per problem instead of 4. The grader, the practice set and the prompt wording are our own.
