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

Every edit tried so far, sorted by kind: [a map of weight edits](https://tinkerings.dev/posts/weight-edits/). Every search and rollout, turn by turn: [trajectories](https://tinkerings.dev/posts/weight-edits/traces/).

Qwen3.5-0.8B fails at math in one way: it writes "Wait..." and rechecks until it runs out of tokens. Every gain above comes from stopping that. Transcripts and scores are in `results/searches/`.

The RL loop has run one full step on a small test (4 rollouts, 8 forward passes each): rewards of +13.3, +4.7, 0 and 0 points, one policy-gradient update, one checkpoint. Rollouts are in `results/rl/smoke3/`. A trained editor has not been compared with an untrained one yet.

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
- **Problem sets.** Train: MATH and GSM8K training problems, with a ready-made index and disjoint splits for diagnosing and choosing (`select`) and for checking the final edit (validation: `confirm_1`, `confirm_2`). Reward set: hidden MATH training problems that score each rollout. Test: MATH-500, run once on standalone searches.
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
```

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
