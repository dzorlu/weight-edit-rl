# Changelog

Changes to the prompt, the tools and the RL setup, newest first, with the reason for each.

## 14 November 2026

**The publish loop commits the run page and trajectories to the site's main** before each publish. Two sessions publish the site (one for the blog post, one for the runs), and a publish from a checkout without the latest run page replaced it with an old one.

**One page per run.** `/run/` links to each run. A run's page has one table per step (step 1, step 2, ...), one row per rollout and a mean row: reward, test, edited accuracy, hit cap, tensors edited, turns, minutes, forward passes, Python calls, how it ended. `nolib1`'s page starts with step 1, shared with `hero3`. The trajectory viewer numbers `nolib1`'s steps the same way.

**`hero3` stopped after step 2** (17:54). Step 1 mean reward +13.6, step 2 +24.3; entropy 0.44 → 0.27. Step 1's edits on the test set: +0.6, +11.1, +16.7, +15.2, +32.8, no edit, +2.5, +29.4, in line with their rewards. Step 2's edits are being scored on the test set. The step-2 checkpoint is kept (`hero3-step2`). The live page and the repo now follow `nolib1`.

**A second arm without the library: `nolib1`** (17:48). Starts from `hero3`'s step-1 checkpoint, same prompt and settings, but no web tools and no `/library`. Resumed from the state checkpoint `tinker://2ca0a9ef-7226-5b53-a9d0-673f54ba489d:train:0/weights/hero3-step1`, the first training update. In `hero3`, step 2's rollouts read step 1's notes, including the best rollout's full recipe, so step 2's gain mixes the training update with copying. `nolib1`'s first step measures the update alone; `hero3` continues as the arm with the library.

**Each step's edits are scored on the test set** (`rl.loop::test`). The reward set trains the editor, so it cannot also measure it. Each rollout's submitted edit is now scored on MATH-500, which the RL loop never uses, against one cached set of the original weights' answers. 2 GPUs, alongside the rollouts' 8.

**The sampler-trainer KL is exactly 0, and that is real.** Given fake sampler log-probabilities, Tinker's trainer returned its own computed values, which match the sampler bit for bit at the same weights. With one update per batch there is no mismatch to correct.

**Trainer-versus-sampler metrics, and the hero run restarted as `hero3`** (14:00).
Each RL step now records, on the action tokens it trains on: the KL between the sampler's and the trainer's log-probabilities (first- and second-order estimates, as in the Tinker cookbook), the largest per-token difference, the share of tokens differing by more than 0.2, and the entropy of the sampled tokens. The original weights' answers on the reward set are cached, so a run no longer spends 12 minutes regenerating them.
Why: without these, a mismatch between Tinker's sampler and trainer, or a collapse in entropy, would go unnoticed.

**Hero run restarted as `hero2`** (13:45). In `hero1` a target machine ran out of memory and was replaced, which reset that rollout's sandbox, including its saved edit, without telling the editor. Rollout logs were also only saved at the end of a step.
- Target memory doubled to 96 GB.
- The harness checks after every tool call whether the target machine was replaced, and if so tells the editor that `/work`, its variables and its edit are gone. Each rollout records how many times this happened.
- Rollout logs are saved to the volume every 8 turns, so a step can be watched while it runs.

**Hero run started** (`hero1`, 13:06): 6 RL steps of 8 rollouts, 32 forward passes and 120 minutes per rollout, web and library on, reward on 128 problems × 2 samples. First run with every change below.

**Problem sets named train, validation, test and reward set** in the README and on the website. The editor's prompt still says "practice", to keep rollouts comparable.

**Baseline summary lists only frequent words.**
The list of words that open lines more often in cut-off answers now only counts words seen at least 200 times.
Why: the first version ranked by ratio and was led by rare words such as "Diego".

**Original answers are cached.**
The original weights' answers on the `select`, `confirm_1` and `confirm_2` splits are generated once and stored in `/practice/original/`. The forward tool serves them from the cache, so a paired test only generates the edited side. Only the text and its grading are cached, not activations.
Why: every rollout regenerated the same original answers, and generation is nearly all of a rollout's GPU time.

**Practice index and ready-made splits.**
`/practice/index.tsv` lists every problem. `/practice/splits.json` has disjoint lists: `baseline`, `select`, `confirm_1`, `confirm_2`, `extra`.
Why: 14 forward calls in 8 rollouts failed on made-up problem ids, and every rollout built its own splits.

**The first diagnosis is given.**
`/practice/baseline/` holds the original weights' answers to 512 generations and a summary. The prompt tells the editor to start from it.
Why: 5 of 8 rollouts spent their first forward pass and about 13 minutes on the same diagnosis.

**Always submit the best edit.**
The RL prompt asks for the best edit to be left in place every time. A harmful edit scores below zero.
Why: one rollout dropped an edit worth +25 points on practice because it had no budget for a 300-answer confirmation, and scored the same 0 as a rollout that found nothing. The strict rule stays for standalone searches.

**Free rein on the kind of edit.**
The listed edit forms are now starting points. Any tensor inside the transformer blocks may be edited.
Why: every edit so far was on an MLP or attention output matrix.

**Web search, page reading and a shared library** (`--web`).
`web_search` and `web_read` call Exa. `web_read` saves the page to `/library`, which is pushed into each rollout and merged back, so notes persist across rollouts and steps. Pages that publish an edit for this exact model are filtered out.
Why: let the editor look up and read methods. Without the filter the published fix for this model can be found with one query.

**Countdown line.**
After every tool result the editor sees the turn number and what is left: forward passes, Python runs, web calls, minutes and context.
Why: time and context ran out before the forward-pass budget did.

**Sandbox notes in the prompt.**
Tensor name patterns, tokenizer and `save_edit` usage, and the limits of one generation call.
Why: about 10 Python errors in 8 rollouts came from guessing these.

**Tool output is cut at 6,000 characters** in the editor's context.
Why: context reached 80,000 to 144,000 tokens after only 8 forward passes, and the limit is 256,000.

**Checkpoints expire.**
Sampling checkpoints after 7 days, resumable ones after 2.
Why: one Inkling checkpoint is 17 GB and a resumable one 50 GB. The first RL run stopped when Tinker's free 100 GB filled up.

**Flat arguments for the forward tool, and a repair step.**
`forward` takes `type`, `model`, `texts` and so on directly. A tool call that cannot be parsed is completed and rerun.
Why: with the request nested three objects deep, Inkling dropped a closing brace in 3 of 5 test calls, and rollouts spiralled. With flat arguments 10 of 10 parsed.

**Inkling's own tool-call format.**
The RL loop declares the tools to the renderer and reads native tool calls.
Why: with a text protocol and direct sampling, Inkling sometimes switched to its native format and did not end its turn.

**Each rollout gets its own target container, and the reward is scored there** after the editor has finished.

**RL loop added** (`rl/loop.py`): rollouts sampled from the current Tinker checkpoint, reward, one policy-gradient step, new checkpoint.

## 13 November 2026

**Consolidated guide in the prompt.** One section replaces "what counts" and the methods list, with each method named after its source.

**The editor may finish early** with a reply starting `FINAL SUBMISSION`.
Why: with no way to stop, an editor burned forward passes on one-answer requests to run the budget down.

**Search drivers run on non-preemptible capacity and refuse to restart** over an existing run.
Why: a preempted driver restarted a search from turn 1 on top of the first session's files.

**The held-out test is spawned, not called.**
Why: a blocking call was cancelled when the local command ended.
