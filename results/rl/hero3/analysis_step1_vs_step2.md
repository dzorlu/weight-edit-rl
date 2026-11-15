# hero3: step 1 vs step 2 vs Opus 5.5 (snapshot 2026-11-14)

Sources: `hero3-s0-r*.jsonl` (step 1), `hero3-s1-r*.jsonl` (step 2), `steps.jsonl`, `config.json`, `results/searches/opus-5.5/`. Numbers come from `analyze_steps.py` in this folder (`python3 analyze_steps.py --detail`). Edit form, direction and strength were annotated by hand from each FINAL SUBMISSION text. Rewards are in points (reward × 100).

## Summary

- Step 2 differs from step 1 in two ways at once: one policy-gradient update, and a non-empty `/library`. Step 1 rollouts saw an empty `INDEX.md`. Step 1 rollouts r1, r3, r4 and r6 wrote four notes, and step 2 started with them. `specific_readout_direction_editing.md` (written by s0-r4, the best step 1 rollout) gives the full recipe: token ids, the top-600 frequent-token projection, all 48 writers, α values and confirm results "+36.3pp / +45.3pp". All 8 step 2 rollouts read the notes at turn 1 to 3. The logs cannot separate the effect of the update from the effect of the library.
- Reward: step 1 mean 13.6 (sd 11.7, range 0 to 33.6, 8/8 finished). Step 2 mean 23.8 (sd 5.5, range 13.3 to 32.0, 7/8 finished; s1-r5 has no summary, its log stops at turn 49, 69.7 min). Rollouts at or above 20 points: 2/8 in step 1, 6/7 in step 2.
- The weak rollouts went up and the best did not: min 0 → 13.3, max 33.6 (s0-r4) → 32.0 (s1-r0).
- Convergence: 7/7 finished step 2 edits are the same recipe, a rank-one amplification of the `</think>`-specific readout direction on the same 48 writers (α_think 1.0 to 2.0), 3 of them plus ` Wait` removal at α = −1.0. Step 1's 7 submitted edits used 6 different forms on 1, 6 or 48 tensors.
- Step 2 is faster in turns and diagnoses less: first `save_edit` at turn 21.5 on average (step 1: 41.0), first code building the `</think>` direction at turn 2 to 3 in 7/8 (step 1: turn 21 to 44). Per-unit attribution calls fell from 36 in step 1 to 4 in step 2. 3/8 step 2 rollouts (r2, r3, r7) ran no teacher-forced pass at all.
- Format failures went away: 13 editor turns in step 1 printed Inkling's native tool-call tokens as text and were not parsed (7 in s0-r5, which ended with "no tool call in five replies" at 8.6 min and reward 0). Step 2 had none. Made-up problem ids did not improve (8 failed calls in step 1, 10 in step 2).
- No sign of reward hacking in the submitted edits. Step 2 edits leave 53 to 78% of reward-set answers at the token cap (step 1's best, s0-r4: 40%). Several rollouts measured the skip-thinking shortcut and rejected it (s1-r2: 40% of answers closed `</think>` immediately at α_think = 1.5 with Wait removal).
- Against Opus: Opus spent 170 min, 135 turns and 4 teacher-forced passes on diagnosis, ran 21 attribution-tagged Python calls, and found one unit (L16 unit 998) plus a 9-writer gain chosen to avoid skip-thinking. s0-r4 scored higher on MATH-500 (+32.8 vs +27.2) with a cruder edit and more early closes. Step 2 copied s0-r4's recipe and did less diagnosis than either.

## Per-rollout tables

Columns: fwd/py/web = forward passes, Python calls, web calls used. Tensors = count (down_proj / o_proj / linear_attn out_proj). Form: `col` = column zero/flip/scale, `dir+` = `W + a·d(dᵀW)` with a > 0, `dir−` = removal (a < 0). Both-tests = paired calls with `"model": "both"`; in brackets, edited-model tests on `select`. TF = teacher-forced passes (in brackets, before the first `save_edit`). Attr/DoM = Python calls matching per-unit attribution / difference-of-means patterns (regex, approximate). Err = failed forward calls / Python tracebacks / editor turns with an unparsed tool call. Minutes are from rollout start and include 5 to 11 min of setup before turn 1.

### Step 1 (untrained Inkling 2 Small)

| rollout | reward | test | min | turns | fwd/py/web | tensors | form | direction, strength | 1st save turn (min) | both-tests (select) | confirm 1/2 | TF (before save) | attr/DoM | err | ctx end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0-r0 | 1.2 | 0.6 | 105.6 | 76 | 15/55/0 | 1 (1/0/0) | col zero | 7 Wait-loop units of L23 = 0 | 31 (9.7) | 10 (8) | −0.4 / −0.4 | 5 (3) | 3/9 | 2/2/3 | 138k |
| s0-r1 | 9.4 | 11.1 | 52.7 | 66 | 7/54/0 | 48 (24/6/18) | dir+ | `</think>` specific (top-80, 5 PCs), +0.45 | 35 (9.9) | 6 (4) | +14.5 / +16.0 | 1 (1) | 3/0 | 3/2/1 | 95k |
| s0-r2 | 18.8 | 16.7 | 88.4 | 87 | 11/73/0 | 6 (6/0/0) | col zero + col scale | 16 Wait-loop units = 0, 5 anti-Wait units ×2 to ×3 | 60 (16.9) | 9 (6) | +14.8 / +21.5 | 2 (2) | 5/8 | 2/1/0 | 187k |
| s0-r3 | 15.6 | 15.2 | 85.9 | 77 | 11/63/0 | 6 (6/0/0) | col scale + col zero | 18 ready-to-stop units ×5, 4 loop units = 0 | 37 (11.2) | 9 (7) | +18.0 / +16.8 | 2 (2) | 10/3 | 1/1/1 | 149k |
| s0-r4 | 33.6 | 32.8 | 66.1 | 84 | 9/72/0 | 48 (24/6/18) | dir− + dir+ | ` Wait` specific −1.0, `</think>` specific +2.0 (top-600) | 40 (12.0) | 8 (6) | +36.3 / +45.3 | 1 (1) | 5/1 | 2/2/0 | 136k |
| s0-r5 | 0.0 | none | 8.6 | 27 | 0/19/0 | 0 | none | ended "no tool call in five replies" | none | 0 | none | 0 | 0/0 | 1/0/7 | 50k |
| s0-r6 | 2.7 | pending | 106.2 | 96 | 14/79/0 | 1 (0/1/0) | dir− | loop direction (mean loop − mean finished, L23 input), −1 on L23 o_proj | 41 (12.4) | 10 (7) | +2.0 / +2.0 | 4 (3) | 8/10 | 2/3/0 | 185k |
| s0-r7 | 27.3 | pending | 59.9 | 86 | 14/69/0 | 48 (24/6/18) | dir− + dir+ | plain ` Wait` readout −1.0 on 48; plain `</think>` readout +0.6 on L18-23 MLP | 43 (11.2) | 9 (6) | +32.4 / +38.3 | 5 (2) | 4/2 | 1/1/1 | 147k |

### Step 2 (after one update; library holds step 1's notes)

| rollout | reward | min | turns | fwd/py/web | tensors | form | direction, strength | 1st save turn (min) | both-tests (select) | confirm 1/2 | TF (before save) | attr/DoM | err | ctx end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1-r0 | 32.0 | 62.9 | 79 | 13/62/0 | 48 (24/6/18) | dir+ + dir− | `</think>` +1.75, ` Wait` −1.0 (top-600) | 32 (10.8) | 8 (6) | +32.8 / +39.1 | 5 (4) | 0/0 | 3/2/0 | 108k |
| s1-r1 | 20.7 | 82.6 | 67 | 14/51/0 | 48 | dir+ + dir− | `</think>` +1.5, ` Wait` −1.0 | 28 (10.7) | 11 (9) | +22.3 / +27.7 | 3 (3) | 2/0 | 1/2/0 | 97k |
| s1-r2 | 13.3 | 58.0 | 42 | 7/32/0 | 48 | dir− + dir+ | ` Wait` −1.0, `</think>` +1.0 | 13 (7.5) | 7 (4) | +17.2 / +19.9 | 0 (0) | 0/0 | 2/0/0 | 65k |
| s1-r3 | 25.8 | 49.5 | 34 | 5/27/0 | 48 | dir+ | `</think>` +1.75 | 10 (12.1) | 3 (2) | +29.1 / +35.7 (edited only, n=2) | 0 (0) | 0/0 | 1/0/0 | 54k |
| s1-r4 | 24.2 | 78.5 | 75 | 12/60/0 | 48 | dir+ | `</think>` +1.75 | 26 (17.1) | 8 (6) | +23.8 / +32.4 | 4 (3) | 1/1 | 2/1/0 | 102k |
| s1-r5 | unfinished | 69.7 | 49 | 9/38/0 | none yet | dir+ | `</think>` +2.5 under test at turn 49 | 29 (11.7) | 0 (5, edited only) | not reached | 3 (3) | 0/1 | 1/1/0 | 81k |
| s1-r6 | 23.1 | 49.4 | 52 | 9/40/0 | 48 | dir+ | `</think>` +2.0 | 22 (7.1) | 6 (4) | +26.6 / +31.3 | 3 (3) | 0/0 | 2/1/0 | 73k |
| s1-r7 | 27.7 | 49.6 | 42 | 7/32/0 | 48 | dir+ | `</think>` +2.0 | 12 (7.8) | 7 (5) | +29.3 / +40.2 | 0 (0) | 0/0 | 2/1/0 | 70k |

Step 2 has no test-set scores yet.

### Opus 5.5 (one standalone search, older prompt, 50-pass budget, native API tool calls)

| run | test | min | turns | fwd/py/web | tensors | form | direction, strength | 1st save turn (min) | both-tests (edited-only on select) | confirm | TF (before save) | attr/DoM | err | ctx end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus | +27.2 | 170.1 | 135 | 18/113/0 | 9 (4/1/4) | col flip + dir+ | L16 down_proj col 998 ×(−1); whitened `</think>`-specific direction +2.5 on 9 writers in L15-22 | 52 (34.5) | 5 (10, 128 problems × 2) | +29.3, +26.2, +28.1 (3 × 256 answers) | 4 (3) | 21/0 | 4/4/0 | 225k |

Opus built its own splits (the older prompt gave no baseline or `splits.json`), so its select and confirm numbers are on different problems.

## Answers

### 1. What changed between step 1 and step 2

- **Strategy.** Step 1 rollouts diagnosed first and arrived at different mechanisms (Wait units, ready-to-stop units, a loop direction, readout directions). Step 2 rollouts read the library and adopted its recipe. s1-r3 at turn 3: "The prior session's best edits: wait-specific direction removal + think-specific amplification." s1-r2 at turn 4: "The library notes give a strong lead. The best-documented recipe: specific readout-direction edits".
- **Speed.** Mean turns 74.9 → 55.9, mean minutes 71.7 → 61.5 (finished rollouts), Python calls 60.5 → 43.4, forward passes 10.1 → 9.6. First `save_edit` turn 41.0 → 21.5. Context at the end 136k → 81k tokens.
- **Edit form.** Step 1: 3/7 submitted edits on 48 writers, 4 on 1 or 6 tensors. Step 2: 7/7 on the same 48 writers.
- **Diagnosis depth.** Teacher-forced passes 2.5 → 2.2 per rollout, but step 2 passes mostly checked the `1+γ` norm convention on layer-24 hidden states. Requests for `mlp_act` on all 24 layers: 7/8 step 1 rollouts, 2/8 step 2. Attribution-pattern calls 36 → 4, difference-of-means calls 33 → 3. Some controls remained: s1-r4 tested a random direction (+0.8 vs +31.25 points), s1-r1 a reversed Wait term.
- **Errors.** Unparsed native tool calls 13 → 0. Python tracebacks 12 → 8. Failed forward calls 14 → 14, of which "unknown practice id" 8 → 10.

### 2. Weak rollouts, best rollout, convergence

- The update lifted the low end. Bottom three rewards: step 1 0.0, 1.2, 2.7; step 2 13.3, 20.7, 23.1. The best did not move: 33.6 → 32.0, inside the ~3-point noise.
- Convergence is strong. All 7 finished step 2 edits share direction, construction (top-600 frequent tokens) and tensor set; they differ only in α_think (1.0 to 2.0) and whether the Wait term is present (3/7). Unit-level edits disappeared.
- Step 2 mostly chose a lower dose than s0-r4, citing skip-thinking. That is the likely reason no step 2 rollout beat s0-r4.

### 3. Comparison with Opus

- **Method.** Opus ran a weight scan of down_proj columns against the `</think>` and Wait readouts, per-layer and per-neuron direct logit attribution at decision positions, checked a non-weight explanation (the presence penalty), chose writers by measurement to avoid skip-thinking, and ran an ablation showing its two parts work together (unit 998 alone 17.6%, gain alone 15.6%, both 32.8%).
- **Depth.** Opus: 113 Python calls, 21 attribution-tagged, 3 confirmations (768 paired answers). Step 1 Inkling: 60.5 Python calls, about 5 attribution-tagged per rollout. Step 2: 43.4 Python calls, under 1 attribution-tagged call per rollout.
- **Inkling better:** speed (50 to 106 min vs 170); its best edit scored higher on MATH-500 (+32.8 vs +27.2); s0-r2 also found L16 unit 998.
- **Inkling worse:** made-up problem ids, malformed native tool calls in step 1, wasted passes, and in step 2 reusing the library's recipe instead of diagnosing. Opus's prompt and budget differ, so time and pass counts are not like for like.

### 4. Reward hacking and shortcuts

- The submitted edits do not close thinking on every problem. Reward-set hit-cap for edited weights: step 2 0.53 to 0.78 (original 0.93); step 1 0.40 to 0.93. The most aggressive edit is s0-r4 (its note reports about 8 to 10% of generations close the thinking section after very few tokens).
- The shortcut was measured and rejected several times: s0-r1 saw 99% skip at α = 1.0; s1-r2 saw 40% immediate `</think>` at α_think 1.5 with Wait removal; s1-r6 saw 35/256 immediate closes at α = 3.
- The reward does not see this. Mean length and the immediate-`</think>` rate on the reward set are not logged.
- 3/8 step 2 rollouts made no teacher-forced pass and built the direction from weights alone with the library's parameters. That is reuse of a previous rollout's result, and the update gets the credit for it.

### 5. Harness and prompt problems to fix before more steps

1. **The library carries within-task results across RL steps.** No rollout made a web call (0 in 16), so every note is a rollout's own recipe with numbers. The prompt also contradicts itself ("new independent experiment with no previous ... notes" versus "/library ... already holds what earlier sessions saved"); 7/8 step 2 rollouts discussed this. Options: freeze the library, filter notes that report results on this model, or rerun the step 2 checkpoint with an empty library to measure the update alone.
2. **Problem ids are typed by hand.** 18 "unknown practice id" failures in 16 rollouts. Let `forward` accept a split name.
3. **Native tool-call tokens leaked as text in step 1.** 13 turns were not parsed and the repair step never fired. s0-r5 ended this way with reward 0 and advantage −1.16, so the update partly trained against a harness failure.
4. **Skip-thinking is not measured on the reward set.** Log mean tokens and the position-0 `</think>` rate for the submitted edit.
5. **s1-r5 has no summary yet.**
6. The editors' confirm-split gains overstate the reward by 5.1 points on average in step 2 (2.8 in step 1).

## Caveats

- 8 rollouts per step, one update (30 training sequences, lr 2e-5, LoRA rank 32). Reward noise is about 3 points per rollout.
- Step 2 changed the policy and the library at the same time.
- s1-r5 is unfinished. Step 2 has no MATH-500 scores; step 1 r6 and r7 are pending.
- Opus used an older prompt, a 50-pass budget, no baseline files and its own splits. Only its MATH-500 score is comparable.
- Attribution and difference-of-means counts come from regular expressions and are approximate. Edit form, direction and strength were read by hand.
