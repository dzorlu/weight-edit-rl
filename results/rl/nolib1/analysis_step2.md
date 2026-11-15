# nolib1 step 2: why one update did not raise the reward (snapshot 2026-11-14)

Sources: `nolib1-s0-r*.jsonl`, `steps.jsonl`, `config.json` in this folder; `../hero3/hero3-s0-r*.jsonl`, `../hero3/steps.jsonl`. Numbers come from `analyze_step2.py` (`python3 results/rl/nolib1/analyze_step2.py --detail`), which reuses `analyse()` from `../hero3/analyze_steps.py`. Edit form, direction and strength were read by hand from each FINAL SUBMISSION text. Rewards in points. "Step 1" = hero3 s0 (the rollouts the update trained on); "step 2" = nolib1 s0 (sampled from the checkpoint after that update, prompt without the web/library section, no web tools).

## Summary

1. The difference in means (13.6 vs 13.5) is far below what 8 vs 8 rollouts can detect. Rollout-to-rollout sd is 12.5 (step 1) and 8.2 (step 2); the SE of the difference is 5.3, so a 5% two-sided test with 80% power needs about 16 points. Scoring noise alone (2.5 points per rollout at 21% edited accuracy on 256 answers) would allow about 4.5 points; the spread comes from which recipe each rollout found.
2. The step 1 behaviour with the highest reward did not become more common. Step 1 had 2/8 rollouts with a ` Wait` readout removal plus a `</think>` readout amplification (r4 33.6, r7 27.3). Step 2 has 1/8 (r0, 21.1), and its `</think>` term is weaker (+1.0 on L23 o_proj only). No step 2 rollout used the top-600 "specific" direction construction. The mix of recipe families is about the same as in step 1.
3. The update was small relative to the signal it had to carry. 679k sampled tokens all received their rollout's advantage; turns that call `save_edit` are 7.1% of them, final replies 1.2%, and turns that paste problem ids into `generate` calls 24.0%. The token-weighted advantages nearly cancel (Σ adv × tokens = +57k against Σ |adv| × tokens = 579k). One optimizer step at lr 2e-5 on a rank-32 LoRA. The mean sampled-token negative log-probability fell from 0.436 to 0.407 nats.
4. The harness did not hide an improvement. All 8 rollouts finished and were scored; each editor's own confirm-split estimate matches its reward within 3.0 points on average, so the low rewards (0.4, 3.1) are the edits' real effect.

## 1. Did the policy change?

Columns: fwd/py = forward passes / Python calls. TF = teacher-forced passes (before first `save_edit`). attr/DoM = Python calls matching per-unit attribution / difference-of-means patterns (regex, approximate). hit-cap = share of reward-set answers cut at 8,192 tokens.

### Step 1 (hero3 s0, before the update)

| rollout | reward | adv | min | turns | fwd/py | tensors | family | edit | 1st save turn (min) | TF | attr/DoM | hit-cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0-r0 | 1.2 | −1.06 | 105.6 | 76 | 15/55 | 1 | unit | 7 Wait-loop units of L23 = 0 | 31 (9.7) | 5 (3) | 3/9 | 0.93 |
| s0-r1 | 9.4 | −0.36 | 52.7 | 66 | 7/54 | 48 | readout | `</think>` specific (top-80) +0.45 on 48 writers | 35 (9.9) | 1 (1) | 3/0 | 0.82 |
| s0-r2 | 18.8 | +0.44 | 88.4 | 87 | 11/73 | 6 | unit | 16 units = 0, 5 units ×2 to ×3 | 60 (16.9) | 2 (2) | 5/8 | 0.79 |
| s0-r3 | 15.6 | +0.18 | 85.9 | 77 | 11/63 | 6 | unit | 18 units ×5, 4 units = 0 | 37 (11.2) | 2 (2) | 10/3 | 0.75 |
| s0-r4 | 33.6 | +1.71 | 66.1 | 84 | 9/72 | 48 | readout | ` Wait` specific −1.0 + `</think>` specific +2.0 (top-600), 48 writers | 40 (12.0) | 1 (1) | 5/1 | 0.40 |
| s0-r5 | 0.0 | −1.16 | 8.6 | 27 | 0/19 | 0 | none | ended "no tool call in five replies" | none | 0 | 0/0 | none |
| s0-r6 | 2.7 | −0.93 | 106.2 | 96 | 14/79 | 1 | loop | difference-of-means loop direction −1 on L23 o_proj | 41 (12.4) | 4 (3) | 8/10 | 0.91 |
| s0-r7 | 27.3 | +1.18 | 59.9 | 86 | 14/69 | 48 | readout | plain ` Wait` readout −1.0 on 48 writers + plain `</think>` +0.6 on L18-23 down_proj | 43 (11.2) | 5 (2) | 4/2 | 0.59 |

### Step 2 (nolib1 s0, after the update)

| rollout | reward | min | turns | fwd/py | tensors | family | edit | 1st save turn (min) | TF | attr/DoM | hit-cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r0 | 21.1 | 47.1 | 81 | 9/68 | 48 | readout | plain ` Wait` readout −1.0 on 48 writers + plain `</think>` +1.0 on L23 o_proj | 35 (8.6) | 5 (4) | 3/2 | 0.70 |
| r1 | 11.3 | 73.0 | 69 | 12/54 | 4 | unit | 6 Wait units ×(−1): L16 u998, L20 u1111, L22 u194, L23 u36/u269/u346 | 41 (8.8) | 5 (5) | 8/0 | 0.80 |
| r2 | 16.8 | 58.8 | 61 | 12/47 | 48 | readout | plain ` Wait` readout −1.0 on 48 writers | 31 (8.1) | 5 (5) | 5/1 | 0.75 |
| r3 | 21.1 | 88.8 | 91 | 13/73 | 1 | readout | plain `</think>` readout +6 on L23 o_proj | 47 (10.5) | 3 (2) | 11/0 | 0.69 |
| r4 | 0.4 | 80.2 | 63 | 9/50 | 1 | unit | L19 down_proj unit 31 ×2 | 36 (8.9) | 1 (1) | 7/0 | 0.93 |
| r5 | 13.3 | 119.5 | 72 | 13/53 | 18 | unit + loop | 7 Wait units = 0, L16 u998 ×(−1.5), loop direction −1 on L15-23 writers | 26 (8.8) | 6 (3) | 6/7 | 0.83 |
| r6 | 21.1 | 78.0 | 84 | 14/67 | 1 | readout | plain `</think>` readout +6 on L23 o_proj | 42 (9.2) | 5 (4) | 6/0 | 0.70 |
| r7 | 3.1 | 59.0 | 68 | 8/56 | 48 | loop | difference-of-means loop direction −1 on 48 writers | 43 (10.7) | 3 (3) | 10/5 | 0.90 |

### What changed and what did not

- Recipe families. Step 1: readout 3/8 (mean 23.4), unit 3/8 (mean 11.9), loop 1/8 (2.7), none 1/8 (0). Step 2: readout 4/8 (mean 20.0), unit 2/8 (5.9), unit + loop 1/8 (13.3), loop 1/8 (3.1). 3/8 → 4/8 readout edits is not distinguishable from no change with 8 rollouts.
- The rewarded combination did not spread. The two rollouts with the largest positive advantages (s0-r4 +1.71, s0-r7 +1.18) combined ` Wait` removal on 48 writers with `</think>` amplification. In step 2 only r0 combined them, with a `</think>` term on one tensor at α = 1.0; its select test of that `</think>` term alone gave +1.6, the combination +24.6. r2 used the ` Wait` removal alone (16.8); r3 and r6 used `</think>` amplification alone, on L23 o_proj at α = 6 (21.1 each). No step 2 rollout tested ` Wait` removal together with a strong `</think>` term.
- New in step 2: the `</think>` amplification on L23 o_proj only, at α = 6, found by attribution (r3: L23 attention gives about +20.7 of the +22 `</think>` logit at a genuine close; r6: +21.5 of about 23). r3's select sweep: α 0.75 → 7.4%, 2 → 8.6%, 4 → 17.2%, 5 → 24.6%, 6 → 30.9%, 7 → 22.3%, 8 → 9.8%. r3 reports a random-direction control at α = 6 scoring 36.3% on select, but no code in its log builds a random direction; that request re-tested the same α = 6 weights, so the control is not supported (found in a later audit of all trajectories).
- Strength of the main lever: hit-cap for the best step 2 edits is 0.69 to 0.70; for s0-r4 and s0-r7 it was 0.40 and 0.59.
- Diagnosis: teacher-forced passes 2.5 → 4.1 per rollout (before first `save_edit` 1.8 → 3.4). Attribution-pattern calls 38 → 56 in total, difference-of-means calls 33 → 15. First `save_edit` turn 41.0 → 37.6, minute 11.9 → 9.2. Mean turns 74.9 → 73.6, Python calls 60.5 → 58.5, forward passes 10.1 → 11.2 (of 32), minutes 71.7 → 75.5 (of 120). Step 1 includes s0-r5, which stopped at 8.6 minutes.
- Errors: turns where the editor printed Inkling's native tool-call tokens as text and nothing was parsed: 13 turns in 5 rollouts → 3 turns in 3 rollouts; rollouts ended by this: 1 (s0-r5) → 0. This is the one change in the direction of the gradient (s0-r5 had advantage −1.16), but 1 vs 0 rollouts is not evidence on its own. Python tracebacks 12 → 22; failed forward calls 14 → 20 (section 4).
- Entropy: `steps.jsonl` logs the mean negative log-probability of sampled tokens under the sampling policy: 0.4357 nats over 678,730 tokens (step 1), 0.4065 over 616,591 (step 2), −6.7%. Contexts differ, so this does not measure how far the policy moved; the KL between checkpoints is not logged (`kl_sample_train` is 0 because it compares sampler and trainer on the same weights).
- Prompt: step 1's prompt included the web/library section and web tools (step 1 made 0 web calls and saw an empty library index); step 2 dropped both. The update was trained on the longer prompt. Whether this matters is not determinable from the logs.

## 2. Good vs bad rollouts within step 2

Different recipes, not one recipe at different strengths.

- The three 21.1 rollouts (r0, r3, r6) all edit a token readout direction (` Wait` removal and/or `</think>` amplification). All three ended at edited accuracy 28.5% (73/256), with hit-cap 0.703, 0.691, 0.695. r3 and r6 submitted the same form and strength.
- r4 (0.4) scaled one MLP unit (L19 u31, ×2). Select stayed between +3.5 and +5.5 over 5 tests; confirm +1.2 and +2.3. It tried a "specific" `</think>` direction on L19-23 MLPs and attention outputs and reports all variants ≤ +5.5; it did not try L23 o_proj alone at the strengths r3 and r6 used.
- r7 (3.1) removed a difference-of-means loop direction (Wait-predicting positions minus all others) from 48 writers: select +9.0 and +9.4, α = −2 gave 0%, confirm +2.0 and +5.5. Step 1's s0-r6 used the same idea on one tensor and scored 2.7.
- r1 (11.3) and r5 (13.3) edited Wait-writing MLP units (r5 added the loop direction on L15-23). Step 1's unit edits scored 1.2, 15.6, 18.8.
- Each rollout's own evidence predicted its reward: mean of final confirm_1/confirm_2 paired differences minus reward is +3.0 points over 7 step 2 rollouts (step 1: +2.8). r4 estimated +1.8 and got 0.4; r7 estimated +3.7 and got 3.1. The prompt tells the editor to always submit its best edit.
- Budget was not the limit: the 21.1 rollouts stopped with 18 to 23 of 32 forward passes unused and 31 to 73 minutes left; r0 ran 2 select tests and no sweep. r4 and r7 finished with 23 and 24 passes and 40 and 61 minutes left.

## 3. Size of the training signal

- The update used 30 training sequences, 1,870,355 training tokens (prompt and tool results counted each time a sequence is split) and 678,730 sampled tokens, each weighted by its rollout's advantage. `loss_sum` = −57,002 matches −Σ advantage × sampled tokens (−57,061) at importance ratio 1: the loss is a sum over tokens, so longer rollouts weigh more.
- Token-weighted advantages: s0-r4 +148,915, s0-r6 −120,012, s0-r7 +101,201, s0-r0 −92,150, s0-r2 +52,119, s0-r5 −28,892, s0-r1 −20,018, s0-r3 +15,898. The 129k-token s0-r6 (loop direction, 2.7) weighs more than s0-r7. s0-r5's format failure carries only −28,892 because it stopped after 24,907 tokens.
- Tokens tied to the choice of edit: turns calling `save_edit` hold 48,378 tokens (7.1%), final replies 8,479 (1.2%). The recipe choice is also made in reasoning in earlier turns, so the exact share is not determinable from the logs. Turns issuing `generate` calls (mostly 256 problem ids typed inline) hold 162,692 tokens (24.0%) and get the same advantage as the edit decision.
- One Adam step at lr 2e-5 (β1 0.9, β2 0.95) on a rank-32 LoRA. If hero3 started from a fresh LoRA this was Adam's first step, which moves each parameter by about lr in the sign of its gradient regardless of gradient size; `config.json` does not record whether hero3 had a `start_from`. The logs contain no direct measure of the change in output distribution. A large change from one such step is not expected, and the behaviour tables show none beyond what 8 rollouts can resolve.
- Detectable difference, 8 vs 8: sample sd 12.5 and 8.2, SE of difference 5.3, observed −0.1 (t = −0.01). About 16 points needed for 80% power at 5% two-sided; with only 2.5 to 3-point scoring noise it would be about 4.5 points, but between-rollout spread from recipe choice dominates.

## 4. Harness: anything that would mask an improvement

- All 8 step 2 rollouts ended "editor finished", 0 target restarts, all scored. The reward baseline was the cached 256 answers (7.42%, hit-cap 0.930), as in hero3.
- Tensor counts in the summaries match each FINAL SUBMISSION (48, 4, 48, 1, 1, 18, 1, 48).
- Failed forward calls (not counted against the budget): 20 in step 2: 5 CUDA out-of-memory on teacher-forced passes (r0 ×2, r5 ×2, r7 ×1), 6 "items × n exceeds" limits, 6 unknown problem ids (typed by hand), 1 empty request, and 2 `KeyError: 't'` in r5, which wrote safetensors into `/work/edited` with `save_file` under the tensor name instead of `save_edit`'s key `t`. r5 fixed the keys; its final edit was scored on 18 tensors. Step 1 had 14 failed calls, mostly unknown ids and size limits.
- Python tracebacks 12 → 22, ordinary coding errors (shapes, keys), none from the harness.
- r5 finished at 119.5 of 120 minutes; its tests used `"model": "edited"` (512 answers) rather than paired calls. No rollout hit the hard time limit.
- Nothing found that would lower a reward that should have been higher.

## Caveats

- 8 rollouts per step, one update. Step 1 and step 2 also differ in prompt (web/library section and web tools removed).
- Families and strengths were read by hand from final texts; attribution and difference-of-means counts come from regular expressions and are approximate.
- nolib1 step 2 has no MATH-500 scores yet.


## Corrections from an audit of all trajectories (2026-11-14)

- nolib1-s0-r3's "random-direction control" re-tested the same α = 6 weights; no random direction was built (corrected in section 1 above).
- Tests without a paired standard error (edited side only, original from cache): nolib1-s0-r5 (all its tests) and, in step 3, nolib1-s1-r6. Their numbers are not paired comparisons.
- Reward is edited accuracy minus 7.42% on the reward set (256 answers) for every rollout, by construction (`rl/loop.py`, reward scoring).
