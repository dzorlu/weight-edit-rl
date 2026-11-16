# cur4g16: how the editor's behaviour changed over training (snapshot 2026-11-16)

Sources: `cur4g16-s{0..7}-r{0..15}.jsonl`, `steps.jsonl` and `config.json` in this folder; the practice splits (`/data/math_practice/splits.json`); the held-out sessions `/eval/untrained-orig-math500/` and `/eval/cur4g16-s4-orig-math500/` (4 sessions each) on the `glimmer-runs` volume. All numbers come from `analyze_cur4g16.py` (`python3 results/rl/cur4g16/analyze_cur4g16.py [--detail]`). Edit kind, target and strength were read by hand from each FINAL SUBMISSION text (the `EDITS` table in the script). Counts of attribution, difference-of-means and readout code, of controls and of topics come from regular expressions and are approximate. Steps are numbered 1–8 (log files `s0`–`s7`).

Setup differences that confound a trend across all 8 steps:
- Steps 1–2 and steps 3–4 had different prompts: from step 3 on, a test is capped at 128 answers per model (256 before). The same-setup comparisons are step 1 vs 2, step 3 vs 4, and steps 5–8 among themselves.
- Steps 5–8 ran on `cur4g16-k1` (Qwen plus one earlier edit, 44.5% of the reward set), and the prompt listed that edit. `steps.jsonl` records the edit as `cur4g16-s2-r11`, the step-3 session at +37.5 (the best of steps 1–4), not a step-4 session.
- Two sessions failed with a harness error and are excluded: `s5-r14` (step 6) after 1 turn and `s7-r1` (step 8) after 68 turns.
- The aborted-start artefact was not found. No log restarts its turn numbers, clock or budget counters, and no test above 128 answers was accepted in step 3.

Noise: with 16 sessions per step, the sd between sessions is 11–34 minutes of editor time, 2.8–4.5 forward passes and 108–177 completion tokens per turn, so the SE of a step mean is about a quarter of these. A share of 8/16 has an SE of 12.5 points, and a drop from 9/16 to 4/16 gives Fisher p = 0.15. p-values are two-sided permutation tests on session values unless marked Fisher.

## Summary, ranked by strength of evidence

1. **On the updated starting model (steps 5–8), most sessions submitted nothing, and their own tests support that.** 37 of 62 sessions ended with no net edit (9/16, 8/15, 13/16, 7/15), with no trend (Fisher p = 0.61 for steps 5–6 vs 7–8). Sessions were about half as long (46.7 vs 87.0 editor minutes), the last save came at 39 minutes instead of 61, and 7 minutes followed it instead of 26. Only 1 of the 37 no-edit sessions ever saw a paired test above +2 SE, against 17 of the 25 edit sessions. This responds to the new task; it is not a change across updates.
2. **Steps 5–8 copied the form of the listed earlier edit and went after a new failure.** 10 of 25 edits were a rank-one write of a token readout gated by a difference-of-means input direction, the form of the earlier edit (4 of 64 edits in steps 1–4; Fisher p < 0.01). 60 of 62 sessions discussed strengthening the earlier edit. 47 of 62 FINAL texts name repeated `</think>` after the answer has started (10 of 64 in steps 1–4), and 7 edits suppress `</think>` instead of amplifying it.
3. **Shorter reasoning per turn after 4 updates, in both clean comparisons.** Reasoning characters per turn fell from 1,513 to 1,213 from step 3 to 4 (p = 0.01); in the held-out sessions they were 1,607 (untrained) and 1,215 (step-4 editor) (p = 0.08, 4 vs 4). Within steps 1–4, shorter reasoning went with higher reward (Spearman −0.22 pooled; −0.74 in step 1). Step 1 → 2 showed no change (p = 0.69), and on the new starting model reasoning was longer again (2,014–2,313).
4. **Format and testing errors seen only in step 1 disappeared.** Two step-1 sessions submitted without any forward pass (−6.2, +1.2), against 0 of 48 in steps 2–4 (Fisher p = 0.06). Sessions with a repaired tool call went from 8/16 in step 1 to 2/16 in step 2 (Fisher p = 0.05). Failed forward requests fell from 5.2 to 2.0 per session (mostly mistyped practice ids). In the direction of the update, but on few events.
5. **Time after the last save rose within both same-setup pairs**: 20.0 → 37.5 minutes (p < 0.01) and 15.4 → 31.1 (p = 0.02). It correlated negatively with advantage in steps 1 and 3 (−0.25, −0.49), so the change is not in the direction of the training signal; its cause is not determinable from the logs.
6. **The mix of edit kinds drifted, but not toward what was rewarded.** Sessions amplifying `</think>` numbered 9, 7, 7, 4 of 16 in steps 1–4, although this target correlated +0.52 with advantage in step 1. Sessions targeting Wait or loops rose from 9 to 14 of 16 (Fisher p = 0.11). Across all measures with |Spearman| ≥ 0.3, the next step moved with the advantage 21 times and against it 21 times.

The update check shows each update did what the objective asks on the sampled tokens (correlation between a session's change in log-likelihood and its advantage 0.70–0.98; KL 0.0044–0.0055 per token). At 16 sessions per step, that movement shows up in behaviour as at most the changes above.

## 1. Effort and pacing

| | step 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | held-out untrained | held-out step-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| reward (pts) | 16.1 | 15.2 | 17.0 | 19.7 | 0.7 | 0.9 | 0.0 | 1.1 | 17.8 | 17.7 |
| editor minutes | 83.6 | 106.7 | 74.2 | 83.4 | 44.6 | 48.7 | 44.9 | 48.8 | 83.0 | 73.2 |
| turns | 84.9 | 77.8 | 84.1 | 80.9 | 86.8 | 83.1 | 85.1 | 79.1 | 58.2 | 65.0 |
| forward passes (of 32) | 10.3 | 13.1 | 15.1 | 13.4 | 11.5 | 12.2 | 11.4 | 12.7 | 8.5 | 10.0 |
| python calls | 68.1 | 61.8 | 66.4 | 64.8 | 72.8 | 68.1 | 71.2 | 63.8 | 46.2 | 52.2 |
| first save_edit (min) | 8.6 | 5.7 | 6.6 | 8.1 | 8.8 | 8.3 | 8.5 | 7.5 | 6.6 | 6.7 |
| last save_edit (min) | 63.6 | 69.2 | 58.7 | 52.3 | 36.4 | 41.4 | 37.7 | 42.1 | 54.4 | 50.8 |
| minutes after last save | 20.0 | 37.5 | 15.4 | 31.1 | 8.2 | 7.3 | 7.3 | 6.7 | 28.6 | 22.4 |

- The 128-answer cap shortened sessions (steps 1–2: 95.2 minutes; steps 3–4: 78.8, p = 0.01) and raised the number of tests (7.5 → 10.4 per session, p < 0.01).
- The first save came at 5.7–8.8 minutes in every step, with no trend.
- Held-out sessions (90-minute cap, no 128 cap): 2 of 4 untrained sessions ended at the time limit, 0 of 4 for the step-4 editor.

## 2. Testing discipline

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| paired tests per session | 6.1 | 8.1 | 10.7 | 9.8 | 7.1 | 8.0 | 6.9 | 9.2 |
| of them on confirm_1/2 | 1.7 | 2.3 | 3.5 | 3.6 | 1.9 | 2.3 | 1.3 | 2.4 |
| sessions submitting with no edited test | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sessions mentioning a random-direction control | 5 | 4 | 2 | 3 | 3 | 3 | 4 | 4 |
| sessions mentioning a reversed-sign control | 8 | 3 | 5 | 4 | 7 | 12 | 8 | 12 |

- Every session that submitted an edit in steps 2–8 ran at least one confirm test. Whether each mentioned control was run is not determinable from the regex; a random vector appears in edit-saving code in only 2 sessions.

## 3. Diagnosis

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | held-out untrained | held-out step-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher-forced passes | 3.6 | 4.7 | 4.0 | 3.7 | 4.4 | 3.9 | 4.4 | 3.5 | 2.8 | 3.5 |
| attribution calls | 5.9 | 5.8 | 6.3 | 4.9 | 3.9 | 3.7 | 3.1 | 3.1 | 6.2 | 4.5 |
| difference-of-means calls | 2.8 | 4.5 | 3.8 | 1.8 | 4.3 | 4.3 | 4.1 | 2.6 | 1.8 | 2.0 |
| readout-through-embedding calls | 9.2 | 9.2 | 8.9 | 7.9 | 6.7 | 6.1 | 4.3 | 4.3 | 7.8 | 6.2 |

Every session read `/practice/baseline/SUMMARY.txt`. Only the two untested step-1 sessions saved an edit computed from the weights alone.

## 4. Edits

Sessions using each kind (a session can use several):

| step | readout removed (mostly Wait) | readout amplified (`</think>`) | units zeroed / scaled / given a readout | loop direction removed | gated rank-one write | head zeroed | no net edit | distinct combinations |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 7 | 6 | 3 | 1 | 0 | 0 | 10 |
| 2 | 8 | 6 | 1 | 4 | 1 | 1 | 0 | 8 |
| 3 | 4 | 5 | 6 | 3 | 2 | 0 | 0 | 10 |
| 4 | 6 | 3 | 8 | 1 | 0 | 0 | 0 | 9 |
| 5 | 2 | 0 | 2 | 0 | 3 | 0 | 9 | 3 |
| 6 | 1 | 0 | 2 | 0 | 4 | 0 | 8 | 4 |
| 7 | 1 | 0 | 0 | 0 | 2 | 0 | 13 | 2 |
| 8 | 5 | 0 | 2 | 0 | 1 | 0 | 7 | 3 |

- Diversity did not collapse: steps 1–4 each have 8–10 distinct combinations among 16 edits.
- Strengths: removals almost always α = −1; `</think>` amplification +0.1 to +6 (large values only on layer 23's o_proj); gated writes 0.01–0.15; unit scaling ×2 to ×8.
- Steps 5–8: edits target Wait (17 sessions), `</think>` suppression (7) and answer-start or "boxed" tokens (3); mean reward +1.8 (range −3.5 to +7.0).

## 5. Outcome links within steps (steps 1–4, n = 64, Spearman with reward minus step mean)

- The edit's effect: edited hit-cap −0.95; two or more kinds +0.35; number of tensors +0.32; gated `</think>` write +0.23 (4 sessions, mean 26.0).
- Process: submitting untested −0.29; reasoning characters per turn −0.22; FINAL text length −0.29.
- Effort is not consistently linked to reward: forward passes +0.42 in step 1, then +0.17, −0.25, −0.64.
- Mean reward by kind: gated write 26.0 (n = 4), unit scaling 20.4 (9), readout removal 18.8 (22), readout amplification 17.1 (21), unit zeroing 16.8 (15), loop direction 16.1 (11), head 5.5 (1). By target: Wait and `</think>` together 24.0 (8), `</think>` alone 16.3 (18), Wait alone 16.3 (19), loop alone 14.0 (5).

## 6. Steps 5–8: no-edit decisions

- Starting model: 45.3% correct, 28.5% cut at the cap, 34.2% with no boxed answer. 47 of 62 FINAL texts describe a failure absent in Qwen as downloaded: after the earlier edit closes thinking, the model restarts the answer and emits `</think>` again, up to the cap.
- No-edit sessions ran 5.8 edited tests and 5.3 save_edit calls on average before giving up; in 36 of 37, the best paired difference seen was at most +2 SE, and many saw large losses (−46.1, −39.1, −46.9, −36.4 points).
- The listed earlier edit: 60 of 62 sessions discussed strengthening it; 29 of 62 saved a variant of its form; no session restored or reversed it.
- New ideas: suppressing `</think>` in the answer section (7 sessions); writing answer-start tokens or "boxed" at loop positions ("boxed" in edit-saving code in 22 of 62 sessions, 7 of 64 before); targeting the Wait loops left inside thinking.
- Sampling entropy rose from 0.43–0.47 (steps 1–4) to 0.60–0.69.

## 7. Reasoning style

| | 1 | 2 | 3 | 4 | 5–8 | held-out untrained | held-out step-4 |
|---|---|---|---|---|---|---|---|
| completion tokens per turn | 1,030 | 1,025 | 873 | 747 | 1,015 | 1,045 | 917 |
| reasoning characters per turn | 1,718 | 1,653 | 1,513 | 1,213 | 2,120 | 1,607 | 1,215 |
| python tracebacks per session | 3.1 | 2.4 | 2.3 | 3.2 | 2.3 | 1.8 | 0.5 |
| failed forward requests | 5.2 | 2.0 | 1.4 | 1.6 | 1.4 | 2.0 | 1.5 |
| repaired tool calls | 0.9 | 0.1 | 0.1 | 0.1 | 0.2 | 1.5 | 0.5 |

## 8. Changes in the direction of the training signal

Across 24 measures and the within-setup updates (1→2, 3→4, 5→6, 6→7, 7→8), where |ρ| ≥ 0.3, 21 changes went with the advantage and 21 against it. With: fewer repaired tool calls; no untested submissions after step 1; shorter reasoning per turn; more paired tests after step 1. Against: `</think>` amplification (ρ +0.52 in step 1, then 9 → 7 sessions); time after the last save; submitting an edit after step 6. The behaviours of each step's top sessions did not become more frequent in the next step.

## 9. Held-out sessions: untrained vs step-4 editor (same task, same prompt, 4 vs 4)

| | untrained | step-4 editor |
|---|---|---|
| reward-set reward | 20.3, 19.1, 15.2, 16.4 (mean 17.8) | 17.2, 29.3, 14.1, 10.2 (mean 17.7) |
| MATH-500 gain | 17.1, 0*, 13.8, 16.2 (mean 11.8) | 16.4, 28.8, 14.0, 12.2 (mean 17.9) |
| editor minutes; time-limit endings | 83.0; 2 | 73.2; 0 |
| turns; tracebacks | 58.2; 1.8 | 65.0; 0.5 |

\* This session's edit (+19.1 on the reward set) was lost at the time limit, a harness bug fixed afterwards, and is counted as 0.

On the reward set the two editors are tied (17.8 vs 17.7). Most of the MATH-500 gap (11.8 vs 17.9) comes from the lost edit; without it the untrained mean is 15.7. The step-4 editor's one large edit (+28.8) combines Wait removal with a gated `</think>` write, the two highest-reward forms in steps 1–4.

## Not determinable from the logs

- Whether a mentioned reversed-sign or random-direction control was actually run.
- Why the time after the last save rose from step 1 to 2 and from step 3 to 4.
- Whether the three no-op submissions (tensors identical to the starting weights) were bit-identical when scored; their rewards (0.0, −1.9, −1.9) are consistent with scoring noise.
