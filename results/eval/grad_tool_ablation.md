# Gradient-tool ablation (snapshot 2026-11-17)

Editor: the step-8 editor of run `cur4g16` (`tinker://80fdacdc-ec4e-5c2e-afaf-de75995e07e4:train:0/sampler_weights/cur4g16-step8-keep`), no training. Starting model: `cur4g16-s8best` (Qwen3.5-0.8B plus the step-8 editor's best edit; 39.5% on MATH-500), so the loop is mostly fixed already and gains have to come from something else. 4 sessions per arm, 90-minute cap, 2 sessions per GPU, 32 forward passes. Gains are over the starting model: reward set (256 answers) and MATH-500 (first 250 problems × 2). Script: `grad_tool_ablation.py`.

- Arm A: current tools (`python`, `forward` generate / teacher_forced).
- Arm B: the same plus the forward tool's `gradient` request (`--grad-tool`): one backward pass on up to 8 texts of up to 4,096 tokens; returns activation × gradient attributions of a target (log-prob of given tokens, or the loss) per layer, per MLP unit and per attention head. No optimizer; weights unchanged (checked: teacher-forced log-probs identical before and after).

| | Arm A (current tools) | Arm B (+ gradient tool) |
|---|---|---|
| reward-set gain per session | +3.9, −1.2, +0.4, +1.6 (mean +1.2) | +1.6, 0, 0, 0 (mean +0.4) |
| MATH-500 gain per session | +2.4, −1.2, −4.2, −1.8 (mean −1.2) | +5.6, 0, 0, 0 (mean +1.4) |
| sessions with no edit | 1 (r3: tensors saved byte-identical to the start) | 3 |
| sessions whose best paired test was above +2 SE | 1 (z 2.1) | 1 (z 2.4) |
| gradient requests | – | 3, in 1 of 4 sessions (2 succeeded, 1 out of memory) |

Edit kinds: every submitted edit adds another rank-one `</think>`-readout term to the same 48 writer matrices as the starting model's edit (A r0, A r2, B r0), or rescales a few `</think>`-writing MLP units (A r1). No session found a different failure to fix.

How the gradient tool was used (session B r1, the only one that used it):
- Turn 22: it passed a Python expression as the text (`"open(f'/work/diag/fail_{i}.txt').read() for i in range(4)]"`), got attributions for that literal string, and noticed: "I passed a literal string which got treated as one raw text. The attribution numbers are for that garbage text."
- Turn 26: a ~4,000-token looping answer ran out of GPU memory.
- Turn 33: a 1,500-token excerpt worked (target: log-prob of `</think>`; residual attribution peaked at layer 14, +20.3). The session then tested candidate edits, all at or below the start (best paired z −7.2 … ), and submitted no edit.

Reading: on this starting model neither arm finds a real gain (all differences are within about ±5 points on 500 answers and 4 sessions per arm); the gradient tool did not change that and was barely used (1 of 4 sessions). Two problems limit it as built: forward requests take texts as literal strings, so the editor cannot point the tool at answers saved in its sandbox and has to paste them into the call; and texts near the 4,096-token limit run out of memory next to the packed vLLM servers (2,048 is a safer limit). Both apply to teacher-forced requests too.
