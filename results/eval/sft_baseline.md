# Fine-tuning baseline, no editor (snapshot 2026-11-18)

Script: `harness/sft.py` (run `/runs/sft-baseline`). Qwen3.5-0.8B as downloaded is fine-tuned on its own correct, finished answers to practice problems, on the tensors an editor may change (transformer blocks; 279 tensors changed).

- Training data: 162 correct, finished answers (18 cached, the rest from 2,048 new answers to 1,024 practice problems outside the select and confirm splits); 122 fit the 6,000-token limit.
- Training: AdamW, 4 sequences per step, loss on the answer tokens, learning rate {1e-6, 3e-6, 1e-5}, checkpoints at 16, 48 and 144 steps.
- Selection: accuracy on 128 select problems × 1 sample (8,192-token cap). Generation budget: 2,048 + 10 × 128 = 3,328 answers, under one editor session's limit of 32 × 128 = 4,096.

Select accuracy (original 7.0%, 96% of answers cut at the cap):

| learning rate | 16 steps | 48 steps | 144 steps |
|---|---|---|---|
| 1e-6 | 29.7 | 16.4 | 24.2 |
| 3e-6 | 23.4 | 20.3 | **34.4** |
| 1e-5 | 23.4 | 7.8 | 25.8 |

Best checkpoint (3e-6, 144 steps) on MATH-500, first 250 problems × 2, paired: 7.8% → 31.4%, **+23.6 ± 2.2** (118 answers wrong → right, 0 right → wrong); answers cut at the cap 94.4% → 69.0%.

Against the editor on the same 250 problems (`grad_tool_armC.md`): current tools mean +7.7 (4 sessions), with weight gradients mean +21.9, best session +31.2; the step-8 editor on 16 November, 120-minute cap, mean +23.8, best edit 38.5%.

Reading: plain fine-tuning on 122 of the model's own finished answers matches the editor's mean session and is below its best sessions. The select grid is not monotone in steps (128 answers per cell, SE about 4 points), so the choice of checkpoint is partly noise. The fine-tuned model changes 279 tensors in full; the editor's edits change 1–48 tensors, mostly at rank 1.

## Models side by side (MATH-500, first 250 problems × 2, 8,192-token cap)

| Model | Accuracy | Answers cut at the cap | Tensors changed |
|---|---|---|---|
| Qwen3.5-0.8B as downloaded | 7.8% | 94.4% | – |
| Fine-tuned, no editor (3e-6, 144 steps) | 31.4% | 69.0% | 279, in full |
| `cur4g16-s8best`: the RL-trained (step-8) editor's best edit | 40.2% | 34.8% | 48, rank-1 terms |

`cur4g16-s8best` on all 500 problems: 39.5%. The edit was the best of 4 step-8 sessions on 16 November (chosen on MATH-500 itself among those 4, so its 40.2% is a best-of-4, not a session mean). Editor session means on these 250 problems: +16.5 (current tools, 12 sessions), +18.9 (with gradients, 12 sessions); fine-tuning +23.6.
