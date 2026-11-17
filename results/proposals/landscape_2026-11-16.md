# Weight-space programming: landscape and next ablations (snapshot 2026-11-16)

Sources read in full unless noted: Takehi et al., "Sparks of Weight-Space Programming" (MIT CSAIL blog, via citations); Hu et al., arXiv 2611.09382 (via the earlier sweep); WSP-Gym (README, github.com/emeraldant/WSP-Gym); Naik, Liu, Gangal, "Auto Weight-Space Programming (Part 1)"; Marasini, Khadka, Rahimi, "WeightBench", arXiv 2611.18553; Mehta et al., "Interpretable weight-space programming builds transparent transformers and brain encoding models", arXiv 2611.18114; Ostrow, Qian, Fiete, Liu, "A Mathematical Theory of the Jacobian Lens", arXiv 2610.31955; Cohen and Shapira, "Can we compress backpropagation into a model?" (roicohen.org); the ulam-2-small LinkedIn post (pasted text). Also read: Marthi, "Suppressing 'Wait' stops a small model's reasoning loops. The Jacobian lens added nothing in my tests" (LinkedIn, 16 Nov 2026; text pasted by the user).

## Directions

1. **Agents that edit with forward passes** (Takehi et al.; Hu et al.; ulam-2-small; this project). An agent diagnoses, edits and tests. Qwen3.5-0.8B MATH-500: 7.9 → 48.1% (GPT-6 Astra, 50 passes; 12–14% with 10–25 passes); 8.35 → 37.8% (Hu et al.); 6.6 → 39.4% single edit, 42.5% stacked (ours, 8k cap). ulam-2-small runs the same loop at 3B scale with each round's best checkpoint as the next base, and no SFT or RL.
2. **Edits from weights alone, forward passes restricted** (WeightBench; WSP-Gym answer-at-once; AWSP reading). Frontier agents score near 0 on WeightBench's LM reasoning-loop repair; WSP-Gym's trained editor repairs small networks at 82.9% but no better than random edits on Qwen3.5-0.8B.
3. **Trained weight writers** (WSP-Gym: SFT on planted faults, then RL; Cohen–Shapira: a meta-model writes ΔW for Qwen3.5-2B from a task description, trained by regression, then loss through the base model, then reward; AWSP: writing weights fails). The trained writers mostly remove a tendency (a favourite answer, a loop) and add no knowledge; with reasoning on, Cohen–Shapira's update cuts MATH-500 22.8 → 10.5.
4. **Models written from scratch** (arXiv 2611.18114): agents program small transformers and brain encoders in code; 100% on algorithmic tasks, then compression to ~166 parameters; "current agents do not invent genuinely new computational principles."
5. **Localisation theory** (Jacobian lens, arXiv 2610.31955): averaged-Jacobian directions pick out modes that persist across tokens and contexts even at low variance; toy models only.

**Marthi (LinkedIn, 16 Nov 2026).** On Qwen3.5-0.8B, 96 templated arithmetic problems, 2,048 tokens, temperature 0.3, one sample: a direction built from hesitation tokens ("Wait", "Hmm", "But", "Actually", "Okay") through the Jacobian lens, ablated in activations at layers 15–19, cut looping 30 → 17 and raised correct answers 28 → 44; the logit-lens direction reached 46; a plain sampling-time ban on those tokens did best (16 loops, 48 correct, 2 of 28 baseline-correct problems lost vs 8–14 for the ablations); random directions did nothing. At layers 8–12 the Jacobian lens did not help (looping 41 vs 30 baseline). A direction from contrasting looping vs non-looping generations was a null result. Re-running the same 14 looping prompts looped 6 of 14 times. Cites a closed-form MEMIT-style edit of one layer-23 MLP output matrix ("Embedding", vendor post) with a KL gate on protected text (reject if > 0.03 KL); without it, WikiText perplexity exploded. Conclusion: suppressing hesitation tokens reduces loops; the lens finds nothing a ban misses.

## What the field agrees on

- On the Qwen loop task, edits remove one habit. A decode-time "Wait" ban gives 38.8% (WSP-Gym, 32k cap, 1 answer); random edits inside a table that names the failure reach 37.6%. Our best edits sit at this level.
- Gains come from removing an obstacle to an existing skill, not from adding skill (Takehi et al., Cohen–Shapira, 2611.18114, WeightBench's META framing).
- Forward-pass budget decides results (Takehi: 12% at 10 passes vs 48% at 50).
- Most of a trained editor's gain is supervised in WSP-Gym (RL +2.6 points); pure RL from the base collapses to "no edit" when success is rare — the same collapse as our curriculum plateau.

## Ranked ablations for hill-climbing on this task

1. **Ban baselines on our harness** (~$15; Marthi's ban beat every ablation, including on problems already solved): "Wait" ban; "Wait" ban + our best edit; at 8k and 32k caps. Decides whether our edits exceed the ban and whether the 48.1% vs 39% gap is the token cap.
2. **Truncation and length diagnostics** (free, from logs): accuracy on problems the base already solves, mean length, "Wait"/reflection-token rates, early `</think>` rate for every edit. Catches gains that come from cutting reasoning.
3. **Random matched controls** (~$10): equal-norm, equal-location, equal-rank random edits on the editor's chosen tensors.
4. **Ban as the new base**: train and evaluate the editor on Qwen with the loop already removed (ban or the best edit), with reward = gain over that base plus a guard penalty on a non-math set (WeightBench's collateral term). Forces gains beyond the loop.
5. **Supervised warm start** (~$60): SFT the untrained editor on the top-quartile sessions (WSP-Gym and Cohen–Shapira both get most of their gain from a supervised stage), then RL; compare with the 8-step RL editor.
6. **Describe-and-rebuild and data-independence checks** (2611.18114): rebuild the best edit from a 200-word description in a fresh session; rerun the edit code on other problems.
7. **Forward-pass budget sweep** (8, 16, 32, 50) for the step-8 editor: separates reasoning about weights from search.
8. **Planted-fault gym on real LMs** (WSP-Gym extended): ground-truth repairs, short sessions, many tasks; the build with the most RL signal.
