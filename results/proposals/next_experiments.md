# Next experiments: literature map and ranked proposals (snapshot 2026-11-16)

Written from a literature sweep and the results in `results/eval/SUMMARY.md` and `results/rl/*/analysis*.md`. Entries marked [nr] were restated from prior knowledge, not re-read; check before citing. Two X posts (Sparse Rewiring; Lazare, veni foras) could not be fetched.

## Key external fact

On this target, an edit can score high without editor skill. WSP-Gym ran the same failure on Qwen3.5-0.8B (MATH-500, 32k-token cap, 1 answer per problem): unedited 8.2%; banning "Wait" at decoding (no weight change) 38.8%; the best random edit to the same 6-token × 64-neuron table 37.6%; untrained Inkling 2 Small's edit 38.2%. Over layers 16–23 the mean gain of random valid edits (+10.3) was not below any proposer's. Our best single edit (39.4%) and stacked edits (42.5%) sit at the same level. The reward probably measures "stopped the loop", which many edits reach.

Our own strongest effect: with the same editor checkpoint, the notes library raised mean reward from 13.5 (`nolib1` step 2) to 24.3 (`hero3` step 2), more than any RL update here.

## (a) Literature map

**A. Weight-space programming and LLMs editing weights**
- Takehi, Teoh, Zhang, Khattab (MIT CSAIL), "Sparks of Weight-Space Programming", blog, 2 Nov 2026 — https://rikiyat.github.io/blog/weight-space-programming/ . GPT-6 Astra edits Qwen3.5-0.8B (one direction subtracted from 64 output columns in layers 15 and 19); with 50 forward passes MATH-500 7.9 → 48.1%, GSM8K 13.8 → 67.1%, IFEval 21.0 → 62.3%; with 10 or 25 passes 12.3% and 14.1%. Gemma 3 270M math: no gain. "Not a method"; edits added no abilities the model lacked.
- Hu et al., "Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations", arXiv 2611.09382. Qwen3.5-0.8B MATH-500 8.35 → 37.80 from one layer-23 down_proj column; IFEval +36.72; Qwen3.5-9B gains ≤ ~3; 6 of 16 trajectories kept the original weights; controls on 48 questions: base 6, edit 20, ablation 4, reversed 0, random direction 10; three layer-23 searches gave +29.45, +5.00, +27.30.
- @emeraldant, WSP-Gym, 2026 — https://github.com/emeraldant/WSP-Gym . Small ReLU networks with 9 planted fault types; Inkling 2 Small trained on Tinker (SFT on the inverse fault, then RL) repairs 82.9% in-distribution in ~38 tokens; unseen families 58.6%, unseen fault types 19.9%; best of 8 with verification 90.5%; ~$174. Label-free RL on imperfect networks +12.0% (second seed +7.4%) vs +5.0% random. Qwen3.5-0.8B controls as above.
- Naik, Liu, Gangal, "Auto Weight-Space Programming (Part 1)", blog, Nov 2026 — https://atharva-naik.github.io/blog/awsp-part-1/ . A Qwen3.5-4B LoRA reads the logit lens of 128-coordinate down_proj slices at 99.8% (4-way); writing fails (recall@8 1.0–1.4% vs 27.7% analytic); diversity of training neurons matters; proposes an intervention "oracle" and an L0/L1/L2 ladder.
- von Csefalvay et al., "IWIBench", arXiv 2611.08647 (abstract only): 10 editing objectives on Gemma 3 4B IT; only the strongest models do well; success transfers poorly across task families.
- Kocabay, "Weight-Space Reading", alphaXiv, 13 Nov 2026: Qwen3.8-27B predicts behaviour change of subtle edits at 0.730 balanced accuracy vs 0.578 from text; readers predicted changes for 53–88% of 40 null edits.
- Lazaridis et al., "When Can One Neuron Fix Repetition Loops in LLMs?", arXiv 2606.13705: held-out loops 46/384 → 12/384 on Gemma 4 IT; "doom looping" over unrecallable facts not fixed.
- Pipis et al., "Wait, Wait, Wait... Why Do Reasoning Models Loop?", ICML 2026, arXiv 2512.12895: smaller and distilled models loop more.
- Tigges et al. (Goodfire), "Neurons as an Interpretable Basis", 10 Nov 2026 (activation interventions).
- Mukherjee et al., "RL Finetunes Small Subnetworks in LLMs", arXiv 2505.11711. Goel et al., "Learning to Interpret Weight Differences", arXiv 2510.05092. Li et al., "Training Language Models to Explain Their Own Computations", arXiv 2511.08579. Zweiger et al., SEAL, arXiv 2506.10943 [nr]. Darwin Gödel Machine, arXiv 2505.22954; Gödel Agent, arXiv 2410.04444 [nr]; STOP, arXiv 2310.02304 [nr].

**B. Interpretability tools and editing operators**
- Attribution patching (Nanda 2023 [nr]); AtP*, arXiv 2403.00745; ACDC, arXiv 2304.14997 [nr]; Syed et al., arXiv 2310.10348 [nr]; EAP-IG, arXiv 2403.17806 [nr]; circuit-tracer (Qwen3 0.6B–14B and Gemma 3 transcoders; not Qwen3.5).
- Qwen-Scope, arXiv 2605.11887 (SAEs for Qwen3.5-2B and larger, not 0.8B; repetition features). Gemma Scope 2 (Gemma 3 270M–27B).
- Against SAEs downstream: Smith et al. (GDM), Mar 2025; AxBench, arXiv 2501.17148 [nr] (difference-of-means beats SAEs).
- Direction edits: Arditi et al., arXiv 2406.11717 [nr]; ThinkEdit, arXiv 2503.22048; Venhoff et al., arXiv 2506.18167.
- Decode-time baselines: NoWait, arXiv 2506.08343; SEAL (steerable reasoning calibration), arXiv 2504.07986.
- Knowledge editing: ROME, MEMIT [nr]; AlphaEdit, arXiv 2410.02355. Merging: Task Arithmetic, TIES, DARE [nr]; Text-to-LoRA, arXiv 2506.06105.
- Agents doing interpretability and auditing: MAIA, arXiv 2404.14394 [nr]; Investigator Agents, arXiv 2502.01236 [nr]; Marks et al., arXiv 2503.10965 [nr]; Bricken et al. (Anthropic) auditing agents, Jul 2025; AuditBench, arXiv 2602.22755 ("tool-to-agent gap").
- Ground-truth testbeds: Tracr [nr]; InterpBench, arXiv 2407.14494; Sleeper Agents [nr]; BEEAR, arXiv 2406.17092; SteerEdit, arXiv 2604.12359; WMDP [nr].

**C. RL for long-horizon agents**
- GiGPO, arXiv 2505.10978 (step-level anchor groups); VinePPO, arXiv 2410.01679 [nr]; MT-GRPO, arXiv 2505.11821; TRACE, arXiv 2607.13988 (secondary summary).
- Yang, He-Yueya, Liang, "RL for ML Engineering Agents", arXiv 2509.01684 (duration-scaled updates, partial credit).
- CANOPY, arXiv 2609.01245 (outcome RL plateaus from "signal starvation"; more same-task rollouts, KL anchoring, loss on own action tokens). SandMLE, arXiv 2604.04872 (small synthetic tasks, >13× faster).
- Variance and sampling: RLOO, Dr. GRPO [nr]; PODS, arXiv 2504.13818; Foster et al., arXiv 2502.12272; SFL [nr]; Miller, arXiv 2411.00640 [nr].
- Spurious Rewards, arXiv 2506.10947 [nr]. Expert iteration vs RL: Havrilla et al., arXiv 2403.04642; RAFT++, arXiv 2504.11343; ReST-EM, STaR [nr]; SWE-Gym, arXiv 2412.21139. Yue et al., arXiv 2504.13837.

**D. Search, archives, research agents**
- AlphaEvolve, FunSearch [nr]; ShinkaEvolve, arXiv 2509.19349; ELM, arXiv 2206.08896; QDAIF, arXiv 2310.13032; OMNI, OMNI-EPIC, POET.
- Exploration: Outcome-based Exploration, arXiv 2509.06941; Pass@k Training, arXiv 2508.10751; TTT-Discover, arXiv 2601.16175.
- Research agents: AIRA, arXiv 2507.02554; AIDE, arXiv 2502.13138; MLGym [nr]; RE-Bench, arXiv 2411.15114; AI Scientist [nr]; AI co-scientist [nr]; AI CUDA Engineer [nr].
- Learning from experience in context: GEPA, arXiv 2507.19457 (beats GRPO with up to 35× fewer rollouts); ACE, arXiv 2510.04618; Training-Free GRPO, arXiv 2510.08191; Memento, arXiv 2508.16153; Voyager, ADAS [nr].

## (b) Ranked experiments (impact per dollar; ~$1.8 per L40S-hour; a 16-session RL step ≈ $60, 2 h)

1. **Skill-free baselines on our harness (~$15, half a day, no editor sessions).** Score on the reward set and MATH-500 (8k cap, paired): decode-time ban on " Wait"/"Wait"; thinking off; " Wait" readout removal on 48 writers; 3 random equal-norm rank-one directions on the same writers (±); a readout row swap. Repeat on the plateau base `cur4g16-k1` with a `</think>`-after-answer ban. If a skill-free arm reaches the editor's mean reward, define reward as gain over the best skill-free arm or move RL to experiment 2.
2. **Planted-fault gym on real LMs with ground truth and short sessions (~2 days to build; ~$25 per RL step).** Faults with exact inverses (sign flips, scaled heads, injected rank-one directions, swapped readouts) and trained-in behaviours (trigger backdoors, counterfactual facts, format breakage, sycophancy) on Qwen3.5-0.8B and on hosts with dictionaries (Qwen3-0.6B, Gemma 3 270M/1B); ~15% no-fault targets. Reward: share of damage repaired on probe prompts (teacher-forced or ≤512-token generations) minus guard damage, plus a small overlap term with the planted ΔW. Sessions 15–20 min, 8 forward passes, 4 per GPU; faults sampled by learnability. Measure on held-out fault families and targets vs untrained editor and random or brute-force baselines at equal passes; transfer to the loop task. Risk: WSP-Gym transferred poorly to unseen fault types (19.9%).
3. **Library as the learner, editor frozen (~$120, 1 day).** Untrained editor; an LLM writes method-level library entries from each group's best and worst sessions, with model-specific numbers filtered out. Library vs empty, paired, on Qwen3.5-2B (MATH-500), NuminaMath and the gym. Measure held-out gain vs experiment-1 baselines and the number of distinct edit kinds.
4. **Turn-level credit from the editor's own paired tests, plus forked sessions (~1 day of code).** Confirm-split tests give r_t = max(0, gain − best so far)/SE for the turns since the previous test, mixed 0.3:0.7 with the terminal reward; fork sessions at the first save_edit into 4 continuations (GiGPO-style anchor groups); loss only on action tokens; down-weight problem-id lists. Risk: retesting until noise gives a gain (count only confirm tests; cap tests per edit).
5. **Expert iteration on the existing archive vs Dr. GRPO (~$60–100, 1 day).** SFT the untrained LoRA on the top-quartile sessions (~250 logged), and a RAFT++/TOPR arm using the bottom quartile as negatives; evaluate on 2B, NuminaMath and the gym. Risk: distils the loop recipe and narrows exploration.
6. **Evolutionary search over edits with an LLM mutation operator and a MAP-Elites archive (~$80–150 per run).** Edit programs as genomes; descriptors edit kind × layer band × tensor type × target token; fitness 64 problems at the 8k cap; novelty rejection; a mandatory random-mutation arm at the same budget.
7. **Best-of-group objective with a novelty bonus over edit kinds (~$60 per step).** Pass@k-style advantages plus a historical-count bonus per LLM-labelled edit kind; only on targets with several real fixes.
8. **Tool library for the editor (~$100).** Forward-only activation and path patching, AtP* attribution, an AlphaEdit null-space projector, and transcoders or SAEs on hosts that have them; ablate one tool at a time on gym targets. Expect patching and difference-of-means to matter more than SAEs (AxBench, AuditBench).

## (c) First

Experiment 1. It costs about $15, needs no editor sessions, and decides whether further RL on the MATH loop task can show editor skill. Build the experiment-2 gym in parallel; it does not depend on the result.
