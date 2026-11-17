# Decode-time ban baselines (snapshot 2026-11-16)

Question: do the editor's weight edits do more than banning "Wait" at decoding? All runs: MATH-500, 500 problems × 2 samples, 8,192-token cap, the same seeds and sampling as every held-out score in this repo (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, thinking on), paired by problem and sample. A ban sets the token's logit bias to −100 at every generated position, thinking included; no weight changes. Code: `harness/target.py::ban_eval`, `rl/loop.py::ban`. Raw summaries: `results/eval/ban/`.

Token ids (Qwen3.5-0.8B tokenizer): " Wait" 13428, "Wait" 13784, " Hmm" 85152, "Hmm" 77264, " Actually" 32645. "But" and "Okay" were not banned: they occur in ordinary math text.

| Condition | MATH-500 | Cut at 8,192 tokens | Mean tokens | vs Qwen as downloaded (paired) | Right → wrong |
|---|---|---|---|---|---|
| Qwen3.5-0.8B as downloaded | 6.6% | ~93% | – | – | – |
| + ban " Wait"/"Wait" | 22.9% | 77.8% | 7,367 | +16.3 ± 1.4 | 3 |
| + ban " Wait"/"Wait"/" Hmm"/"Hmm"/" Actually" | 23.1% | 77.5% | 7,351 | +16.5 ± 1.4 | 3 |
| Step-8 editor's best edit (`cur4g16-s8best`), no ban | 39.5% | 34.0% | 4,041 | – | – |
| `cur4g16-s8best` + ban " Wait"/"Wait" | 40.9% | 29.4% | 3,679 | +34.3 ± 1.8 | 6 |
| Best stacked child (`rebase2-k2`, `rebase1-k3`), no ban | 42.5% | – | – | – | – |

`cur4g16-s8best` + ban against `cur4g16-s8best` alone: +1.4 ± 1.4 points (100 wrong→right, 86 right→wrong).

Reading:
- At our 8,192-token cap the ban alone reaches 22.9%; 78% of answers still run to the cap. The model keeps re-checking with other words. WSP-Gym's 38.8% for the same ban used a 32,768-token cap and one sample per problem, so the two numbers are not comparable.
- The editor's edit (39.5%) beats the ban by about 17 points at this cap, and it shortens answers far more (34% cut vs 78%). It is not just the ban written into the weights.
- Adding the ban on top of the edit adds +1.4 ± 1.4 points: the edit already removes almost all of what the ban removes.
- Widening the ban to "Hmm" and "Actually" changes nothing (23.1% vs 22.9%).
