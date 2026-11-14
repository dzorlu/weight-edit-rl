# Notes (session 1)
## Diagnosis (gen_1: original, 256 diag MATH problems x2, 8192 tok)
- acc 9.6%, 92.8% cut at 8192 tokens, 79% no boxed. When </think> emitted (7.6%), 95% correct.
- 254/475 truncated texts contain gold answer string; gold first appears median 7% into text -> model finds answer but never stops.
- Truncated texts: ~116 'Wait' per text; loops of re-verification. Almost no genuine wrap-up statements.
- TF (fwd_3, fwd_4): P(</think>) at line starts ~0 except at wrap-up lines (0.04-0.27). </think> readout built across L14-22 (biggest: L22 linear attn).
## Edits tried
- E_w(0.5): rank-1 gain along </think>-specific readout dir w (whitened, lam=0.1) on attn-out+down_proj L14-22. gen_5 (sel[:128]x2, both): orig 7.42% vs edit 7.03%, trunc 94%->93%. NO EFFECT.

- KEY FINDING: L16 MLP neuron 998 is a pure 'wait' token neuron (down_proj col promotes wait/Wait/ WAIT...), fires strongly NEGATIVE right after a result is computed ('$14 = r$', '= -40$', '\boxed{14}$.'), contributing +2-3 logits to ' Wait' at Wait positions (largest single neuron).
- E_998(0): zero L16 down_proj col 998. gen_6 (sel[:128]x2, both): orig 7.03% -> edit 14.45%, diff +7.42 +- 2.08 (22 w->r, 3 r->w). trunc 94.9->87.5%, Wait/1k tok 14.8 -> 2.6. Saved copy in /work/best_1.
- n998 dose: beta=-1 (flip col) gen_7 (sel[:128]x2 edited only): 17.6% (trunc 83.6%, Wait/1k 0.14). Monotone: orig 7.0, beta0 14.5, beta-1 17.6.
- Wait circuit: n998 output dir is read by other Wait neurons L17 n1915, L18 n881, L20 n405, L22 n573 (gate alignment 0.2-0.44).
- beta=-1 + E_w(alpha=1.0, L14-22) gen_8: 25.4% (trunc 75%). E_w helps once Wait loops are removed (wrap-up lines in truncated texts rose 0.5->2.0 per text with beta=-1).
- beta=-1 + alpha=2.0 gen_9: 37.1% (trunc 58.6%). Strong alpha dose-response (0:17.6, 1:25.4, 2:37.1).
- full-set alpha=2 causes thinking SKIP (25% of closures <100 tokens; </think> at token 0) -> reject. Start-of-thinking position has w-readout similar to ordinary positions but higher base logit.
- Selective set S+ = attn L17,18,19,20,21 + MLP L15,16,19,22 (chosen from fwd_3/4: hi(wrap-up) vs lo and vs think-start). S+ alpha=2.5,beta=-1 gen_10: 32.8% (trunc 66.8%), only 1 closure <200 tok. vs orig6 +25.8+-3.4.
- S+ alpha=3.5 gen_11: 35.9% but 22 closures <200 tok (skipping) -> prefer alpha=2.5.
- CANDIDATE A = S+ alpha=2.5 + n998 beta=-1, saved /work/best_2. Confirming on conf split (held out).
- Remaining failure: doubt via other phrasings ('Is there any reason to doubt', 'Let's check', 'What if'), verbatim repetition truncs dropped 132->40.
- CONFIRM 1 (gen_12, conf[:128]x2, both): orig 6.64% -> A 35.94%, diff +29.3 +- 3.4 (75 w->r, 0 r->w).
- CONFIRM 2 (gen_13, conf[128:256]x2, both): orig 8.2% -> A 34.4%, diff +26.2 +- 3.55 (68 w->r, 1 r->w). Candidate A CONFIRMED (512 paired, ~+27.7pp).
- beta=-2 (S+ 2.5) gen_15 sel: 34.8% vs beta=-1 32.8% (paired diff small, see below). L18 n1261 = generic 'let' neuron driving 'Let me double-check' at doubt points (not doubt-specific; not edited).
- Ablation: S+ alpha=2.5 alone (n998 intact) gen_16 sel: 15.6% (trunc 84%). n998 flip alone 17.6%. Both 32.8% -> synergistic (Wait loops block reaching wrap-ups; weak </think> blocks exit).
- Early closures (<200 think tok) with A on conf: 13/512 (after 'Analyze the Request' restatement lines), 6/13 correct.
- V1 (S+ alpha=3.0, beta=-2) gen_17 sel: 35.9% vs A 32.8%.
- CONFIRM 3 (gen_18, conf2[:128]x2, both): orig 5.47% -> A 33.59%, diff +28.1 +- 3.4 (73 w->r, 1 r->w).
- FINAL: candidate A (identical to /work/best_2) in /work/edited: L16 down_proj col 998 negated; rank-1 gain (1+2.5 w w^T) on writers attn L17-21 (L19 o_proj, others linear_attn.out_proj) + mlp down_proj L15,16,19,22 along whitened </think>-specific readout direction w.
