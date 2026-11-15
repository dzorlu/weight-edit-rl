"""What the editor is told: the task prompt, and the section on what counts as an edit and how to find one.

Every statement about the target's architecture is filled in from its config and tensor names (harness.target.target_arch):
make_prompt(arch), make_guide(arch), model_sentence(arch), tensor_names_text(arch), layer_ranges(arch).
PROMPT and GUIDE are the texts for Qwen3.5-0.8B, used by the standalone editors (editors/); the RL loop builds its own per target."""

PROMPT_TEMPLATE = r"""TASK. You are given the weights of a language model, <<MODEL>>. Your job is to improve its capabilities by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails at the target tasks and repair that with targeted changes.

HOW IT WILL BE MEASURED. The 500 problems of MATH-500, 2 samples per problem, separately for original and edited weights. You never see this test set. Metric: accuracy of the final \boxed{} answer. Prompt protocol: the problem text followed by "Put your final answer in \boxed{}." as a single user message. The chat template has thinking enabled. Sampling is temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 and at most 8192 new tokens. Config, tokenizer and decoding are fixed; only weight values can change. This is a new independent experiment with no previous edits, notes, trajectories or published editing recipe.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it fails at the target tasks, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Any specific, evidence-supported failure mechanism is in scope. The objective is improved correctness on unseen problems. Possible failure categories include incorrect knowledge retrieval, calculation or logical inference, misunderstanding constraints, answer selection or expression, and failure to complete a useful response. These are examples, not a required diagnosis or an exhaustive list. Out of scope, and refused by save_edit: the final norm, the input embedding / output layer, the vision tower, and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for, and neither is making the model skip its thinking or close it early regardless of the problem.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. The model's files are read-only in /model, practice problems are in /practice (read /practice/README.txt), and /work is your read-write folder. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx, safetensors, tokenizers. Defined in advance (do not reassign these names): NAMES (all tensor names), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (a tokenizers.Tokenizer: tok.encode(text).ids, tok.decode(ids), tok.id_to_token(i)). Gradients are disabled. To run code, call the `python` tool with the parameter `code`; if the last line is an expression, its value is shown. One execution may take up to 900 seconds; output is cut at 10,000 characters. You may run code up to 300 times. If the kernel is restarted, its variables are lost and you are told so.
2. Practice problems: 3,000 from the MATH training set and 1,500 from the GSM8K training set, each with a reference solution. They are disjoint from the held-out test after normalized text matching. You may partition them for your own confirmation.
3. A forward-pass tool with a budget of 50 forward passes in total. A forward pass is one of two things, and each counts as one pass:
 (a) GENERATION: the model answers up to 512 generations (problems x samples) with the evaluation's protocol. Call the `forward` tool with the parameter `request` set to JSON of this form:
 {"type": "generate", "model": "original", "edited" or "both", "problems": ["<practice id>", ...], "prompts": ["<your own user message>", ...] (optional, used verbatim), "n": samples per item (items x n <= 512; with "both" <= 256 per model), "max_tokens": at most 8192}
 "original" runs the unedited weights, "edited" your current edits (the tensors saved with save_edit at that moment), and "both" runs both on the same items and returns a paired comparison (the accuracy difference edited - original with its standard error, and how many samples went from wrong to right and back). The sampling seed of every item is fixed by the problem and sample index, in every pass and for both models. Every answer to a practice problem is graded. You get the accuracy (overall and by source), the mean length in tokens, the share cut at max_tokens and the share with no boxed answer; texts, extracted answers and grades are in /work/gen_<k>.jsonl (with "both": /work/gen_<k>_original.jsonl and /work/gen_<k>_edited.jsonl), one JSON per line with qid, sample, source, prompt, gold, text, extracted, correct, tokens, finish. Activations are not saved during generation; to get them, send the generated texts through a teacher-forced pass.
 (b) TEACHER-FORCED: the model reads up to 256 texts of up to 8,192 tokens each, at most 262,144 tokens in total (no generation). Call the `forward` tool with `request` set to:
 {"type": "teacher_forced", "model": "original" or "edited", "texts": ["raw text", ...] or "chats": [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...] (add "enable_thinking": true for the thinking template), "return": {"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers <<HS>>], "mlp_act": [layers <<MLP>>]}}
 The results are .npy files (with meta.json) in /work/fwd_<k>/, right-padded to the longest text (at most 3 GiB of activations per pass; attention maps are not available). A request that fails does not count against the budget.

HOW TO WORK.
1. Diagnose first. Compare correct and incorrect practice responses, including completed wrong answers, missing answers and token-limit truncations. Separate observable symptoms from proposed causes. Consider competing explanations before committing the remaining budget. Choose what to investigate from this run's own evidence.
2. Localize causally. That a component is more active on correct answers is not enough. Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Accuracy over n generations has a standard error of about 50/sqrt(n) percentage points. Compare original and edited weights on the SAME problems with "model": "both". Keep an edit only if its gain is clearly larger than its standard error on problems that were not used to choose it. A lower share of cut-off answers does not by itself mean more correct answers: judge edits by accuracy.
4. Plan your forward passes: for example one diagnostic generation, teacher-forced passes for dense measurements, and final paired generations on problems you have not used.
5. Work efficiently: state the hypothesis of each python call in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays; print short excerpts of generated texts (at most about 300 characters each); outputs are shown in ASCII only. Keep /work/notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit); it is shown to you again if the conversation is restarted. Keep copies of good edits (for example copy /work/edited to /work/best_1/).

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) internet access. Allowed: any analysis of the weights and of forward-pass results, any use of the practice problems, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. Each of your replies contains one tool call (python or forward), until you finish. You are told the remaining budget after every result. You may finish at any time; you do not have to use the whole budget, and unused forward passes cost nothing. When all 50 forward passes are used or the time limit is reached, you must finish. To finish, reply without a tool call, starting your reply with the words FINAL SUBMISSION, and say what you changed, the evidence for it and why you expect it to help. The tensors in /work/edited at that moment are submitted. Submit an edit only if a paired comparison on at least 300 generations that were not used to choose it shows a gain larger than 2 standard errors. Otherwise call drop_edit on every edited tensor and submit the original weights; that is a valid result and better than submitting an edit the evidence does not support."""


GUIDE_TEMPLATE = """WHAT COUNTS, AND HOW TO GET THERE.

Goal. Improve correctness on unseen problems by changing weights you can justify: find what goes wrong when the model fails, find the components responsible (specific units, heads or directions), and change them. Any specific failure mechanism that your evidence supports is in scope. Possible kinds of failure include wrong knowledge retrieval, wrong calculation or inference, misread constraints, a wrong or badly expressed final answer, and never completing a useful response. These are examples, not a required diagnosis.

Out of scope, and refused by save_edit: the final norm, the input embedding / output layer, the vision tower, and any edit that only multiplies a whole tensor by a constant. Also not wanted: making the whole output distribution sharper or flatter, and making the model skip its thinking or close it early regardless of the problem.

The steps below are established methods from mechanistic interpretability and model editing. They are procedures only. Which tokens, layers and units matter must come from your own measurements in this run.

A. Diagnose from behaviour first.
1. Open with one large generation on the original weights (256 to 512 generations in one forward pass). Count how many are cut at the token limit, and find which words or phrases are far more frequent in the failed texts than in the completed ones.
2. Before any analysis, split the practice problems into fixed disjoint sets and save them (for example /work/splits.json): one for diagnosis, one for choosing between edits, and one or two confirmation sets never used for any choice.

B. Locate the component.
3. Direct logit attribution (Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits"; the logit lens, nostalgebraist 2020). To find what produces a next token T, send the model's own failed generations through a teacher-forced pass that returns mlp_act and hidden_states for many layers, and look at the positions right before T. The final score of T is dot(r(T), h_final) / rms(h_final), where <<READOUT>> and gamma the final norm weight (check the norm formula against the config). The MLP of layer L adds down_proj_L @ act_L to the residual stream, so unit j of layer L contributes act_L[j] * dot(r(T), down_proj_L[:, j]) / rms(h_final). Compute this for every unit of every layer, rank, and repeat at several positions. Attention outputs can be split per layer from the hidden states. This measures only the direct effect on the output; a unit can also act through later layers.
4. When does it fire. For the top units, look at their activation over all positions to learn what they respond to.
5. Difference of means (Turner et al. 2023, "Activation Addition"; Rimsky et al. 2024, "Steering Llama 2 via Contrastive Activation Addition"). Average the activations at positions that show the behaviour and subtract the average at positions that do not (for example late in a loop against the first check, or failed against completed answers). The result is a direction, or a ranking of units, for that behaviour. Use many positions from many texts; two texts are not enough.
6. A wanted token that is too weak (for example the token that ends the thinking section) is handled with the same attribution as in 3: compare, per component, how much is written along its readout direction where it should come next against ordinary positions. A readout direction specific to one token can be made by removing what it shares with the frequent tokens.

C. Write the edit. These forms all change a matrix that writes into the residual stream (MLP down_proj, attention output projection).
7. Ablation: set one down_proj column or one attention head's output block to zero.
8. Column scaling: multiply one down_proj column by a factor (try 0, -1, -2).
9. Rank-one edit (Meng et al. 2022, "Locating and Editing Factual Associations in GPT"): W + outer(a, b), where b is a pattern of input activations that triggers the edit and a is what is then written. A single-column change is the case where b is one-hot.
10. Removing or amplifying one output direction d (Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction", which calls removal weight orthogonalization): W + alpha * outer(d, d @ W) for a unit vector d on a chosen set of writer matrices. alpha = -1 removes everything those matrices write along d; alpha > 0 amplifies it. Sweep alpha.
11. Deleting single weights (Lu et al. 2026, "The Surprising Effectiveness of Deleting Weights in LLM Post-Training") zeroes individual entries of the MLP gate projections. There the entries are chosen by gradient training, which is not allowed here; without it, expect little.

D. Test causally, then confirm.
12. Every candidate is tested with a paired generation ("model": "both") on at least 256 generations. Sweep the strength for a dose-response. For a combined edit, also measure each part alone. Useful controls: the same edit with the direction reversed, and a random direction of the same size.
13. Read the edited model's texts for side effects (for example thinking that closes after very few tokens).
14. Confirm the final candidate on the untouched confirmation sets, keep /work/notes.md current, and copy each good edit to its own folder.
15. Forward passes are scarce: use few of them with large batches, and never spend one on fewer than 64 generations."""


ORDINAL = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 8: "eighth"}


def layer_ranges(a):
    """(hidden_states range, mlp_act range) as the forward tool accepts them: 0 = embeddings .. L = after the last layer."""
    return f"0..{a['layers']}", f"0..{a['layers'] - 1}"


def model_sentence(a):
    """The model's name and the architecture facts the task statement gives."""
    full = ", ".join(map(str, a["full_attn"]))
    head = f"{a['name']} (a {a['layers']}-layer text transformer, hidden size {a['hidden']}, MLP intermediate size {a['mlp']}; "
    attn = f"full attention with {a['heads']} heads of width {a['head_dim']}"
    if not a["linear_attn"]:
        return head + f"every layer uses {attn})"
    if a["full_every"] in ORDINAL:
        where = f"every {ORDINAL[a['full_every']]} layer, i.e. layers {full} counting from 0, uses"
    else:
        where = f"layers {full} (counting from 0) use" if len(a["full_attn"]) > 1 else f"layer {full} (counting from 0) uses"
    return head + f"{where} {attn}, the other layers use linear attention)"


def tensor_names_text(a):
    """The sandbox note on tensor names."""
    p, L = a["prefix"], a["layers"]
    mlp = ",".join(a["mlp_proj"]); proj = ",".join(a["attn_proj"])
    t = f"{p}.layers.<L>.mlp.{{{mlp}}}.weight for L = 0..{L - 1}. "
    if not a["linear_attn"]:
        return t + f"Every layer has full attention (\u2026layers.<L>.{a['attn_mod']}.{{{proj}}}.weight)."
    full = ", ".join(map(str, a["full_attn"]))
    return t + (f"Layers {full} have full attention (\u2026layers.<L>.{a['attn_mod']}.{{{proj}}}.weight); "
                f"the other layers have linear attention (\u2026layers.<L>.{a['linear_mod']}.\u2026, see NAMES).")


def make_prompt(a):
    hs, mlp = layer_ranges(a)
    return PROMPT_TEMPLATE.replace("<<MODEL>>", model_sentence(a)).replace("<<HS>>", hs).replace("<<MLP>>", mlp)


def make_guide(a):
    if a["tied"]:
        readout = "r(T) = E[T] * (1 + gamma), E is the embedding matrix"  # tied: the output layer is the input embedding
    else:
        readout = "r(T) = U[T] * (1 + gamma), U is the output layer (lm_head; it is separate from the input embedding)"
    return GUIDE_TEMPLATE.replace("<<READOUT>>", readout)


# Qwen3.5-0.8B as target_arch reads it (the standalone editors run without the weights volume at import time).
QWEN35_08B = {"name": "Qwen3.5-0.8B", "layers": 24, "hidden": 1024, "mlp": 3584, "full_attn": [3, 7, 11, 15, 19, 23],
              "linear_attn": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22], "full_every": 4, "heads": 8, "head_dim": 256,
              "tied": True, "prefix": "model.language_model", "mlp_proj": ["gate_proj", "up_proj", "down_proj"], "attn_mod": "self_attn",
              "attn_proj": ["q_proj", "k_proj", "v_proj", "o_proj"], "linear_mod": "linear_attn"}
PROMPT = make_prompt(QWEN35_08B)
GUIDE = make_guide(QWEN35_08B)
