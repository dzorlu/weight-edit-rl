"""Checks the Tinker pieces the RL loop depends on, on a two-turn toy conversation.

    TINKER_API_KEY=... uvx --python 3.12 --with tinker-cookbook --with openai python scripts/tinker_check.py

1. render a prompt for Inkling 2 Small and sample with log-probabilities;
2. append the reply and a second user turn, render again, and check that the new prompt
   extends (old prompt + sampled tokens), so a whole rollout can be one training sequence;
3. turn the two turns into training data and take one optimizer step;
4. save, and sample from the updated weights (native client and the OpenAI-compatible endpoint).
"""
import os, time, json
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Transition, Trajectory
from tinker_cookbook.rl.data_processing import trajectory_to_data, _flatten_chunks

MODEL = os.environ.get("TINKER_MODEL", "thinkingmachines/Inkling-2-Small:peft:262144")
EFFORT = 0.2
t0 = time.time()
tok = get_tokenizer("thinkingmachines/Inkling-2-Small")
rname = model_info.get_recommended_renderer_name("thinkingmachines/Inkling-2-Small")
renderer = get_renderer(rname, tok)
print("renderer", rname, type(renderer).__name__)

svc = tinker.ServiceClient()
tc = svc.create_lora_training_client(base_model=MODEL, rank=32).result()
print("training client up", round(time.time() - t0), "s")
sampler = tc.save_weights_and_get_sampling_client().result()
params = tinker.SamplingParams(max_tokens=600, stop=renderer.get_stop_sequences())


def turn(messages):
    ob = renderer.build_generation_prompt(messages, effort=EFFORT)
    r = sampler.sample(prompt=ob, num_samples=1, sampling_params=params).result()
    s = r.sequences[0]
    msg, ok = renderer.parse_response(s.tokens)
    return ob, s, msg, ok


messages = [{"role": "user", "content": "End your reply with one fenced block:\n```python\nprint(2+2)\n```"}]
ob1, s1, m1, ok1 = turn(messages)
print("turn 1: prompt", ob1.length, "tokens; sampled", len(s1.tokens), "tokens; logprobs", s1.logprobs is not None, "; stop", s1.stop_reason, "; parsed ok", ok1)
print("  message keys:", list(m1.keys()) if isinstance(m1, dict) else type(m1), "| text:", get_text_content(m1)[:120].replace("\n", " "))
print("  cache-hit field:", getattr(r if False else s1, "prompt_cache_hit_tokens", "n/a"))
messages2 = messages + [m1, {"role": "user", "content": "[result of the python tool]\n4\n\nNow reply with the words FINAL SUBMISSION."}]
ob2, s2, m2, ok2 = turn(messages2)
f1, f2 = _flatten_chunks(ob1.chunks) + list(s1.tokens), _flatten_chunks(ob2.chunks)
is_prefix = f2[:len(f1)] == f1
print("turn 2: prompt", ob2.length, "tokens; sampled", len(s2.tokens), "| new prompt extends old prompt + sampled tokens:", is_prefix)
if not is_prefix:
    k = next((i for i, (a, b) in enumerate(zip(f1, f2)) if a != b), min(len(f1), len(f2)))
    print("  first difference at position", k, "of", len(f1), "| expected:", repr(tok.decode(f1[k:k + 12])), "| got:", repr(tok.decode(f2[k:k + 12])))
    print("  tail of sampled turn 1:", repr(tok.decode(list(s1.tokens)[-12:])))

traj = Trajectory(
    transitions=[Transition(ob=ob1, ac=TokensWithLogprobs(tokens=list(s1.tokens), maybe_logprobs=list(s1.logprobs), sequence_id=getattr(s1, "sequence_id", None)), reward=0.0, episode_done=False),
                 Transition(ob=ob2, ac=TokensWithLogprobs(tokens=list(s2.tokens), maybe_logprobs=list(s2.logprobs), sequence_id=getattr(s2, "sequence_id", None)), reward=1.0, episode_done=True)],
    final_ob=tinker.ModelInput.empty())
data = trajectory_to_data(traj, 1.0, include_provenance=False)
for d in data:
    d.loss_fn_inputs.pop("mask", None)  # importance_sampling takes only target_tokens, logprobs, advantages
print("training sequences from the two turns:", len(data), "| lengths", [d.model_input.length for d in data])
t1 = time.time()
fb = tc.forward_backward(data, loss_fn="importance_sampling")
op = tc.optim_step(tinker.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8))
res = fb.result(); op.result()
print("forward_backward + optim_step ok in", round(time.time() - t1), "s | metrics", {k: round(v, 4) for k, v in list(res.metrics.items())[:4]})
ck = tc.save_weights_for_sampler("smoke").result()
print("sampler checkpoint:", ck.path)
sampler = tc.save_weights_and_get_sampling_client().result()
ob3, s3, m3, ok3 = turn(messages)
print("sampled from updated weights (native):", len(s3.tokens), "tokens")
from openai import OpenAI
c = OpenAI(base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", api_key=os.environ["TINKER_API_KEY"], timeout=300)
r = c.chat.completions.create(model=ck.path, max_tokens=200, reasoning_effort="low", messages=[{"role": "user", "content": "Say hi."}])
print("sampled from the checkpoint via the OpenAI-compatible endpoint:", (r.choices[0].message.content or "")[:80].replace("\n", " "))
print("total", round(time.time() - t0), "s")
