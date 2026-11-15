# Two rollouts per L40S: change and test (14 November 2026, 23:05)

## What changed

- `rl.loop::rl --rollouts-per-gpu 2` (default 1, the old one-GPU-per-rollout `Target`). Rollouts i and i+1 of a step share one `PackedTarget` container (`gid = <exp>-s<step>-g<i//2>`); every method takes the rollout id first.
- Per rollout (`_Slot`): its own `/work` and `/library` (real folders `/slots/<n>/…`), its own python kernel, its own edited vLLM server (port 8002+n, 0.30 of the GPU), its own boot id and request counter. The kernel runs in its own user, mount and pid namespace (`unshare`): it sees its folders at `/work` and `/library`, `/slots` is empty, it cannot see or kill other processes, and it has a 36 GiB address-space limit (`MemoryError` instead of taking the container down).
- Shared: the original vLLM server (port 8001, 0.15 of the GPU), started only when a request needs answers that are not cached; the teacher-forced transformers model, capped at 0.13 of the GPU (`set_per_process_memory_fraction`), so it gets a CUDA OOM before any vLLM server does. One GPU lock serializes vLLM start-ups and teacher-forced passes (weights swap, hooks, forward, restore).
- Teacher-forced passes (both modes): the output head runs on 1,024 positions at a time instead of full-vocabulary log-probs for the whole batch (that allocation was 11 GB and failed even with one rollout per GPU); a CUDA OOM returns "the teacher-forced pass ran out of GPU memory (it does not count) …" and the weights are restored. Packed batches are 8,192 tokens.
- `PackedTarget(mps=1)` starts CUDA MPS; the daemon starts on Modal but this mode is untested and the loop does not use it.
- Outputs (`/work/gen_k*.jsonl`, `/work/fwd_k/`, tool result JSON, reward dict) have the same format and paths as before.
- Test driver: `scripts/packing_test.py` (`::packtest`: same steps on `Target` then on a pair; `::packstress`: worst case).

## Results

Normal sequence (16 uncached practice problems, 8192 cap; old teacher-forced code):

| step | 1 per GPU | 2 per GPU (A / B, concurrent) |
|---|---|---|
| paired generation, 16 problems (includes server start-ups) | 260 s | 717 / 718 s |
| edited generation after save_edit (restart) | 103 s | 106 / 219 s |
| reward, 128 problems × 2 | 407 s | 853 / 812 s |

- Isolation held: each kernel saw only its own `/work` marker, `/slots` empty, 1 visible pid, memory limit enforced; reward `edited_tensors` were each rollout's own tensor (layer 23 vs 20); boot ids unchanged; snapshots per rollout.
- The teacher-forced pass of that run failed with CUDA OOM in all three cases, including one rollout per GPU (old code); fixed as above.

Worst case (new code, 0.30 / 0.30 / 0.15 + 0.13):

- 7,781-token text, `hidden_states` 0–24 and `mlp_act` 0–23, `topk` 20, while the other rollout was starting servers and generating: passed.
- A: "both" 256 answers per model (uncached, 4096 cap) with the shared original server and A's edited server busy; meanwhile B generated 256 edited answers, saved a new edit and restarted its edited server during A's generation, then started its own "both" 256: no errors, no OOM in any vLLM log. A's 512 answers took 1,153 s.
- GPU memory plateaued at 41.8 of 46.1 GB with 100% utilization (first run at 0.32/0.32/0.16 peaked at 45.2 GB).
- Simultaneous reward scoring of both rollouts was not reached: the run was stopped at 23:02 once memory had plateaued. Rewards use the same preallocated servers; the first run scored both rewards concurrently without error.

## Cost and speed

When both rollouts generate, each runs about 2× slower (the GPU time-slices between the vLLM processes; total throughput is about that of one rollout per GPU). The saving comes from the time a rollout leaves the GPU idle (~30%). Rollouts will hit `rollout_minutes` sooner; consider a higher cap.

## Launch

    uvx modal run --detach -m rl.loop::rl --exp <name> --steps <n> --group-size 8 --forward-budget 32 --rollouts-per-gpu 2
