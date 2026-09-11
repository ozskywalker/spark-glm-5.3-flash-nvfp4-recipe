# Experiment 01 Results: GLM-5.3-Flash-TrellisMX-MXFP8

Status: **PAUSED (2026-09-11), not blocked**. See `HYPOTHESIS.md` for the full
background and the evidence that already closed the question of whether the
TrellisMX routed-expert overlay itself can run on this fleet (it cannot --
confirmed from the checkpoint's own shipped source, not inference). Track A
(the carrier boot) is paused after 11 boot attempts on an earlyoom OOM that
has resisted three targeted fixes in a row (TileLang cache persistence + gmu
0.84, gmu 0.84 alone, max_num_batched_tokens 512) -- see attempts 9-11 below.
Two untested, more invasive levers remain (128K max_model_len floor, or a
further gmu cut) -- user chose to stop spending boot cycles for today rather
than try either blind. No containers left running; both hosts clean.

## Track A: carrier-only fit & performance

Recipe: `recipes/glm-5.3-flash-nvfp4-experiment01-trellismx-carrier-vllm.yaml`

- [ ] Carrier download (`local-inference-lab/GLM-5.3-Flash-NVFP4` @
      `520de24eabf507659eaef7c70f14fd584527facc`, ~199 GB) -- in progress
- [ ] Replicate to both nodes (this fleet's `/models` is per-node local disk,
      not shared -- confirmed via `df -h` on each host)
- [x] Boot, confirm real `snapshots/<hash>` path matches the recipe -- matched
      (`config.json`: `model_type=glm5_next`, `quant_method=modelopt`)
- [x] **Real finding, not yet a clean boot**: two boot-time bugs hit and
      fixed before any successful boot -- (1) `--moe-backend marlin`
      (inherited from the production template) crashes on this checkpoint's
      unquantized MTP MoE layer, fixed by dropping the flag; (2)
      `--load-format instanttensor` (also inherited) hit earlyoom TWICE in a
      row, ~85-90% through load each time, confirmed via live `free -h` to
      be real RSS exhaustion (121Gi/121Gi used) from InstantTensor staging
      close to the full ~184GB-per-rank checkpoint in host memory on a node
      with only 121GB total RAM -- not the page-cache/fragmentation problem
      `prelaunch_flush.sh` targets (confirmed by running a fresh flush
      immediately before each attempt; both still hit the wall identically).
      This is itself a real fit finding for this checkpoint's size relative
      to production's. Switched to vLLM's default loader as the fix; full
      detail in the recipe file's header comment. (3) With the default
      loader, checkpoint load succeeded clean (44/44 shards, ~12 min, memory
      stayed >100GiB available throughout -- confirms (2)'s diagnosis was
      right) but MTP/speculative-decode weight loading then crashed:
      `KeyError: 'model.layers.45.mtp_block.mlp.experts.routed_experts.
      w2_weight_scale'`. Root cause, well-evidenced not guessed: TrellisMX's
      own source explicitly special-cases layer 45 (the MTP block) to defer
      to the carrier's own quant handling, and their runtime overlay ships a
      95KB patched `modelopt.py` we don't have -- stock vLLM here builds
      that layer as unquantized (consistent with finding #1 above) but the
      checkpoint's weight names for it still assume a quantized target.
      Pragmatic fix: dropped `--speculative-config` entirely to get a
      working baseline. **This means decode throughput from this recipe is
      not MTP-accelerated and not directly comparable to the production
      MTP-4 baseline** -- flagged wherever compared. (4) After dropping
      speculative decoding, checkpoint load itself started failing
      *inconsistently* on the head node (127.0.0.1/spark-276f specifically
      -- the remote node 10.7.0.87 has stayed healthy every single time)
      under identical recipe/flush conditions: one clean success (fully
      loaded, >100GiB available throughout), then two earlyoom kills in a
      row on retries with a fresh flush immediately before each, one
      escalating to SIGKILL with swap fully exhausted. No leftover process
      or container found to explain the difference (`ps`/`docker ps -a`
      checked clean between attempts). Not yet root-caused -- current best
      guess is contention from this same node also carrying this whole
      session's other background load (the 186GB HF download earlier, ~527GB
      of local Docker images/build cache from other work this session,
      ongoing git/file activity), not something wrong with the recipe
      itself, since one attempt under identical settings did succeed
      cleanly. Tried the leading hypothesis (this session's own Docker/
      scratch footprint on that node): removed a ~25GB probe image, pruned
      ~21GB of build cache, removed ~1GB of root-owned scratch files --
      ~46GB reclaimed, node verified clean (`ps`/`docker ps -a` empty,
      112GiB free immediately before retry). **Attempt 8 hit the identical
      earlyoom kill anyway.** That hypothesis is now ruled out, not just
      unconfirmed. 4 of the last 5 attempts on this exact node have now
      failed this way; the other node has been clean every single time.
      **Paused again to report to the user rather than keep guessing** -- 8
      boot attempts so far, 3 distinct real recipe bugs found and fixed (all
      confirmed real, independent of this memory issue), but the remaining
      blocker needs either information this session doesn't have (cgroup/
      reservation config on this node?) or a different diagnostic approach,
      not another blind retry-with-a-fix-guess cycle.
- [x] **Attempt 9, 2026-09-10, post-reboot**: both physical hosts were
      rebooted (spark-276f 5min uptime, spark-2dd4 3min uptime) before this
      attempt -- `prelaunch_flush.sh` fragmentation check came back exceptionally
      clean on both (~30M contiguous free blocks in zone Normal, vs. the
      2,000 warning threshold and the ~49K/147K previously seen as "healthy"),
      confirming the reboot did clear whatever host-level state attempts 5-8
      were fighting. **This attempt failed differently, and the failure
      reframes the earlier "flaky head node" theory.** sparkrun's own greedy
      scheduler assigned Head this time to 10.7.0.87/spark-2dd4 -- the node
      that had been clean in every one of attempts 1-8 -- while 127.0.0.1/
      spark-276f (the node previously suspected of being the bad one) was
      Worker. Both loaded weights cleanly this run: worker (127.0.0.1) logged
      "Loading weights took 201.09 seconds" / "Model loading took 89.1 GiB
      and 218.855292 seconds" at 21:52:44 UTC, no memory pressure during
      load. The crash came **~10 minutes later, in a different phase**:
      post-load KV-cache/encoder setup, specifically while TileLang
      JIT-compiles this image's own fused-norm kernels
      (`mhc_pre_big_fuse_with_norm_tilelang`, `mhc_post_tilelang` --
      completed compiling at 22:02:10 UTC). At 18:02:17 local (=22:02:17
      UTC) -- 7 seconds after that JIT compile finished --
      `journalctl -u earlyoom` on **spark-2dd4 (10.7.0.87, the head this
      run)** shows `sending SIGTERM to process ... "VLLM::Worker_TP":
      badness 996, VmRSS 5494 MiB`, a failed kill retry ("Timer expired"),
      then `process exited after 0.0 seconds` at 22:02:27 UTC -- earlyoom
      killed the head's own local TP worker process. 5 seconds earlier
      (22:02:22 UTC), the *other* rank's container (127.0.0.1/node_1)
      independently logged `Worker proc VllmWorker-1 died unexpectedly
      (exit code: None)` -- consistent with a cascade: killing one rank's
      worker breaks the TP process group, and the peer rank's own executor
      then dies too. The head's `EngineCore` finished unwinding with
      `RuntimeError: cancelled` out of `shm_broadcast.acquire_read`, i.e.
      the same IPC layer named in `nccl_shm_stall_mtp_underflow_lead`, but
      here as a *downstream symptom* of the earlyoom kill, not an
      independent stall. **Bottom line: it wasn't the same node both times
      (attempts 1-8 kept failing on 127.0.0.1/spark-276f specifically;
      this time the failure followed the Head *role* onto whichever node
      sparkrun assigned it to) and it wasn't the same phase (checkpoint
      load, previously; post-load KV-cache/encoder setup + TileLang JIT
      this time).** The "one physical host is flaky" theory from attempts
      1-8 is no longer well-supported as a complete explanation -- open
      hypothesis now: something about the **Head role specifically**
      (its distinct init responsibilities, or the local TileLang JIT
      compile step, or both) has a memory-pressure profile that's more
      OOM-prone than the same work on a Worker, independent of which
      physical box carries that role. Not yet tested: whether a Worker-role
      TileLang JIT compile shows the same mem-avail oscillation (12-14.6%,
      seen for ~10 minutes straight on the head before the kill, per host
      `journalctl` polling) without tipping into a kill. Both containers
      were left in a zombied `Up`-but-dead-engine state after the crash
      (docker still reported them running; `/health` was unreachable) --
      cleaned up with `sparkrun stop`. 9 attempts total now; still paused
      to report rather than guess a 10th time blind.
- [x] **Attempt 10, 2026-09-10, two fixes applied first**: traced the
      recipe's provenance and found it was adapted from the stale
      `recipes/glm-5.3-flash-nvfp4-vllm.yaml` template rather than the
      maintained exl3-v19 lineage, so it never got two fixes already
      validated fleet-wide on the *same image* (`radixark/vllm-glm53-flash:
      sm121-v9`): (1) `TILELANG_CACHE_DIR: /cache/runtime/tilelang` --
      without it the cache falls back to the container-ephemeral `/tmp/
      .tilelang/cache`, and attempt 9's kill landed 7s after this image's
      `mhc_pre_big_fuse_with_norm_tilelang` kernel finished its first-ever
      JIT compile, the same kernel already implicated in a prior production
      earlyoom crash; (2) `gpu_memory_utilization` 0.85 -> 0.84, the
      fleet-wide value shipped after the kernel-1032 earlyoom regression
      (`gmu_086_earlyoom_regression`), never ported to this template. Both
      applied to the recipe, fresh `prelaunch_flush.sh` re-run (still
      exceptionally clean, ~25M contiguous free blocks both hosts), retried.
      **Crashed again, same signature, at both nodes near-simultaneously
      this time.** Load itself was asymmetric and slow: worker (127.0.0.1)
      finished loading weights in 202.6s (matching attempt 9's pace,
      confirming the load-phase fix from attempts 1-4 still holds), but
      head (10.7.0.87) took far longer -- only 68% through 44 shards at the
      10-minute mark, meaningfully slower per-shard than the worker despite
      identical checkpoint/settings, most likely page-cache/IO contention
      from running that much closer to the memory ceiling while also
      carrying the APIServer + orchestration role. The worker then sat idle
      ~9 minutes (22:27:33 to 22:36:44 UTC) waiting on the still-loading
      head before both proceeded into KV-cache setup together. **Crucially,
      this time the kill fired BEFORE TileLang JIT compile even started** --
      `journalctl -u earlyoom` shows near-simultaneous `sending SIGTERM to
      process ... "VLLM::Worker_TP"` on **both** spark-276f (127.0.0.1) and
      spark-2dd4 (10.7.0.87) within a 10-second window (18:37:12-18:37:22
      local), and on spark-276f a second kill even hit the parent `vllm`
      process itself (badness 971). The container-log traceback pins the
      exact call site: `EngineCore._initialize_kv_caches ->
      determine_available_memory() -> collective_rpc(...) ->
      shm_broadcast.dequeue -> RuntimeError: cancelled` -- vLLM's own
      dummy-batch memory-profiling forward pass (which measures activation
      memory by intentionally running close to the GPU/unified-memory
      ceiling) is what tips this into the earlyoom kill zone, independent
      of the TileLang kernel entirely this time. **Diagnosis refined, not
      yet fixed**: the two applied fixes were real, well-evidenced, and
      worth keeping (load-phase memory is unaffected either way), but
      neither targets the actual failing step. The proximate cause is
      specifically vLLM's KV-cache memory-profiling dummy run on this
      checkpoint's much larger 89.1 GiB resident footprint (vs.
      production's smaller EXL3 checkpoint) inside the same 121 GiB
      unified-memory budget -- (1-gmu)*total only buys ~19.4 GiB of
      headroom at 0.84, and that's not enough margin for this specific
      checkpoint's profiling-step peak at this max_model_len/
      max_num_batched_tokens. Untested levers, in order of how directly
      they target the failing step: (a) drop `max_num_batched_tokens`
      (2048 currently) to shrink the profiling dummy-batch's own footprint;
      (b) test at the 128K context floor instead of the 262144 ceiling
      first -- a smaller `max_model_len` shrinks the KV-cache sizing math
      the profiling step has to do; (c) a further `gpu_memory_utilization`
      cut below 0.84, accepting this checkpoint may just need a lower value
      than production's smaller model did. 10 attempts total. Containers
      cleaned up (`sparkrun stop`). Paused again to report rather than
      guess an 11th time.
- [x] **Attempt 11, 2026-09-10/11**: dropped `max_num_batched_tokens` 2048
      -> 512 (4x) to shrink the KV-cache-profiling dummy batch, the most
      directly-targeted of the three untested levers from attempt 10.
      Fresh flush, retried. **Crashed again, same signature**: both nodes'
      `VLLM::Worker_TP` earlyoom-killed within a 2-second window of each
      other (20:18:32-20:18:34 local), same `determine_available_memory ->
      shm_broadcast dequeue -> RuntimeError: cancelled` traceback on the
      head, same load-phase asymmetry (worker done in ~221s, head still
      loading past minute 9). `max_num_batched_tokens` was not the lever --
      a 4x cut produced no observable change in when or how the kill
      happens. This narrows things further: the KV-cache profiling dummy
      batch's own size isn't what's tipping the balance, which points more
      strongly at (b) or (c) from attempt 10's list -- the sheer size of
      `max_model_len` (262144) driving the KV-cache-sizing math itself, or
      `gpu_memory_utilization` needing to go lower than 0.84 for this
      checkpoint's 89.1 GiB resident footprint, independent of batch shape.
      11 attempts total. Containers cleaned up (`sparkrun stop`). Three
      fix-and-retry cycles in a row have now failed to clear this --
      pausing to report rather than attempt a 12th blind guess.
- [ ] Correctness: `probe_sanity.py` (coherent output, no reasoning leak,
      finish_reason=stop)
- [ ] **ModelOpt token-corruption check**: this carrier is ModelOpt-format
      NVFP4 (confirmed via `trellismx.py`'s `ModelOptNvFp4FusedMoE` import) --
      the same format class as `LibertAIDAI/GLM-5.3-Flash-NVFP4`, which this
      project already found has a documented, reproducible corruption bug
      (see `recipes/glm-5.3-flash-nvfp4-vllm.yaml` header, vLLM #54150).
      Different publisher, different weights -- run this project's existing
      CJK/emoji reproduction methodology directly against this checkpoint
      before trusting anything else from this boot.
- [ ] Context floor: confirm clean boot and correctness at 128K
- [ ] Context target: confirm clean boot and correctness at 192-256K
- [ ] Performance: TTFT / prefill / decode at 16K, 64K, 128K, 192-256K vs.
      the current production baseline (`v19-indexercompat`, EXL3 4bpw):
      decode ~27-30 tok/s, prefill @16K ~1,612 tok/s (TTFT ~9.9s)
- [ ] Concurrency: `probe_concurrency_pipeline.py` at a few levels
- [ ] Verdict: does this carrier fit and perform competitively, independent
      of the (confirmed-unavailable) TrellisMX routed-expert weights?

## Track B: SM121a kernel silicon-compatibility (bounded side-investigation) -- DONE

Full writeup in `HYPOTHESIS.md`'s "Track B result" section; evidence in
`evidence/track-b-sm121a-kernel-probe/`.

- [x] Triton vs. precompiled-CUDA determination -- neither exactly: CUTLASS
      Python DSL (JIT per-device, but hand-tunes MMA instruction selection,
      not automatically arch-portable like Triton)
- [x] SM120-specific hardcoding beyond the top-level capability guard -- none
      found on the tested K4/K5 decode path; a real one exists on the
      untested K2/K3 path (Hopper/SM90 WGMMA imports)
- [x] Isolated kernel invocation attempt on real GB10 hardware -- ran
      (`world_size=4, tp_rank=0`, not `world_size=1` which the constructor
      rejects). JIT-compiled and executed cleanly, correct output shape,
      confirmed `(12,1)` capability. Output all-NaN with synthetic weights --
      inconclusive on numerics (no real codebook data to test against).
- [x] Verdict: **looks like a fixable-by-us gate, not a fundamental
      incompatibility** -- the SM120 guard is likely just conservative, not
      an ISA wall. Does not unblock the checkpoint on its own: TP4 remains a
      separate, independent, unsolved blocker (2 physical GPUs vs. 4
      required, no resharding path for the sidecar files).

## Bottom line

Not yet written -- fill in once both tracks report real results, not before.
