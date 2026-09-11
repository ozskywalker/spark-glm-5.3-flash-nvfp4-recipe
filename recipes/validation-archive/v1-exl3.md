# VALIDATION.md archive: v1 EXL3 (lane bring-up)

_Archived from `recipes/VALIDATION.md` lines 550-1238 (git-blame-verified,_
_no content altered). See `recipes/VALIDATION.md` for the current index and_
_where to look for common problems._

---


## DFlash2 lane: first boot attempt crashed during warmup (2026-08-31)

`glm-5.3-flash-nvfp4-vllm-dflash2.yaml` — RedHatAI checkpoint, DFlash2
drafter (`incoai/GLM-5.3-Flash-DFlash2`, `num_speculative_tokens=7`),
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` image (built FROM
sm121-v8, no InstantTensor). Same 3 GiB KV pin, gmu 0.85, CUDA graphs
enabled (no `--enforce-eager`) as the shipped MTP-4 baseline. Job
`91f55faeb9fcc52a_5ce1214c28bf`.

**Boot progressed cleanly through everything new to this image**:
architecture resolution for both `Glm5NextForConditionalGeneration` and
`DFlash2DraftModel`, NCCL init, MoE backend resolution (`MARLIN`, as
expected — no scheme change), target weights loaded (90.67 GiB, 431.6s —
plain safetensors shard read, no direct I/O, matches the documented ~15
min no-InstantTensor boot), drafter weights loaded (2.18 GiB, 7.4s), Eagle3
aux layers registered `(6, 15, 25, 34, 43)`, KV cache resolved against the
existing 3 GiB pin (310,292 tokens, consistent with the pin), spec-decode
rejection sampler kernels warmed up, MoE router GEMM kernels warmed up,
two new TileLang-JIT-compiled kernels (`mhc_pre_big_fuse_with_norm_tilelang`,
`mhc_post_tilelang`) compiled without incident, FlashInfer autotuner ran to
completion.

**Then, 14 seconds after the autotuner finished** (21:45:05 → 21:45:14),
`Worker proc VllmWorker-0 died unexpectedly (exit code: None)` — the exact
silent-kill signature documented throughout this file: no worker-side
traceback, no dmesg trace. The executor-side stack is just the inevitable
consequence (`RuntimeError: cancelled` on a `dequeue()` waiting for a
response that will never come, inside `compile_or_warm_up_model` →
`collective_rpc`). Log evidence pins the death to the CUDA-graph-capture
window: it's the next step after autotuning in vLLM's init sequence, and no
"Capturing CUDA graphs" line appears before the crash.

**Host memory checked immediately after** (`free -h`, `swapon --show`,
`/proc/sys/vm/swappiness` on the node still running the job): 113 GiB free,
515 MiB swap used of 16 GiB, swappiness=1 — fully healthy. This was a
transient in-pipeline allocation spike, not a leak or persistent corruption;
consistent with every other "silent NVRM first-touch kill" this project has
hit (see the top-of-file KV pool ceiling section and the
`max_num_batched_tokens=3584` entry) rather than a new failure mode.

**Working theory**: DFlash2 adds real one-time memory pressure at exactly
the CUDA-graph-capture step that the MTP-4 baseline doesn't carry — the
drafter's own decode graphs, the mHC contraction kernels, and the
rejection-sampler's extra shapes all need graphs captured on top of the
target model's own, and this cluster's graph-capture headroom on top of a
3 GiB KV pin was already the tightest-proven margin (raising
`max_num_batched_tokens` past 2048 alone was enough to hit this same kill
with MTP-4, zero drafter). Next step: relaunch with `--enforce-eager` added
(pin, gmu, everything else held constant) to test this in isolation — this
is the identical technique already used successfully to prove CUDA graphs
safe on the MTP-4 baseline (toggle enforce-eager alone, pin constant). If
eager mode boots clean, that confirms graph capture as the trigger and
gives a path forward (ship DFlash2 eager-only, or find headroom to re-enable
graphs later); if eager mode *also* dies, the cause is elsewhere in
DFlash2's warmup and needs further bisection.

### Isolation result: --enforce-eager got further, but still died — new theory

Relaunched with `--enforce-eager` added, pin/gmu/everything else held
constant (same job ID `91f55faeb9fcc52a_5ce1214c28bf`, fresh
`prelaunch_flush.sh` run first). **Partial result**: this time the boot
completed successfully all the way through `Application startup complete`
(weights: 90.67 GiB/445.2s; drafter: 2.18 GiB/instant; KV cache resolved
against the same 3 GiB pin; spec-decode rejection sampler + router GEMM
kernels warmed up; FlashInfer autotuner ran to completion; TileLang
compiled the mHC kernels **repeatedly, for multiple distinct shapes** this
time — `mhc_pre_big_fuse_with_norm_tilelang` and `mhc_fused_tilelang` each
compiled 3x for what look like different batch/shape buckets, vs. exactly
once each under CUDA-graph mode). The last worker-side line before the
server went ready: `Kernel JIT monitor activated; monitored JIT
compilations during inference will use mode=warn` — i.e. this image's own
code anticipates JIT compiles happening **during live serving**, not just
warmup.

The server then sat fully idle (zero log lines — expected, since
`--enable-logging-iteration-details` only logs when iterations run) for
**~2h41m** before dying with the identical silent-kill signature (`exit
code: None`) — but this time triggered from inside
`_handle_client_request` → `RuntimeError("Executor failed.")`, i.e. an
actual request reached the engine and the worker was gone by the time it
tried to serve it. No probe or curl was run against this endpoint during
that window from this side — the trigger was either sparkrun's own
deferred readiness/health verification, or some other periodic internal
vLLM task riding the same request-queue path (`fault_tolerance/
engine_core_sentinel.py` wraps the busy loop, per the traceback). Container
stayed "Up" per Docker/sparkrun status throughout — only the inner `vllm
serve` process died; confirmed via `curl /health` returning connection
refused after the crash.

**Revised theory**: `--enforce-eager` fixed the graph-capture-time crash
(boot now completes, which it never did before), but this image
JIT-compiles TileLang mHC kernels **per new shape encountered**, including
shapes that only show up under real traffic (not covered by the fixed
warmup shapes). A live recompile triggered by an actual request's shape —
on top of the resident 90.67 GiB of weights, the pinned 3 GiB KV pool, and
zero slack budgeted for a surprise compiler invocation — is a plausible
new trigger for the same driver-level ceiling this cluster has hit
repeatedly, just moved from "warmup time" to "first-real-shape time."

**Next test**: relaunch again (eager mode kept — it's strictly better,
gets further), and send a real completion request **immediately** upon
readiness instead of waiting, to convert the uncontrolled ~2h41m gap into
a controlled, fast reproduction. If it dies on the first deliberate
request, that confirms "any real request kills it" as a hard blocker
independent of timing. If it survives that first request, the trigger is
something else entirely (e.g. a periodic internal task on a fixed
interval) and worth chasing separately.

### Third attempt: crashed within seconds, same signature — plus a log-ordering finding

Relaunched a third time, identical config (eager mode kept, same 3 GiB
pin), intending to fire a real request the instant it was ready. Never got
the chance: this attempt died **4 seconds** after `Started server process
[74]` / `Waiting for application startup.` — the identical `Worker proc
VllmWorker-0 died unexpectedly (exit code: None)` →
`_handle_client_request` → `RuntimeError("Executor failed.")` signature as
attempt 2, just much earlier in wall-clock time.

**Important ordering finding**: `Application startup complete` was
logged *after* the crash and shutdown sequence had already started (the
worker died at 00:54:12, the fatal-error traceback and executor shutdown
completed by 00:54:14-21, and only then does `INFO: Application startup
complete.` appear at 00:54:21, immediately followed by the async output
handler discovering the dead engine). This means FastAPI's own "startup
complete" line does **not** reliably indicate the engine was healthy at
that point — it can print after a fatal engine error already occurred, if
that error surfaced via the async output-handler task rather than the
startup path itself. **This retroactively weakens the attempt-2 read**
("booted clean, served nothing for ~2h41m, then a real request killed
it") — the `_handle_client_request` calls in both attempt 2 and attempt 3
are consistent with an *internal* request vLLM's own engine issues as part
of its own startup/warmup sequence (not necessarily sparkrun's readiness
probe or genuine external traffic), and attempt 2's apparent "clean boot"
may just be the same log-ordering artifact stretched across whatever
caused that particular run's internal warmup call to be delayed ~2h41m
(cause unknown — possibly a queued/retried internal task, not yet
understood).

**Net result across all three attempts**: DFlash2 has never successfully
served a single request on this cluster, under either CUDA-graph or eager
mode. Every attempt hits the identical silent-kill signature; only the
timing varies. This is consistent with a hard memory-margin problem
specific to DFlash2's extra footprint (drafter weights/activations, mHC
TileLang kernels compiled per-shape, extra spec-decode-rejection sampler
state) landing on top of a KV pin (3 GiB) that was tuned for the lighter
MTP-4 drafter, not headroom for a fundamentally different bug.

**Next test**: reduce `kv_cache_memory` below the existing 3 GiB pin
(trying 2 GiB) to trade KV capacity for scratch headroom during
warmup/serving, keeping `--enforce-eager`. If this also fails, the
conclusion is that DFlash2 is not viable on this specific 2-node cluster's
memory margin without upstream patches this project doesn't carry, and the
disposable recipe gets documented as a negative result and deleted per its
own header instruction.

### Fourth attempt: 2 GiB undershot the bare minimum (clean error, not a crash)

Reduced `kv_cache_memory` to 2 GiB (2147483648) and relaunched. This time
vLLM raised a **clean, legible `ValueError`** before the silent-kill point
was ever reached — a real fix, not a forensics problem:

```
ValueError: To serve at least one request with the model's max seq len
(262144), (2.52 GiB KV cache is needed, which is larger than the
available KV cache memory (2.0 GiB). ... estimated maximum model length
is 165888.
```

DFlash2's own KV accounting needs **2.52 GiB minimum** just to serve one
request at `max_model_len=262144` — higher than MTP-4's minimum, because
verifying `num_speculative_tokens=7` drafted tokens per step needs more
per-sequence KV headroom than a 4-token MTP draft. This narrows the usable
range to **2.52-3.0 GiB**: below 2.52 GiB the engine won't even start
(clean rejection); at 3.0 GiB (three attempts) it starts but dies via the
silent kill. That's a very thin unexplored band — worth one more direct
test at the midpoint (2.75 GiB) before concluding the KV-pin lever is
exhausted.

### Fifth attempt: midpoint (2.75 GiB) also died — verdict

Relaunched at `kv_cache_memory=2952790016` (2.75 GiB), the midpoint of the
only band where DFlash2 both starts and can theoretically serve a request.
**Died with the identical signature** — `Worker proc VllmWorker-0 died
unexpectedly (exit code: None)` → `_handle_client_request` →
`RuntimeError("Executor failed.")`, `Application startup complete` again
printed after the fatal error (same log-ordering artifact as attempt 2/3).

**Verdict: DFlash2 is not viable on this 2-node cluster with the current
image and config.** Five attempts, every point in the only KV-memory band
where the engine can even start (2.52-3.0 GiB) hits the same silent
NVRM-first-touch kill, independent of:
- CUDA graphs vs. `--enforce-eager` (attempt 1 vs. 2-5) — eager mode
  changed *when* it dies (sometimes during warmup, sometimes after
  `Application startup complete`), never *whether*.
- KV pin size within the valid band (3.0 / 2.75 GiB) — no headroom found
  by trading KV capacity for scratch memory.

The KV-pin tuning lever that fixed every prior memory-ceiling issue on
this cluster (documented throughout this file) does not fix this one,
because DFlash2's minimum required KV (2.52 GiB) already consumes most of
the narrow band this cluster has to give — there's no room left to trade.
The remaining unknowns (why attempt 2 survived ~2h41m before dying, the
exact nature of the internal `_handle_client_request` call that triggers
the kill) would need instrumentation this project already ruled out as
non-viable on this hardware (5/5 live-GPU-profiling crashes, see the MoE
Marlin investigation above) to chase further, or upstream engagement with
tonyd2wild's own image/patches — out of scope for this recipe's bring-up
process.

**Disposition**: `glm-5.3-flash-nvfp4-vllm-dflash2.yaml` deleted per its
own header instruction ("if not, delete this file"). The shipped
`glm-5.3-flash-nvfp4-vllm.yaml` (RedHatAI checkpoint, MTP-4, 3 GiB pin,
CUDA graphs) remains the validated, working recipe — see the "RedHatAI
checkpoint performance baseline" section above for its numbers
(21.56 tok/s short-bench, 19.40 tok/s longer-form median). DFlash2's
own 2.15x-speedup claim over MTP-4 was never able to be measured on this
hardware; if revisited later, start from the 2.52-3.0 GiB KV band
finding above rather than re-deriving it.

## max_num_seqs tuning (2026-09-01): the DFlash2 crash signature isn't DFlash2-specific

Following community reports (tonyd2wild's DFlash2 repo: raising
`max_num_seqs` 6→64 and `max_num_batched_tokens` 8192→16384 significantly
improved throughput/prefill on their hardware), tested `max_num_seqs`
tuning on the **shipped, fully-validated baseline recipe**
(RedHatAI checkpoint, MTP-4, CUDA graphs, 3 GiB pin — everything unchanged
except the one override), via `-o max_num_seqs=32`, to isolate this one
variable.

**Booted clean** (InstantTensor load, ~80s weights + warmup, same as
every prior boot on this recipe), reached `Application startup complete`,
responded HTTP 200 to `/health` and `/v1/models`. Then, within seconds of
the first real requests hitting it, **died with the exact same silent
NVRM-first-touch-kill signature documented throughout the DFlash2 chase
above**: `Worker proc VllmWorker-0 died unexpectedly (exit code: None)` →
`_handle_client_request` → `RuntimeError("Executor failed.")`. Host
memory checked immediately after: 116 GiB free, healthy — same transient
in-pipeline pressure signature as every other occurrence, not a leak.

**This is an important correction to the DFlash2 postmortem's framing**:
the "DFlash2 isn't viable" conclusion was written as if the crash mode
were specific to DFlash2's extra footprint (drafter, mHC kernels, etc).
It is not — the *identical* signature reproduces on the plain MTP-4
baseline with nothing changed except `max_num_seqs` 6→32. The real
culprit is `max_num_seqs` (or its interaction with CUDA-graph capture
bucket sizing — more concurrent-batch-size buckets means more graphs
captured, more static memory reserved) landing on this cluster's already
razor-thin margin. DFlash2's failure was very likely this same mechanism,
just triggered by DFlash2's own baseline footprint rather than a
`max_num_seqs` change — the two investigations converge on one finding
rather than being separate problems.

**Next**: bisect `max_num_seqs` between 6 (proven safe) and 32 (crashes)
to find the actual safe ceiling on this cluster, instead of assuming
tonyd2wild's 64 (measured on different, presumably less memory-constrained
nodes) transfers here.

### Bisection result: 16 crashes at idle, 8 crashes under load — verdict

- `max_num_seqs=16`: died within ~15s of `Application startup complete`,
  **before any external request was sent** — same signature as 32. This
  points to a mechanism independent of real traffic entirely: more
  concurrent-batch-size buckets to capture CUDA graphs for scales with
  `max_num_seqs`, and this alone appears to exhaust the margin during
  warmup, matching the DFlash2 chase's own "graph-capture-adjacent memory
  pressure" theory.
- `max_num_seqs=8`: booted clean, survived 20s idle, served a real
  single-stream completion correctly (`probe_sanity.py`: 19.97/23.78/24.83
  tok/s, median 23.78 — in line with the 6-seq baseline's 21.56, no
  meaningful change, as expected since a single-stream test doesn't
  exercise scheduling depth). **Then died the moment it was hit with 8
  concurrent requests** (a custom aggregate-throughput probe,
  `concurrency_bench.py`, sending 8 concurrent chat completions at
  `max_tokens=200`). This time the crash dump is actually legible and
  conclusive — captured via `--enable-logging-iteration-details`'s dump-
  on-fatal-error:

  ```
  SchedulerStats(num_running_reqs=6, num_waiting_reqs=2,
  kv_cache_usage=0.9714285714285714, ...)
  ```

  97% KV pool occupancy, mid `_process_engine_step` (not even admitting a
  new request — this was a normal decode step on already-running
  requests). **This is not a mysterious silent kill — it's the 3 GiB KV
  pool genuinely running out of room under concurrent load.**

**Why this cluster's concurrent-request ceiling is lower than the raw
token math suggests**: `block_size=2304` (chosen for kpool*64 x MLA-128
alignment, `--enable-mfu-metrics` era config) means every request reserves
KV space in 2304-token chunks regardless of actual prompt/generation
length. The "365K token pool" framing used throughout this file describes
*aggregate* capacity, not concurrent-request headroom — with a block this
large, admitting several requests simultaneously consumes blocks much
faster than the token-count math implies, especially combined with this
model's hybrid mamba/MLA/indexer layers each needing their own
block-aligned reservation per request (the "attention block size padded to
match mamba page size" line logged at every boot).

**Verdict**: `max_num_seqs=6` is not an arbitrarily-conservative legacy
value — it's very close to this cluster's actual concurrent-capacity
ceiling given the 3 GiB pin and `block_size=2304`. Raising it independently
crashes via two distinct mechanisms depending on the value: graph-capture
memory pressure at idle (16+), or KV-pool saturation under real concurrent
load (8, once actually pushed to its own configured limit). tonyd2wild's
6→64 finding does not transfer to this cluster — it was very likely
measured on hardware with meaningfully more KV headroom than this
cluster's demonstrated ~4-6 GiB deficit. **`max_num_seqs` stays at 6 in
the shipped recipe; not raising `max_num_batched_tokens` either** — the
`max_num_batched_tokens=3584` finding from earlier in this document
already showed that lever crashes independently before `max_num_seqs` was
ever a factor, so there's no reason to expect the paired 16384 value from
the same community report to fare any better.

An important secondary finding worth carrying forward: **the crash-dump
diagnostics (`--enable-logging-iteration-details`) actually work and are
informative** — this is the first crash in this entire project's history
where the failure came with real, actionable scheduler state
(`kv_cache_usage`, running/waiting counts) instead of a bare `exit code:
None`. Every future memory-ceiling investigation on this cluster should
lean on this instrumentation rather than treating these crashes as
un-diagnosable.

## MFU counter turned into an actual percentage (2026-09-01)

The `--enable-mfu-metrics` / `--enable-logging-iteration-details` combo
already prints a periodic `MFU: X TF/s/GPU` line (`perf.py:1529`) and a
paired `Y GB/s/GPU` memory-bandwidth figure — this was captured as a raw
counter (`vllm:estimated_flops_per_gpu_total`) in the earlier baseline
section but never converted to a percentage for lack of a peak-FLOPS
denominator. Sourced GB10's actual specs to close that gap:

- **BF16 dense tensor peak: ~11-12 TFLOPS/GPU** (measured figure from
  community benchmarking, not the marketing "1 PFLOP" number — that
  figure is 2:4-sparse NVFP4, a completely different precision/kernel
  path GB10 is specifically built around; **Marlin (weight-only NVFP4)
  dequantizes to BF16 for the actual GEMM, so BF16 peak is the correct
  denominator for this recipe's kernels, not the FP4 figure**).
- **Unified memory bandwidth peak: 273 GB/s** (LPDDR5x, shared CPU+GPU,
  per GB10 node/rank).

Measured a sustained single-stream 400-token generation on the shipped,
unmodified baseline recipe (max_num_seqs=6, everything else as validated)
and captured the periodic log line during active decode (`Running: 1
reqs`, 25.3 tok/s generation throughput):

```
MFU: 0.4 TF/s/GPU    102.4 GB/s/GPU
```

- **Compute utilization: 0.4 / ~11.5 TFLOPS ≈ 3.5% of peak BF16 FLOPs.**
- **Memory utilization: 102.4 / 273 GB/s ≈ 37.5% of peak bandwidth.**

**This is a coherent, expected result, not a red flag**: single-stream
MoE decode is intrinsically memory-bandwidth-bound, not compute-bound —
each token only activates ~18B of the model's 320B total parameters, but
every byte of those active parameters must stream from memory before the
(comparatively tiny) matmul runs. Low FLOPs utilization at batch=1 is the
normal signature of this access pattern on any hardware, not evidence of
an inefficiency to chase.

**This directly connects to the max_num_seqs finding above**: the
textbook fix for low decode-phase MFU is deeper batching — more
concurrent sequences amortize each expert-weight fetch across more useful
compute, raising FLOPs utilization without changing memory traffic much.
That is exactly the lever the max_num_seqs investigation just proved
closed on this cluster (KV-pool saturation at 8 concurrent, graph-capture
memory pressure at 16+). **The two findings together describe one
structural ceiling**: this cluster's ~4-6 GiB memory deficit versus
reference nodes forecloses the batching depth that would otherwise raise
MFU. This isn't a software tuning gap — it needs more memory headroom
(a larger KV pin than this cluster can sustain, or more/bigger nodes),
not another config flag.

## persistent_topk SM-count gate re-validated under CUDA graphs (2026-09-01)

Closed the open item flagged after `--enforce-eager` was dropped: the
`persistent_topk` SM-count gate (hard `RuntimeError` past ~24K context on
GPUs with <78 SMs; GB10 has 48, fixed via a bind-mounted indexer patch in
the image) was proven safe under eager mode, but never explicitly
re-confirmed for **decode** specifically under CUDA graphs — the earlier
250K-token long-context test measured prefill TTFT and confirmed
retrieval worked, but wasn't framed as a targeted decode-path check for
this exact gate, and CUDA graph capture's static shapes could in
principle route decode differently than eager execution.

Ran a dedicated, faster (30K tokens, not the full 250K) `probe_longctx.py`
pass on the shipped baseline recipe (max_num_seqs=6, CUDA graphs enabled,
current instrumentation flags) — comfortably past the 24K threshold:

- `prompt_tokens=29938` (target 30000), TTFT 31.4s, all 4 planted codes
  retrieved, `finish_reason=stop`, 27 completion tokens generated cleanly.
- No `persistent_topk`, SM-count, `RuntimeError`, or indexer-related
  warnings anywhere in the logs for this request.

**Gate confirmed holding under CUDA graphs for decode past 24K context.**
This open item is closed — no further action needed.

## EXL3: the "bigger swing" this session's findings point toward (2026-09-01)

Every tuning lever this document has tried (checkpoint format, MoE
backend, KV pin size, max_num_seqs, CUDA graphs vs eager) converges on
the same structural wall: this cluster's memory headroom under NVFP4/
Marlin is too tight to batch deeper, run DFlash2, or meaningfully raise
FLOPs utilization. Re-examined the two community repos surveyed earlier
this session (Reederey87/glm53-flash-exl3-2x-dgx-spark,
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) with the specific question:
does EXL3 avoid this wall, and is it worth the switch? Both repos serve
the **same model** on the **same 2x GB10 hardware class**, so this is
about as close to an apples-to-apples comparison as exists.

**EXL3 is a genuinely different execution path, not a config tweak.**
Both repos state explicitly: **"Do not pass `--moe-backend marlin"`** —
EXL3's MoE experts run via `exllamav3_ext.exl3_moe`, "one fused launch per
layer," entirely bypassing Marlin. This matters because this project's own
`gb10-kernel-bench` work found MoE Marlin dominates decode cost 30-50x
over every other kernel on this hardware — EXL3 sidesteps the bottleneck
this session spent most of its effort working *around*, rather than
*through*.

**Why GB10 needs Marlin at all for NVFP4, per Reederey87's own README**:
"EXL3 (the only quantization GB10 can actually run — it lacks the
instruction NVFP4 compiles to)." This lines up exactly with our own logs
from this session's DFlash2 attempts (`marlin_utils_fp4.py:354`): *"Your
GPU does not have native support for FP4 computation but FP4 quantization
is being used. Weight-only FP4 compression will be used leveraging the
Marlin kernel."* GB10 has no native FP4 tensor-core path; Marlin exists
specifically to dequantize-and-multiply-in-BF16 as a software workaround.
EXL3 (a fused INT4-trellis format with its own hand-tuned aarch64/sm_121
kernels) apparently doesn't need that workaround.

**Performance, independently corroborated by both repos** (DFlash2 k=7,
2026-08-28/31 measurements):

| | Reederey87 | MiaAI-Lab | This project's NVFP4/MTP-4 baseline |
|---|---:|---:|---:|
| Structured/code decode | ~67 tok/s (1.0000 accept) | 65.1 tok/s (0.959 accept) | not measured (no structured-output bench run) |
| Prose decode | 28-31 tok/s | 27.1 tok/s | **19.40-21.56 tok/s** |
| MTP-only baseline (no DFlash2) | — | ~24.6 tok/s | 19.40-21.56 tok/s |

Even MiaAI-Lab's own **MTP** baseline on EXL3 (24.6 tok/s, no speculative
decoding upgrade) beats this project's NVFP4 MTP-4 result outright — the
gap isn't only from DFlash2, weight format matters on its own.

**Memory headroom, the theory this session's crashes point toward**: EXL3
packs the 320B experts at ~164 GiB total / ~82 GiB per node (uniform-K4,
4bpw) vs. our NVFP4 checkpoint's ~184 GiB. MiaAI-Lab's DFlash2 config
captures CUDA graphs for concurrent batch sizes **1 2 4 8 16 24 32** — the
exact scaling that killed every attempt in this project's own DFlash2 and
max_num_seqs investigations above. The ~20 GiB smaller footprint, plus not
needing Marlin's extra dequant scratch, plausibly explains why
community-reported high-`max_num_seqs`/DFlash2 configs work on EXL3 and
don't transfer to this project's NVFP4 setup — not a difference in raw
hardware, a difference in how much memory the quantization scheme itself
consumes before any request even arrives.

**Quality is a second, independent reason to consider this — not just
speed.** An independent teacher-logit KLD panel (25 sealed windows, 51,175
positions) cited in MiaAI-Lab's README:

| Checkpoint | Mean KLD (nats, lower=closer to teacher) | Size |
|---|---:|---:|
| TR3 K6 (6bpw) | 0.0137 | 254 GB |
| Official FP8 | 0.0206 | 328 GB |
| **EXL3 4bpw** | **0.0246** | **176 GB** |
| NVFP4 (same base stack) | **0.0605** | ~180 GB |

On the same underlying model, NVFP4 shows **~2.5x worse fidelity to the
teacher distribution** than EXL3 4bpw, despite being a similar file size —
EXL3 4bpw actually matches full FP8's fidelity almost exactly at roughly
half the bytes. This doesn't necessarily transfer number-for-number to
RedHatAI's specific NVFP4 conversion (different quantizer/calibration than
whatever "brandonmusic stack v44" refers to), but it's a real, measured
signal that NVFP4 may be leaving quality on the table independent of the
speed story.

**Honest cost of pursuing this**: not a config change — a full new
bring-up. New ~164-176 GiB checkpoint download, a custom-built image
(`exllamav3` compiled for aarch64/sm_121, not available prebuilt for this
cluster's exact driver/CUDA pairing), a different KV dtype
(`fp8_ds_mla`, not `fp8_e4m3`) and attention backend
(`FLASHINFER_MLA_SPARSE_SM120`/`SM121`, not the current NoPE-MLA path),
and both source repos describe extensive independent hardening work
(prefix-cache page-boundary fixes, hybrid-KDA-specific patches, DFlash2
drafter integration) beyond just swapping the quantization flag — this
project's own GLM-5.3-Flash NVFP4 bring-up took "seven day-0 bugs" to reach
a working baseline, and EXL3 would need its own equivalent bring-up pass
from scratch, not a port of anything already validated here.

**Recommendation**: this is the correct next investment if further
performance is worth a multi-day bring-up effort — it's the only lever
surveyed this session (including the entire DFlash2 chase) that plausibly
escapes the structural memory ceiling documented throughout this file,
with two independent, same-hardware data points backing the performance
claim. It is explicitly **not** a quick win — treat it as a new project
phase (its own bring-up, following this repo's own
`new-recipe-bringup` skill/checklist), not a follow-up experiment to slot
into the current NVFP4 recipe.

# EXL3 lane bring-up (2026-09-01)

New recipe: `glm-5.3-flash-exl3-vllm.yaml`. Ported from
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (read directly from their
`start.sh`, `.env.example`, `Dockerfile` — not just the README prose).
Checkpoint (`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, 164 GiB, 120 shards)
and the DFlash2 drafter checkpoint were already present on this cluster's
`/models` share from earlier session work; verified complete (120/120
shards, `main` ref) before use. Image
(`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`) confirmed pullable
via `docker manifest inspect`. No downloads needed.

First-pass config deliberately more conservative than MiaAI-Lab's own
production defaults, per this project's established bring-up discipline
(isolate one variable at a time): `--enforce-eager` (their default is
CUDA graphs on), no speculative decode (their default is DFlash2 k=7),
`max_model_len=262144` (their default is 1,000,000), `gpu_memory_
utilization=0.85` (theirs 0.87), `max_num_seqs=4` (their own conservative
value, kept). See the recipe's own header comment for full rationale.

## Base config: booted clean on the first attempt

Unlike every NVFP4+DFlash2 attempt this session (5/5 failures), the EXL3
base config **booted successfully on the first try** — no crash forensics
needed. `Application startup complete`, verified with a direct `/health`
check (HTTP 200, learned not to trust that log line alone after the
DFlash2 log-ordering finding) and a real chat completion (coherent,
correct: "The capital of France is Paris.").

- `probe_sanity.py` (corrected `--model glm-5.3-flash-exl3` — the probe's
  hardcoded default model name is the NVFP4 lane's; not a real failure):
  **ALL SHORT-CONTEXT CHECKS PASSED**. Decode **13.52-13.53 tok/s**
  (unusually tight variance — expected for eager mode with zero
  speculative decode, no acceptance-rate noise), TTFT 0.17-0.26s.
- `probe_soak.py` (2 rounds x 2 waves): **PASSED** — 10/10 + 10/10
  sequential, 3/3 + 3/3 concurrent, endpoint alive after.

**This number (13.52 tok/s) is not yet a fair comparison against the
NVFP4 baseline** — it's EXL3's least-optimized configuration (no graph
capture, no speculative decode) vs. the NVFP4 lane's most-optimized one
(CUDA graphs + MTP-4, 21.56 tok/s median). The correct apples-to-apples
point is NVFP4's own bare-eager baseline, which was measured earlier this
session at 21.8 tok/s — EXL3 is currently behind even that. This is not a
red flag: EXL3's fused `exl3_moe` kernel is architecturally different from
Marlin and community numbers (all measured with graphs on) suggest it
benefits significantly from graph capture. Next step: isolate CUDA graphs
the same way this project validated them safe on the NVFP4 lane — pin
everything else constant, drop only `--enforce-eager`.

## CUDA graphs: booted clean, modest uplift — the real lever is elsewhere

Dropped `--enforce-eager`, everything else held constant (same job ID,
fresh prelaunch flush). **This is the exact phase that killed every
NVFP4+DFlash2 attempt this session (5/5 failures) — it booted clean on
the first try here.** Graph capture: PIECEWISE 4/4 + FULL 3/3 buckets
(matching `max_num_seqs=4`), **3 seconds, 0.18 GiB** — trivial cost,
strong confirmation of the memory-headroom theory from the EXL3 research
section above.

- Weights: 80.45 GiB, 369-370s (consistent with the eager-mode boot,
  confirming EXL3's smaller footprint is real and repeatable on this
  cluster's own hardware, not just a community-reported number).
- `probe_sanity.py`: ALL PASSED. Decode **14.07-14.19 tok/s** (median
  14.09) vs. 13.52 eager-mode — only a **~4% uplift**, much smaller than
  the 15-20% CUDA graphs gave the NVFP4/Marlin lane. Plausible reason:
  `exl3_moe`'s fused kernel already collapses what Marlin does as several
  discrete kernel launches into one, leaving less kernel-launch overhead
  for graph capture to amortize.

**Conclusion: CUDA graphs are safe here (unlike the NVFP4 lane's DFlash2
combination) but are not where the big EXL3 performance win comes from.**
Every community number that beats this session's NVFP4 baseline (27-31
tok/s prose, even MiaAI-Lab's own MTP-only reference at ~24.6 tok/s) used
speculative decode. That's the next lever, not further base-config tuning.
Promoted: CUDA graphs stay on for every experiment from here.

## MTP k=2: beats the NVFP4 baseline outright

Added `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
— the checkpoint's own native MTP block (layer 45), k=2 matching
MiaAI-Lab's own documented rollback default (`MTP_TOKENS=2`; their own
testing found higher k degrades acceptance too much on this checkpoint).
Everything else held constant (CUDA graphs on, same job ID, fresh flush).

**Booted clean.** Weights 350.8s + MTP head 28.5s = 82.38 GiB total (only
+1.93 GiB over the no-spec-decode config). Graph capture: 5s, 0.41 GiB —
again trivial, again the phase that killed every NVFP4+DFlash2 attempt.

- `probe_sanity.py`: ALL PASSED. Decode **26.49-28.29 tok/s** (median
  **26.75**) — nearly **2x** the graphs-only result (14.09) and **beats
  the shipped NVFP4 baseline outright** (21.56 tok/s median, CUDA graphs +
  MTP-4) by ~24%, using a *lighter* speculator (k=2 vs. k=4). Slightly
  exceeds MiaAI-Lab's own reported MTP-only reference (~24.6 tok/s) on
  different hardware — first independent confirmation this transfers.
- Acceptance: 71-77% average draft acceptance rate, mean acceptance
  length 2.4-2.5 (of a possible 3 with k=2) — healthy, not marginal.
- `probe_soak.py` (2 rounds x 2 waves): **PASSED** — including 3
  concurrent requests, near this config's own `max_num_seqs=4` ceiling —
  the exact kind of concurrent-load stress that triggered the NVFP4
  lane's KV-saturation crash at `max_num_seqs=8`. No issue here.

**This is the first config this session that beats the shipped NVFP4
recipe on a genuine apples-to-apples short-bench comparison.** Next:
DFlash2 (k=7, the community's own headline config) for the full
performance ceiling — same incremental methodology, one variable at a
time, full crash forensics if it fails.

## DFlash2 k=7: boots clean where NVFP4 failed 5/5 — workload-dependent win

Swapped MTP for `{"method":"dflash","model":".../GLM-5.3-Flash-DFlash2/
snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe",
"num_speculative_tokens":7,"kv_cache_dtype":"auto",
"draft_sample_method":"probabilistic","rejection_sample_method":
"standard","draft_tensor_parallel_size":2}` — matches MiaAI-Lab's and
Reederey87's exact JSON construction. **Drafter pinned to commit
`7d74cdd`, not the current `main`/`dc77ff1c` snapshot** — Reederey87's own
README: "the Hub repo has shipped three different weights under the same
name; the two newer ones were A/B-tested here and won nothing (one loses
6% on prose)." Both snapshots were already present locally; used their
vetted pin. Everything else held constant (CUDA graphs on, same 262144
context, same job ID, fresh flush).

**Booted clean on the first attempt.** Eagle3 aux layers registered
identically to the NVFP4 lane's own DFlash2 attempts (`(6, 15, 25, 34,
43)` — same architecture, same patch). Target weights 351.2s, drafter
6.8s, total **82.02 GiB**. **Graph capture: 38s, net -1.35 GiB** (CUDA
graph memory estimate returned to the KV pool — matches MiaAI-Lab's
documented `CG_ESTIMATE` behavior). This is the exact phase — DFlash2 +
CUDA graph capture — that killed all 5 attempts on the NVFP4 lane. Zero
issues here.

**Performance is workload-dependent, matching both community repos'
own findings exactly:**

| Content type | Decode tok/s | Draft acceptance | Mean accept length (of 8) |
|---|---:|---:|---|
| Prose (`probe_sanity.py` short bench) | 22.11-24.53 (median 24.30) | 24.8-30.0% | 2.7-3.1 |
| Structured (counting 1→100, 300 tokens) | **56.35** | **88.9-91.4%** | **7.2-7.4** |

The structured number closely tracks MiaAI-Lab's own reported 65.1-67
tok/s (same order of magnitude, likely the gap is this recipe's more
conservative `max_num_seqs=4` vs. their tuned config). **On prose, DFlash2
(24.30 median) is actually slightly *behind* MTP k=2 (26.75 median) — a
~9% regression** — consistent with DFlash2's block-diffusion drafter
losing confidence sharply at longer look-ahead distances on
low-predictability content (per-position acceptance rate here: 0.7 at
position 0 down to 0.06 by position 6, vs. structured content's 0.85-1.0
held flat across all 7 positions).

- `probe_sanity.py`: ALL PASSED (prose numbers above).
- `probe_soak.py` (2 rounds x 2 waves): **PASSED** — 10/10 + 10/10
  sequential, 3/3 + 3/3 concurrent, endpoint alive after.

**Recommendation**: neither speculator is a strict winner — the choice is
workload-dependent. For a deployment expecting significant structured/
code-generation traffic (coding agents, tool-calling-heavy workloads),
DFlash2's 2x+ structured advantage outweighs its small prose regression.
For a primarily conversational/prose deployment, MTP k=2 is simpler,
lighter (no separate ~2 GiB drafter checkpoint, no `draft_tensor_
parallel_size` complexity) and slightly faster. Both are legitimate
shipping choices, unlike the NVFP4 lane where DFlash2 was never an option
at all.

## Full session comparison: NVFP4 vs EXL3, with and without DFlash2

| Configuration | Decode tok/s (prose) | Decode tok/s (structured) | Boots on this cluster? |
|---|---:|---:|---|
| NVFP4 + MTP-4 + CUDA graphs (shipped baseline) | 19.40 (400-tok) / 21.56 (short) | not measured | Yes |
| NVFP4 + DFlash2 k=7 + CUDA graphs | — | — | **No — 5/5 attempts crashed** (silent NVRM kill, every KV pin in the valid 2.52-3.0 GiB band, both eager and graph mode) |
| EXL3 + MTP k=2 + CUDA graphs | 26.75 (short) | not measured | Yes |
| EXL3 + DFlash2 k=7 + CUDA graphs | 24.30 (short) | **56.35** | Yes |

Model-loading footprint: NVFP4 ~90.67-91.23 GiB vs. EXL3 ~80.45-82.38 GiB
(~10 GiB smaller, consistent across every EXL3 boot this session) — the
most direct, measured confirmation of the memory-headroom theory that
motivated trying EXL3 in the first place. Every phase that killed NVFP4
(graph capture, DFlash2, concurrent load near `max_num_seqs`) passed
cleanly on EXL3 with meaningful margin to spare (5.62 GiB KV pool
estimate not even engaged — no pin was needed at 262144 context,
unlike the NVFP4 lane's hard 3 GiB pin requirement).

Not yet tested on the EXL3 lane (future work, same incremental
methodology): the 1,000,000-token context MiaAI-Lab and Reederey87 both
run in production (this session validated at 262,144 to match the NVFP4
baseline), higher `max_num_seqs`/`max_num_batched_tokens` bisection
(started at their own conservative defaults, never pushed), prefix
caching effectiveness (`--enable-prefix-caching` is on but unmeasured),
and MTP's own structured-content number (only DFlash2 was tested on the
structured prompt) for a fully symmetric comparison.
