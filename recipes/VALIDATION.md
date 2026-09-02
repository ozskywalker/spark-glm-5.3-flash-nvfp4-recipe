# GLM-5.3-Flash-NVFP4 sparkrun recipes — validation record (2026-08-27)

Cluster: sparkrun `default` — head `10.7.0.87` (spark-2dd4, rank 0) + worker
`127.0.0.1` (spark-276f, rank 1). Recipe: `glm-5.3-flash-nvfp4-vllm.yaml`
(TP=2, image `radixark/vllm-glm53-flash:sm121-v9`, weights pre-placed under
`/models` on both nodes, chat template at `/models/glm53-flash-chat_template_mm.jinja`).

## Result: ALL CHECKS PASSED

| Check | Result |
|---|---|
| Model load (InstantTensor, 91.23 GiB/rank) | ~80 s weights + 40 s warmup to serving |
| `/v1/models` serves `glm-5.3-flash` | PASS |
| Prometheus metrics (`/metrics`, 417 `vllm:*` series) | PASS |
| Short-context chat (coherent, thinking-off, no `<think>` leak) | PASS |
| Streaming TTFT / decode | 0.29-0.31 s / 23.7-27.4 tok/s (MTP-4) |
| Stability soak (2x10 sequential + 2x3 concurrent) | PASS, endpoint alive after |
| Cache continuation (shared ~6K prefix, cold/partial/warm) | PASS, no degenerate output |
| Long context **249,951 tokens** (95% of 262,144 max) | PASS — TTFT 180.6 s, 4/4 planted codes retrieved at 25/50/75/97% depth |

## Key finding: KV pool ceiling on this cluster

The upstream repo's TP2 configs (`4445787956` = 4.14 GiB, and the 672K-token
`5905580032` = 5.5 GiB record) **do not survive on spark-2dd4/spark-276f**:
three boots died in the silent NVRM first-touch kill (worker `exit code: None`,
no Python traceback, no dmesg/journal trace) during warmup or seconds after
startup. A boot without the pin showed vLLM's profiler computing
**"Available KV cache memory: -0.02 GiB"** at gmu 0.85 — these nodes have
~4-6 GiB less effective unified-memory headroom than the upstream record nodes.

The shipped recipe therefore pins `--kv-cache-memory 3221225472` (3 GiB/rank
~= 370K fp8 KV tokens = 1.4x max_model_len), which boots clean and passed the
full suite including the 250K-token prefill. Raise only with live memory
watching + the flush ritual (`scripts/prelaunch_flush.sh`).

## Operational notes

- Pre-launch ritual is mandatory: `scripts/prelaunch_flush.sh <hosts> --during-load`
  (drop_caches on all ranks + cache-flusher loop during load).
- Boot to ready: ~4.5 min with warm page cache (image pre-distributed).
- The TP4 recipe (`glm-5.3-flash-nvfp4-vllm-tp4.yaml`) mirrors the upstream
  tp4 script but is NOT validated here (cluster has 2 nodes).
- Probes: `probes/probe_sanity.py`, `probe_soak.py`,
  `probe_cache_continuation.py`, `probe_longctx.py --tokens 250000`.
  All dependency-free (urllib); sized via the server's `/tokenize` endpoint.

## NVFP4 KV cache lane: not servable on sm121-v9 (confirmed 2026-08-27)

`glm-5.3-flash-nvfp4-kvnvfp4-vllm.yaml` (`--kv-cache-dtype nvfp4`) fails fast
at VllmConfig validation, before weight load:

```
nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention)
backends. Please use a different --kv-cache-dtype (e.g., 'fp8' or 'auto')
for MLA models such as DeepSeek.
```

No MLA-sparse backend in the image supports nvfp4 (SM90: auto/bf16/fp8/fp8_e4m3;
SM120: + fp8_ds_mla; TRTLLM-SM10: fp16/bf16/fp8; plain FlashInfer nvfp4 requires
the SM100 non-MLA trtllm-gen kernel). The custom `nvfp4_ds_mla` kernel work
required is specced in docs/NVFP4-KV-BUILD-SPEC.md but was never built. The
recipe is kept as the target lane for that build.

## TP2 unpinned-KV + CUDA-graph experiment: confirmed failure (2026-08-31)

`glm-5.3-flash-nvfp4-vllm-testing.yaml` (uncommitted at the time) dropped three
things from the validated recipe simultaneously: the `--kv-cache-memory
3221225472` pin, `--enforce-eager`, and bumped `gpu_memory_utilization`
0.85 -> 0.87 — betting that auto-sized KV would now be safe given everything
learned since (persistent_topk fix, driver-wall forensics).

**Result: fails cleanly at engine init, before weight-adjacent memory is
touched further.** Weights load fine (InstantTensor, 27.45 s). KV-cache
sizing then raises a plain `ValueError`: "available KV cache memory (1.27
GiB)" vs "2.13 GiB KV cache is needed" for max_model_len=262144 — i.e. gmu
0.87 without a pin leaves **less** than half the headroom the working 3 GiB
pin uses. sparkrun's own dry-run VRAM estimator predicted 14.6 GB available
at gmu 0.87 (681K-token capacity) — off by >10x from the real number; treat
that estimator as directional only on this cluster, not load-bearing.

This is a *different* failure signature from the documented silent NVRM
first-touch kill (worker `exit code: None`, no dmesg trace): here the engine
raises a real exception and exits with a message, before any KV slab is
pinned. The APIServer process then hangs post-exception with the container
still reporting `Up` in `docker ps`/`sparkrun status` — another confirmation
of the "don't trust liveness, probe `/health`" rule (see fleet_watchdog.sh).

Suspected primary cause: removing `--enforce-eager` re-enables CUDA graph
capture, which reserves activation memory *before* the KV-cache sizing step
— shrinking the pool available to the KV check independent of the gmu bump.
This matches an independent community finding (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
issue #47, EXL3/vLLM on the same hardware): full-depth CUDA graph capture at
1M context OOM'd even under an explicit KV-memory-bytes pin, and
`ENFORCE_EAGER=1` + explicit pin was their workaround. Follow-up test below
isolates this variable.

Not yet re-tested: whether `index_topk` on this image needs the same
`2044` cap that Reederey87/glm53-flash-exl3-2x-dgx-spark found necessary on
the same NoPE/sparse-attention kernel family (their fix for a 2048-wide
buffer overflow) — worth checking against our persistent_topk crash
(a5c4b19) as a separate experiment; not exercised by this test.

## TP2 CUDA-graph isolation test: PASSED, promoted to the shipped recipe (2026-08-31)

Follow-up to the failure above: held the validated 3 GiB pin + gmu 0.85
constant, changed only one variable — dropped `--enforce-eager`. This
isolates whether CUDA graph capture itself (as opposed to the missing pin)
was responsible for the earlier OOM.

**Result: full suite PASSED**, incl. the 250K-token prefill — the same
memory-pressure test the original enforce-eager config was validated against.

| Check | Result |
|---|---|
| Model load (InstantTensor) | clean, `Application startup complete` |
| Sanity (`probe_sanity.py`) | PASS — TTFT 0.31-0.33s, decode **25.1-26.3 tok/s** (vs 21.8 tok/s baseline with `--enforce-eager`, +~15-20%) |
| Soak (`probe_soak.py`, 2 rounds x 2 waves) | PASS — 10/10 + 10/10 sequential, 3/3 + 3/3 concurrent, endpoint alive after |
| Cache continuation | PASS — cold/partial/warm all retrieved planted code, no degenerate output |
| Long context 249,951 tokens | PASS — TTFT 195.3s, 4/4 planted codes retrieved |

Conclusion: the OOM in the experiment above was caused by dropping the
`--kv-cache-memory` pin (and/or the gmu 0.85->0.87 bump), not by CUDA graph
capture. With the pin held constant, CUDA graphs are safe and strictly
improve decode throughput. **`--enforce-eager` removed from the shipped TP2
recipe** (`glm-5.3-flash-nvfp4-vllm.yaml`); `glm-5.3-flash-nvfp4-vllm-testing.yaml`
retired now that its finding is folded into the main recipe.

## max_num_batched_tokens tuning: 2048 baseline OK, 3584 hits the memory wall (2026-08-31)

Prompted by an unverified third-party (Twitter) claim that vLLM's
`long_prefill_token_threshold` silently caps prefill chunks at 1792 tokens
regardless of `--max-num-batched-tokens`. **Checked directly against our
image's vLLM source (`0.1.dev20051+g487ecf187`) before touching config:**
`long_prefill_token_threshold` defaults to `0` (`vllm/config/scheduler.py:70`)
and the scheduler only applies it when `0 < threshold < num_new_tokens`
(`scheduler.py:526`,`907`) — at 0 it's a no-op. No `1792` literal exists
anywhere relevant in our image (the only hits are unrelated: H100 MoE
autotune JSON configs, an image-processor pixel constant, an LFM2 model
default). **The claimed bug does not apply to this build/recipe** — we had
also never explicitly set `--max-num-batched-tokens` (silently defaulting to
vLLM's own default of 2048), so this was a legitimate blind spot worth
testing on its own merits, independent of the debunked claim.

Made `max_num_batched_tokens` an explicit recipe default (2048, i.e. no
behavior change) and tested raising it:

| Value | 64K-token cold-prefill TTFT | Result |
|---|---|---|
| 2048 (baseline, now explicit) | 53.3 s (~1198 tok/s effective) | PASS |
| 3584 | — | **FAIL: silent worker death** |

At 3584, `Worker_TP0` produced its last log line at boot (kernel warmup,
graph capture finished, "Breakable CUDA graph enabled") and then **vanished
with zero traceback** ~75 s later, mid-request, during the 64K-token
prefill: `Worker proc VllmWorker-0 died unexpectedly (exit code: None)` —
the exact silent NVRM first-touch-kill signature from
docs/KV-HUNT-672K-TP2-RECORD.md and the 2026-08-31 unpinned-KV failure
above, this time triggered by a larger per-iteration prefill activation
footprint (`--max-num-batched-tokens`) rather than KV pool size or CUDA
graph capture itself. **`max_num_batched_tokens` stays at 2048 in the
shipped recipe** — this cluster's memory ceiling is sensitive to prefill
batch size too, not just `kv_cache_memory`/`gpu_memory_utilization`.

Caution for follow-up experiments: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
issue #47 reports that on 610.43.02-class drivers, memory freed by a killed
CUDA process can stay "trapped" until a full reboot (not released by
`nvidia-persistenced` restart), eroding the usable floor by ~0.5-1 GiB per
failed launch. Checked directly after this failure via `torch.cuda.mem_get_info()`
in a throwaway container on both nodes: 121-125 GB free of 130.66 GB total —
no sign of erosion, well clear of MiaAI's ~101-102 GB reported ceiling. (Our
driver/kernel combo, `580.173.02` / `6.17.0-1031-nvidia`, matches the
Canonical-packaged pairing MiaAI flagged as showing the OS/CUDA memory-
accounting gap, for reference if this resurfaces.)

## index_topk buffer-overflow check: not applicable, already fixed differently (2026-08-31)

Follow-up to Reederey87/glm53-flash-exl3-2x-dgx-spark's finding that
`index_topk=2048` "overflows the kernel's 2048-wide buffer arithmetic" with
their kpool-tail KV layout, fixed there via `--hf-overrides
'{"index_topk":2044}'`. Our model config matches their setup closely enough
to be worth checking directly: `index_topk=2048`, `index_kpool=4`,
`index_kpool_always_select_tail=True` (from the checkpoint's `config.json`).

**Checked against our own image source — the bug class doesn't apply to us,
for two independent reasons:**

1. **The persistent_topk crash we already fixed (a5c4b19) never reaches this
   path on GB10.** `docker/sparse_attn_indexer_kpool_sm121.py` gates
   `persistent_topk` on `multi_processor_count >= 78`; GB10 has 48 SMs, so
   every decode routes to `top_k_per_row_decode` instead, unconditionally.
2. **Our build's `topk_indices_buffer` is already sized with the tail
   margin Repo 1 was missing.** `vllm/models/glm5next/nvidia/model.py`
   allocates `buffer_width = topk_tokens + (index_kpool - 1)` (2048 + 3 =
   2051, rounded up to the next multiple of BLOCK_N=128 -> 2176 — this is
   the "2176-wide kpool index buffer" already referenced in this repo's
   README), not a bare 2048-wide buffer. `expand_pools_and_append_tail`
   writes `topk + pool_size - 1` columns and the buffer already has room for
   exactly that. Repo 1 was evidently on an older/different vLLM commit
   without this fix; ours already carries it.

No config change made or needed; `index_topk` stays at the model's default
(2048). Also note: forcing `index_topk=2044` would be actively wrong on the
`FLASHINFER_MLA_SPARSE_SM120` backend, which hard-asserts
`index_topk == 2048` (`flashinfer_mla_sparse.py:227`) — irrelevant to us
since we run `FLASHINFER_MLA_SPARSE_SM90` on GB10, but worth knowing if this
ever comes up again for a different backend/hardware combo.

## Instrumentation: cheap observability flags shipped (2026-08-31)

Added `--enable-mfu-metrics`, `--kv-cache-metrics`, `--cudagraph-metrics`,
`--enable-logging-iteration-details` to the shipped TP2 recipe — all
documented as low/no-overhead in vLLM's own config and safe alongside the
CUDA-graph default (unlike `--enable-layerwise-nvtx-tracing`, which explicitly
isn't). Boots clean, sanity + soak pass with no observable perf change.

Confirmed live via `/metrics` and container logs:
- `vllm:estimated_flops_per_gpu_total` / `_created` (MFU building blocks) —
  present.
- `vllm:cache_config_info` and related KV series — present (pre-existing +
  `--kv-cache-metrics` sampling).
- Per-iteration log lines (`Iteration(N): ... context requests ... generation
  tokens, iteration elapsed time: Xms, GPU KV cache usage: Y%`) — present,
  one per engine step.

Not yet confirmed: a distinct `cudagraph`-named metric series never appeared
in `/metrics` after boot, sanity, or a full soak pass — no error either, so
it may just need a trigger condition (a specific dispatch-mode transition?)
not yet exercised. Flag stays on; revisit if it matters later.

## Built-in torch profiler: reproducibly crashes the worker on this stack (2026-08-31)

Attempted a detailed per-op/kernel timeline via vLLM's built-in torch
profiler (a temporary `--enforce-eager` + `--profiler-config.*` launch,
`recipes/glm-5.3-flash-nvfp4-vllm-profiling.yaml`, deleted after this
investigation — not archived, not shipped). Correction along the way: the
env var `VLLM_TORCH_PROFILER_DIR` does NOT exist in this build ("Unknown
vLLM environment variable" warning) — this is a newer vLLM with a
`--profiler-config.*` CLI group instead (`profiler=torch`,
`torch_profiler_dir=...`, plus `record_shapes`/`with_flops`/schedule
knobs — see `vllm/config/profiler.py` in the image).

**Three attempts, three different crashes, all at the same call site**
(`POST /stop_profile`, worker process, right after `profiler_stop`):

1. `record_shapes=true, with_flops=true`, unbounded (~200 decode
   iterations recorded): worker died with the familiar silent NVRM
   signature (`exit code: None`, no traceback) during trace export/gzip.
2. Dropped record_shapes/with_flops, bounded the capture via the
   profiler's own schedule (`warmup_iterations=2, active_iterations=15`):
   different failure — a genuine **segfault** in torch's `PythonTracer`
   destructor (`~PythonTracer()` -> `stop()` -> GIL acquisition on an
   already-torn-down thread) when the schedule auto-completes under
   vLLM's multi-process (`mp`) executor. Not a resource issue.
3. Back to plain manual start/stop, no schedule, no record_shapes/
   with_flops, trivial 40-token request (smallest possible capture):
   **same silent NVRM-style worker death as attempt 1**, despite the
   tiny trace size — ruling out "trace too big" as the root cause.

Each attempt: only a lightweight `*.async_llm.*.pt.trace.json.gz` survived
(the API-server/front-end Python-call-stack trace — `cat: python_function`
only, zero CUDA/kernel events, ~95K events of mostly idle thread-wait
noise; the one real signal in it is the request-handling call chain itself
costing ~85ms, negligible next to GPU time). The actual **worker-side GPU
kernel trace never wrote once, in any of the three attempts** — the crash
happens before or during that export every time. Verified after each crash
via `torch.cuda.mem_get_info()` on both nodes: no trapped/eroded memory
(121-126 GB free of 130.66 GB each time), so this isn't compounding damage
from repeated failed launches either — it's a real, reproducible bug in
this build's torch-profiler-plus-mp-executor path, independent of capture
size or settings.

**Conclusion: vLLM's built-in torch profiler is not currently usable on
this stack for kernel-level instrumentation.** Options not yet tried:
`--profiler-config.profiler=cuda` (the plain CUDA profiler mode, a
different code path from `torch`) as a lower-risk alternative; or a real
Dockerfile addition of `nsys` into the image (present on the host, not in
`sm121-v9`) to profile from outside the crashy in-process mechanism
entirely. `--enable-layerwise-nvtx-tracing` (incompatible with CUDA
graphs, would need the same `--enforce-eager` temporary launch) hasn't
been tried and is independent of this profiler subsystem — worth a shot
on its own before writing off per-layer visibility entirely.

The cheap, always-on metrics shipped separately above
(`--enable-mfu-metrics`, `--kv-cache-metrics`, `--cudagraph-metrics`,
`--enable-logging-iteration-details`) are unaffected by any of this —
those come from vLLM's ordinary Prometheus/logging path, not the profiler
subsystem, and remain the working instrumentation baseline.

## nsys in-image: also crashes, conclusively ruling out per-op GPU profiling on this cluster (2026-08-31)

Built `nsys` (Nsight Systems 2026.1.3.425, arm64) into a new image layer —
`docker/Dockerfile.glm53-sm121-v10`, `FROM sm121-v9`, installed via the CUDA
apt repo already trusted by the base image (adding `cuda-keyring` on top
conflicts with it — reused the existing source as-is). Built, verified
(`nsys --version`), distributed to both nodes (`docker save | ssh ... docker
load`). This sidesteps vLLM's in-process profiler entirely — nsys traces
from outside, no dependency on the crashy code paths above.

**Two attempts, two more crashes — same silent NVRM-style signature, zero
traceback, at two completely different points:**

1. `nsys profile --capture-range=cudaProfilerApi` paired with vLLM's
   `--profiler-config.profiler=cuda` (bare `cudaProfilerStart`/`Stop`
   markers — no Python tracer object, structurally can't hit the segfault
   class above). Booted clean, served fine, but calling `POST
   /start_profile` (which triggers `cudaProfilerStart()` in the worker)
   killed the worker instantly — `Worker_TP0`'s last log line is ~44s
   before the death, zero traceback in between.

   First attempt at this also surfaced a real, separate, easily-fixed
   issue worth keeping: `nsys`'s own CUPTI instrumentation costs ~19 GiB of
   GPU memory before vLLM even starts (102.76/121.69 GiB free vs. the
   103.44 GiB the coarse `gpu_memory_utilization=0.85` admission check
   wants) — lowering gmu to 0.78 for nsys-wrapped launches clears it.

2. Plain `nsys profile` with **no** capture-range and **no**
   `--profiler-config.*` at all (unconditional capture from process
   start, meant to be stopped externally via `SIGINT` after sending a
   request — zero interaction with any vLLM profiler-activation code).
   Never even reached ready: died mid-boot, inside
   `determine_available_memory()` (vLLM's own peak-activation memory
   profiler, which runs a dummy forward pass during warmup) — 13s after
   the last TileLang compile log line, zero traceback.

**Five consecutive crashes now, across two fundamentally different
profiling mechanisms** (vLLM's built-in torch profiler x3, nsys x2),
**triggered at different call sites** (`stop_profile` export,
`cudaProfilerStart()`, a plain memory-profiling step with no profiler API
involved at all) — the common thread is not a specific API bug, it's GPU/
CUDA instrumentation overhead itself (CUPTI buffers, driver-level trace
hooks) landing on top of GB10's already-thin memory margin on this
cluster (see docs/GB10-KV-MEMORY-LADDER.md, docs/SM121-CRASH-FORENSICS-
2026-08-27.md) and tipping it over. Verified after every crash via
`torch.cuda.mem_get_info()` on both nodes: consistently clean, no trapped/
eroded memory (123-126 GB free of 130.66 GB each time) — so this isn't
compounding damage across attempts, it's the same wall, hit five different
ways.

**Conclusion: live, in-cluster GPU kernel-level profiling (vLLM's built-in
profiler or nsys, in-process or wrapping) is not currently viable on this
2-node cluster.** Not a config problem to keep iterating on. If per-kernel
visibility is needed later, more promising directions: an isolated
single-kernel microbenchmark harness (bypassing the full serving pipeline
and its memory footprint entirely — closer to `probes/bench_glm53.py`'s
shape than to live profiling) or dropping `kv_cache_memory` much lower / a
much smaller `max_model_len` specifically for a profiling boot to buy nsys
more headroom (untried — the two attempts here both used the shipped 3 GiB
pin and 262144 max_model_len; a purpose-built minimal-footprint profiling
config, rather than the production recipe plus nsys bolted on, might have
enough margin left).

The nsys image layer (`sm121-v10`) and its Dockerfile are kept — the build
process itself worked cleanly and may be useful again with a
lower-footprint launch config. The disposable profiling recipe used for
this investigation is deleted, same as the torch-profiler one above.

## MoE backend investigation: why Marlin, and what else was tried (2026-08-31)

Prompted by `gb10-kernel-bench`'s finding (https://github.com/ozskywalker/
gb10-kernel-bench) that the Marlin NVFP4 MoE GEMM costs 30-50x more than
attention or MHC at the same batch size — the likely dominant cost in real
decode. Investigated whether a faster backend is available.

**Why Marlin wins (source-read, `vllm/model_executor/layers/fused_moe/
oracle/nvfp4.py`, no launch needed):** the oracle tries backends in a fixed
priority order and picks the first whose `is_supported_config()` passes.
`FLASHINFER_TRTLLM` and `FLASHINFER_CUTEDSL` hard-require
`is_device_capability_family(100)` — genuine SM100-class Blackwell
(B100/B200 datacenter), not GB10's SM121. `FLASHINFER_CUTLASS`'s NVFP4
path requires **both** weight and activation to be NVFP4-quantized
(`(kNvfp4Static, kNvfp4Dynamic)`, i.e. w4a4) — our checkpoint
(LibertAIDAI/GLM-5.3-Flash-NVFP4) is weight-only NVFP4 (w4a16, activations
stay bf16), so this backend is never reached regardless of hardware.
Marlin is the first backend in the list whose scheme check actually
matches w4a16 NVFP4. **This is a checkpoint-quantization-scheme
constraint, not a fixable/overly-conservative capability gate** — unlike
several past findings in this project (persistent_topk's SM-count gate,
the PDL capability≥9 gate), there's no obvious wrong check to relax here.

**Two backends do support our exact scheme and pass their device checks
on this hardware** (confirmed via direct `_supports_current_device()`
calls, no launch needed): `HUMMING` (`(kNvfp4Static, None)`,
`has_device_capability((7,5))`) and `flashinfer_b12x`
(`(kNvfp4Static, None)`, `is_device_capability_family(120)`) — both never
get reached in auto-selection because Marlin sits earlier in the priority
list and already succeeds. `flashinfer_b12x` is additionally excluded
from auto-selection entirely by an explicit code comment: "intentionally
excluded... until the upstream CUTLASS SM121 MMA op guard is resolved."
Notably, `flashinfer_b12x` (the `ghcr.io/spark-arena/dgx-vllm-eugr-
nightly-b12x` image) is already used in production on this same fleet for
the DeepSeek-V4-Flash recipes archived in `~/ai/recipes`.

**Tested `-o moe_backend=humming` directly on the real recipe (real
weights, not synthetic) — crashed twice, same signature both times:**

1. Default gmu (0.85): booted, loaded main weights fine (27.65s), then
   during the second (MoE-conversion) weight pass hit `RuntimeWarning:
   Shrink io_depth from 256 to 186 due to memory limit`, slowed ~20x
   (349 MB/s vs. the usual 6-9 GB/s), then the worker died silently
   (`exit code: None`) at ~1% into that pass.
2. Lowered `gpu_memory_utilization` to 0.78 (the fix that worked for
   nsys's overhead earlier) — same `io_depth` warning (shrunk to 203 this
   time, slightly better), same slow pass, same silent death at ~1%.

Both crashes verified clean afterward via `torch.cuda.mem_get_info()` on
both nodes (125+ GB free of 130.66 GB) — not trapped/compounding memory,
a real per-attempt memory-pressure failure in HUMMING's weight-conversion
path specifically. **Conclusion: HUMMING is not viable on this cluster
without a deeper fix to its loading-time memory footprint** — this isn't
a quick gmu tweak away, unlike the earlier CUDA-graph/nsys cases.

**Tested `-o moe_backend=flashinfer_b12x` directly — cleanly rejected,
not a crash:**

```
ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp. Use
'flashinfer_trtllm', 'flashinfer_cutlass', 'flashinfer_cutedsl', 'cutlass',
'marlin', or 'humming' instead.
```

GLM-5.3-Flash's config sets `swiglu_limit=10.0` (a SwiGLU activation clamp
for numerical stability); `flashinfer_b12x`'s kernel doesn't implement
that clamp, and vLLM's own oracle (`NVFP4_BACKENDS_WITH_CLAMP` in
`oracle/nvfp4.py`) correctly refuses the combination rather than risk
silently-wrong output. This is a **second, independent reason** B12X is
unusable here, on top of the "upstream CUTLASS SM121 MMA op guard"
exclusion noted earlier — not something fixable by a flag; it would need
a real kernel change (implementing the clamp) to ever work for this
specific model, even though B12X works fine for DeepSeek-V4-Flash (no
swiglu_limit) on this same fleet.

**Conclusion: both untried alternatives are genuine dead ends for THIS
model**, for two different, well-understood reasons — HUMMING crashes
from memory pressure in its weight-loading path, B12X is correctly
blocked by a real architecture-compatibility check. Marlin is the only
NVFP4 w4a16 MoE backend that actually works for GLM-5.3-Flash on this
image. Verified clean afterward via `torch.cuda.mem_get_info()` (no
trapped memory). Further MoE speedup on this exact model/hardware/quant
combo would need either a genuinely different checkpoint (full w4a4
NVFP4, unlocking FLASHINFER_CUTLASS) or a kernel-level fix to one of the
rejected backends — not a recipe-level config change.

## Checkpoint swap: LibertAIDAI -> RedHatAI (correctness fix, 2026-08-31)

Prompted by tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark's README, which
documents a reproducible token-corruption bug in ModelOpt-quantized NVFP4
builds of this model (vLLM #54150): intermittent corrupted token IDs,
invisible in English, surfacing as U+FFFD inside CJK/emoji output. Measured
by tonyliu312/GLM-5.3-Flash-DFlash2-TP4-1M-Context against real gateway
traffic: 16/1997 requests (0.80%) corrupted, and reproduced directly
against `LibertAIDAI/...NVFP4` (our previous checkpoint) 4/4 runs on
Korean text, 0/4 on `RedHatAI/GLM-5.3-Flash-NVFP4` (a compressed-tensors
conversion of the same base model).

Checked the actual published `quantization_config` before assuming
anything from the README: **RedHatAI's checkpoint is weight-only NVFP4
too** (`input_activations: None` in both quant groups) — contrary to the
README's claim that it's W4A4, this does NOT unlock `FLASHINFER_CUTLASS`
for MoE (see the MoE backend investigation section above). One wrinkle:
layer 45's experts are 8-bit weight-only (`float-quantized`) rather than
4-bit — every other layer is uniform NVFP4. Confirmed at boot: a *second*,
separate `Using MARLIN Fp8 MoE backend` selection fires for that one
layer, alongside the usual NVFP4 Marlin selection for the other 41.

Downloaded via `hf download RedHatAI/GLM-5.3-Flash-NVFP4` (~185 GiB, 21
files) to the head node, synced to the worker over the internal fabric
(`rsync`, faster than a second independent download). Tested via a
disposable `-o`-free copy of the shipped recipe with only `model_path`
changed — everything else identical (Marlin, fp8 KV, MTP-4, CUDA graphs,
3 GiB pin, the new metrics flags).

**Result: booted clean, full suite passed, zero corruption reproduced:**

- Boot: weights loaded in 30.66s (first pass) + 26.84s (second/MTP pass,
  now routes through a distinct Fp8 Marlin path for layer 45) — comparable
  to the previous checkpoint, no regression.
- MoE backend: still `MARLIN` for the bulk of layers, as expected (scheme
  unchanged) — `Using 'MARLIN' NvFp4 MoE backend out of potential
  backends: [...]` logged identically to before.
- KV pool: 365,621 tokens — matches the previous checkpoint's figure
  exactly (same architecture, same config).
- `probe_sanity.py`, `probe_soak.py` (2 rounds x 2 waves),
  `probe_cache_continuation.py`: all PASSED.
- **Corruption check** (reproducing tonyliu312's methodology): 4
  generations at `temperature=0` — Korean prose, Traditional Chinese with
  emoji, an emoji-definition list, a Japanese haiku — **zero U+FFFD**
  across all four, vs. the documented reproduction on the old checkpoint.

**Promoted to the shipped recipe.** `model_path` now points at
`RedHatAI/GLM-5.3-Flash-NVFP4`'s snapshot; header comment updated with the
full rationale. The disposable test recipe is deleted. Weights are on both
nodes at `/models/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/
36c184c6cda000a481711306df5adde42f63321a` — the old LibertAIDAI weights
are left in place (not deleted) in case of rollback.

## RedHatAI checkpoint performance baseline (2026-08-31)

The correctness validation above only ran the 3-sample short bench built
into `probe_sanity.py` — not enough for a real performance comparison
before chasing DFlash2. Ran a proper capture on the shipped recipe
(RedHatAI checkpoint, CUDA graphs, 3 GiB pin, current metrics flags) to
serve as the baseline DFlash2 gets compared against:

- **Short bench** (`probe_sanity.py`, same methodology as every prior
  CUDA-graph/metrics measurement in this doc): 21.13 / 24.61 / 21.56 tok/s
  (median 21.56), TTFT 0.25-0.32s. Matches the existing CUDA-graph baseline
  (21.8-26.3 tok/s across prior measurements) — **no regression from the
  checkpoint swap**, within normal run-to-run noise.
- **Longer-form decode** (400 max_tokens, prose explanation prompt, 8 runs,
  `stream_options.include_usage` for real token counts — not previously
  measured at this length on either checkpoint): median **19.40 tok/s**
  (range 17.88-20.60), median TTFT 0.393s (range 0.28-0.40s). Lower than
  the short-bench number, consistent with MTP's known content-dependent
  acceptance (prose drafts worse than structured/code content per this
  project's own README and the community DFlash2 writeups) — not a
  checkpoint regression, a different (and more realistic multi-turn-length)
  workload. **This is the number to compare DFlash2 against**, since
  DFlash2's own published benchmarks are likewise measured on
  code/structured prompts at comparable length, not a 10-token sanity
  check.
- **Long-context** (`probe_longctx.py`, 250K tokens, same exact test as the
  original validation): TTFT **198.2s**, 4/4 planted codes retrieved — vs.
  195.3s on the old checkpoint. Within 1.5%, no meaningful prefill
  regression.
- **Soak** (2 rounds x 2 waves): all PASSED, timing comparable to prior
  runs (sequential median 2.1-2.2s, concurrent x3 median 3.96-12.4s).
- `vllm:estimated_flops_per_gpu_total` (the new `--enable-mfu-metrics`
  counter): 2.68e15 by end of this session's traffic — a real number now
  exists to compute MFU against once there's a clean way to pair it with
  wall-clock GPU-active time; not yet turned into an actual MFU percentage.

Not yet re-validated: the persistent_topk SM-count gate (a5c4b19) was proven
out before `--enforce-eager` was dropped (2026-08-31, above). CUDA graph
capture uses static shapes and could in principle route decode differently;
worth a dedicated >24K-context decode soak under the current CUDA-graph
config to confirm the gate still holds, since that's a different question
than the one this section answers.

## Reproduce

```bash
./scripts/prelaunch_flush.sh 10.7.0.87,127.0.0.1 --during-load
sparkrun run glm-5.3-flash-nvfp4-vllm.yaml          # uses default cluster
python3 probes/probe_sanity.py --base-url http://10.7.0.87:8000
python3 probes/probe_soak.py --base-url http://10.7.0.87:8000 --rounds 2 --waves 2
python3 probes/probe_cache_continuation.py --base-url http://10.7.0.87:8000
python3 probes/probe_longctx.py --base-url http://10.7.0.87:8000 --tokens 250000
```

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

# EXL3 v2: validating four new upstream PRs (2026-09-01)

MiaAI-Lab merged four PRs today (all `main`@`c190db1`) after v1 was
already validated and in production use: PR77 (fat-expert prefill CUDA
kernels), PR86 (indexer workspace rightsizing), PR63 (prefix-cache
thinking-toggle fix), PR96 (spinwait tuning) — see each PR's own title,
body, and diff (read directly via `gh api`, not the social-media summary
that prompted this investigation) for full technical detail. A fifth
claim ("fairer scheduling") maps to `GLM53_MIXED_PREFILL_CHUNK=skip`,
already set in v1 — not a new change.

**None of the four had reached the published `ghcr.io/miaai-lab/...:exl3`
image tag** — confirmed empirically by pulling it fresh and grepping the
baked-in `chat_template.jinja` for PR63's fix (absent). Built our own
image from a fresh clone at `main`@`c190db1`
(`docker build -t glm53-exl3-v2:c190db1 .`) rather than waiting on a
republish. New recipe: `glm-5.3-flash-exl3-v2-vllm.yaml`.

## Incident: the build itself caused a production outage

Ran the ~7-minute CUDA-extension compile directly on `spark-276f`
(10.7.0.142) — one of the two nodes actively serving the user's live v1
deployment. The build's CPU/memory load coincided almost exactly with
`shm_broadcast.py`'s "No available shared memory broadcast block found in
60 seconds" warnings appearing on the live server, and generation requests
began timing out (confirmed via direct `curl`, 15-20s with no response,
`/health` still returning 200 the whole time — a genuine engine hang, not
a crash, and not detectable via health checks alone). Load average had
already dropped back to normal (0.29) by the time this was diagnosed,
confirming it wasn't ongoing contention resolving on its own — the engine
needed an actual restart. Stopped, flushed, relaunched v1; verified
recovery with a real generation request (not just `/health`) before
declaring it fixed.

**Lesson for any future build/compile work on this cluster**: never run
a heavy build on a node that's also carrying live production traffic,
even when the build itself doesn't touch the GPU — CPU/memory contention
alone can hang a co-located vLLM worker. Coordinate build windows with
whoever is depending on the live service.

## Boot: four attempts before a clean serve

1. **gmu=0.87** (matching v1): failed at the startup free-memory check —
   `Free memory on device cuda:0 (104.93/121.69 GiB) ... less than desired
   ... (0.87, 105.87 GiB)`. Real, not page-cache (persisted through a fresh
   `prelaunch_flush.sh`).
2. **gmu=0.87, retry after fresh flush**: failed again, `105.31/121.69
   GiB` — closer but still short. Confirms this is genuine, current
   headroom on this node, not a caching artifact. (Both this and the
   previous v1 baseline run below hit the identical failure class at 0.87
   — see the "second incident" note.)
3. **gmu=0.86** (MiaAI-Lab's own documented fallback for tighter-margin
   nodes — not this project's usual 0.85, since v1's earlier 0.85 failures
   were a *different* problem, post-weight-load KV sizing, not this
   startup check): passed the memory check, booted through weight load and
   graph capture, then died with `KeyboardInterrupt` inside vLLM's own
   init-timeout watchdog, specifically during the API server's
   video-processor warmup (`glm5next.py` video-patch reshape) — unrelated
   to any of the four PRs (none touch multimodal code).
4. **gmu=0.86, `--language-model-only`** (vision/video tower off — not
   needed for this validation, and the crashing code path): **booted
   clean**. Verified with a real chat completion (not just `/health`),
   `probe_sanity.py` and `probe_soak.py` both fully passed.

## Second incident: v1 hit the identical 0.87 startup failure too

While re-launching v1 for a same-day prefill baseline, it hit the exact
same `Free memory ... 105.75/121.69 GiB ... less than 105.87 GiB` failure
at its own established gmu=0.87 — a config that had booted cleanly
multiple times earlier the same day. The free-memory figure crept upward
across each of these three consecutive near-misses (104.93 -> 105.31 ->
105.75) but never quite cleared 105.87. This is consistent with
MiaAI-Lab's and Reederey87's own documented GB10/driver finding: memory
freed by a killed CUDA process can stay "trapped" until a full reboot,
eroding the usable ceiling by a fraction of a GiB per failed launch — very
plausibly compounding from today's repeated crashed/killed launches
(the v2 boot attempts, the hang-recovery cycle). Worked around by
overriding to gmu=0.86 for this run (`-o gpu_memory_utilization=0.86`,
not a permanent recipe change). **If boots keep landing short of 0.87
after this session, a full node reboot — not just cache flushing — may be
the real fix**, per that documented driver behavior.

## Results: three of four claims independently confirmed, one inconclusive

**PR86 (indexer workspace rightsizing) — confirmed, exact formula match.**
Boot log: stock = 10,485,760 entries (= `262144*40`, matches the documented
formula exactly); rightsized = 262,148 entries (=
`min(4,2048) * cdiv(262144+2, 4)`, matches exactly); **~1287 MiB reclaimed**
at our 262,144 context. Their own headline (~4.5-4.8 GiB) was measured at
1,000,000 context — the workspace scales with `max_model_len`, so this
was never going to transfer 1:1; scaling their figure by our context
ratio (262144/1000000 = 26.2%) predicts ~1320 MiB, matching the observed
1287 MiB closely. The mechanism is real and behaves exactly as documented.

**PR63 (prefix-cache thinking-toggle fix) — confirmed clean.** ~16.2K-token
system+tools+filler prompt (matching PR63's own test shape): cold 14.67s,
warm same-effort repeat 4.79s (real ~3x cache benefit), **toggle
thinking=False 4.78s** — indistinguishable from the warm repeat, not
falling back to a cold-like re-prefill. Toggling thinking no longer costs
a full re-prefill on this deployment.

**PR77 (fat-expert prefill kernel) — confirmed, real gain, smaller than
their own measurement.** Same-day A/B, `probe_longctx.py`, both under
identical config otherwise (context, speculator, gmu overridden to 0.86 on
both arms for a fair comparison):

| Context | v1 (stock fat-expert path) | v2 (E2 kernel) | Gain |
|---|---:|---:|---:|
| ~100K tokens | 816.6 tok/s (122.4s TTFT) | 933.3 tok/s (107.1s TTFT) | **+14.3%** |
| ~250K tokens | 920.5 tok/s (271.5s TTFT) | 1048.4 tok/s (238.4s TTFT) | **+13.9%** |

Both correctness-passed (4/4 planted retrieval codes at both lengths, both
versions). Consistent ~14% gain at two context lengths — a real, robust
effect, not noise — below their own reported +20-21%, expected given the
PR author's own caveat that the effect is "geometry-sensitive" and had
already failed to reproduce on one other deployment. This deployment
reproduces a smaller but still substantial version of the claimed effect.

**PR96 (spinwait tuning) — inconclusive at this measurement's resolution,
no regression.** `docker stats` CPU sampling during matched 400-token
decode bursts: v1 (stock) ~107-122% (avg ~111%), v2 (`GLM53_SPINWAIT_MS=16`)
~107-110% (avg ~108%) — a small, directionally-consistent reduction, not
the claimed 85.3%. This doesn't falsify their claim: `docker stats`
measures whole-container CPU (GPU worker's real compute dominates during
active decode), while their claim specifically targets EngineCore's idle
spin-wait threads, which matter most during IPC gaps between requests, not
a continuously-active single-stream decode burst. Properly validating the
85.3% figure would need process-level profiling (py-spy/psutil targeting
the EngineCore PID specifically) under bursty/gappy traffic — out of scope
given this project's own established finding that live profiling tooling
is unreliable on this cluster. Decode throughput itself showed no
regression (24.07-26.26 tok/s short-bench, comparable to v1's 26.75).

## Decision: v2 promoted as the new default

Real, meaningful, independently-confirmed gains (prefill +14%, prefix-cache
fix eliminates a real re-prefill cost, KV capacity mechanism confirmed
correct) with zero observed regressions (decode throughput comparable,
full sanity+soak suites passed, no new crash classes). **v2 replaces v1 as
the deployed default.** Operational note: `glm53-exl3-v2:c190db1` is a
custom-built image, not in any registry — only exists on this cluster's
two nodes (built on spark-276f, synced to the head via sparkrun's own
resource-distribution step, confirmed working for a locally-built,
non-registry image). If this cluster is ever rebuilt from scratch, the
image needs rebuilding from the same commit
(`git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
&& git checkout c190db1 && docker build ...`), not just re-pulled.
`gpu_memory_utilization` shipped at 0.86 (not 0.87) per the memory-ceiling
findings above — revisit upward only with fresh headroom confirmed (a
node reboot may be warranted first, given the "trapped memory" pattern
observed today).

## Vision retest: the multimodal-warmup crash didn't reproduce

Initially shipped v2 with `--language-model-only` (vision/video tower off)
after one boot attempt died with a `KeyboardInterrupt` during the API
server's multi-modal renderer warmup. Traced the actual mechanism: the
exception comes from `api_server.py`'s own `_interrupt_init` —
`signal.signal(signal.SIGTERM, _interrupt_init)`, installed specifically
to convert an *external* SIGTERM arriving before uvicorn's own handlers
are installed into a clean `KeyboardInterrupt`. Not an internal vLLM
timeout. Checked for an external sender: no Docker `HEALTHCHECK` defined
on the image, no custom `StopSignal`/`StopTimeout`, and no `sparkrun
stop`/kill command was issued during that boot window. Source never
conclusively identified.

Given no reproducible mechanism was found, retested the identical config
(gmu=0.86, MTP k=2, vision **on**) cleanly: weights loaded (82.38 GiB),
graph capture (5s, 0.33 GiB), engine init (44.93s), then **`Multi-modal
warmup completed in 13.978s`** and **`Readonly multi-modal warmup
completed in 3.478s`** — both phases that crashed before now complete
without incident. Full `probe_sanity.py` (TTFT 0.227-0.33s, decode
23.55-25.09 tok/s median 24.12) and `probe_soak.py` (10/10 + 10/10
sequential, 3/3 + 3/3 concurrent) passed.

**Conclusion at the time: treated as a one-off.** Restored vision/video
(`--skip-mm-profiling --limit-mm-per-prompt '{"image":4,"video":1}'`,
matching v1's capability) as v2's shipped config.

**Correction (2026-09-01, later the same day): this is real and
intermittent, not a one-off.** The identical `KeyboardInterrupt` during
multi-modal renderer warmup recurred a third time, on a completely fresh
boot after both cluster nodes rebooted (clean memory, `uptime` ~3 min on
both) — ruling out memory pressure or accumulated cruft as the cause. A
second retry (same config, no changes) booted clean again, matching the
original pattern: sometimes crashes, sometimes doesn't, no config
difference between attempts. Genuinely intermittent, not reproducible
on-demand, and the external SIGTERM's actual sender still hasn't been
identified. Current stance: keep vision on (2/3 attempts across this
session's history succeeded), retry once on a boot failure at this exact
signature before considering `--language-model-only` again. If this
recurs, capture the exact SIGTERM sender (host process list / `strace -f`
on the container's
PID 1) before reaching for `--language-model-only` again as a workaround.

## Final decision: v2 (vision on) promoted as the default

Copied to `~/ai/recipes/glm-5.3-flash-exl3-tp2.yaml`, replacing v1's
config in place (same filename — anyone already launching this recipe
gets the improvements automatically). No capability regression, no
performance regression, three of four PR claims independently confirmed
with real measured gains, the fourth neither confirmed nor disproven.
Live deployment left running on this exact validated config.

## v3 attempt: PR80 "mixed-prefill gate v2" — built, deployed, negative result (2026-09-01)

Follow-up investigation after the user asked whether anything else from
the community (specifically Reederey87/glm53-flash-exl3-2x-dgx-spark's
v1.3.2 release and MiaAI-Lab's own recent activity) had been missed.
Found two things: v1.3.2's fat-expert-kernel adoption is the same PR77
already validated above (a second independent same-hardware-class
confirmation: their own numbers, +11-18% across three context lengths,
line up with this project's +14.3%/+13.9%); and PR80 "Mixed-prefill gate
v2" (MiaAI-Lab, still **open/unmerged** against `main` at validation
time, `mergeable: false`) — a real fix, independently validated by both
MiaAI-Lab and Reederey87, for a genuine bottleneck the shipped v2 config
still has: `GLM53_MIXED_PREFILL_CHUNK=skip` holds a fully-cached follow-up
turn for the *entire* duration of any concurrently-running generation
(their own receipts: 15-17s TTFT for what should be a near-instant cached
hit). Gate v2's claimed fix: a request whose uncached remainder fits one
hybrid block bypasses the hold entirely (their receipts: 15-17s → 1.4-1.6s).

### Build: clean, resolved by hand

PR80's branch conflicted against current `main`@`c190db1` (base was an
older commit, `493cb88`, predating PR77/86/63/96). Fetched
`refs/pull/80/head`, merged locally, resolved four conflicted files —
`.env.example`, `README.md`, `start.sh`, `tests/test_start_overrides.py`
— all pure two-sided additions except one real value conflict in
`README.md`'s documented `MAX_NUM_BATCHED_TOKENS` default (`main` had
already moved it to 7168 post-PR77, an upstream change this project
hadn't picked up yet — worth its own follow-up test; resolved in favor of
`main`'s newer value, not PR80's stale 2048). The core scheduler overlay
(`overlay/patch_scheduler_decode_floor.py`) merged with **zero conflicts**
— gate v2's actual logic is additive to the existing decode-floor patch.
Rebuilt (`docker build -t glm53-exl3-v2:c190db1-gate2 .`): 24 seconds total
— the CUDA extension compile fully cache-hit (gate v2 touches no
exllamav3 code, pure Python scheduler overlay), confirming this was a
cheap, low-risk rebuild. Applied this session's hard-won lesson: built
with `nice -n 15`, actively verified the live deployment's health with
real generation requests (not just `/health`) throughout, rather than
fire-and-forget like the incident earlier today. No repeat incident.

Verified the gate v2 markers (`[glm53-decode-floor-v2]`) were genuinely
present in the built image's `vllm/v1/core/sched/scheduler.py` before
deploying, not just trusting the build log.

### Boot: two memory-ceiling failures, same "trapped memory" pattern as earlier today

First attempt at `gpu_memory_utilization=0.86` (matching the shipped v2
value) failed at the identical startup free-memory check hit repeatedly
earlier today: `102.6/121.69 GiB free ... less than desired ... 104.65
GiB`. This confirms the "trapped memory" erosion documented earlier
(105.75 → 105.31 → 104.93 → 102.6 GiB free across today's cumulative
crashed/killed launches) is real and ongoing, not resolved by cache
flushing alone. Dialed to `gpu_memory_utilization=0.82` for real margin
(99.8 GiB budget vs. 102.6 GiB free, not another razor-thin step) — booted
clean. **If this keeps eroding, a full node reboot is the actual fix, not
further gmu reduction** — see `~/ai/research/glm-5.3-flash-gb10/README.md`.

### Result: config genuinely active, claimed benefit did NOT reproduce

`probe_sanity.py` and `probe_soak.py` both passed cleanly (decode median
25.1 tok/s, consistent with every prior v2 measurement — no regression).
Confirmed via a live request that `_glm53_gate_config()` logged the
correct values: `{'warm_tokens': 3584, 'max_wait_ms': 1500, 'late_cap':
512}`.

Reproduced PR80's own test shape directly: primed a ~16K-token cached
prefix, started a 300-token generation on the identical prompt (so it
decodes with a warm cache), then fired a **second identical-prompt
request (max_tokens=1)** while the first was actively decoding — this
should trigger the warm-bypass path immediately (`remaining` computed
against the cached prefix should be near-zero, far under the 3584-token
threshold).

| Attempt | Delay before follow-up | Follow-up TTFT | Concurrent generation's own duration |
|---|---:|---:|---:|
| 1 | 0.8s | 14.09s | 15.84s |
| 2 | 1.5s | 8.73s | 11.53s |

Both runs: the follow-up's wait time scales **proportionally with the
concurrent generation's remaining duration** (89% and 76% respectively) —
exactly the signature of the *old* `skip` hold-until-done behavior, not a
bypass. A control test (sequential warm-cache hit, no concurrent decode)
confirmed the base prefix-cache mechanism itself is intact and unaffected
by gate v2: 4.75-4.82s, matching pre-gate-v2 measurements exactly (4.78-
4.79s) — so this is specifically a gate-timing issue, not a caching
regression.

Read the gate function's actual logic directly from the built image
(`_glm53_mixed_prefill_gate`) to rule out a bad merge: the warm-bypass
check (`remaining <= cfg["warm_tokens"]: return None`) runs *before* the
mixed-prefill policy check and looks structurally correct. The most
likely explanation, not yet confirmed: **PR80 was validated by both its
authors against DFlash2** (Reederey87's and MiaAI-Lab's own receipts both
reference DFlash2-shaped workloads); this deployment runs **MTP k=2**.
Something about how `num_computed_tokens` is populated at the scheduling
point this gate reads from may differ meaningfully between the two
speculators (MTP's simpler single-layer draft vs. DFlash2's aux-capture/
mHC pipeline) in a way that defeats the "remaining uncached prefill"
computation specifically under MTP. Not confirmed — would need scheduler-
level instrumentation this project has already established is unreliable
on this cluster (see the profiling-crashes findings earlier in this
document) to pin down conclusively.

### Decision: reverted, not promoted

Reverted the live deployment to the fully-validated v2 config (four PRs
confirmed, zero known regressions) rather than ship gate v2 with an
unconfirmed benefit and an added (if currently harmless) layer of
unmerged-upstream complexity. Gate v2 is not disproven as a concept —
Reederey87's own independent production deployment validates the
mechanism works *somewhere* — but it does not reproduce here under this
deployment's actual speculator (MTP). Worth revisiting if:
(a) PR80 merges upstream with additional MTP-specific testing, or
(b) this deployment ever switches to DFlash2 as the default speculator
(not recommended right now — see the DFlash2 production-crash findings
earlier in this document).

**New thread surfaced, not yet pursued**: upstream's own `main` branch
has moved `MAX_NUM_BATCHED_TOKENS` from 2048 to 7168 as of the fat-expert-
kernel work (discovered via the README merge conflict above) — this
project's shipped recipes still use 2048. Worth testing in isolation
(this cluster's own history includes hitting a wall raising this value on
the NVFP4 lane, so treat as a fresh experiment, not an assumed win).

## max_num_batched_tokens 2048 -> 7168: substantial confirmed gain, promoted (2026-09-02)

Tested the thread surfaced above, on the shipped v2 config (4 PRs, gmu
back to 0.86 after both cluster nodes rebooted overnight — the "trapped
memory" pattern from the previous day is fully resolved, confirmed via
`uptime` ~3 min and 0B swap used on both nodes at the start of this
session). Isolated via `-o max_num_batched_tokens=7168`, everything else
held constant.

**Booted clean** — graph capture (5s, 0.36 GiB) succeeded without
incident, notable given this exact knob crashed the NVFP4 lane outright
at just 3584 (less than half this value) earlier in this project. Direct,
measured confirmation that EXL3's smaller memory footprint is real usable
headroom, not just a theoretical advantage.

- `probe_sanity.py`: ALL PASSED, decode median 25.68 tok/s — no
  regression, consistent with every prior v2 measurement.
- `probe_soak.py` (2 rounds x 2 waves): **PASSED**, including concurrent
  waves — the exact kind of load that has caused problems elsewhere on
  this cluster.
- `probe_longctx.py`, same-day A/B against the 2048 baseline:

| Context | 2048 (baseline) | 7168 | Gain |
|---|---:|---:|---:|
| ~100K tokens | 933.3 tok/s (107.1s TTFT) | **1198.5 tok/s (83.4s TTFT)** | **+28.4%** |
| ~250K tokens | 1048.4 tok/s (238.4s TTFT) | **1252.5 tok/s (199.5s TTFT)** | **+19.5%** |

Both correctness-verified (4/4 planted codes retrieved at both lengths).
This is a substantial, real, independently-reproducible gain — larger in
absolute terms than the fat-expert-kernel's own +14.3%/+13.9%, and the two
appear to compound (this test already had the fat-expert kernel active).

**Promoted**: `max_num_batched_tokens` raised to 7168 in both the working
repo's `glm-5.3-flash-exl3-v2-vllm.yaml` and the archived
`~/ai/recipes/glm-5.3-flash-exl3-tp2.yaml`. This now matches upstream's
own current production value exactly (discovered via the PR80 merge
conflict — their README already documented "E2 keep 2026-09-01" for this
exact figure, chosen alongside the fat-expert kernel; this project just
hadn't picked it up yet before this test).

### Gap closed: structured-content decode, not just prose (2026-09-02, later)

The promotion above validated `probe_sanity.py` (prose decode), `probe_soak.py`
(concurrent load), and `probe_longctx.py` (prefill/TTFT) — but not a
structured-content decode number, which is the workload DFlash2's own
validation flagged as sensitive to scheduler/batching changes. Since
`max_num_batched_tokens` is nominally a prefill/scheduling knob, no decode-phase
effect was expected, but that was an assumption, not something measured for
this specific recipe. Closed the gap against the live `max_num_batched_tokens=7168`
service (MTP-2, matching the shipped default): same counting-task methodology
used for the DFlash2 lanes above (1→100, one number per line, 300 max tokens,
temp 0, 3 runs), acceptance rate read from `vllm:spec_decode_num_*_tokens_total`
deltas across the 3 runs.

| Run | Decode tok/s | Completion tokens | Finish reason |
|---|---:|---:|---|
| 1 | 32.21 | 200 | stop (counted 1→100 correctly) |
| 2 | 31.17 | 200 | stop |
| 3 | 31.65 | 200 | stop |

**Median 31.65 tok/s, MTP-2 acceptance 96.57% (394/408 draft tokens
accepted)** on structured content — actually *higher* than this same
recipe's prose number (25.68 tok/s from the promotion run above), consistent
with MTP's draft head predicting highly-regular sequences well. Confirms the
assumption: `max_num_batched_tokens` shows no decode-phase regression on
structured content either. No change needed to the shipped recipe.

### Incidental finding: the multimodal-warmup crash is real and intermittent, not a one-off

During the restoration boot that preceded this test (before the
`max_num_batched_tokens` change), the `KeyboardInterrupt`-during-
multimodal-warmup crash documented in the "Vision retest" section above
recurred a **third time** — on a completely fresh boot, immediately after
both cluster nodes rebooted (ruling out memory pressure/accumulated state
as the cause). An immediate retry with identical config booted clean, same
as the pattern established earlier. This is now confirmed **genuinely
intermittent** (crashes ~1 in 3 boots so far, no config correlation found),
not a one-off as originally concluded — see the correction inline in the
"Vision retest" section. Current handling: retry once on this exact
signature before considering `--language-model-only`; the external
SIGTERM's actual sender remains unidentified.

## MAJOR FINDING: NVFP4+DFlash2 works after all — FlashInfer autotune was the root cause (2026-09-02)

Retested the exact NVFP4+DFlash2 config that failed 5/5 times earlier in
this project (`RedHatAI/GLM-5.3-Flash-NVFP4`, `ghcr.io/tonyd2wild/
vllm-glm53-flash:sm121-v11-dflash2`, 3 GiB KV pin, `num_speculative_
tokens=7`) with exactly **one** new variable: `--no-enable-flashinfer-
autotune`. Motivation: every prior crash died with the identical silent-
kill signature within seconds of `flashinfer.jit: [Autotuner]: Autotuning
process ends` logging, and while building the EXL3 lane's own image,
found MiaAI-Lab's own Dockerfile comment independently documenting the
same failure class: "FlashInfer's sparse-MLA autotune and fused_moe
gemm1/gemm2 autotune kill rank 0 on SM121" — they disable both
unconditionally for this exact reason, on the same hardware family. This
project never tried that specific fix before concluding NVFP4+DFlash2 was
a hard wall and pivoting to EXL3.

**Result: boots clean, genuinely serves, fully stable.** Confirmed
`Skipping FlashInfer autotune because it is disabled` in the boot log,
then watched it sail straight through every phase that killed all 5 prior
attempts: target-model CUDA graph capture (PIECEWISE 15/15, FULL 6/6) and
DFlash2's own speculator graph capture (FULL 6/6, one slow first-shape
JIT compile that looked like a stall — `shm_broadcast.py`'s 60s warning —
but resolved and finished normally, not a crash). Full engine init: 93.66s.

- Verified with a genuine generation request (not just `/health`) before
  trusting it, per this project's own established discipline.
- `probe_sanity.py`: ALL PASSED. Decode **23.49-30.44 tok/s (median
  28.19)** — actually **beats the NVFP4+MTP-4 baseline outright** (21.56
  tok/s median), TTFT 0.215-0.32s.
- `probe_soak.py` (2 rounds x 2 waves): **PASSED**, including concurrent
  waves — the same class of load that has caused problems elsewhere on
  this cluster. Genuinely stable, not a lucky single request.
- Structured-content test (counting task, 300 tokens): **66.42 tok/s**,
  final-window acceptance **93.4%**, mean acceptance length **7.54 of a
  possible 8** — this actually **beats the EXL3+DFlash2 lane's own
  56.35 tok/s** structured-content result from earlier this session.

### What this means

**The original "NVFP4+DFlash2 is not viable on this cluster" conclusion
was wrong in its stated mechanism.** It was never really a memory-
headroom problem specific to NVFP4/Marlin's larger footprint (the theory
this project built the entire EXL3 pivot on) — it was FlashInfer's
autotune routine crashing rank 0 during warmup, a bug independent of
which quantization scheme is in use. The EXL3 pivot itself remains fully
justified on its own separate merits (smaller checkpoint, more prefill/
caching wins from the four validated PRs, already in production) — but
the *specific* claim that NVFP4 structurally cannot run DFlash2 on this
hardware is now known to be false, and the credit for "why EXL3 succeeded
where NVFP4 failed" should go partly to this project's own late-arriving
discovery of the autotune bug, not solely to memory headroom as
originally concluded.

**This does not change the shipped default.** EXL3 v2 (now with
`max_num_batched_tokens=7168`) remains the better overall choice — proven
in longer production use today, smaller memory footprint leaving more
margin for the cluster's other volatility (the "trapped memory" pattern,
the intermittent multimodal-warmup crash), and this NVFP4+DFlash2 result
is from a single validation session, not the extended production
exposure EXL3 has had. But it closes an open question this project
carried since the DFlash2 postmortem, and the finding itself
(`--no-enable-flashinfer-autotune` as a fix for SM121 rank-0 crashes) is
worth remembering for any future GB10/SM121 work regardless of
quantization scheme — see `~/ai/research/glm-5.3-flash-gb10/README.md`,
now updated to reflect this confirmation rather than the earlier
"worth retrying" framing.

**Test artifact**: `recipes/glm-5.3-flash-nvfp4-vllm-dflash2-autotune-off.yaml`
in the working repo, kept (not deleted) since it's now a working,
validated config, not a ruled-out one.

## Request pipeline timing: where time actually goes (2026-09-02)

Built a visual "anatomy of a request" breakdown against the live EXL3 v2
service (`max_num_batched_tokens=7168`, MTP-2), not from theory but from
before/after deltas on vLLM's own Prometheus counters
(`vllm:*_seconds_sum`, `vllm:spec_decode_num_*_tokens_total`) across three
single-stream (concurrency=1) workloads, n=6 requests each. Script:
`recipes/probes/probe_pipeline_timing.py`.

| Workload | Prompt&rarr;gen tok | e2e | queue | prefill | decode | overhead* | decode tok/s | MTP-2 accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Short prose | 23&rarr;128 | 4.92s | ~0 | 0.227s (4.6%) | 4.644s (94.5%) | 0.046s (0.9%) | 27.35 | 73.8% (pos0 91.6% / pos1 55.9%) |
| Structured (counting) | 32&rarr;200 | 6.60s | ~0 | 0.222s (3.4%) | 6.310s (95.6%) | 0.068s (1.0%) | 31.54 | 97.1% (pos0 100% / pos1 94.1%) |
| Medium prompt | 275&rarr;73 | 2.97s | ~0 | 0.513s (17.3%) | 2.383s (80.3%) | 0.071s (2.4%) | 30.35 | 93.0% (pos0 100% / pos1 85.9%) |

*overhead = e2e &minus; (queue+prefill+decode): network/JSON/detokenize, not
broken out further by current metrics.

**Findings**:
- **Queue time is genuinely zero at single-stream** (0.00002-0.00003s avg)
  — the scheduler is never the bottleneck at concurrency=1. Not yet tested
  at higher concurrency (`num_requests_waiting_by_reason` exists for a
  future capacity-ceiling probe).
- **Decode dominates e2e for short/medium completions** (80-96%), not
  prefill — confirms the `max_num_batched_tokens` win (a prefill-side
  knob) doesn't touch the thing actually gating short-request latency.
- Cross-referenced against the `max_num_batched_tokens` promotion's
  already-measured 100K/250K-token TTFT (83.4s / 199.5s): the picture
  flips completely at long context — prefill becomes totally dominant.
  Confirms the fat-expert kernel and the batched-tokens raise targeted
  the right regime.
- **MTP-2 acceptance is workload-dependent even with only 2 draft
  positions** (73.8% prose vs. 97.1% structured) — same shape as DFlash2's
  much more dramatic 7-position decay (70%&rarr;6% prose vs. flat
  85-100% structured), just far less pronounced at this shallow depth.
- **Confirmed gap, not yet closed**: vLLM's `/metrics` on this build has
  no sub-forward-pass breakdown — no way to separate MLA attention,
  sparse-indexer, mamba-layer, or MoE/fat-expert-kernel time from each
  other via Prometheus alone. `vllm:estimated_flops_per_gpu_total` and
  the paired read/write-byte counters exist but read 0 (not populated by
  this vLLM version) — a roofline (compute- vs. bandwidth-bound) view
  wasn't possible from metrics. Next step for real kernel-level attribution:
  `torch.profiler` or `nsys` trace on a single prefill+decode step.

**Artifact**: published as "Anatomy of a GLM-5.3-Flash Request" —
pipeline flow diagram, per-workload waterfall, context-length crossover,
and a speculative-decode funnel comparing MTP-2 against the DFlash2
reference numbers above, plus a ranked "where to dig next" list.

# Deep instrumentation night (2026-09-02)

Goal: extend the harness far enough to see under the hood — sub-forward-pass
timing, process-level CPU attribution, and concurrency — then use it. The
production recipe was explicitly cleared for disruption for this work.

New tooling added under `recipes/probes/`:

| Script | What it gets that `/metrics` cannot |
|---|---|
| `probe_concurrency_pipeline.py` | Per-stage server-side splits across a concurrency sweep, plus sampled scheduler gauges (running/waiting/KV%) that are gauges and vanish after the run |
| `probe_cpu_profile.sh` | py-spy speedscope + flamegraph of EngineCore and Worker under a live decode load |
| `analyze_pyspy.py` | Buckets sampled stacks into wait-vs-work (GPU sync / IPC spin / idle / real Python work) |
| `probe_profile_run.py` | Drives `/start_profile` → workload → `/stop_profile` |
| `collect_traces.sh` | Pulls traces out of both TP ranks' containers |
| `analyze_trace.py` | Aggregates a Chrome trace into kernel-family attribution, engine-scope CPU time, and GPU-busy vs wall |

Two container facts worth recording, since both cost time to discover:
py-spy needs `docker exec -u root --privileged` (the container runs as an
unprivileged user under `ptrace_scope=1`, and without both flags py-spy
returns a bare "Permission Denied"); and the profiler's `torch_profiler_dir`
must live under `/tmp`, because that same unprivileged user cannot create a
directory at the filesystem root.

## Concurrency: throughput saturates at c=8, and KV cache is 90% idle

`probe_concurrency_pipeline.py --workload prose --levels 1,2,4,8,16 --waves 3`
against the shipped config (MTP-2, `max_num_batched_tokens=7168`,
`max_num_seqs=4`):

| c | wall | aggregate tok/s | client e2e p50 | server queue avg | queue % of e2e | TTFT avg | running max | waiting max | KV peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.8s | 25.96 | 4.88s | 0.00001s | 0.0% | 0.291s | 1 | 0 | 2.39% |
| 2 | 20.3s | 37.79 | 5.25s | 0.771s | 13.0% | 1.145s | 2 | 1 | 4.78% |
| 4 | 32.7s | 46.91 | 8.33s | 1.114s | 11.3% | 2.728s | 4 | 3 | 9.57% |
| 8 | 53.7s | 57.21 | 12.65s | 6.080s | 45.3% | 6.563s | 4 | 7 | 9.57% |
| 16 | 102.1s | 60.19 | 22.91s | 13.696s | 63.0% | 14.132s | 4 | 15 | 9.57% |

- **Throughput saturates at c=8.** c=8 → c=16 buys +5.2% aggregate throughput
  for +81% client latency. There is no reason to run this deployment above
  c=8 as configured.
- **Queue time becomes the dominant cost, and it is not a memory limit.**
  `num_requests_running` pins at exactly 4 — `max_num_seqs` — from c=4
  upward, while everything else waits. Meanwhile **KV cache usage peaks at
  9.57% and stops rising**, because the cap on concurrent sequences prevents
  the KV pool from ever being used. Over 90% of the KV pool this deployment
  spent its whole bring-up fighting to enlarge is idle at saturation.
- MTP-2 acceptance is unaffected by batching (0.731 → 0.748 across the
  sweep), so deeper batching does not cost speculative-decode efficiency.
- Zero preemptions at every level.

`max_num_seqs=4` is inherited from the NVFP4 lane, where raising it past 6 hit
the memory ceiling documented earlier in this file. That constraint was never
re-tested on EXL3, which has ~10 GiB more headroom — and the KV figure says
the headroom is real and unused. Follow-up test below.

## Process-level CPU: both engine processes are ~97-99% inside an IPC spin

`probe_cpu_profile.sh` (py-spy, 200 Hz, 25 s, during a steady single-stream
900-token generation), analyzed by `analyze_pyspy.py`:

| Process | ipc_spinwait | real Python work | top self-time frames |
|---|---:|---:|---|
| EngineCore | 99.9% | 0.1% | `wait` 27.9%, `acquire_read` 15.8%, `should_warn` 19.8% (two lines), `check` 7.6%, `timeout_ms` 8.5% (three lines) |
| Worker_TP0 | 96.6% | 2.6% | `sched_yield` 32.4%, `memory_fence` 17.3%, `wait` 13.9%, `acquire_read` 19.3% (three lines) |

Everything above is `vllm/distributed/device_communicators/shm_broadcast.py`.
Caveat on method: py-spy ran `--nonblocking` and reported ~33% sampling
errors, which biases toward frames it can unwind (pure Python) and away from
native execution — so treat these as "of the samples that resolved" rather
than as absolute wall-clock shares. The *composition* is the finding, not the
exact percentage.

What the composition shows is that the spin loop is not merely waiting, it is
doing repeated Python bookkeeping while waiting. Each `acquire_read`
iteration calls `check()` → `memory_fence()` (which is
`with _memory_fence_lock: pass`, a real Python lock acquire/release),
then `self._spin_condition.wait(timeout_ms=read_timeout.timeout_ms())` and
`read_timeout.should_warn()` — each of which calls `time.monotonic()` and does
arithmetic, per iteration, forever. Roughly a third of the sampled spin time
is in those helpers rather than in the yield itself.

## Root cause: `GLM53_SPINWAIT_MS=16` has never been applied

The shipped recipe has set `GLM53_SPINWAIT_MS=16` since v2 was promoted.
**It does nothing.** Verified in the live container: the env var is set, and
`shm_broadcast.py` line 134 still reads `busy_loop_s: float = 1,` — stock.

Audit of every EXL3/GLM53 knob the shipped recipe sets, checking where each is
actually read:

| Knob | Read by | Live? |
|---|---|---|
| `EXL3_FUSED_MOE` | `overlay/exl3.py` (installed module, runtime) | yes |
| `EXL3_FAT_KERNEL` | `overlay/exl3.py` | yes |
| `EXL3_MOE_ROW_TILE` | `overlay/exl3.py` | yes |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | patched `vllm/v1/engine/detokenizer.py` | yes |
| `GLM53_MIXED_PREFILL_CHUNK` | patched `vllm/v1/core/sched/scheduler.py` | yes |
| `GLM53_INDEXER_WORKSPACE` | injected `_glm53_workspace_mode()`, reads env per call | yes |
| `GLM53_SPINWAIT_MS` | **only `patch_spinwait.py`**, at patch time | **NO** |

Six of seven knobs are read at runtime by code the Dockerfile bakes in, so
they work regardless of how the container is started. `GLM53_SPINWAIT_MS` is
the exception: `patch_spinwait.py` *rewrites the default parameter value in
the source text*, reading the env var at patch time. Upstream's `start.sh`
runs it at container startup (start.sh lines 1038-1039); this project's
sparkrun recipes set `entrypoint: ""` and call `vllm serve` directly, which
bypasses start.sh entirely. The Dockerfile's own build-time run of that patch
happened with the variable unset, i.e. stock.

**This retroactively explains the PR96 "inconclusive" verdict** recorded
earlier in this file. That validation measured a feature that was never on.
The honest correction is not "the claim is unproven" but "the claim was never
tested" — and the py-spy numbers above are what stock actually looks like:
with `busy_loop_s=1` and decode steps arriving every ~30 ms, the reader never
leaves the busy window, so the zmq park path is unreachable by construction.

Test recipe: `recipes/glm-5.3-flash-exl3-v2-spinwait.yaml`, which runs the
image's own `patch_spinwait.py` before `vllm serve` and echoes the patched
line at boot so the change is visible in the log rather than assumed.

Separately: the image ships a compiled native spin loop
(`vllm/spinloop.abi3.so`, importable) but `VLLM_USE_SPINLOOP_EXT` defaults to
False, so all of the above spinning runs in Python. That is a second,
independent lever on the same code path.

## Profiler gotcha: `max_iterations` kills the server

Worth recording because it cost a boot cycle and the name gives no hint.
`--profiler-config` accepts `max_iterations`, which reads like a trace-size
cap. It is not a cap — it is a benchmark-harness exit. On reaching the limit
mid-request the server logged:

```
[wrapper.py:126] Max profiling iterations reached. Stopping profiler...
[launcher.py:114] [shutdown] API server: shutdown triggered
[utils.py:640] Process manager: force killing remaining process EngineCore
```

...and the deployment went down. It is intended for scripts that profile N
iterations and exit. **On a long-lived server set `max_iterations: 0`** and
bound trace size with a short workload plus an explicit `/stop_profile`.

Also visible in that first attempt, and a finding in its own right: the first
profiled request logged Triton JIT compilation *during inference* for
`_compute_local_logits_stats_kernel`, `_rejection_kernel`, and
`_resample_kernel` — vLLM's own `jit_monitor` flags these as latency spikes.
These are speculative-decode rejection-sampling kernels being compiled on
first use, so the first request after every boot pays a JIT penalty that no
steady-state benchmark will show.

One more limit found the same way: **profiling a long-context prefill kills
the engine.** A `--prefill-only` trace of a ~33K-token prompt produced enough
trace events that the worker missed its shm_broadcast deadline, and the read
raised `RuntimeError: cancelled` → `RuntimeError: Executor failed.` →
`EngineDeadError`. Keep profiled prefills small (a few thousand tokens); the
long-context prefill numbers already come from `probe_longctx.py`, which does
not need the profiler.

## Sub-forward-pass attribution: decode is GEMM+MoE bound, attention is 0.3%

This is what Gap #1 was asking for. Traces via `/start_profile`, collected
from both TP ranks, analyzed with `analyze_trace.py`. **CUDA graphs did not
have to be disabled** — CUPTI resolved kernels inside graph replays, so these
are production-configuration numbers, not an eager-mode approximation.

Prose decode, batch=1, MTP-2 (each step verifies 3 tokens: 1 + 2 draft),
2.53 s window, 74,777 kernel events:

| kernel family | rank 0 | rank 1 | calls | µs/call |
|---|---:|---:|---:|---:|
| gemm | **52.5%** | 49.8% | 13,827 | 96 |
| moe_exl3 | **32.8%** | 31.3% | 2,288 | 363 |
| comms (NCCL) | 7.2% | **11.9%** | 2,600 | 70 / 124 |
| elementwise + memory | 3.6% | 3.4% | 41,208 | ~2 |
| **attention** | **0.3%** | 0.3% | 1,391 | 6 |
| mamba_ssm | 0.2% | 0.2% | 962 | 6 |
| norm / quant / sampling | 0.3% | 0.3% | 2,667 | ~3 |

**GPU busy 96.0% of the wall span.** Decode at batch=1 is GPU-bound, not
CPU-bound — which reframes the py-spy result above: both engine processes sit
in the spin loop because they are waiting on a genuinely busy GPU, not because
CPU orchestration is the bottleneck.

Three findings worth acting on:

**1. Attention is irrelevant at short-context decode (0.3%).** All the
sparse-MLA / indexer machinery that dominates this model's reputation matters
for prefill; it is noise during decode. Optimization effort aimed at decode
should ignore it entirely.

**2. The dominant GEMM kernels are Ampere-era, with 16×16 tiles.** Named in
full:

| kernel | calls | µs/call | total |
|---|---:|---:|---:|
| `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8` | 5,850 | 133.5 | 781 ms (30.8%) |
| `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` | 3,876 | 62.0 | 240 ms (9.5%) |
| `internal::gemvx::kernel<...__nv_bfloat16...>` (cuBLAS GEMV) | 209 | 943.0 | 197 ms (7.8%) |
| `exl3_moe_kernel<4, 256>` | 1,144 | 720.0 | 824 ms (32.5%) |

`cutlass_80` is the SM80/Ampere target and `wmma_tensorop` is the legacy WMMA
path — running on SM121 Blackwell. The tiles are 16×16. At batch=1 with MTP-2
the GEMM's M dimension is **3**, so a 16×16 tile is ~81% empty in M: these are
GEMVs being pushed through tensor-core GEMM kernels that need M≥16 to fill a
tile. That is simultaneously the explanation for the modest decode throughput
and the strongest available lead for kernel work — a Blackwell-native
(SM100/SM120-class, TMA, warpgroup MMA) path for these shapes, or a genuine
GEMV/skinny-GEMM kernel, is attacking 40%+ of decode GPU time.

**3. Cross-node TP costs 7-12% and is asymmetric.** Both ranks issue
identical kernel counts (13,827 gemm / 2,288 moe / 2,600 comms — perfectly
symmetric work division), but rank 1 spends 323 ms in `ncclDevKernel_AllReduce`
against rank 0's 183 ms (124.9 vs 70.5 µs/call). Rank 1 is arriving at the
collective first and waiting on rank 0. That ~140 ms delta is ~5% of the
window, spent idling inside the collective.

## Batching is the dominant decode lever, and the trace says why

Same trace methodology at 8 concurrent requests (`max_num_seqs=16`):

| | batch=1 | batch=8 |
|---|---:|---:|
| GPU busy | 96.0% | 64.1% |
| `exl3_moe_kernel<4,256>` per call | 720.0 µs | 1512.6 µs |
| gemm µs/call (family avg) | 96.2 | 92.7 |
| moe_exl3 share | 32.8% | 46.5% |
| attention share | 0.3% | 0.3% |

**The MoE kernel costs 2.1× more time while serving 8× the sequences**, and
per-call GEMM time is flat. These kernels are dominated by reading weights,
not by the arithmetic on any one sequence, so their cost is almost independent
of how many sequences ride along. That is the mechanism behind the concurrency
numbers, and it is why `max_num_seqs` mattered so much.

(The drop to 64.1% GPU-busy at batch=8 is partly ramp-up/tail — 8 requests of
unequal length in one 13 s window — but 4.7 s of GPU idle is large enough to
be worth a dedicated look; see the open threads at the end.)

## `max_num_seqs` 4 → 16: +171% aggregate throughput *and* lower latency

The concurrency sweep above showed `num_requests_running` pinned at 4 while KV
sat at 9.57%. Re-ran the identical sweep with `-o max_num_seqs=16`, everything
else held constant:

| c | agg tok/s @4 | agg tok/s @16 | change | e2e p50 @4 | e2e p50 @16 | KV peak @16 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.96 | 25.46 | −1.9% | 4.88s | 4.98s | 2.56% |
| 2 | 37.79 | 35.42 | −6.3% | 5.25s | 5.74s | 5.13% |
| 4 | 46.91 | 49.90 | +6.4% | 8.33s | 9.09s | 10.26% |
| 8 | 57.21 | 95.10 | **+66.2%** | 12.65s | **7.87s** | 20.51% |
| 16 | 60.19 | **163.45** | **+171.6%** | 22.91s | **14.13s** | 41.03% |

Latency *improves* at the same time as throughput at c≥8 — this is not a
throughput-for-latency trade. Zero preemptions at every level, MTP-2
acceptance unchanged (0.726-0.750), and KV still only 41% used at c=16, so
the ceiling has not been found. The small regressions at c=1-2 are within
run-to-run noise for this bench but are reported as measured.

**Caveat before promoting this**: every prompt in this sweep is short. KV
demand scales with context length, and at 262,144-token context 16
simultaneous sequences cannot fit — the scheduler would queue and eventually
preempt. `max_num_seqs=4` came from the NVFP4 lane's memory ceiling and was
inherited untested; the right follow-up is a long-context concurrency sweep to
find the safe value, not a blind promotion of 16.

## DFlash2 k=7 re-validated at MNBT=7168: bigger wins, and a new crash

Recipe: `recipes/glm-5.3-flash-exl3-v2-dflash2-7168.yaml` (drafter pinned to
snapshot `7d74cdd`, matching the original measurement). Booted clean; vLLM
logged the expected `max_num_scheduled_tokens is set to 7168 based on the
speculative decoding settings` warning. Available KV cache with the drafter
resident: **14.6 GiB** (versus a materially larger pool under MTP-2 — the
drafter is not free).

| workload | DFlash2 @7168 | DFlash2 @2048 (historical) | MTP-2 @7168 |
|---|---:|---:|---:|
| prose (23→128 tok) | 24.04 tok/s, 28.9% accept | 24.30 tok/s | **27.35** tok/s, 73.8% |
| structured (counting) | **61.07** tok/s, 94.1% accept | 56.35 tok/s | 31.54 tok/s, 97.1% |
| medium summarize (275→73) | **50.52** tok/s, 75.5% accept | not measured | 30.35 tok/s, 93.0% |

Per-position acceptance (out of the 7 draft slots) is where the workload
dependence lives:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose | 222 | 138 | 90 | 36 | 24 | 18 | 6 |
| structured | 158 | 158 | 157 | 156 | 154 | 133 | 125 |
| medium | 70 | 62 | 54 | 46 | 46 | 46 | 46 |

- **The MNBT raise helps DFlash2 too**: structured decode 56.35 → 61.07 tok/s
  (+8.4%). The worry in this recipe's header — that a 7-deep drafter might be
  starved by sharing the scheduling budget — did not materialize.
- **Prose is unchanged** (24.30 → 24.04, noise) and still ~12% behind MTP-2.
- **New result: mid-length semi-structured work is a large DFlash2 win.** The
  summarization workload runs 50.52 vs MTP-2's 30.35 tok/s — +66%. Its
  per-position curve decays and then *plateaus* at 46/70 (66%) rather than
  collapsing like prose. Previous validation only tested prose and a counting
  task, so the useful middle of the workload space had never been sampled;
  DFlash2's win zone is wider than "code and counting".
- **But it crashed.** During a prompt-length sweep (filler prompts up to
  ~8-11K tokens) the worker died with this project's documented silent-kill
  signature — `EngineDeadError`, no CUDA OOM, no assertion, no traceback from
  the worker. Short prompts had been fine for dozens of requests immediately
  prior. With only 14.6 GiB of KV and a 7168-token scheduling budget plus a
  resident drafter, a multi-thousand-token prefill is a plausible memory
  trigger, consistent with every prior instance of this signature on this
  cluster.

**Verdict: still not promotable, for a new reason.** The performance case is
now stronger than it was (two of three workloads win big, and one of those is
newly discovered), but a config that dies on an 8K-token prompt cannot ship.
The next step is a prompt-length bisection under DFlash2@7168 to find where it
breaks, and whether lowering MNBT or `max_num_seqs` back down buys stability
without giving up the structured-content win. MTP-2 remains the default.

## Stretch goal closed: the 1-2% "overhead" bucket is a fixed per-request cost

`probe_overhead_bucket.py` varies one dimension at a time and regresses the
residual (server `e2e` minus queue+prefill+decode) against each. Run on the
restored shipped config, n=4 per point:

| output tokens (prompt fixed at 17) | overhead | % of e2e |
|---:|---:|---:|
| 16 | 12.2 ms | 1.27% |
| 64 | 43.3 ms | 1.40% |
| 160 | 44.1 ms | 0.61% |
| 320 | 65.3 ms | 0.50% |

| prompt tokens (output ~26) | overhead | % of e2e |
|---:|---:|---:|
| 42 | 78.2 ms | 5.97% |
| 978 | 42.4 ms | 1.34% |
| 3,858 | 62.4 ms | 1.48% |
| 9,618 | 61.9 ms | 0.97% |

Fitted slopes: **145 µs per output token, −0.12 µs per prompt token** (i.e.
zero), and a strikingly stable **5.1 ms** gap between client wall time and
server-side e2e across all eight points.

Conclusion: the residual is **a roughly fixed per-request cost of tens of
milliseconds, not a scaling problem**. About 5 ms of it is HTTP transport to
the probe host; the rest is per-request server-side work (response assembly,
detokenizer finalization, usage accounting) that is *completely independent of
prompt length* and grows only weakly with output length. As a share of e2e it
therefore **shrinks** as requests get larger — 5.97% on the shortest request
measured, 0.50% on the longest. Run-to-run scatter at n=4 is comparable to the
trend itself, which is the other reason not to chase this further.

**Not worth optimizing.** It only reaches a few percent on trivially short
requests, and nothing about it degrades under load or context growth. Closing
this thread rather than leaving it open.
