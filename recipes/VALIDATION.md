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
