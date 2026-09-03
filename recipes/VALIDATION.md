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

## Spinwait actually turned on: two new bugs on the way there, then a clean negative result

Dig item #4 from the previous session's list. Goal: get `GLM53_SPINWAIT_MS=16`
genuinely active (it never has been, per the finding above) and re-measure
PR96's claim honestly instead of leaving it "inconclusive against a feature
that was off."

**Attempt 1 — run `patch_spinwait.py` before `vllm serve`, failed.**
Crashed every relaunch:
`PermissionError: ... '.shm_broadcast.py.glm53-spinwait.tmp'`. Root cause,
distinct from the entrypoint/`start.sh` bypass already documented: `docker
inspect` on the live container shows sparkrun runs it as `1000:1000`
(the image itself has no `USER` directive, so this is sparkrun's own
convention, presumably for `/models` bind-mount ownership) —
`shm_broadcast.py` is `root:root 0644` in a `root:root 0755` directory,
unwritable and unreplaceable by that uid. Even a recipe that correctly
invoked `start.sh` would hit this same wall; the constraint is filesystem
permissions, not which script performs the edit.

**Fix — in-process monkeypatch instead of a source edit.** vLLM forks
EngineCore/Worker as OS processes (confirmed:
`envs.VLLM_WORKER_MULTIPROC_METHOD == "fork"`), and fork gives children a
copy-on-write snapshot of the parent's memory, already-imported modules
included. `SpinCondition.__init__` (the only place `busy_loop_s` is
defined) has exactly one defaulted parameter, so
`SpinCondition.__init__.__defaults__` is a 1-tuple — reassigning it before
`vllm serve`'s own top-level process forks anything propagates the change
to every child with zero file writes. `recipes/glm-5.3-flash-exl3-v2-spinwait.yaml`
now writes this as a small shim to `/tmp` (writable) and launches
`vllm.entrypoints.cli.main:main()` directly — the same function the real
`vllm` executable calls, so argument parsing is unaffected.

**Attempt 2 — the shim itself crashed the engine, a second, independent
bug.** First version had no `if __name__ == "__main__":` guard. vLLM's
API-server bootstrap turned out to use a `multiprocessing` **spawn**
boundary above the fork-based EngineCore/Worker layer — separate from
`VLLM_WORKER_MULTIPROC_METHOD`, which only governs the latter. `spawn`
re-imports the launching script to reconstruct child state; with no guard,
that re-import re-executed every top-level statement, including the
`sys.exit(main())` call itself — so the "child" launched a second, fully
independent `vllm serve` from inside what should have been a lightweight
bootstrap. Visible in the log as a second complete startup banner and a
second `[glm53-spinwait-monkeypatch]` line at a later timestamp, followed
by `EngineCore initialization failed`. Standard fix: wrap the patch-and-run
call in the guard. Confirmed by exact log-line counts: exactly one
`[glm53-spinwait-monkeypatch]` line per node after the fix, versus two
before.

**Booted clean.** Verified with a real generation request, both nodes
show exactly one patch-confirmation line.

### The measurement, done honestly this time

py-spy time-in-frame percentages **cannot distinguish real CPU burn from a
blocked syscall** — both look like "the process is in this frame" to a
wall-clock sampler, which is exactly the ambiguity a claim about *CPU*
usage needs resolved. Added `probe_cpu_ticks.sh`, which reads
`/proc/<pid>/stat` utime+stime deltas across an identical 900-token
generation — actual CPU-seconds, not sampled frame occupancy — for both
configs, same hardware, back-to-back:

| Process | stock (busy_loop_s=1s) | patched (busy_loop_s=16ms) | change |
|---|---:|---:|---:|
| EngineCore | 98.5% of wall (1 core) | 99.8% of wall (1 core) | none (noise) |
| Worker_TP0 | 200.1% of wall (2 cores) | 202.0% of wall (2 cores) | none (noise) |

**No CPU reduction at all** — not a smaller effect than claimed, no effect.
py-spy's frame composition did shift with the patch (e.g. Worker's
`sched_yield` self-time share rose from 32.4% to 65.2%), consistent with
the busy window genuinely getting shorter and the reader cycling more
often — but total time spent across the whole spin-wait code path, and
actual CPU-seconds consumed, did not move.

Decode throughput, same three workloads as the pipeline-timing baseline,
same live server, immediately before/after the swap:

| Workload | stock decode tok/s | patched decode tok/s | change |
|---|---:|---:|---:|
| short prose | 26.46 | 27.29 | +3.1% |
| structured (counting) | 30.85 | 31.03 | +0.6% |
| medium summarize | 29.80 | 29.83 | ~0% |

Small, plausibly-real gains roughly in line with PR96's own claimed +0.95%
decode — the throughput half of the claim holds up reasonably. **The CPU
half does not**, now that it has actually been tested rather than measured
against a feature that was off.

**Verdict: genuinely tested now, and it's a clean negative on the headline
number.** Decode is a wash-to-slightly-better; the promised CPU win isn't
there on this cluster's build/hardware. Not promoting `GLM53_SPINWAIT_MS`
into the shipped recipe on this evidence — the earlier "inconclusive" was
too generous; "doesn't reproduce" is the honest read. `VLLM_USE_SPINLOOP_EXT`
(the native spin-loop extension, confirmed importable but disabled by
default) is still an untested, independent lever on the same code path and
remains open.

## The Blackwell-kernel lead investigated: negative result, real pivot found

Dig item #2 from the profiling session: "40%+ of decode GPU time is
SM80-target CUTLASS WMMA with 16x16 tiles, ~81% empty in M — worth checking
whether a newer cuBLAS/CUTLASS already dispatches an SM100/SM120 kernel
before writing anything." Investigated as instructed, in that order.

**Toolchain is already current — not a stale-library problem.** This
container ships CUDA 13.0 (nvcc `V13.0.88`, built Aug 2025), cuBLAS
`13.1.1.3`, PyTorch `2.13.0+cu130`, and `nvidia-cutlass-dsl` `4.6.2`. cuBLAS
still dispatches the SM80-targeted WMMA kernel family for this shape+layout
on SM121 even at this version — confirmed by reproducing the exact call in
isolation (below), so this is a genuine coverage gap in cuBLAS's own kernel
selection for SM121, not an artifact of an old install.

### Identifying the actual GEMM: the LM head, and one real profiling gotcha

Correlated `aten::mm`'s `Input Dims`/`Input Strides` against the dominant
kernel bucket from the earlier trace. Shape: `[M, 4096] x [4096, 77440]`,
strides `[[4096,1],[1,4096]]` — the second operand is accessed **transposed**
(column-major), matching standard `nn.Linear`/`ParallelLMHead` weight
storage (`[out_features, in_features]`, used via `x @ W.T`). 77440 =
154,880 (this checkpoint's real vocab size, confirmed in `config.json`) / 2
(TP=2). This is the LM head / vocab-logits projection.

**CUDA-graph replay breaks normal profiler kernel↔op correlation** — a
limitation this project's own profiling recipe header already flagged, now
hit directly: of 27 captured `aten::mm` events in the decode trace, only 2
(both the rare `M=1` case) had a `cudaLaunchKernel` event whose `External
id`/`correlation` successfully joined to a kernel event. The 25 `M=3` events
— the real, steady-state MTP-2 decode shape — never joined, because a
captured graph replays via `cudaGraphLaunch` without re-emitting individual
`cudaLaunchKernel` calls through the profiler's RecordFunction machinery.
Worked around it by reproducing the exact shape and **weight layout** in
isolation instead (layout matters: a naive contiguous-weight benchmark
dispatched a *different* kernel, `cutlass_80_tensorop_bf16_s16816gemm_
bf16_128x64_32x6_nn_align2`, than the real transposed-weight case).

With the correct layout (`F.linear(X, W)`, `W: [77440, 4096]`), the isolated
repro dispatches the **identical** kernel family seen live in production —
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` —
confirming the reproduction is faithful.

### The tile being "81% empty in M" turns out not to matter

| M | kernel | time | achieved bandwidth |
|---:|---|---:|---:|
| 1 | `internal::gemvx::kernel` (dedicated GEMV path) | 3482.1 us | 182.2 GB/s |
| 3 | `cutlass_80_wmma_..._16x16_128x2_tn_align8` | 2583.1 us | 245.6 GB/s |
| 8 | same kernel | 2582.9 us | 245.6 GB/s |

**M=3 and M=8 take identical time.** This operation is fully
memory-bandwidth-bound, not compute-bound: the kernel's job is to stream a
605 MiB (`77440 x 4096 x 2` bytes) weight matrix out of memory once per
call, and that cost doesn't change whether 3 rows or 8 rows of activation
ride along. A mostly-empty 16x16 MMA tile costs nothing when the bottleneck
is HBM traffic, not tensor-core issue rate — the original framing
("SM80 WMMA kernel, tile mostly empty, therefore inefficient") was
structurally accurate but the inference from it doesn't hold.

### Is 245.6 GB/s actually close to this hardware's ceiling? Yes.

Three independent, purpose-built bandwidth microbenchmarks on the same
node: large elementwise `copy_` (read+write) **231.3 GB/s**, fp32 reduction
(read-dominant) **222.7 GB/s**, bf16 reduction **230.9 GB/s**. **The
production LM-head kernel (245.6 GB/s) already beats all three.** There is
essentially no headroom left for a hand-written kernel to extract from pure
memory throughput on this shape — the "40%+ of decode GPU time" figure is
close to the physical cost of reading a 605 MiB matrix from memory every
decode step, not evidence of a bad kernel implementation.

Checked the same question at smaller sizes — the model's other dense (i.e.
non-MoE) linear layers, all of which run bf16 (see below) — using this
checkpoint's real MLA dimensions (`hidden_size=4096, q_lora_rank=1536,
kv_lora_rank=512, qk_nope_head_dim=v_head_dim=256, num_attention_heads=64`,
TP=2-sharded where applicable):

| Layer (TP=2 per-rank shape) | weight size | time (M=3) | achieved BW |
|---|---:|---:|---:|
| `q_a_proj` [4096→1536] | 12.6 MB | 18.2 us | 691.7 GB/s |
| `kv_a_proj` [4096→512] | 4.2 MB | 5.9 us | 712.9 GB/s |
| `q_b_proj` [1536→8192] | 25.2 MB | 53.7 us | 468.4 GB/s |
| `kv_b_proj` [512→16384] | 16.8 MB | 16.1 us | 1041.8 GB/s |
| `o_proj` [8192→4096] | 67.1 MB | 250.8 us | 267.5 GB/s |
| MLP gate/up [4096→6144] | 50.3 MB | 212.0 us | 237.4 GB/s |
| LM head [4096→77440] | 605.3 MB | 2583.1 us | 245.6 GB/s |

Small matrices (≲25 MB) show much higher apparent bandwidth — almost
certainly L2 cache locality, not real HBM throughput. Past roughly
50 MB the number converges cleanly to the same ~235-270 GB/s band the LM
head and the raw bandwidth tests both landed in. **Every dense GEMM big
enough to matter is already running near this hardware's practical memory
ceiling, on the exact same kernel family, regardless of shape.** This
generalizes the LM-head finding to the whole decode path: there is no
"write a faster kernel for this shape" opportunity here, because the
bottleneck is bytes moved, and the existing off-the-shelf cuBLAS kernel is
already close to moving them as fast as this hardware's memory subsystem
allows.

### The real lever: this checkpoint quantizes only the MoE experts

Every one of the shapes above is bf16 in production. Checked why:
`Exl3Config.get_quant_method()` returns `Exl3MoEMethod` only for
`RoutedExperts` layers and `UnquantizedLinearMethod()` for every other
`LinearBase` — attention projections, the router, and the LM head all fall
through unquantized. This isn't a vLLM integration gap: the checkpoint's own
`config.json` states it explicitly —
`"quantization_config": {"scope": "glm53_routed_experts_only",
"non_routed_dtype_policy": "official_source_native", "head_bits": 16}`.
It's the checkpoint authors' deliberate choice, consistent with common
practice (output/embedding layers are often kept at higher precision across
quantization schemes for sensitivity reasons) — not something to "fix" by
finding a missed flag.

**Given the table above, this is where the real lever is.** These
operations are bandwidth-bound; halving the bytes read (bf16 → fp8 for the
LM head and/or attention projections) would roughly halve their time,
directly cutting the single largest slice of decode GPU time — a much
larger and more tractable win than a hand-tuned Blackwell kernel for bytes
that don't change. Not attempted here (it means a different checkpoint or
an additional quantization pass, out of scope for a kernel-trace
investigation) — flagged as the next real thread, not this one.

**Verdict: the specific lead ("write a Blackwell-native kernel for the
M=3 skinny-GEMM tile-fill problem") is a negative result, investigated
properly rather than assumed.** The tile-fill framing was structurally true
and functionally irrelevant. Nothing was written; nothing should be, for
this specific complaint. The productive version of "reduce decode GPU
time in dense layers" is quantization coverage, not kernel authorship.

### Incident: production went down mid-investigation, cause not confirmed

The isolated-shape benchmarks above ran via `docker exec` directly against
the same container and GPU as the live production `vllm serve` process
(read-only from the server's own perspective — no config or file changed).
~15 minutes after the last benchmark, the API server logged
`[shutdown] API server: shutdown triggered` with no preceding error,
warning, or OOM message anywhere in the log — the same "clean" shutdown
sequence the server runs on a genuine SIGTERM, and the same signature this
file already has two other unresolved instances of (the multimodal-warmup
`KeyboardInterrupt` entries above): an external signal with no identified
sender. Timing is suggestive (during a window of unusually many short-lived
`docker exec python3` processes against the same container) but not proof —
no evidence ties the two together beyond proximity, and the benchmarks
themselves ran to completion and printed correct results beforehand.
Restored production immediately (stop, flush, relaunch, verified with a real
generation request). Flagging rather than concluding: if this recurs during
similarly benchmark-heavy sessions, that pattern would be worth treating as
a real lead rather than coincidence.

## FP8 LM head: real hardware win confirmed, then a hard end-to-end failure

Follow-up to the bf16-vs-fp8 scoping question above. Goal: get from "here's
what it would take" to an actual patched boot with a real quality check.

### Attention/MLP-shaped FP8: bigger and more genuinely Blackwell-native than the LM head

Same isolated-benchmark methodology as the LM head test, applied to this
checkpoint's real MLA attention and MLP dense-layer shapes (TP=2 per-rank):

| Layer | weight | bf16 | fp8 | speedup | fp8 kernel |
|---|---:|---:|---:|---:|---|
| `q_a_proj` [4096→1536] | 12.6 MB | 19.0us | 13.4us | 1.41x | `nvjet_sm121_..._tmaAB` |
| `kv_a_proj` [4096→512] | 4.2 MB | 11.7us | 8.5us | 1.38x | `nvjet_sm121_..._tmaAB` |
| `q_b_proj` [1536→8192] | 25.2 MB | 77.9us | 76.1us | 1.02x | `nvjet_sm121_..._tmaAB` |
| `kv_b_proj` [512→16384] | 16.8 MB | 48.8us | 21.6us | **2.26x** | `nvjet_sm121_..._tmaAB` |
| `o_proj` [8192→4096] | 67.1 MB | 313.4us | 170.2us | 1.84x | `nvjet_sm121_..._tmaAB` |
| MLP gate/up [4096→6144] | 50.3 MB | 249.6us | 91.0us | **2.74x** | `nvjet_sm121_..._tmaAB` |

**These dispatch genuinely SM121-native kernels** (`nvjet_sm121_qqtst_mma_..._tmaAB`,
using TMA — a real Blackwell code path), unlike the LM head's own fp8 kernel,
which fell back to an SM89-targeted one. Five of six shapes show real,
sometimes better-than-2x speedups (MLP gate/up: 2.74x — beyond what halving
bytes alone predicts, meaning the fp8 kernel is also more *efficient* here,
not just moving less data). `q_b_proj` is the outlier at 1.02x, not yet
explained. This is a substantially better result than the LM head's 1.54x —
worth its own follow-up regardless of what happened below.

### Real-checkpoint validation (before booting anything)

Loaded this checkpoint's actual `lm_head.weight` and layer 3's real MLA
attention weights (layer 3 confirmed via `model.safetensors.index.json` as
one of 12 MLA layers out of 45 — the other 34 use mamba/gated-linear
attention) directly via `safetensors.safe_open`, no synthetic data:

- **Zero outlier channels** (>10x mean magnitude) in any checked tensor —
  the classic per-tensor-FP8 failure mode (a few extreme-magnitude channels
  crushing everyone else's precision) doesn't apply here.
- LM head fp8 vs fp32 reference (random probe activations): 100% argmax
  agreement, 93.3% top-10 overlap, matching bf16's own 100%/100%.
- Attention-layer intermediate projections showed lower "argmax agreement"
  under fp8 (66.7% on some), but **so did bf16 on the same layers** — this
  metric isn't meaningful for intermediate hidden states (argmax over an
  arbitrary feature vector has no natural interpretation, unlike logits).
  Correctly read as "isolated synthetic-activation testing has hit its
  limit here," not as a quality signal either way.

This all pointed toward "worth a real boot" — and vLLM's own
`cutlass_fp8_supported()` / `cutlass_block_fp8_supported()` both confirmed
`True` for SM121 beforehand, per the earlier scoping conversation.

### Two multiprocessing bugs before the patch even reached the right layer

Recipe: `recipes/glm-5.3-flash-exl3-v2-fp8-lmhead.yaml`. Same in-process
monkeypatch technique validated for the spinwait fix (`Exl3Config` lives in
root-owned site-packages; this container runs as uid 1000 per sparkrun, so
no file on disk gets edited) — wraps `Exl3Config.get_quant_method()` to
route the LM head through vLLM's own `Fp8LinearMethod`
(`is_checkpoint_fp8_serialized=False`, `activation_scheme="dynamic"`;
`weight_block_size` must be `None` — vLLM's `Fp8Config` rejects block-wise
scaling unless the checkpoint is already fp8-serialized, so per-tensor
dynamic isn't a compromise, it's the only scheme this load-time approach
can use).

**Attempt 1**: `isinstance(layer, LinearBase)` as the gate. Booted clean,
generated real (if untested) text — but silently did nothing. No "routing"
log line ever appeared; GPU memory usage was *higher* than the bf16
baseline (88.34 GiB vs ~86.48 GiB), not lower. Root cause: `ParallelLMHead`
is **not** a `LinearBase` subclass (MRO: `ParallelLMHead` →
`VocabParallelEmbedding` → `PluggableLayer` → `Module`) — confirmed
directly (`issubclass(ParallelLMHead, LinearBase)` → `False`). The
isinstance check silently excluded the one layer this whole recipe exists
to patch.

**Attempt 2**: widened the check to `isinstance(layer, (LinearBase,
ParallelLMHead))`. Verified this is architecturally sound before
relaunching: `Fp8LinearMethod.create_weights`'s positional signature
(`input_size_per_partition, output_partition_sizes, input_size,
output_size, params_dtype`) lines up exactly with how
`VocabParallelEmbedding.__init__` calls `quant_method.create_weights`, and
vLLM's own source explicitly exempts `ParallelLMHead` from the "quant
method must implement `.embedding()`" requirement it enforces for real
embedding layers. Should have worked — **still didn't fire, zero times,
for any layer at all**, confirmed with an unconditional debug print inside
the patched method (not even `RoutedExperts`, which every boot this whole
project has ever done proves gets a real `get_quant_method` call).

Root cause: the `if __name__ == "__main__":` guard that fixed the
*previous* recipe's fork-bomb bug (see the spinwait section above) was
gating the wrong thing here. Two separate multiprocessing boundaries are in
play — (1) vLLM's API-server bootstrap crosses a `multiprocessing.spawn`
boundary that re-imports the launching script as a module, and (2)
EngineCore/Worker — where model construction and `get_quant_method` calls
actually happen — are **forked from that spawned child**, not from the
original top-level invocation. Gating the patch call itself behind
`__main__` (matching the spinwait shim's structure exactly) meant the
spawned child, which re-imports the file with `__name__ != "__main__"`,
never applied the patch at all — and everything forked from it inherited
the unpatched class. The spinwait patch never hit this because
`shm_broadcast` gets used by the original top-level process directly, not
only inside the spawned/forked descendants.

**Fix**: split the two concerns. The monkeypatch itself now runs
unconditionally at import time (every process that loads this file gets
it, `__main__` or not); only the actual `vllm.entrypoints.cli.main:main()`
call stays behind the `__main__` guard, so vLLM still only launches once.
Confirmed working via a `pid`/`__name__` tag on the confirmation log line:
patched in 3 separate processes this boot (`pid=75 __main__`, `pid=175
__mp_main__`, `pid=207 __mp_main__`), and the routing line finally fired
for the real layer: `routing 'language_model.lm_head' (ParallelLMHead)
through Fp8LinearMethod`.

### It booted, CUDA graphs captured clean, memory dropped — and output was garbage

With the patch actually reaching the right layer: booted without error,
`Graph capturing finished in 5 secs, took 0.37 GiB` (CUDA graph capture
survives fine), and consumed weight memory dropped to 83.99 GiB (down from
~86.48 GiB baseline and the two broken attempts' 88-89 GiB) — real
confirmation the fp8 path was structurally active this time.

**Every single generated response was garbage** —
`����...` (Unicode replacement characters, meaning
invalid/nonsensical token IDs) followed by unrelated repeated letters.
Reproduced on two separate prompts, `finish_reason: length` both times (it
ran to the token cap rather than hitting a stop token, consistent with
totally incoherent logits rather than a formatting quirk). This is total
failure, not subtle degradation — the isolated real-weight correctness
check earlier in this session (100% argmax agreement) did not predict it,
because that check used synthetic random activations, not real hidden
states from an actual 45-layer forward pass.

**Root cause, found by reading `create_fp8_weight_parameter`** (not yet
independently reproduced in isolation, so flagged as a strong hypothesis,
not a fully closed loop): it allocates the weight parameter as
`torch.float8_e4m3fn` **from the start** —
`torch.empty(..., dtype=torch.float8_e4m3fn)` — before any weight data has
been loaded. Standard vLLM weight loading then copies the checkpoint's
bf16 tensor into this fp8-typed parameter, which means an implicit,
**unscaled** dtype cast happens at load time, before
`process_weights_after_loading` ever gets a chance to compute a proper
scale. This checkpoint's real lm_head weights are small in magnitude
(max ≈0.22, mean ≈0.015-0.02, confirmed directly from the safetensors
data) — right down in e4m3's subnormal range, where a handful of mantissa
bits leaves almost no usable precision without first scaling the values up
into e4m3's useful dynamic range (up to ±448). A naive unscaled cast at
those magnitudes doesn't lose a little precision, it destroys nearly all
of it. `Fp8LinearMethod`, used this way — attached directly to a layer via
a patched `get_quant_method`, bypassing whatever the real `--quantization
fp8` top-level CLI flag's own model-loading integration does — appears to
assume either an already-fp8-serialized checkpoint or a scale-aware
weight-loading path that a bare per-layer monkeypatch doesn't provide.

**Verdict: the hardware/kernel case for fp8 remains strong (confirmed real
speedups, genuinely SM121-native kernels on 5 of 6 attention/MLP shapes),
but this specific implementation path — attach `Fp8LinearMethod` directly
via a patched `get_quant_method`, dynamic/per-tensor, no checkpoint
changes — does not work as built.** The likely fix is either a custom
weight loader that stages through bf16 and computes+applies a scale before
casting (replicating what a genuine `--quantization fp8` model-loading
integration presumably does), or accepting that real fp8 serving needs an
actual offline-converted, scale-calibrated checkpoint (AutoFP8/
llm-compressor-style) rather than a live monkeypatch — which is also the
more standard way production fp8 deployments are done. Not attempted
further this session. Production restored and verified after every attempt
in this section; nothing here shipped.

### Full probe suite, run through to completion against the known-broken build

The write-up above stopped at two ad-hoc curl checks once garbage output
was confirmed and reproduced. Went back and ran the actual project probe
suite against the same fp8-lmhead boot anyway — partly because "quality
check" deserves the same standard tooling this project uses for every
other recipe, not ad-hoc curl, and partly because a formal run surfaced
real information the curl checks didn't: which specific mechanisms survive
the corrupted LM head and which don't.

**`probe_sanity.py`** — 3 of 9 checks failed, and the *pattern* of what
passed is the useful part:

```
[PASS] models-lists-served-id / metrics-enabled / chat-nonempty
[FAIL] chat-finish-stop — finish_reason=length (never predicts a stop token)
[FAIL] chat-coherent — garbage (Unicode replacement chars)
[PASS] stream-first-token — ttft=0.30s (matches the healthy baseline)
[PASS] stream-usage
[FAIL] stream-counts — can't even count 1 to 5
```

Every structural/mechanical check passes (routing, streaming, usage
accounting, TTFT); every check that touches actual generated content
fails. **Decode throughput also collapsed**: 11.6-11.7 tok/s across three
bench runs, versus the healthy baseline's 25-31 tok/s — a new finding the
earlier ad-hoc checks didn't surface. The likely mechanism: MTP's draft
model still predicts against the *original* (undamaged) hidden-state
distribution, but the corrupted LM head means the target model's verified
tokens no longer match those drafts, so acceptance collapses toward zero
and decode falls back to slow single-token-per-step generation on top of
already-wrong output.

**`probe_soak.py`** — **PASSED**, all 7 checks, including
`endpoint-alive-after-soak`. Worth stating plainly so this isn't
misread out of context: soak only checks that requests complete without
crashing or timing out and the server survives concurrent load — it has
no content-correctness check. A soak pass here means "the broken model
serves garbage stably," not "the model works." The two probes are
answering different questions and both results are exactly what a correct
run should show.

**`probe_longctx.py`** (100K tokens, matching the scale used throughout
this file): **TTFT 83.3s — matches the healthy baseline's 83.4s at the same
context length almost exactly**, and `prompt-tokens-near-target` passed.
This is the most precise localization in this whole investigation: prefill
and context-handling are completely unaffected by the fp8 LM head patch,
exactly as the architecture predicts (prefill only touches the LM head
once, for the first generated token's logits — the other 45 layers of
actual context processing never go near it). `finish-stop` and
`codes-retrieved` (0/4) failed for the same reason as every other content
check. The damage is precisely isolated to the LM head's own weight, not
anything upstream.

**Net effect of running the full suite**: the conclusion is unchanged (this
implementation doesn't work, root cause is the unscaled fp8 cast) but now
backed by the project's standard tooling rather than two curl calls, with
two genuinely new findings — the decode-throughput collapse via broken
speculative-decode acceptance, and the clean prefill/decode damage
boundary. Production restored and re-verified with a real generation
request after this run, same as every other attempt in this section.

## FP8 LM head, attempt #2: scale-aware loader (fixes the root cause, doesn't just work around it)

Options for closing the gap, in the order they're worth exploring if effort
isn't the constraint: (1) a scale-aware dynamic loader that fixes the exact
bug found above, (2) per-channel instead of per-tensor weight scaling on
top of that, (3) extending the same validated mechanism to attention/MLP
(bigger win — ~30.8% of decode kernel time vs. the LM head's ~9.5% — but
larger blast radius, and per-layer isolated checks are known to be
non-diagnostic, so scope only after the mechanism is proven), (4) an
offline-calibrated (AutoFP8/llm-compressor-style) checkpoint as the
eventual production-hardened path. None of these are mutually exclusive —
they compose — the sequencing is about which question each one answers and
what it depends on knowing first.

**Reading vLLM's actual source (not just re-testing) found a second,
worse bug than the one first written up above.** `create_fp8_scale_parameter`
initializes `weight_scale` to `torch.finfo(torch.float32).min` (~ -3.4e38)
as a sentinel, expecting a real fp8-serialized checkpoint to carry a
`*.weight_scale` tensor that the loader matches by name and overwrites.
A plain bf16 checkpoint has no such key — nothing ever overwrites the
sentinel — and `process_weights_after_loading`'s non-block-quant path uses
`layer.weight_scale` completely as-is. So v1's failure wasn't just an
unscaled cast losing precision on subnormal-range weights; the real SM121
GEMM kernel was running against a garbage weight **and** a -3.4e38 scale.
That second bug alone is sufficient to explain "every generation is
replacement characters, never predicts a stop token" — worse than bug 1
alone would produce.

**Fix: `ScaleAwareFp8LinearMethod`**, a subclass of vLLM's own
`Fp8LinearMethod` (`glm-5.3-flash-exl3-v2-fp8-lmhead-v2.yaml`) that overrides
exactly the two methods responsible:
- `create_weights` allocates the weight Parameter in `params_dtype` (bf16),
  not fp8 — the standard loader then does a normal, lossless same-dtype
  copy from the checkpoint. No `weight_scale` parameter is created here at
  all (there's nothing in the checkpoint for the stock loader to match it
  against). Kernel selection (`init_fp8_linear_kernel`, `use_marlin`
  detection) is copied verbatim from upstream so the real SM121 fast-path
  kernel is chosen exactly as before.
- `process_weights_after_loading` runs after the real bf16 weights are
  loaded, computes an actual scale from them (`w.abs().amax() / 448.0` —
  the same math already validated in `real_weight_fp8_check.py`), quantizes
  with that scale, then hands off to the upstream kernel's own
  `process_weights_after_loading` (16-alignment padding etc.) so everything
  downstream of weight prep is stock vLLM code. A `GLM53_FP8_LMHEAD_SCALE_MODE`
  env var (`tensor` default, `channel` available) picks per-tensor vs.
  per-output-row scaling on the same recipe file, for the #2 → #3 A/B
  without a rebuild.

**Boot 1** hit an unrelated external `SIGTERM` mid-boot (during multimodal
video-processor warmup, well after weight loading/quantization/CUDA graph
capture all completed cleanly) — traced to the tooling session's own
foreground process-tracking reaping a long-running launch command, not
anything in this patch. Confirmed from that boot's logs before restarting:
`[glm53-fp8-lmhead-v2] quantized weight scale_mode=tensor scale_shape=(1,)
scale_range=[0.000481742, 0.000481742] weight_max_before=0.2158` — scale is
exactly `0.2158/448`, real weight statistics, not a sentinel — and CUDA
graph capture completed (FULL + PIECEWISE, prefill + decode, "Graph
capturing finished in 6 secs"). Relaunched with the launch command properly
detached from the tooling session (`setsid nohup ... & disown`) rather than
tracked as a long foreground call; booted clean on retry, same quantization
line, same successful graph capture ("finished in 5 secs, took 0.37 GiB").

**Quality check**: two prompts (an explanatory paragraph, a code+reasoning
task), both `finish_reason=stop`, both fully coherent — no replacement
characters, no `finish_reason=length` runaway. A qualitative night-and-day
difference from v1's `���...` on the same two-prompt bar.

**`probe_sanity.py`**: **ALL CHECKS PASSED**. Decode 25.8-28.28 tok/s
(median 27.25), TTFT 0.309-0.336s — squarely inside the shipped bf16-lm-head
baseline's own range (23.49-30.44 tok/s, TTFT 0.215-0.32s) with no
regression, and if anything a touch better at the median, though not
distinguishable from run-to-run noise at n=3.

**`probe_soak.py`**: **PASSED**, all 7 checks — 3 sequential rounds (10/10
each, median 2.68-3.54s), 3 concurrent waves (3/3 each, median 0.76-8.14s),
`endpoint-alive-after-soak`. Unlike the v1 soak run, this one is meaningful
as a correctness signal too, not just a stability one — every response in
every round is real coherent content this time.

**`probe_longctx.py`** (100K tokens): **ALL CHECKS PASSED**. TTFT 85.5s
(vs. the healthy baseline's 83.3-107.1s across prior runs at this scale —
no regression), `finish-stop`, and **4/4 planted codes retrieved
correctly** (`CODE-1-BBBB, CODE-2-PPPP, CODE-3-JJJJ, CODE-4-CCCC`) — the
first successful codes-retrieved result anywhere in this fp8-lmhead thread;
v1 failed this check by construction (garbage output). This is the real,
positive counterpart to v1's clean *damage*-isolation finding: the LM head
was the only thing broken, and now that it's fixed, everything downstream
of it works too.

**Verdict: attempt #2 is a clean, fully-validated positive result.**
Correct per-tensor dynamic FP8 quantization on the LM head — real weights,
real scale, real SM121 kernel — holds up through the complete probe suite
plus real multi-turn quality checks, with throughput matching (not
regressing) the shipped bf16 baseline. Not yet promoted to production;
left running as the active experiment while #3 (per-channel scaling) and
#4 (extending the same mechanism to attention/MLP) are explored, per the
plan agreed this session. `glm-5.3-flash-exl3-v2-fp8-lmhead.yaml` (v1) is
kept as-is, with its header now historical — the bug it exposed is real and
well-documented, this is simply the fix.

## FP8 LM head, attempt #3: per-channel scaling — no measurable difference from per-tensor

Same recipe, `GLM53_FP8_LMHEAD_SCALE_MODE=channel` (env override, no
rebuild). One shape bug caught before booting by reading vLLM's own kernel
padding code rather than trusting the `cutlass_scaled_mm` docstring's
broadcast description: the docstring implies `scale_b` for a `[K,N]`
weight should be `[1,N]`, but `CutlassFP8ScaledMMLinearKernel.process_
weights_after_loading`'s own padding logic (`.view(-1, *weight_scale.shape
[1:])` after flattening) only reshapes cleanly if the real convention is
`[N,1]` — confirmed against real executable code, not documentation, before
spending a boot cycle on it.

Booted clean: `scale_shape=(77440, 1)`, `scale_range=[8.99e-05, 4.82e-04]`
— a genuine >5x spread across vocab rows, so there IS real per-row
variation for per-channel scaling to capture. CUDA graphs captured (5s,
0.36 GiB). Quality check: identical coherent output to per-tensor mode on
the same prompt. `probe_sanity.py`: **ALL PASSED**, decode 27.66-29.04
tok/s (median 28.64) — statistically indistinguishable from per-tensor's
25.8-28.28 (median 27.25).

**Verdict: no measurable win from per-channel over per-tensor, on this
layer.** Consistent with the original correctness check's finding of zero
outlier channels (>10x mean) in `lm_head.weight` — the checkpoint's actual
weight distribution is well-behaved enough that a single global scale
already captures nearly all the precision per-channel could add. Not worth
carrying the extra complexity for this layer; per-tensor (attempt #2)
remains the simpler, equally-correct choice. Skipped a full soak/longctx
re-run here — the underlying mechanism is identical to the already
fully-validated #2, and the only variable (scale granularity) shows no
signal worth chasing further with more probe time.

## FP8 LM head, attempt #4: widen scope — real finding is a correction, not a benchmark

`glm-5.3-flash-exl3-v2-fp8-wide.yaml` widened the same `ScaleAwareFp8LinearMethod`
routing to every `LinearBase`/`ParallelLMHead` layer except `RoutedExperts`
(prefix contains "experts") and the MoE router (`mlp.gate` — exact last
path segment "gate", deliberately NOT matching `mlp.gate_proj`, which is
the dense-MLP layer we do want in scope). The intent, per the plan agreed
this session, was to reach attention (both the MLA and gated-linear-
attention/mamba families) plus dense MLP — attention/MLP's combined ~30.8%
decode-kernel-time share is the bigger prize next to the LM head's ~9.5%.

**It booted clean, passed the full probe suite, and the real finding is
that it didn't reach attention at all — for a reason worth understanding,
not a bug to route around.** Only **8 layers** got routed: the 3
dense-layer MLPs (`layers.{0,1,2}.mlp.gate_up_proj` + `down_proj`, dense
because `first_k_dense_replace=3`) and both LM heads (main +
MTP draft). Zero attention layers matched — not because the exclusion
predicate was wrong, but because attention never reaches
`Exl3Config.get_quant_method` at all. Read directly from this container's
`vllm/models/glm5next/nvidia/model.py`:

```
self.self_attn = Glm5NextMLAAttention(
    ...
    quant_config=None,  # MLA projections are BF16 in checkpoint
    ...
```

— an explicit, hardcoded `None` at the call site, not a config value. Since
`LinearBase.__init__` only calls `quant_config.get_quant_method(...)` when
`quant_config is not None` (falling back straight to `UnquantizedLinearMethod`
otherwise), no monkeypatch of `Exl3Config` — however the routing predicate
inside it is written — can ever see these layers. The same pattern repeats
lower in the file for the mamba/gated-linear-attention layers ("pattern
(quant_config=None for BF16 submodules)"). This is a deliberate choice made
in vLLM's own model code for this architecture, not a gap in this
checkpoint's `Exl3Config` scope declaration.

**This changes the addressable-surface picture from the Blackwell-pivot
investigation.** That investigation's ~30.8% "attention+MLP" kernel-time
share and the 1.4x-2.74x real-weight speedup measurements were both
produced by extracting checkpoint tensors directly via `safetensors` and
benchmarking them in isolation — a path that never goes through vLLM's
model construction or `get_quant_method` at all, so it correctly measured
what the *hardware* can do on those shapes, but couldn't reveal that the
*model code* hardcodes attention to bf16 regardless of quant config.
Reaching attention for real would mean patching `Glm5NextMLAAttention.__init__`
(and its mamba counterpart) directly to override the hardcoded `None` —
a qualitatively different, more invasive intervention than a
`get_quant_method` monkeypatch, and one that overrides a choice the model
authors made on purpose (plausibly: MLA's compressed latent KV cache
compounding precision loss over long-context attention is a different risk
profile than a single LM head or an FFN's independent-per-token output —
worth treating as a real signal, not just an obstacle, until investigated
further). Not attempted this session.

**What actually validated, on its own merits**: LM head + the 3 dense-MLP
layers, fp8'd together. Quality check: coherent on the same code+reasoning
prompt used throughout this thread. `probe_sanity.py`: **ALL PASSED**,
decode 25.35-29.11 tok/s (median 28.35) — no regression, consistent with
the earlier LM-head-only numbers (expected: 3 dense-MLP layers out of 45
is a small slice of total FFN compute next to 42 layers of untouched,
already-EXL3-quantized routed experts). `probe_soak.py`: **PASSED**, 7/7.
`probe_longctx.py` (100K tokens): **ALL PASSED**, TTFT 82.9s (matching
baseline), 4/4 planted codes retrieved.

**Verdict: a real, validated, small-but-genuine widening (LM head + 3
dense-MLP layers), not the attention win originally targeted.** Production
restored and re-verified with a real generation request after this run.

## Summary for review: where this leaves the four options

- **#2 (scale-aware dynamic loader on the LM head)**: fully validated,
  clean win, matches baseline throughput. Ready to promote if desired.
- **#3 (per-channel scaling)**: tested, no measurable difference from #2's
  per-tensor default. Not worth the extra complexity for this checkpoint.
- **#4 (widen scope)**: the mechanism cleanly extends to any *bf16 Linear
  layer that vLLM's model code actually routes through a quant config* —
  which turned out to be only the LM head and the 3 dense-MLP layers for
  this architecture, not attention. That's still validated and real, just
  smaller than planned.
- **Attention fp8 (the bigger prize — ~30.8% of decode kernel time)**
  remains unreached: it requires patching the model's attention module
  construction directly (overriding a hardcoded `quant_config=None`), a
  materially different and more invasive change than anything tried this
  session, and one that overrides what looks like a deliberate precision
  choice by the model's authors rather than an oversight.

## Does #2 actually move throughput? Rigorous A/B says yes, clearly

The n=3 `probe_sanity.py` numbers throughout this thread (23.49-30.44 tok/s
baseline vs. 25.8-29.11 across #2/#3/#4) all overlap — not enough signal to
call a win, given a back-of-envelope ceiling (LM head ~9.5% of decode
kernel time x 1.54x speedup) suggested only ~3.3 percentage points of
possible improvement, smaller than that noise band. Built
`probe_throughput_ab.py` to actually resolve this: one fixed prompt,
temperature=0, 400-token forced completions (many more decode steps per
run averages out per-request scheduling jitter), n=20 per config.

| Config | n | mean tok/s | median | stdev | min | max |
|---|---|---|---|---|---|---|
| Baseline (bf16 lm_head) | 20 | 23.658 | 23.552 | 0.498 | 22.835 | 25.108 |
| #2 (fp8 lm_head, tensor) | 20 | 25.599 | 25.520 | 0.716 | 24.447 | 27.269 |

**+1.941 tok/s, +8.2%, pooled-SE t-stat ≈ 9.95 — unambiguous, not noise.**
Larger than the theoretical ceiling estimate, most likely because that
estimate used a kernel-time share measured without accounting for MTP-2
speculative decoding invoking the LM head multiple times per accepted
step (draft verification + new-token prediction), so its real contribution
to wall-clock decode time is higher than a single-pass kernel-time-share
figure suggested. **This is the first real, statistically solid throughput
win in the whole fp8 investigation.** Production restored and re-verified
after this benchmark.

## Attention fp8: investigated the "why", not just the "how" — traces to the model creators' own quantization scheme, not a vLLM gap

Per request, dug into *why* attention is excluded before writing any
override. `vllm.models.glm5next` (this container's model implementation)
is not on vLLM's released `main` branch at all — it ships from
**vllm-project/vllm#53906, "[Model] add GLM-5.3-Flash support", still
OPEN**, an in-flight PR the container was built from at commit
`487ecf187` (2026-08-25). That alone would be reason for caution, but the
actual evidence goes further and is more specific than "unfinished PR":

`model.py` has a function, `_try_load_fp8_attn_proj`, whose docstring
reads: *"Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on
load. The FP8 checkpoint stores these as block-FP8 (weight +
weight_scale_inv) ... When the model target is BF16 (no weight_scale_inv
param) we dequantize; otherwise we return False so the normal
stacked/direct path loads the FP8 tensor as-is."* Two things this proves:

1. **The real, official `zai-org/GLM-5.3-Flash` FP8 checkpoint DOES store
   these attention projections in block-FP8** — with real, calibrated
   `weight_scale_inv` tensors, produced by the model's own creators, not
   a gap anyone needs to fill.
2. **vLLM's model code actively converts them back to BF16 for serving,
   on purpose** — the function only fires because the attention Linear
   layers are constructed with `quant_config=None` (so no
   `weight_scale_inv` param exists on the model side to receive the
   checkpoint's real fp8 data); if that weren't the case, the function
   explicitly backs off ("return False") and lets the checkpoint's own
   real fp8 weight+scale load through untouched.

So the mechanism is conditional, not absolute — but the condition it's
conditioned on (`quant_config=None`, hardcoded uniformly for every
sub-projection of both `Glm5NextMLAAttention` and the KDA/mamba
counterpart, per the two near-identical comments found in `model.py` and
`kda.py`) is itself a choice, and it means: **even the official
zai-org checkpoint's own calibrated FP8 attention weights get discarded
and reconstructed as BF16 by this vLLM support code, every time.** This
isn't an unimplemented feature or a conservative default pending
follow-up — it's the vLLM integration faithfully mirroring a quantization
scope decision the model's own creators made when they produced the
official FP8 release: MoE experts, dense MLP, LM head get FP8; every
attention projection (both architectures) stays BF16, unconditionally.
`o_proj` is additionally confirmed excluded via `modules_to_not_convert`
per the same docstring.

**Conclusion**: attention fp8 is not "harder to implement," it's working
against a specific, traceable, first-party accuracy decision — encoded
independently in two places in the vLLM integration (the hardcoded
`None`s, and the dequantize-on-load fallback that only exists because
real fp8 attention data needed somewhere to go). Overriding it is still
possible (same monkeypatch technique used throughout this session would
work mechanically), but it would be deliberately serving attention at a
precision the model's own creators evaluated and rejected, not exploring
an open question. Flagged back to the user for a decision before any
further work here.

**Decision: promote #2, leave attention alone** — not worth the error risk
of going against a calibrated, first-party decision, given #2 alone is
already a real, statistically solid +8.2% win.

## Promoted: glm-5.3-flash-exl3-v4-vllm.yaml is now the shipped default (2026-09-02)

`glm-5.3-flash-exl3-v4-vllm.yaml` = v2's fully validated base (all four
PR77/86/63/96 changes, MTP-2, MNBT=7168, gpu_memory_utilization=0.86)
plus the `ScaleAwareFp8LinearMethod` LM-head patch from attempt #2, scale
mode `tensor` (the validated default — `channel` remains available via
`GLM53_FP8_LMHEAD_SCALE_MODE=channel`, same recipe, no rebuild). Verified
byte-for-byte against `glm-5.3-flash-exl3-v2-fp8-lmhead-v2.yaml`'s diff
against `glm-5.3-flash-exl3-v2-vllm.yaml` before merging, so nothing
production-relevant was dropped in the promotion.

Booted clean on the first attempt: patch fired
(`scale_shape=(1,) scale_range=[0.000481742, 0.000481742]
weight_max_before=0.2158` — matches every prior #2 boot exactly), CUDA
graphs captured (5s, 0.38 GiB), real generation request coherent
(finish_reason=stop), `probe_sanity.py` all-pass (decode 25.23-28.77
tok/s, matching the validated #2 numbers). This is now the live default;
`glm-5.3-flash-exl3-v2-vllm.yaml` remains on disk as the pre-FP8 baseline
(and the fallback if anything regresses), same pattern as v1 being kept
after v2 superseded it.

## Applied the K-pool tail cache fix (vcruz305's report): v5 promoted

While researching *why* attention fp8 is excluded (the prior GitHub
investigation), found `vcruz305`'s report on the same PR thread
(vllm-project/vllm#53906): a real, reproducible crash bug in this
container's hybrid-model support, reproduced on the SAME hardware class
(DGX Spark GB10, EXL3 pack) and the SAME vLLM commit (`487ecf187`) this
container runs. Any generation exceeding ~2.2K *generated* tokens crashes
with a CUDA illegal memory access, or silently corrupts a neighboring
layer's KV data — independent of `max-model-len`. Confirmed present in
this exact container before patching: both anchor lines
(`MambaHybridModelState.prepare_attn`'s missing `positions=` kwarg,
`compute_kpool_tail_slot_mapping`'s `slot_mapping.clone()`) matched
byte-for-byte in the installed site-packages.

**Root cause**: every hybrid-architecture forward pass takes
`MambaHybridModelState.prepare_attn`, which calls `build_attn_metadata(...)`
without `positions=` (unlike the plain-transformer path in `default.py`,
which does). The K-pool tail cache's slot-mapping builder needs
`positions` to compute each request's own circular tail block; without
it, the tail group falls through to the generic paged mapping, which
indexes a one-entry block-table row and produces garbage block ids that
both kpool kernels write through with no bounds check. A second,
compounding bug: once `positions` are supplied, the tail-mapping function
used to return `slot_mapping.clone()` every step — a fresh tensor whose
address CUDA graph capture bakes in, so replay reads a since-freed/reused
buffer (the actual illegal-access trigger with graphs on).

**Fix, applied as a live monkeypatch** (source-text surgery + `exec` against
the real installed source, not a file edit — same constraint as every
other patch this session, uid 1000 container, root-owned site-packages):
`MambaHybridModelState.prepare_attn` and `compute_kpool_tail_slot_mapping`
are re-derived from `inspect.getsource()` of the CURRENT installed
functions with the two buggy blocks replaced (asserting the anchor text
matches exactly once, so the patch fails loudly rather than silently
no-op'ing if this vLLM build ever changes), then reassigned onto the
class/module. Mirrors vcruz305's own published file-edit patch
(`scripts/patch_kpool_tail_positions.py`) exactly, just applied in-memory.

One real bug caught before booting: the anchor text needed 8-space
indentation after `textwrap.dedent()`, not the 12-space indentation the
raw (non-dedented) class-method source shows — `textwrap.dedent()` only
strips the *common* leading whitespace (4 spaces, the method's own
class-body indent), it doesn't re-normalize every line to zero. Caught by
dry-running the exact patch function against the live container before
writing it into the recipe, not by trusting the transcription.

**Validation**: both patch markers fired at boot
(`[glm53-fp8-lmhead-v4]` and `[glm53-kpool-tail-fix]`, confirming clean
coexistence with the v4 FP8 patch — different files/classes, no
interaction). The critical test: a **4,096-token forced completion**
(temperature 0, well past the ~2.2K crash threshold, matching the
reporter's own 4,096/8,192-token validation bar) completed cleanly —
`finish_reason=length`, `completion_tokens=4096`, coherent technical
content throughout, no crash, no corruption, server healthy afterward.
`probe_sanity.py`: **ALL PASSED**, decode 24.17-26.74 tok/s — matching v4's
numbers, no regression from the fix.

**Promoted as `glm-5.3-flash-exl3-v5-vllm.yaml`** — v4's validated base
(FP8 LM head) plus this crash fix. This is now the live default. Given
how directly this bug applies to real usage (any sufficiently long
generation, not an edge case), this took priority over the throughput
investigation items still open.

## Unexplained production shutdown during the long-context concurrency sweep (2026-09-03, 01:10)

Started `probe_longctx_concurrency.py` against the freshly-promoted v5
(2 concurrent 20K-token prefills, the first of 6 planned cells). At
01:10:22 — mid-request — the API server logged a completely clean,
graceful shutdown sequence (`[shutdown] API server: shutdown triggered`
-> SIGTERM to EngineCore -> HTTP server shutdown -> `Application shutdown
complete`), no traceback, no CUDA error, no OOM
(`docker inspect --format '{{.State.OOMKilled}}'` -> `false`). The
container itself never died (`docker ps` showed it still "Up", just the
server process inside was gone) — an external SIGTERM, not an engine
crash. `journalctl`/`dmesg` on the node were inaccessible from this
session (permission-restricted) so the actual sender could not be
identified this time either.

**This is now the THIRD occurrence of this exact signature in this
project's history**: once noted in the v2 recipe's own header (during
multimodal warmup, sender never identified), once during this session's
first fp8-lmhead-v2 boot (traced that time to this tooling session's own
background-task reaping of a long-running foreground launch — fixed by
switching to `setsid nohup ... & disown` for all subsequent launches),
and now this third time, well after a stable 26-minute boot, during
legitimate concurrent read load. The tooling-reaping explanation from the
second occurrence does NOT apply here (the launch itself had completed
and been serving successfully for 26 minutes). `fleet_watchdog.sh`
(this repo's own auto-recovery script) was checked and ruled out — it
targets an entirely different, inactive topology (TP4/4-node,
`vllm_glm53` container naming) and was not running.

Restored via stop/flush/relaunch, reverified with a real generation
request. Not treated as a regression from anything shipped this
session (the shutdown sequence is textbook external-SIGTERM, structurally
distinct from the K-pool tail crash just fixed, which would show a CUDA
illegal-access traceback, not a clean shutdown). Flagged to the user
before resuming the concurrency sweep, given the pattern is now
recurring and still uncorrelated with a specific trigger.
