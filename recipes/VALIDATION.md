# GLM-5.3-Flash sparkrun recipes — validation record

**This file holds only the currently-promoted version's validation record.**
Everything before it lives in `recipes/validation-archive/`, one file per
prior version, in the order they were promoted (or, for unpromoted
experiments, the version they were investigated under). Read this file
first; go to the archive only when you need history a specific past version
found, fixed, or ruled out.

## How to use this file (for any future agent/session)

- **Before touching the serving recipe**, read this file's own entry below
  in full — it's the currently-shipped configuration's reasoning, not just
  its flags.
- **Before re-investigating something that looks like a new bug**, grep
  `recipes/validation-archive/*.md` first. This project has hit the same
  handful of failure classes repeatedly (see "Where to look for common
  problems" below) — a new-looking crash is very often an old one.
- **When a new candidate is promoted**, move this file's current content
  into a new `recipes/validation-archive/vNN-<name>.md` (keep the exact
  content, just relocate it — don't summarize on the way out, the raw
  investigation detail is what makes the archive useful later), then replace
  this file's entry with the new version's own record. Keep this stub
  section (everything above and including "How to use this file") unchanged.
- **Auto-memory** (this session's persistent memory, not this repo) tracks
  faster-moving state — which recipe is live right now, open investigation
  threads, incidents in progress. This file is the durable, per-promotion
  record; memory is the current-status layer on top of it. If they disagree,
  trust a fresh `sparkrun status` / `git log` over either.

## Tracked upstream/sibling repos

Checked periodically ("routine upstream check") for anything applicable to
this fork. Add new repos here when the user names one, rather than letting
the rotation live only in memory/session context.

- **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** — the actual upstream this
  fork is built from (pinned, currently `c190db1`-lineage, drifted since).
  Source of the base image, most of the `overlay/patch_*.py` mechanisms,
  and the majority of upstream PRs this project backports. Checked to
  `main` @ `1caea9a` (2026-09-11) and again to `HEAD` as of 2026-09-13 --
  quiet round, see "Routine upstream check, 2026-09-13" below.
- **Enntity/sparkglm** — sibling project, same checkpoint family, same GB10
  hardware. Source of the grouped-prefill fat-expert kernel (v9-fatfork)
  and the TileLang JIT-cache-persistence fix. Moved its `main` branch to an
  NVFP4-focused research preview; EXL3-relevant work now lives on its
  `exl3` branch specifically — check that branch, not `main`. That branch
  has itself shifted toward a research/experiments structure (qualification
  protocols, rejected-candidate writeups) rather than shipped features as
  of 2026-09-13 -- checked `exl3` @ `a8aaa229..HEAD`, see below.
- **mmastrac/mentat** — a Ray-replacement control-plane project. Tracked
  but not applicable to this fork (uses vLLM's `mp` executor, not Ray) —
  see the mentat-track-closed history if this is ever reconsidered.
- **mmastrac/glm-5.3-flash-4x-gx10** — a 4-node/TP4 GLM-5.3-Flash
  deployment. Source of the OOM-observer diagnostic mechanism
  (`TORCH_MEM_FRACTION` + CUDA-allocator observer).
- **tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark** — surfaced the
  ModelOpt-format NVFP4 token-corruption bug (vLLM #54150) that's part of
  why this fork's NVFP4 lane uses a compressed-tensors checkpoint instead.
- **AEON-7/vllm-ultimate-dgx-spark** — added 2026-09-12. Independent (not a
  fork of MiaAI-Lab/sparkglm/etc.) from-source vLLM build for GB10/sm_121a,
  currently tracking vLLM v0.29.0, very active (pushed same-day as this
  check, real external contributors, 141 stars). Targets NVFP4/ModelOpt/
  compressed-tensors models (Gemma-4, Qwen3.6/3.8) with DFlash/DSpark spec
  decode -- **no EXL3, no GLM-5.3-Flash** as a first-class target, so
  nothing transfers at the quantization-scheme level. AEON's own patches
  are MIT-licensed (clean to port from). First check surfaced real,
  actionable leads on this fleet's two worst unsolved operational problems:
  see "AEON-7 initial triage" below.

## Where to look for common problems

This fleet (2x NVIDIA DGX Spark GB10, unified host/GPU memory, TP=2) has a
small number of failure classes that recur across almost every version.
Check these first before assuming a new bug:

- **A boot or request dies with no Python traceback, no CUDA error, "exit
  code: None," or the server just goes quiet.** Check `journalctl -u
  earlyoom` on both hosts *first*, before anything else. This fleet runs
  close to its memory margin under load (unified host/GPU memory on GB10),
  and earlyoom killing the API server or a worker process produces exactly
  this signature. It is usually **not** correlated with whichever candidate
  is serving — see the v20 entry below for a same-session controlled A/B
  proving this against production. Standard remedy: `sparkrun stop` on the
  job, `recipes/scripts/prelaunch_flush.sh <hosts> [--during-load]`, retry.
  Auto-memory has `earlyoom_root_cause_correction` and
  `gmu_086_earlyoom_regression` with more detail (fleet-wide, not this
  file's concern to re-derive each time).
- **`torch.AcceleratorError: CUDA error: an illegal memory access was
  encountered` during `Capturing CUDA graphs (PIECEWISE)`.** Hit
  intermittently (not deterministically) across multiple versions,
  including on a completely clean, unmodified retry of the exact same
  image. Treated as transient GB10 flakiness, not a code bug, unless it
  starts reproducing deterministically. Standard remedy: same as above
  (stop + flush + retry).
- **The recurring NCCL/shm-broadcast stall** ("No available shared memory
  broadcast block found in 60 seconds," repeating indefinitely instead of
  self-resolving). This is vLLM issue #51921 (GB10/sm_121-specific), still
  open upstream as of this file's writing. A candidate fix exists (PR
  #54929) but was assessed and rejected — see the v20 entry below for why.
  Standard remedy: stop + flush + retry; if a boot is left in this state
  unattended, production stays down until someone intervenes — this has
  actually happened (a ~6.5h incident, see `validation-archive/v19-
  indexercompat.md`). **Never leave a boot in this state unsupervised past a
  sane timeout.**
- **A KV-cache / host-memory-pressure crash on very long context (~240K+
  tokens).** Traced to host memory fragmentation (`nr_free_pages_blocks`,
  not raw `%MemAvailable`), below PyTorch's own allocator layer — see
  `validation-archive/v15-combined.md` and the fragmentation pre-flight
  check already wired into `prelaunch_flush.sh`.
- **`docker ps` shows both containers "Up" but the server is unreachable
  and hasn't logged anything in minutes.** Don't trust container uptime as
  a proxy for the service being alive — check `/health` and the actual log
  tail first. A real incident (2026-09-11/12): the head's API server got an
  external SIGTERM (not a crash — no traceback, no CUDA error, no
  earlyoom, no dmesg event preceding it; looked like a `docker stop`/
  `sparkrun stop` that only reached one host) and shut down cleanly, but
  the *worker* rank never got the same signal — its process stayed alive,
  stuck in a broken NCCL/TCPStore retry loop against a head that no longer
  existed, logging a "Broken pipe" error roughly once a second and
  actively burning CPU (74 minutes of CPU time accumulated) for the
  **entire ~25 hours** until someone checked. `docker exec ... ps aux`
  showed the truth immediately: the head container had nothing left but a
  shell and `sleep infinity`; the worker still had a live but wedged
  `vllm serve --headless` process. Same underlying lesson as the #51921
  entry above (vLLM v1 can't recover a half-dead engine) plus a new one: a
  **partial** stop (one rank only) is worse than no stop at all, since it
  leaves the surviving rank spinning indefinitely rather than exiting.
  Standard remedy: `sparkrun stop` (stops both), flush, relaunch.
- **A boot-time "weights already present" false positive**, or a shard/
  snapshot count that doesn't match reality. `start.sh`'s `count_shards()`
  and `count_dflash_shard()` scope to the active (`refs/main`) snapshot
  specifically (fixed in v20) — if this regresses, it's almost certainly a
  stale-snapshot-directory issue, not a real download failure.
- **Anything touching sparse-MLA / the K-pool indexer / mamba state
  copies.** This subsystem has produced the most distinct production
  crashes in this project's history (four independent bugs across v5-v8,
  all in `validation-archive/v5.md` through `v8.md`). If a new crash looks
  like "silent worker death, no traceback," check whether it matches one of
  those four shapes before assuming it's novel.

## Currently promoted: `glm-5.3-flash-exl3-v20-upstreamsync-vllm.yaml` (2026-09-11)

Routine-upstream-check candidate, built on `v19-indexercompat` (previous
default). Sourced from a triage across MiaAI-Lab's
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks` fork (`9c0794b..1caea9a`, 65 commits),
Enntity/sparkglm's `exl3` branch, and vLLM main since v0.29.0. Full build
detail: `recipes/build/glm53-exl3-v20-upstreamsync/NOTES.md`.

**Four items shipped:**

1. **Chat template `None` leak fixed** (`files/chat_template.jinja`,
   upstream MiaAI-Lab#156). `visible_text()`'s catch-all branch rendered the
   literal string `None` for an assistant turn with `content: None`
   (tool-calls-only, common in agent traffic) — now guarded with
   `elif content is not none`. Only this hunk was ported; #156's other
   hunks target sort-loop structure that's already diverged here from
   independent local changes. New test: `test_chat_template.py::
   NullContentTests` (mutation-verified against the pre-fix template).
2. **`count_shards()` false-completeness bug fixed** (`start.sh`, upstream
   MiaAI-Lab#153). Was counting `*.safetensors` recursively across *every*
   cached snapshot instead of just the active (`refs/main`) one — stale
   shards from an old build could satisfy the check while the real
   snapshot is incomplete. Fixed for both the shared target-model path and
   this fork's own separate DFlash2 call site (same bug, different code,
   not covered by the upstream PR). New test: `test_snapshot_scoped_count.py`.
3. **`LONG_PREFILL_TOKEN_THRESHOLD` opt-in plumbed** (upstream
   MiaAI-Lab#157), **left empty/off by default**. This fork already answers
   the identical "long prefill freezes a warm session" symptom via its own
   `overlay/patch_scheduler_decode_floor.py` gate — running both together
   is untested and could double-throttle or conflict. Do not enable without
   an A/B against that existing gate first. New test:
   `test_long_prefill_threshold.py`.
4. **`BUILD_MIN_MEM_GIB` build-host guard** (ported from Enntity/sparkglm,
   their own original work). Refuses `docker build` under 32 GiB
   `MemAvailable` by default — build-host-side only, no runtime effect. New
   test: `test_build_headroom.py`.

**vLLM PR #54929 assessed and NOT vendored.** It looked like a candidate
fix for the fleet's recurring #51921 shm-broadcast stall, but on
investigation: targets `flashmla_sparse.py`, a backend enum this fork
doesn't use at all (production runs `FlashInferMLASparseSM120Impl`) — zero
overlap with this fork's existing sparse-MLA/indexer patches, meaning
adopting it would require re-deriving this fork's entire bespoke
NoPE-on-SM120 workaround from scratch against 6,848 lines of unreviewed
code. The PR itself is unmerged, has known open correctness bugs
(CodeRabbit-flagged), was never tested against a NoPE/zero-padded,
`index_kpool>1` config like this one, and its root-cause match to #51921
is contested even among the issue's own commenters. Revisit only after
upstream review/merge or a confirmed root-cause match.

**Build**: PASSED, 63/63 steps incl. full build-time self-test suite.

**tinyGLM gate**: PASSED after one transient CUDA-graph-capture crash
(unrelated to any of the four items — none touch that code path); clean
retry matched v19-indexercompat's dispatch-correctness confirmation lines
exactly, determinism confirmed.

**Real-checkpoint validation**: PASSED. `probe_sanity.py` all passed,
decode 28.17-29.99 tok/s (matches v19's 27.4-29.28 tok/s, no regression).
Item 1 verified live with a real tool-calling round-trip (`content: null` +
`tool_calls`) — coherent reply, no `None` artifact. `probe_longctx.py`
@16K: two of the first three v20 attempts hit the ambient earlyoom pattern
described above — more than this project's usual one-retry precedent, so
it was run to ground with a **direct controlled A/B against unmodified
v19-indexercompat** rather than dismissed. v19 hit the identical earlyoom
signature too (mid-request, on a different process, zero effect on the
outcome), and a third v20 attempt with zero code changes passed clean.
**Confirmed ambient/fleet-wide, not a v20 regression.** TTFT@16K across all
clean runs, both versions: 9.6-11.9s, consistent with prior baselines. Full
performance numbers: `recipes/SPEED.md`.

**Verdict: promoted.** No blocking issues. `v19-indexercompat`,
`v18-gb10gemm`, `v15-combined`, `v12-combined`, and `v9-fatfork` all remain
as known-good rollback targets, in that order of preference.

## Pipeline re-measurement: decode kernel trace + dense-FP8 A/B, both re-run against v20 (2026-09-11)

Follow-up to a data-sufficiency question about the pipeline: is the
2026-09-02 deep-instrumentation-night kernel trace still accurate, and does
the v16-densefp8 prefill/decode tradeoff (measured against the stale
v15-combined baseline) still hold on the current kernel stack (E3, GB10
router-GEMM, FlashKDA, MoE-gate-dedup all postdate both original
measurements)? Two fresh measurements, both against production v20.

**Decode kernel trace (batch=1, prose, CUDA graphs on, MTP-2) — composition
is essentially unchanged from 2026-09-02:**

| category | v20 today (rank0/rank1) | 2026-09-02 (rank0/rank1) |
|---|---|---|
| gemm | 53.0% / 52.6% | 52.5% / 49.8% |
| moe_exl3 | 34.1% / 32.8% | 32.8% / 31.3% |
| comms | 5.0% / 6.6% | 7.2% / 11.9% |
| attention | 0.5% / 0.5% | 0.3% / 0.3% |
| mamba_ssm (KDA) | 0.3% / 0.3% | 0.2% / 0.2% |
| GPU busy | 97.0% / 96.5% | 96.0% |

**The single largest kernel is still the same undersized-tile Ampere WMMA
GEMM** (`cutlass_80_wmma_tensorop_bf16_s161616gemm...`, 36.2-36.3% of GPU
time, 16x16 tiles at decode's M=3) -- the custom-kernel opportunity
identified in the original trace is confirmed still live, unaddressed by
any promoted kernel work since. One real change: an SM120-family cutlass
GEMM now appears (4.5-4.8%, absent from the original trace) -- some
Blackwell-native GEMM is engaging, plausibly from the GB10 router-GEMM fix
-- but it's a small slice; the bulk of decode GEMM time still runs the
legacy path. Cross-node comms asymmetry is smaller now (~1.7% delta vs. the
original ~5%). CPU/IPC-spin profile (py-spy) also unchanged: EngineCore
99.9% ipc_spinwait (was 99.9%), Worker_TP0 97.7% (was 96.6%) -- confirms
decode at batch=1 is still GPU-bound, not CPU-orchestration-bound.

**Dense-FP8 A/B, re-run against v20 instead of v15-combined.** Built a
scratch measurement image (`recipes/build/glm53-exl3-v20-densefp8-ab`,
NOT a promotion candidate) by porting v16-densefp8's `overlay/exl3.py`
diff (the `Glm53DenseFp8Method` dispatch class -- `patch_dense_fp8.py`
alone is only half the mechanism, confirmed the hard way: first boot had
zero runtime confirmation lines, because the dispatch logic that routes
`dense`/`kda`/`mla` prefixes to the FP8 path lives in `exl3.py` itself in
v16's tree, not in the separately-portable patch script) onto v20's
`exl3.py`. Same `GLM53_DENSE_FP8=dense,kda` config as the original test.

| Metric | v20 baseline (this session) | v20+densefp8 (this session) | delta | original (v15->v16) |
|---|---|---|---|---|
| Decode | 27.96-29.90 tok/s | 32.59-34.15 tok/s | **+14-17%** | +12% |
| Prefill @16K | ~1,375 tok/s (TTFT 11.6s) | ~1,437 tok/s (TTFT 11.1s) | +4.5% (noise) | **-12.4%** |
| Prefill @64K | ~1,791 tok/s (TTFT 35.7s) | ~1,512 tok/s (TTFT 42.3s) | **-15.6%** | -14.7% |

**Verdict: the tradeoff holds, essentially unchanged in magnitude.** The
16K prefill hit didn't clearly reproduce (within this fleet's normal
9.6-11.9s TTFT@16K noise band, even slightly favorable) but the 64K
regression reproduces almost exactly (-15.6% vs. the original -14.7%),
consistent with the original's own two-point trend (64K's regression
slightly worse than 16K's). Decode's win is, if anything, a bit larger now
(+14-17% vs. the original +12%). **None of the kernel work landed since
the original A/B (E3, GB10 router-GEMM, FlashKDA, MoE-gate-dedup) moved
this tradeoff's economics in decode's favor** -- the prefill cost at
realistic (64K+) context is materially unchanged, and this fleet's one
real-traffic sample (`validation-archive/v9-fatfork.md`, ~101,500 avg
prompt tokens vs. ~181 avg output tokens, 93.4% cache-hit) is far past the
64K point where the regression is clearly visible, not the ~16K point
where it washes out in noise. **The missing piece is still the workload
question, not the kernel-tradeoff question**: that one real-traffic sample
is 5 days old at the time of this re-measurement and was never repeated --
confirming or updating it (has traffic actually shifted toward shorter,
decode-heavier requests) is the one measurement that would actually change
this recommendation, and it's still not done. Absent that, the original
v16-densefp8 rejection reasoning stands on the same evidence it always
did, now confirmed current rather than 9 days stale.

Scratch artifacts kept in the tree for reproducibility:
`recipes/build/glm53-exl3-v20-densefp8-ab/`,
`recipes/glm-5.3-flash-exl3-v20-densefp8-ab-vllm.yaml`,
`recipes/glm-5.3-flash-exl3-v20-profiling-vllm.yaml` (adds
`--profiler-config` for `/start_profile`, matching `v6-profiling.yaml`'s
config) -- none are promotion candidates.

## Dense-FP8 promoted to default, maxprefill sibling added (2026-09-11)

Given the numbers above, the user's own framing: prefill throughput has
grown substantially since v1 (~860-900 -> ~1,500-1,700+ tok/s across the
version history in `recipes/SPEED.md`), and this fleet runs a mixed
~85/15 heavy/short traffic split with growing agentic use -- worth
sacrificing some of that prefill headroom for a real decode win, contrary
to this project's longstanding prefill-first bias but not a bad trade
given how much headroom now exists. Decision: **ship
`GLM53_DENSE_FP8=dense,kda` on by default**, with a sibling recipe for
workloads that still want maximum prefill.

**Folded the dense-FP8 support into the actual `v20-upstreamsync` build
tree** (not just the scratch A/B image) -- `overlay/patch_dense_fp8.py`
plus the `Glm53DenseFp8Method`/`_glm53_dense_fp8_group()` dispatch logic
in `overlay/exl3.py` (see the porting gotcha above: the patch script alone
is inert, the dispatch lives in `exl3.py` itself). `docker build` rebuild
reused the exact same layer digest as the scratch image
(`sha256:62049b...`), confirming byte-identical content -- `glm53-exl3-
v20-upstreamsync:local` is now one shared image for both recipes below,
the toggle is purely the `GLM53_DENSE_FP8` env var, matching the feature's
original runtime-gated design intent.

**Coherence/accuracy check run before shipping this as default** -- the
one validation step this project's own rules required and the original
v16-densefp8 investigation never completed (it was throughput-only,
explicitly flagged as a gap). Six representative prompts (short factual,
a reasoning riddle, code generation, summarization, a CJK translation --
this project's own known corruption-risk case from the ModelOpt checkpoint
history -- and a multi-turn memory check), same image, temp=0, dense-FP8
on vs. off, side by side:

- All six: coherent, correct, no garbling, zero U+FFFD or corrupted CJK
  output.
- Every divergence between on/off was the ordinary paraphrase-level drift
  any precision change causes under greedy decoding at a branch point
  (e.g. "so 9 survive" vs. "the other 8 died" -- same correct answer,
  different phrasing after the tokens diverge) -- never a wrong answer,
  never incoherent, never a different final conclusion.

**Two recipes now share the one image**, toggled by one env var:

- `glm-5.3-flash-exl3-v20-upstreamsync-vllm.yaml` -- **the default**,
  `GLM53_DENSE_FP8=dense,kda`. Re-confirmed end-to-end on the actual final
  recipe file (not just the scratch build): `probe_sanity.py` ALL PASSED,
  decode 31.54-36.76 tok/s, 210 `[glm53-dense-fp8]` per-layer confirmation
  lines fired across both ranks.
- `glm-5.3-flash-exl3-v20-upstreamsync-maxprefill-vllm.yaml` -- sibling,
  identical in every other respect, `GLM53_DENSE_FP8` unset. For workloads
  that are consistently long-context/prefill-dominated and want the old
  tradeoff back.

Still PROVISIONAL per upstream's own commit message ("changes target
numerics; needs a KLD panel") -- the coherence check above is a real,
meaningful check but is not a substitute for one. Revisit if a future
session has the tooling to run an actual KLD panel, or if real production
traffic surfaces a quality regression this spot-check didn't catch.

`v19-indexercompat` remains available as a same-family rollback target if
either v20 recipe needs to be backed out; it predates this change entirely.

## AEON-7/vllm-ultimate-dgx-spark initial triage (2026-09-12)

Added to the tracked-repo rotation at the user's request; first pass
found real, specific leads on two of this fleet's worst unsolved
operational problems. Investigation-only so far -- nothing below has been
tested against this fleet yet.

**Most actionable: `patches/patch_cudagraph_align.py` (their main tree).**
Fixes a real vLLM gap -- spec-decode capture-size alignment (rounding
capture sizes to multiples of `1+num_speculative_tokens`) is gated to
`cudagraph_mode==FULL` only in stock vLLM, silently skipped under
`PIECEWISE` (upstream vLLM #28015/#28207/#29091, fixed by #29102/#23679
upstream but not yet in whatever base we're on). Their own characterization
of the resulting symptom -- `cudaErrorIllegalAddress` mid-decode on
partial-acceptance steps under PIECEWISE -- matches this fleet's own
recurring, never-root-caused `torch.AcceleratorError: CUDA error: an
illegal memory access was encountered` during CUDA-graph capture *exactly*
in class, though not yet confirmed to be the same mechanism. This fleet
runs MTP-2 (num_speculative_tokens=2, so partial acceptance is a real,
frequent runtime state) under PIECEWISE-capable graphs. **Worth a direct
test**: check whether our installed vLLM already has #29102/#23679, and
if not, whether backporting closes the gap. This is the single most
promising lead this project has had on this crash class since it was
first observed.

**Second lead, same repo, DeepSeek-V4-Flash GB10 branch
(`deepseek-v4-gb10`, commit `5e2420e5e0c8d5034aa728965b04f1b11eb55adf`,
open PR, external contributor `gilby`).** DeepSeek-V4-Flash is
architecturally the closest model in that whole repo to GLM-5.3-Flash
(sparse-MLA hybrid MoE). Three findings:
- Documented root cause for a *different* NCCL failure class on 2-node
  GB10 TP2 (`c10::DistBackendError`, not our `#51921` shm_broadcast stall,
  but the same "NCCL + CUDA graphs on multi-node GB10" failure family):
  full decode-graph replay desyncs NCCL between ranks. Fix: force
  `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` -- collectives
  must stay uncaptured on this fabric. We already don't force FULL
  unconditionally, but worth confirming our actual capture mode.
  Corroborated: a version-pin landmine (`tilelang==0.1.12` silently aborts
  with a duplicate type-attr registration on their sparse-MLA decode path;
  pin `0.1.11`) -- worth checking our own pinned TileLang version against
  this if any TileLang-related instability recurs.
- `overlay-deepseek-v4-gb10/vllm/model_executor/layers/sparse_attn_indexer.py`
  (~lines 337-370) patches the same `sparse_attn_indexer.py` file family
  our own `glm5_next` sparse-indexer touches, for a `cooperative_topk`
  landmine: GB10/sm12x has **no thread-block-cluster launch support**
  (`cooperative_topk` fails "invalid argument"), must fall back to
  `persistent_topk`. Hardware-general fact worth confirming we already
  handle correctly (our own K-pool/indexer patches suggest we do, but
  cross-check against this exact anchor).
- Cites DeepGEMM's `nv_dev` fork (`deepseek-ai/DeepGEMM#324`, commit
  `a6b593d`) as shipping genuine native SM120 kernels (vendored DeepGEMM in
  most trees is sm90/sm100-only) -- the closest thing found anywhere to
  real Blackwell-native GEMM work, directly relevant to this project's own
  still-unaddressed decode-time Ampere-fallback GEMM bottleneck
  (`cutlass_80_wmma_tensorop_bf16_s161616gemm...16x16`, ~36% of decode GPU
  time, identified 2026-09-02, confirmed unchanged 2026-09-11). Worth a
  look before considering any from-scratch custom kernel effort.

**Third lead, worth an A/B, not yet a fix**: their documented "dual-Spark
TP=2 over RoCE" stability recipe (`capture_error_mode="thread_local"` at
all `torch.cuda.graph` sites, `fuse_allreduce_rms:false`,
`--disable-custom-all-reduce`, `VLLM_ALLREDUCE_USE_FLASHINFER=0` --
they found v0.29's FlashInfer-all-reduce-on-by-default causes garbled
output, not a hang, on RoCE Sparks specifically). Different symptom than
our shm_broadcast stall, but the closest available "someone else
stabilized dual-GB10-TP2-CUDA-graphs" playbook found anywhere.

**Fourth, worth tracking**: their issue #9 (closed) on GB10 unified-memory
pressure closely parallels this project's own MemFree/MemAvailable-gap and
silent-kill investigation tracks -- two real mechanisms found (`--kv-
cache-memory-bytes` skips vLLM's own UMA/cudagraph memory-profiling clamp
via an early-return path; Docker cgroup memory limits don't police
`cudaMalloc` on GB10's unified pool, so container RSS looks compliant
right up until the driver fails to service an allocation). Does **not**
mention host memory fragmentation (`nr_free_pages_blocks`) specifically --
that finding still appears to be unique to this project.

**Not applicable**: no EXL3/exllamav3/trellis quantization anywhere in the
repo (nothing transfers at the quant-scheme level); no GLM-5.3-Flash as a
target model; no mention of `earlyoom` specifically.

## cudagraph_align hardening ported; 2026-09-12 23:41 UTC incident root-caused

Following up on the AEON-7 triage above: the user asked to pursue the
`patch_cudagraph_align.py` lead specifically because production had just
crashed (a ~3-hour uptime run, after a period of heavy agentic traffic).
Two separate pieces of work came out of this.

**1. cudagraph_align ported and shipped.** Read this image's own
`vllm/config/compilation.py` (commit `g487ecf187`) directly and confirmed
the gap AEON-7 described is real here too:
`resolve_cudagraph_mode_and_sizes` only rounds `cudagraph_capture_sizes` to
multiples of `uniform_decode_query_len` (3, for this fleet's MTP-2) when
`cudagraph_mode.decode_mode() == CUDAGraphMode.FULL`; every other decode
mode silently skips it, and `cudagraph_dispatcher.py`'s
`_create_padded_batch_descriptor` then asserts
`num_tokens_padded % uniform_decode_query_len == 0` for any uniform decode
batch. AEON-7's own patch text didn't apply byte-for-byte (this image adds
a `not use_v2_model_runner` clause their version predates), so
`recipes/build/glm53-exl3-v20-upstreamsync/overlay/patch_cudagraph_align.py`
is a from-scratch port matching our actual anchor, broadening the
condition from `decode_mode() == FULL` to `!= NONE`. New
`tests/test_cudagraph_align.py` covers both an unpatched and
already-patched source (host fixture + a real extracted `compilation.py`
from this image); full docker rebuild passed all self-checks including the
new one. Shipped directly into `glm53-exl3-v20-upstreamsync:local` (no new
version number -- same build tree, patch added). See
`recipes/build/glm53-exl3-v20-upstreamsync/NOTES.md`, "Amendment
2026-09-12: cudagraph_align hardening" for full detail. **This is a
confirmed-real, independently-justified gap closure, not a fix verified
against a reproduction** -- our own boot log shows
`cudagraph_mode=FULL_AND_PIECEWISE` resolving normally, so the FULL-only
gate is very likely already satisfied in normal operation; this change is
a no-op today and a hardening against any future config/backend-support
path that leaves decode mode at PIECEWISE.

**2. The actual 2026-09-12 23:41 UTC incident -- root-caused, and it is
NOT the cudagraph_align gap.** The user supplied the actual crash log
(`Worker proc VllmWorker-0 died unexpectedly (exit code: None)`, no
traceback of its own, followed by the downstream `shm_broadcast`
"cancelled" error and `EngineDeadError`). The dying step's
`dump_input.py` output showed `num_scheduled_tokens=5832` for a single
request already at `num_computed_tokens=179200` -- a large one-shot
continuation chunk for a ~180K-token-deep request, not a small MTP-2
decode step. That's far above this config's `max_cudagraph_capture_size=96`,
so the step ran eager -- ruling out cudagraph capture/replay
(and therefore cudagraph_align) as the mechanism for this specific crash.

Host-level investigation (dmesg + `journalctl -u earlyoom`, both nodes,
which the user's own earlier `sparkrun stop` at 19:43 had left no other
trace of -- containers were already destroyed) found the real cause:

- **Both hosts hit severe, simultaneous host memory exhaustion** starting
  ~19:40:45 EDT: this host (node_1/rank1) pinned at **3600 MiB avail out of
  124610 (2.89%)** continuously for 29+ seconds; the head (node_0/rank0,
  10.7.0.87) dropped to **2508 MiB (2.01%)** at the same time.
- **19:40:51.14 EDT**, head node: earlyoom crossed its SIGTERM threshold
  and sent SIGTERM to `1712017 uid 1000 "VLLM::Worker_TP"` (badness 984,
  VmRSS 2320 MiB).
- **19:41:01.22 EDT** (10s later): earlyoom logged **`kill failed: Timer
  expired`** -- the SIGTERM did not take effect within earlyoom's wait
  window. Plausible explanation: the worker was mid-flight on the large
  5832-token batch, blocked in CUDA/NCCL work and unable to act on the
  signal promptly.
- **19:41:06-09 EDT**: memory on both hosts suddenly recovered to
  86-92% avail -- consistent with the worker actually dying/releasing its
  memory around then, matching the `23:41:06 UTC` (=19:41:06 EDT) `Worker
  proc ... died unexpectedly` timestamp in the user's log almost to the
  second.
- This host (node_1/rank1) never logged its own SIGTERM/SIGKILL line in
  this window despite being equally starved (2.89% avail) -- it just sat
  at the low-memory warning level without earlyoom escalating to a kill
  here; the head's kill (successful or not) was enough to end the episode
  for both ranks (TP=2 -- losing either rank kills the whole engine).

**This extends, rather than replaces, the project's existing "earlyoom
root-cause correction" finding** ([[earlyoom_root_cause_correction]]):
previously documented as "check earlyoom first, it's usually the silent
killer." New wrinkle, not previously seen: **earlyoom's own kill attempt
can itself fail/time out** against an unresponsive GPU/NCCL-blocked
target, which likely explains why some past "silent kill" incidents were
hard to pin to a specific earlyoom action from timestamps alone -- the
SIGTERM fires, doesn't land immediately, and the process dies later for
reasons that then look uncorrelated unless you specifically check for a
"kill failed" line.

**New data point for the standing host-memory-pressure track**
([[host_fragmentation_xid31_reboot_risk]], the 240K-context fragmentation
finding): this episode happened at ~185K total tokens (179200 computed +
5832 scheduled), well under the ~240K neighborhood previously implicated.
The common factor isn't a fixed context-length threshold -- it's **a
single large one-shot continuation/recompute batch** (5832 tokens in one
step, vs. MTP-2's normal 1-3-token decode steps) for a request already
deep in a long context. Worth checking whether the scheduler's chunking
policy (`patch_scheduler_decode_floor.py`'s mixed-prefill-decode policy)
can be made to cap continuation-chunk size for already-long-context
requests specifically, rather than only floor-ing decode-side batching --
not yet investigated further; flagging for a future session.

**Not yet done**: identifying what specifically was consuming host RAM in
that window (vLLM's own host-side buffers scaling with the 5832-token
batch vs. something fragmentation-related vs. an unrelated host process);
no host-side memory profiler was attached at the time and the containers
are gone. Production was relaunched on the patched image
(`glm53-exl3-v20-upstreamsync:local`, now carrying `patch_cudagraph_align.py`)
after this investigation; see the top of this file for the current state.

## Routine upstream check, 2026-09-13 -- quiet round, nothing folded in

MiaAI-Lab main `1caea9a..HEAD` (32 commits, 6 PRs), Enntity/sparkglm `exl3`
branch `a8aaa229..HEAD` (15 commits), plus pulse checks on the other three
tracked repos and vLLM itself. Honest result: **nothing directly
actionable for this fork's production recipe this round** -- every
substantive item is either launcher plumbing this fork bypasses (sparkrun
runs `vllm serve` directly, not MiaAI-Lab's `start.sh`), a DFlash-specific
feature this fork doesn't ship (MTP-2 stays the production speculator),
or a correctness fix already present in our own vLLM base. Detail:

**MiaAI-Lab, by PR:**
- **#130** (merged): opt-in per-KV-cache-group prefix-cache retention +
  safe replay for DFlash's sliding-window drafter cache. DFlash-only
  (`GLM53_APC_RETENTION_INTERVAL_SWA` requires `SPEC_METHOD=dflash`) --
  not applicable, we run MTP-2.
- **#169** (merged): computes the CUDA-graph capture-size list for
  DFlash's "adaptive-k" verification-length feature from
  `GLM53_ADAPTIVE_K_SET`/`DFLASH_TOKENS`/`MAX_NUM_SEQS` instead of a fixed
  list. Same *class* of bug this fork just spent a session on
  (cudagraph capture sizes not accounting for variable spec-decode token
  counts -- see the cudagraph_align entry above) but the feature itself
  (DFlash adaptive-k) is DFlash-only -- not applicable. Worth noting as
  corroboration that this bug class is real and recurring industry-wide,
  not specific to us or to MTP.
- **#172** (merged): launcher's RoCE GID preflight only checked the first
  HCA on a dual-rail (`HEAD_CX7_IB=dev1,dev2`) kit, silently skipping
  validation on the second -- an unpopulated GID there kills that rank
  ~60s into a run. Fix + new test. Their own measurement: **dual-rail NCCL
  all-reduce hit 20.9 GB/s peak busbw vs. 12.8 GB/s single-rail** on their
  2x GB10 kit. The preflight fix itself is start.sh-only, not applicable
  (sparkrun handles our own networking setup, abstracted behind
  `transfer_interface: cx7` in `~/.config/sparkrun/clusters/default.yaml`)
  -- but that ~63% bandwidth number is worth checking against our own
  setup: **not yet verified whether sparkrun is using both RoCE rails on
  our CX7 cards or just one.** Flagging for a future session -- if we're
  single-rail today, this could be a real, free TP=2 communication
  bandwidth win.
- **#175** (merged): makes `start.sh`'s hardcoded
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` overridable (for
  derived images with a KV connector that can't tolerate expandable
  segments). Not applicable as a fix -- we already set this directly in
  our own recipe YAML, unconditionally, bypassing `start.sh` entirely.
  Confirms our existing setting matches upstream's own recommended
  default.
- **#136, #173, #176** (merged): benchmark-script bearer auth, launcher
  Jinja2-interpreter discovery, a docs typo. Tooling/docs only, no
  runtime-behavior relevance.

**Enntity/sparkglm (`exl3` branch)**: this branch has shifted into a
heavier "research/experiments" structure (qualification protocols,
rejected-candidate writeups) rather than shipped features. Two things
worth recording:
- **`research/experiments/exl3-direct-epilogue`**: a candidate
  micro-optimization to the fat-expert grouped-MoE kernel's epilogue
  (keep the transformed row in its owning warp, skip a shared-memory
  round-trip). **Rejected by their own performance screen** -- compiled
  and passed correctness, but was slower than the reference. Nothing to
  port; recorded for completeness (same rigor this project applies to its
  own rejected experiments).
- **`research/experiments/exl3-e3/UPDATE_REVIEW.md`** flagged **vLLM PR
  #55234** as "especially easy to backport incorrectly" (MLA cache-group
  capability handling for non-causal draft paths, plus a `MambaSpec.merge`
  assertion-preservation fix under `python -O`). Checked directly against
  our own vendored vLLM (`g487ecf187`,
  `vllm/v1/kv_cache_interface.py:487-488`): **already present** (the `any()`
  aggregation the fix restores is exactly what our source has). Nothing to
  do. Their update review also independently confirms two things this
  project already concluded on its own: their reported E3-vs-E2 speedup
  "is not transferable" since SparkGLM (and this fork) already has grouped
  M64 prefill + cooperative K4 decode, and "replacing DFlash2 with MTP"
  is flagged as its own separate hypothesis requiring independent
  measurement, not something to blend silently -- same conclusion this
  fork reached independently months ago.

**Pulse-checked, no new relevant activity**: mmastrac/glm-5.3-flash-4x-gx10
(last push 2026-09-07, before this round's window), tonyd2wild/GLM-5.3-
Flash-NVFP4-DFlash2-2x-DGX-Spark (2026-09-02), AEON-7/vllm-ultimate-dgx-
spark (2026-09-12, before yesterday's full triage -- nothing since).
mmastrac/mentat had a burst of activity (a 0.10.0 release) but it's all
within its existing Ray-replacement scope -- track stays closed, see
[[mentat_track_closed]].

**vLLM upstream**: still v0.29.0 (published 2026-09-09), no new release
since the backport-candidates round closed 2026-09-10. Nothing new to
triage there.

**Net**: a genuinely quiet round. One follow-up flagged (verify RoCE
dual-rail usage in our own sparkrun cluster config) but nothing shipped or
changed in this fork as a result of this check.
