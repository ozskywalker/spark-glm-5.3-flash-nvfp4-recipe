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
  `main` @ `1caea9a` (2026-09-11), `HEAD` as of 2026-09-13 (quiet),
  `HEAD` again as of 2026-09-14 (`f906ee990..d8ad18311` -- a vision-
  encoder host-OOM fix, assessed not urgent, see "TP=4 vs. 2x TP=2
  assessment... 2026-09-14" below), and `HEAD` again as of 2026-09-15
  (`d8ad18311..HEAD`, 91 commits/~15 PRs -- two shipped, see "Routine
  upstream check, 2026-09-15" below).
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
  (`TORCH_MEM_FRACTION` + CUDA-allocator observer). Deep-dived 2026-09-14
  for its TP=4-specific experience (RoCE fabric fault modes, GID
  instability, no multi-replica comparison) -- see "TP=4 vs. 2x TP=2
  assessment... 2026-09-14" below. Three secondary backport candidates
  flagged, not yet investigated: a spin-wait CPU patch, a GPU_MEM_UTIL
  ceiling data point, a `thinking_token_budget` bug report.
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
- **cbertucci33/vllm-v29-glm53flash-exl3-dgx** — added 2026-09-15. Same
  target as this fork almost exactly: GLM-5.3-Flash EXL3 on 2x DGX Spark
  TP=2, vLLM 0.29.0 base. Single-author, documented as a 23-step
  integration history rather than an ongoing project (2 commits since
  2026-09-12, may go quiet). Companion HF assets: target checkpoint
  `cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced` (an abliterated/
  uncensored variant, **not** stock GLM-5.3-Flash) and a published DFlash2
  drafter `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` trained against
  that specific target. See "cbertucci33 triage" below.

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

## TP=4 vs. 2x TP=2 assessment, and routine upstream check, 2026-09-14

Prompted by the user contemplating scaling from 2 to 3 or 4 DGX Sparks.
Conversational recommendation for both cases: don't extend the TP group
(TP=3 hits an odd-sharding-degree risk class this project has never
validated at any TP degree; TP=4 doesn't have that specific risk but
multiplies this fleet's single most-proven fragility, the RoCE/NCCL
fabric, across twice the physical link count). Prefer independent
replicas -- a 3rd node as its own TP=1 lane for the ~15% short-traffic
slice, or 2x TP=2 pairs instead of one TP=4 group -- both reuse the
entire already-validated TP=2 stack with zero new sharding risk and, more
importantly given this project's own incident history, halve blast
radius instead of concentrating it.

**mmastrac/glm-5.3-flash-4x-gx10 deep-dive (fanned out to a subagent,
production untouched throughout)**: confirms the TP=4 concern directly.
Their repo is genuine flat TP=4 across 4 separate physical GB10 boxes
over RoCE (not PP, not DP -- some unused PP plumbing exists in the repo
but isn't what's actually invoked). Their own troubleshooting docs
describe a RoCE fabric fault mode independent of anything we've hit:
NCCL all-reduce silently crawling to ~12 Gb/s (vs. 196 Gb/s healthy)
after a cable hot-plug, with **zero error-counter signal** -- `ib_write_bw`
reads fine, only a Ring-vs-Tree NCCL microbenchmark catches it -- and
requiring a full power-off (not a reboot) to clear. Their own words: "the
existing 4x recipes each solve a different subset and none of them
mention the fabric fault, which is the one that costs half your prefill
throughput while every metric reads healthy." Second independent fragility
axis: the RoCE GID index is per-node-*and-per-boot*, not a stable value --
they derive it dynamically every boot rather than trust a pinned config,
since a stale pin silently breaks TP init. A useful, independently-derived
data point: their own note that per-token all-reduce payload (~720KB) is
"a fraction of a millisecond against a 45ms token -- weigh it only at
TP>2," and that dual-rail RoCE "bought nothing" for them at TP=2 --
direct confirmation the fabric-dependency cost is roughly free at TP=2
and non-trivial past it. GLM-5.3-Flash's NoPE sparse MLA (compressed
latent KV, not per-head) means the "replicate KV heads when TP exceeds
head count" bug class other models hit doesn't apply here regardless of
TP degree. They never evaluated or discuss a multi-replica alternative
anywhere in the repo -- so no counter-evidence to our thesis, only more
confirmation of the exposure. **Not adopted as a topology**; recorded as
the deciding evidence for staying off TP=4.

Secondary backport candidates surfaced by the same dig, not yet acted on:
a spin-wait tuning patch (`busy_loop_s` 1.0->0.002s cut their vLLM CPU
185%->109% and *raised* decode throughput on GB10 by exploiting the
unified CPU/GPU power budget -- this project already has its own
spinwait patch, `patch_spinwait.py`, worth comparing constants);
`GPU_MEM_UTIL=0.90` silently wedges their box hours later while passing
every startup check, same shape as this project's own 0.85/0.86 GMU
regressions -- worth comparing their observed ceiling against our 0.84
one; a report that `thinking_token_budget` is silently ignored at
temp 0/1 -- worth checking against our own build. None investigated
further this round.

**Rest of the tracked-repo rotation** (all fanned out in parallel,
production untouched):
- **MiaAI-Lab main**, `f906ee990..HEAD` (2 commits, one PR, #183):
  "fix(vision): cap per-image tokens so a chat video cannot OOM the
  host." Real incident on their side 2026-09-14: an 11.9MB chat video
  decomposed into ~33 full-res images, `SKIP_MM_PROFILING=1` (mandatory
  on their UMA kit) meant nothing was ever reserved for the vision tower,
  prompt hit 236,544 vision-encode tokens, host OOM-killed the worker.
  Fix: `LIMIT_MM` image cap 100->48, new `MM_IMAGE_TOKENS`/`--mm-
  processor-kwargs max_image_tokens` knob, new `MM_PROCESSOR_CACHE_GB`
  (vLLM defaults to reserving 4 GiB host RAM for processed media --
  real cost on UMA regardless of per-prompt limits). **Assessed: not
  urgent for us.** Our own `limit_mm: {"image":4,"video":1}` is already
  far tighter than even their patched 48-image cap, so we're not exposed
  to their crash scenario. We do NOT explicitly set `--mm-processor-
  cache-gb` or `max_image_tokens`, so we're on vLLM's stock 4 GiB/8000
  defaults -- a real but currently-inert standing host-memory cost, worth
  revisiting alongside the broader host-memory-pressure track
  ([[cudagraph_align_shipped_earlyoom_timeout_found]]) rather than
  urgently now.
- **Enntity/sparkglm `exl3`**: still `2da6a1c31`, confirmed no drift/
  force-push. Quiet.
- **tonyd2wild**: still last-pushed 2026-09-02. Quiet.
- **AEON-7/vllm-ultimate-dgx-spark**: still last-pushed 2026-09-12,
  no new commits/issues/PRs touching CUDA-graph capture, attention
  kernels, Blackwell/SM12x, or RoCE/NCCL. Quiet.
- **vLLM upstream**: still v0.29.0, no new release -- no full backport-
  candidates round warranted. Two things worth tracking: **PR #54929**
  ("Portable Triton sparse-MLA fallback for SM12x") explicitly claims
  `Fixes #51921 (on SM12x)` -- our long-standing open shm_broadcast-stall
  issue. Root cause per the PR: DSA models (DeepSeek-V3.2, GLM-5.2/5.3)
  have no working native sparse-attention kernel on SM12x, the native
  extension livelocks under sustained load (GPUs pin 100% until the NCCL
  watchdog kills the server -- matches our own stall signature), and this
  PR binds a portable Triton fallback (derived from #49026, already
  validated on sm_121/GB10) instead. **Still open, has merge conflicts as
  of 2026-09-14** -- not yet mergeable, nothing to backport yet, but this
  is the most promising #51921 lead found to date; recheck next round.
  Separately, **PR #55737** ("Use FlashKDA for KDA chunked prefill")
  merged 2026-09-14 upstream -- the same approach this fork's own
  `v15-combined` already shipped; not new work for us, just confirms
  upstream converging on what we already have. Worth a quick diff-check
  next backport round, not urgent.

**Net**: the TP=4 question is answered (don't); nothing else this round
needs immediate action. Two things to revisit next time someone's in
here: PR #54929's merge-conflict status (the #51921 fix candidate), and
whether the mm-processor-cache-gb/max-image-tokens defaults are worth
pinning explicitly given the standing host-memory-pressure track.

## Routine upstream check, 2026-09-15 -- two backports shipped

Second fan-out round (6 parallel agents again, production untouched
throughout). MiaAI-Lab main had genuinely moved this time
(`d8ad18311..HEAD`, 91 commits, ~15 real PRs) -- the rest of the rotation
was quiet (sparkglm exl3 still `2da6a1c31`, tonyd2wild still silent since
09-02, AEON-7 still silent since 09-12 -- its 09-13 `pushed_at` bump
turned out to be a `WatchEvent`/star, not a push, confirmed by checking
every branch's actual commit history).

**vLLM PR #54929 re-checked** (the #51921 stall fix candidate): still not
safe to pull. `mergeable: MERGEABLE` (conflict-free this hour, after the
author merged `main` in again just ~2h before this check) but
`mergeStateStatus: BLOCKED`, `reviewDecision: REVIEW_REQUIRED`, and
`pre-run-check` CI is currently failing. Zero maintainer review despite
the author pinging the relevant code owners on 2026-09-04. Scope has also
grown since first seen (a second commit added a "7.4x decode attention"
split-K path, unvalidated by anyone but the author). A week of
conflict/rebase churn, not a stable target. Recheck again in a few days.

**Two items shipped this round** (both in `recipes/build/glm53-exl3-
v20-upstreamsync/`, see NOTES.md's "Amendment 2026-09-15" for full
technical detail):
- **`patch_default_max_new_tokens.py`** (MiaAI-Lab PR #51): omitted-`max_tokens`
  decode-hygiene default. `DEFAULT_MAX_NEW_TOKENS=65536` now set in both
  the default and maxprefill recipes' `env:` blocks. Without this, a
  client that never sets `max_tokens` gets vLLM's own fallback of
  `max_model_len - prompt_len` -- up to our full 262144-token ceiling on
  a short prompt -- large enough for one such client to decode until it
  preempts every other session via KV pressure. Explicit client
  `max_tokens` is completely unaffected. All three patch anchors matched
  our vLLM byte-for-byte; ported near-verbatim from an already
  well-built, tested upstream implementation.
- **`prelaunch_flush.sh`'s `check_memory_available()`** (MiaAI-Lab PR
  #39): pre-boot check comparing host `MemAvailable` against
  `gpu_memory_utilization x MemTotal + headroom` on both nodes, run right
  after the existing drop_caches + fragmentation steps. Catches a
  host-memory hold (not a container -- distinct from any existing
  container-level check) BEFORE the image pull and weight load, instead
  of dying mid-bring-up with a `ValueError: Free memory on device ...
  less than desired` deep in a worker log. FATAL by default
  (`GLM53_PREFLIGHT_SKIP_MEMORY_CHECK=1` to override) -- deliberately
  different policy from the fragmentation check next to it, which stays
  advisory-only, because this failure mode is a guaranteed deterministic
  hard-stop rather than a probabilistic risk. Directly relevant to this
  project's own MemFree/MemAvailable-gap and earlyoom tracks. Verified
  via `bash -n` and a standalone arithmetic check against real host
  numbers; not yet exercised on a live boot (would require running it
  against a host with production traffic, not done this round).

**`thinking_token_budget` bug (flagged via the mmastrac/4x-gx10 dig,
2026-09-14) -- checked and confirmed NOT applicable.** The bug lives in
vLLM's V2 model runner's sampler
(`vllm/v1/worker/gpu/sample/sampler.py`'s `_requires_logits_processing()`
gate, which never checks thinking-budget state, silently skipping budget
enforcement whenever temperature is 0 or 1.0 and no other sampling knob
is active). Read our own vendored vLLM directly: GLM-5.3-Flash+EXL3
resolves to the V1 model runner (established during the cudagraph_align
work), and V1's own sampler (`vllm/v1/sample/sampler.py`) has a
structurally different implementation -- `apply_logits_processors`
applies thinking-budget logic unconditionally whenever there are tracked
requests, with no temperature-based short-circuit gate at all. The buggy
code technically exists in our installed vLLM package (both V1 and V2
ship unconditionally) but is dead code for our runtime config. Confirmed
by reading the actual code, not assumed from the upstream report.

**Also assessed, not adopted this round** (all from the MiaAI-Lab range):
PR #170 (long-prefill boot-warmup ladder extension to 3584/7168/14336/
65536-token rungs -- we're prefill-heavy, plausibly worth it, just not
done yet), PR #94 (`GLM53_KV_CAPACITY_LOG`, informational-only logging
clarifying the boot "GPU KV cache size" line for hybrid MLA+mamba+drafter
models), PR #70 (`spec-accept-gate.sh` -- a diagnostic script checking
for vLLM #53030, CUDA graphs pinning per-position spec-decode acceptance
at exactly 1.00; useful given we run MTP-2, but tooling not a serving-path
change), PR #41 (`spark_doctor.sh`, ops diagnostic tooling). PR #186/#187
(fair-v5 mixed-prefill scheduler, now MiaAI-Lab's own TP=2 default,
measured real TTFT/throughput tradeoffs on their kit) is explicitly
**not** a drive-by candidate -- a real scheduling-behavior change on our
exact topology, needs its own deliberate A/B before ever being
considered, not bundled into a routine-check round. Not applicable at
all: PR #31/#37 (bench convenience endpoint), PR #75 (EXL3 SM121 kernel
lab, dev-only/no GPU tested), any TP=3/TP=4-specific commits (we run
TP=2 only), PR #129 (docs-only), PR #189 (cosmetic).

**mmastrac/glm-5.3-flash-4x-gx10 secondary candidates, resolved**: the
spin-wait patch (`busy_loop_s` 1.0->0.002, same lever as this fork's own
`patch_spinwait.py`, +5.5% decode / 20C cooler on their kit) is worth a
constant-diff check against our own value -- not yet done. The
`GPU_MEM_UTIL=0.90` wedge report is informative but not actionable as a
config change (their shipped ceiling is 0.88, higher than our 0.84; their
failure mode -- total unresponsive wedge, no SSH, OOM killer can't even
intervene on UMA -- is corroborating evidence for this project's own GMU
root-cause understanding, not a new lever). The `thinking_token_budget`
item is covered above.

**Net**: a real, productive round -- two backports shipped (memory
preflight + max-tokens hygiene), one flagged bug ruled out with actual
code verification rather than assumption, several more items identified
and deliberately deferred rather than rushed. Production was never
touched; the image was rebuilt under the same tag (`glm53-exl3-
v20-upstreamsync:local`) but not relaunched.

## cbertucci33/vllm-v29-glm53flash-exl3-dgx triage (2026-09-15)

User-directed look at a single-author repo targeting the same hardware/
model combination as this fork almost exactly. Read-only research: repo
tree via `gh api`, README, and a direct file-existence diff against
`vllm-project/vllm@v0.29.0` to separate genuinely custom work from stock
vLLM. No production access needed or used.

**First-pass mistake, corrected by the user**: initially read the
DFlash2 drafter as unpublished (the README says "model weights are
published separately" without a link). The user supplied the actual HF
links -- both the target (`cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-
Sliced`) and the drafter (`local-inference-lab/GLM-5.3-Flash-DFlash2-
MXFP8`) are public. The material point survives anyway: the target is an
**abliterated/uncensored** checkpoint, not stock GLM-5.3-Flash, and the
drafter was trained against that specific target's activations. Spec-
decode drafters are distilled against one target's output distribution --
there's no reason to assume this drafter's acceptance rate transfers to
our stock (non-abliterated) GLM-5.3-Flash EXL3 checkpoint, and abliteration
is known to shift a model's logit distribution in exactly the ways that
matter for draft-token acceptance. Unverified either direction; flagging
the mismatch rather than assuming it's fine or assuming it's broken.

**What's stock vLLM 0.29.0, not their invention** (confirmed via
`gh api repos/vllm-project/vllm/contents/<path>?ref=v0.29.0`, checking for
a 404): DFlash2 (`vllm/v1/spec_decode/dflash.py`), B12X
(`vllm/model_executor/layers/fused_moe/b12x.py`), and the TopK CUDA
kernel family (`csrc/libtorch_stable/persistent_topk.cuh`) all resolve
in stock v0.29.0. Their README's step-by-step framing implies these are
their own additions; they aren't -- this ecosystem's PRs land upstream
fast. `vllm/models/glm5next/` (their path) is genuinely absent from
v0.29.0 (only `vllm/models/{common,deepseek_v32,deepseek_v4,dots3_note,
hy_v4,inkling,kimi_k3,minimax_m3,qwen4_exp}` exist at that tag) -- so
their GLM-5.3-Flash model support (imported from vLLM PR #53906 per
their README) is a real backport of not-yet-released upstream work, same
category of thing this fork's own overlay patches do.

**What's genuinely custom to them, in priority order**:
1. **Sparkinfer** (`github.com/gittensor-ai-lab/sparkinfer`, real
   separately-maintained project, corroborated by multiple unrelated
   repos in this ecosystem citing it) -- a native EXL3 Trellis execution
   engine used *instead of* stock ExLlamaV3 for the actual quantized
   matmul path. The single most architecturally distinct choice in the
   repo. We (`overlay/exl3.py`) run stock ExLlamaV3 directly, same as
   most EXL3-on-vLLM efforts. Real potential upside if Sparkinfer's
   kernels are faster on GB10 specifically, but unproven against our
   workload and a nontrivial integration cost (own CUTLASS DSL pin, a
   custom ARM64 build with x86 AVX units stripped). Research lead, not a
   backport candidate yet.
2. ~~A GB10-safe exact TopK kernel for MoE expert routing~~ **CORRECTED
   2026-09-16, see below**: this is not the MoE router's TopK at all --
   it's FlashInfer's own JIT topk module
   (`flashinfer.jit.topk.gen_topk_module`, built by their
   `build/build_flashinfer_topk.sh`), used by their custom
   `FLASHINFER_MLA_SPARSE_SM120` attention backend for sparse-MLA
   page/candidate selection. Same subsystem as the already-tracked
   `cooperative_topk`->`persistent_topk` indexer finding, not a second,
   independent call site. See "cbertucci33 items 1/2 resolved" below for
   the actual verified answer.
3. **A lossless EXL3 checkpoint pre-slicer**
   (`tools/slice_exl3_checkpoint.py`) -- splits routed-expert tensors
   per-TP-rank offline, verifies bit-for-bit reconstruction, before
   deploy. Relevant to this project's own boot-time host-memory-pressure
   track (`memfree_memavailable_gap`, the new `check_memory_available()`
   preflight) -- pre-sliced per-rank checkpoints could lower the
   per-node peak memory during weight load, which is exactly the phase
   the new preflight check guards. **Not yet evaluated against our own
   checkpoint format/loader.**
4. Their own EXL3 quantization layer (`vllm/model_executor/layers/
   quantization/exl3.py`, confirmed absent from stock vLLM) -- same
   category as our own `overlay/exl3.py`, expected to exist independently
   in any EXL3-on-vLLM effort since vLLM doesn't ship EXL3 natively.

**Their claimed performance, treated skeptically**: 24.5 tok/s weighted
decode, +13.3% over a "previous runtime, same hardware" baseline of 21.62
tok/s, from a 197-request live sample with DFlash2 (7 proposals). Two
reasons not to read this as "DFlash2 beats MTP-2": (1) it's unclear what
the 21.62 tok/s baseline actually was -- no spec decode, MTP-2, or
something else isn't stated; (2) their own "improved" 24.5 tok/s sits
inside the range this fleet is *already* measuring for stock MTP-2 on the
same hardware class (p50 26.7 tok/s, p95 20.5 tok/s decode-only,
`request_time_per_output_token_seconds` over the last 3h as of this
check -- see the mcp-grafana probe from the same session). More likely
their baseline was weaker than what we already run, not evidence DFlash2
itself is faster.

**Net, and what's queued for the next routine round** (production
untouched this session; these are research-only next steps against our
own build tree, not against a live host):
- Check our MoE router's TopK path for the same GB10 shared-mem-limit
  exposure as items 2 above -- cheap, do this first.
- Evaluate `tools/slice_exl3_checkpoint.py` against our own checkpoint
  layout for the boot-memory-pressure angle.
- Sparkinfer stays a flagged research lead (real, active, corroborated
  project) -- not actionable without a dedicated eval, given the native
  dependency cost.
- DFlash2/their published drafter: not adoptable as-is (trained against
  an abliterated target we don't run); would need our own DFlash2
  drafter trained against stock GLM-5.3-Flash to be a fair comparison at
  all, which is out of scope for a routine check.
- Added `cbertucci33/vllm-v29-glm53flash-exl3-dgx` to the tracked-repo
  rotation above, flagged as likely low-activity (single-author,
  integration-history framing, may not get further commits).

## Routine upstream check, 2026-09-16 -- and items 1/2/3 from the cbertucci33 triage

Fan-out round on a production traffic break (6 parallel agents, read-only
research, production never touched). Also closed out the three action
items queued from the 2026-09-15 cbertucci33 triage.

**MiaAI-Lab main, `d8ad18311..HEAD` (155 commits, mostly merge noise from
a 09-15 PR-cleanup burst).** Two items need a dedicated A/B, not a
routine-round adoption:
- **PR #186/#194/#198 -- "fair v5" mixed-prefill scheduler is now
  MiaAI-Lab's own TP=2 default** (`GLM53_MIXED_PREFILL_CHUNK`:
  `skip`->`fair`, `GLM53_FAIR_PREFILL_SHARE=0.30`, `MAX_STEP_MS=1000`).
  A real decode/prefill-interleaving policy change on our exact topology.
  Not adopted -- same standing rule as PR #186/#187 from the 09-15 round.
- **PR #200/#201 -- default image now `:exl3-instanttensor`,
  `LOAD_FORMAT=instanttensor`**: a new third-party direct-I/O safetensors
  loader (`instanttensor==0.2.0`, installed `--no-deps` to dodge an NCCL
  conflict) replacing the core weight-loading path. New dependency in a
  correctness-critical path -- needs its own validation before we'd point
  at it. Not adopted.

Low-risk, applicable backport candidates identified (not shipped this
session -- see "Net" below for why): PR #37 (`/reset_prefix_cache` admin
endpoint, opt-in), PR #42 (pipefail-safe container health checks), PR #81
(`GLM53_EXTRA_ENV` diagnostic passthrough, opt-in), PR #94
(`GLM53_KV_CAPACITY_LOG`, boot-time log line only, default-on, useful
given our memory-pressure tracks), PR #95 (`GLM53_APC_NO_STORE`
per-request prefix-cache skip, inert unless a caller opts in), PR #170
(long-prefill boot-warmup metadata-kernel fix). Not applicable: PR #184
(TP=3), #115/#187/#188 (TP=4, or fair-scheduler-off confirmation for
TP=3/4 -- confirms the TP=2 fair-default is specifically what needs our
A/B), #137 (abliterated-weights preset, different model), #75 (EXL3
SM121 kernel lab, still dev-only, passively watching), #129/#189/#192
(docs/cosmetic/test-path).

**Enntity/sparkglm `exl3` branch, `a8aaa229..2da6a1c3` (3 commits).**
Nothing to backport -- every substantive item is the sibling project
re-deriving conclusions we already reached independently (E3-vs-E2 not
transferable once you already have grouped M64 + cooperative K4, matches
our own v12-combined-era finding; cooperative-decode-32 not worth it,
matches our serving limit of 16; MXFP8 DFlash2 draft still fails their
own arithmetic semantic gates). One operational signal: the branch's
README now explicitly marks it "retained" (frozen/archival), read
together with the 09-13 shift toward a research/experiments structure --
EXL3-relevant activity from this sibling may be tapering off. Flag for
future rounds: check whether relevant work migrates to `main` (now
NVFP4-focused, likely irrelevant) or simply stops.

**mmastrac/glm-5.3-flash-4x-gx10.** No repo activity since 2026-09-14
(last push still 2026-09-07). Resolved the deferred spin-wait diff-check:
their value is `busy_loop_s=0.002` (2ms), applied via a `sed` bind-mount
override targeting `shm_broadcast.py`, reporting CPU 185%->109%, ~20C
cooler, decode 66.9->70.6 tok/s on their kit. **No action needed on our
side** -- our own `patch_spinwait.py` docstring already records that we
tested 2ms ourselves in our own frozen TP=2/MNBT=2048 sweep and it *lost*
1.68% decode versus our chosen 16ms, which beat stock on both decode
(+0.95%) and CPU (-85.3%). Their optimum for their workload isn't ours;
we'd already tested their exact candidate and rejected it with real data.

**tonyd2wild and AEON-7.** Both confirmed still silent (tonyd2wild: no
commits past 09-02 on either branch; AEON-7: no commits past 09-12 on any
of its four branches). The AEON-7 `updated_at` bump noted in the 09-15
round remains a non-push event. Nothing to review.

**cbertucci33/vllm-v29-glm53flash-exl3-dgx, `29640cb..96483c3`.** One new
commit, README-only (+2 lines, notes on base-model provenance and
cross-quant-variant compatibility). Confirms the "low-activity,
single-author" read from the initial triage.

### cbertucci33 items 1/2 resolved

**Item 1 (GB10 TopK exposure) -- corrected and closed, no action
needed.** Direct verification against our own running production
container (`docker exec`, read-only) rather than assumption:
- Our MoE expert router (`vllm/model_executor/layers/fused_moe/router/
  grouped_topk_router.py`, the path `glm5next`'s `use_grouped_topk=True`
  config selects) calls `ops.grouped_topk(...)`, a dedicated small-k
  CUDA kernel built for the "top-8-of-288-experts" routing problem. It
  never touches `cooperative_topk`/`persistent_topk` at all -- confirmed
  by grepping every file in our installed vLLM package for those two
  symbols: they appear ONLY in `sparse_attn_indexer.py` and
  `sparse_attn_indexer_kpool.py`. **There is no "MoE router TopK GB10
  exposure" -- that framing in the 2026-09-15 entry was wrong, based on
  an unverified assumption about which subsystem cbertucci33's fix
  targeted.** Their fix (confirmed by reading their
  `build/build_flashinfer_topk.sh`, which calls
  `flashinfer.jit.topk.gen_topk_module`) is inside FlashInfer's own JIT
  module for their custom `FLASHINFER_MLA_SPARSE_SM120` backend -- a
  third code path, distinct from both of vLLM's own.
- More importantly: **our actual sparse-indexer TopK gate already
  handles GB10 correctly**, read directly from
  `sparse_attn_indexer.py`:
  ```
  use_cooperative_topk = (
      current_platform.is_cuda()
      and topk_tokens in (512, 1024, 2048)
      and num_rows <= 32
      and logits.stride(0) % 4 == 0
      and current_platform.has_device_capability(90)
      and not current_platform.is_device_capability_family(120)
  )
  use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
      512, 1024, 2048,
  )
  ```
  `not current_platform.is_device_capability_family(120)` explicitly
  excludes GB10 (family 120) from the cooperative path; GB10 falls
  through to `use_persistent_topk`, which has no such exclusion. This
  **directly confirms**, for the first time by reading the actual gate
  rather than inferring from patch presence, the item the AEON-7 triage
  left open ("worth confirming we already handle correctly"). Closed,
  no code change needed.
- We don't use FlashInfer's native sparse-MLA backend at all (our
  FlashInfer is stock 0.6.17, not their custom-built 0.6.18 with the
  `GLM53_NOPE` SM120/SM121 kernels from FlashInfer PRs #4802/#4947), so
  we were never exposed to whatever shared-mem issue exists in
  FlashInfer's own topk JIT module either. Not applicable to us on two
  independent grounds.

**Item 2 (checkpoint pre-slicer) -- real, plausible lever, not yet
trialed.** Read our own `overlay/exl3.py`'s weight-loading path:
`_narrow_tp()` (and `shard_exl3_col`/`shard_exl3_row`) slice each routed-
expert tensor by `tp_rank`/`tp_size` via `.narrow(dim, ...).contiguous()`
-- applied AFTER vLLM's standard loader has already materialized the
FULL, un-sharded tensor from the checkpoint into host memory (vLLM's
default safetensors loader path calls `safe_open(...).get_tensor(name)`,
which copies the complete tensor out of the mmap rather than returning a
view). That means, per large routed-expert tensor, each rank transiently
holds both the full tensor AND its own narrowed half at once, before the
full one is freed -- a real, avoidable peak-memory spike during the load
phase specifically. This lines up directly with this project's own
`memfree_memavailable_gap` and boot-time earlyoom tracks. A pre-sliced
checkpoint (`slice_exl3_checkpoint.py`'s approach: split per-TP-rank
offline, each rank's file only ever contains its own half) would remove
this transient entirely -- each rank's `get_tensor()` call only ever
materializes already-halved data.
Caveat: we run TP=2 across 2 *separate* hosts (one rank per host), so we
don't have the "N ranks competing for one host's RAM simultaneously"
multiplier a single-host TP=2/4 setup would -- the exposure here is the
single-rank-per-host transient 2x-on-the-largest-tensor, not a
cross-rank pile-up. Real, but its actual magnitude (how big the largest
single routed-expert tensor actually is, and whether it's the dominant
term in our known boot-memory pressure or a minor contributor next to
other allocations) is **not yet quantified** -- would need a scratch
trial: slice our own checkpoint with their tool (or an equivalent), boot
against it, and diff peak host RSS during load against our current
un-sliced boot. Not done this session (production was on a brief break,
not available for a full boot trial). Queued as a real candidate for the
next dedicated (not routine) round.

**Item 3 (Sparkinfer dedicated eval) -- downgraded from "research lead"
to "unverifiable dependency," not worth a trial.** Dedicated research
pass materially overturns the 2026-09-15 framing:
- Sparkinfer (`gittensor-ai-lab/sparkinfer`) is real and very active
  (1,568 commits, pushed same-day, MIT-licensed) but tied to Bittensor
  subnet SN74 -- contributors paid in a crypto-incentive token for
  verified speedups, judged by the project's own automated eval bot. A
  full-repo code search for `exl3`/`trellis` returns **zero hits**. It
  supports GGUF and NVFP4/ModelOpt for Qwen3.8/3.6 only -- no GLM, no
  EXL3, nothing resembling cbertucci33's claimed integration surface.
  `sm_121` is a genuine build target, but every published benchmark
  (+86% decode/+127% prefill vs llama.cpp, DSpark speedups) is measured
  on RTX 5090; DGX Spark/GB10 is listed only as an unstarted roadmap
  item, and the "verifiable" eval log is self-hosted/self-reported, not
  third-party audited -- structurally exactly the setup where narrow
  overfitting to the pinned incentive-eval hardware is a real risk, not
  a hypothetical one.
- Inside cbertucci33's own `exl3.py`: generic/dense EXL3 matmuls already
  run through stock `exllamav3_ext.exl3_gemm` -- the same path we use.
  Sparkinfer is invoked ONLY for the routed-MoE-expert path, gated on
  pre-sliced checkpoints, and the module's own docstring calls the
  relevant API "Sparkinfer's **planned** full-rotation Trellis MoE API"
  -- their own word, "planned." Their pinned Sparkinfer commit
  (`d4438d490691f79022fdfc8149e1c5f161d15445`) returns 404 against the
  real public repo; no fork or mirror containing it is findable anywhere
  on GitHub. Their own test for this path monkeypatches
  `_load_sparkinfer_trellis()` rather than exercising a real build. This
  dependency is not obtainable -- their MoE-path performance claims rest
  on something that, as far as can be verified from outside their own
  machine, doesn't exist in public form.
- If it did exist, integration cost would actually be modest (the
  Sparkinfer-specific glue in their `exl3.py` is a small, isolated
  slice, not smeared through the file) -- the blocker is entirely the
  missing artifact, not architectural invasiveness. And yes, it would
  force our checkpoint onto their pre-sliced-per-rank schema, layered on
  top of (not replacing) the current dense-tensor path.
- **Verdict: watch, don't act.** Re-check `gittensor-ai-lab/sparkinfer`
  for `exl3`/`trellis` on the next routine rotation (cheap grep); re-open
  only if that appears or cbertucci33's repo gets a resolved pin. No
  trial possible today -- there's nothing installable to trial against.

### Net

Two real MiaAI-Lab (c)-category items flagged for future dedicated A/Bs
(fair-v5 scheduler, instanttensor loader) and six low-risk (b)-category
backport candidates identified but **not shipped this session** --
volume (6+ new patches) and the higher-priority live decode-speed
investigation reported by the user this same session took precedence;
queued for the next implementation pass. The cbertucci33 GB10-TopK item
is now fully resolved (corrected framing, verified we're safe on both
counts). The checkpoint pre-slicer stays a real, well-reasoned but
unquantified lever. The Sparkinfer lead is downgraded to a cheap watch
item, not a live research thread -- its load-bearing dependency doesn't
verifiably exist. Production untouched throughout.

## Long-context decode-speed investigation, 2026-09-16 (5-9 tok/s at c=1/c=2)

User report: after ~3 days of stability (1 reboot, 1.2-1.3B input tokens
served), long-context decode has "crawled to 5-9 tok/s decode even on
c=1 and c=2" specifically during subagent-heavy sessions (openchamber +
opencode harness). Investigated via mcp-grafana (Prometheus), DCGM host
metrics, and re-reading this project's own prior profiling work --
production traffic was on a brief break, no new live capture was
triggered against it (see "not done this session" below).

**GPU hardware ruled out.** `DCGM_FI_DEV_SM_CLOCK` held a steady
2400-2540 MHz on both nodes across the full 3-day window (no throttling,
running at/above base clock throughout). `DCGM_FI_DEV_GPU_TEMP` cycled
39-79C with load, well within normal range, no thermal runaway or
sustained high-temp plateau. Clean on both counts -- this is not a
hardware degradation story.

**Aggregate server-side metrics don't show a broad regression, but the
tail does.** `request_time_per_output_token_seconds`-derived decode
throughput: p50 over the full 3-day window held flat at ~26.4-26.6 tok/s
the entire time (barely moved). But the **p95 over just the last 6h
dropped to 10.2 tok/s**, versus ~20.5 tok/s measured in yesterday's probe
of this same metric. Typical/short requests are unaffected; the WORST
requests in the distribution have gotten meaningfully worse recently.
That's consistent with a problem concentrated in the long-context tail
specifically, diluted away in the aggregate p50 -- exactly where
subagent-harness traffic (deep, growing multi-turn context, per the
mcp-grafana probe from two sessions ago: median prompt 122,860 tokens,
95.1% prefix-cache hit rate) would land.

**This project already has an unresolved, matching finding from before
this session.** `docs/DESIGN-indexer-workspace.md` (`patch_indexer_
workspace.py`'s design doc, written during the v19-indexercompat work) 
states directly: *"the research train's arithmetic put the indexer's
entire context-proportional decode cost at ~1 ms/step at 100K KV, ~0.5%
of the measured 200 ms long-context delta... roofline arithmetic cannot
prove the indexer is not a contributor -- so this PR simply does not
make a latency claim in either direction."* Read plainly: at 100K KV
context, someone already measured a **~200ms/step decode delta** versus
short-context decode, confirmed the sparse indexer's own theoretically-
expected linear-scaling cost is not the cause (only ~1ms of the 200ms),
and then **explicitly left the actual cause unresolved** -- this was
never root-caused, just ruled out as "not the indexer." **The magnitude
matches the user's report almost exactly**: a healthy short-context
decode step (~26 tok/s implies ~38ms/token) plus a +200ms/step delta at
100K KV gives ~238ms/token, i.e. **~4.2 tok/s** -- squarely inside the
user's observed 5-9 tok/s range. This was not surfaced or connected to
the current complaint until now; it was written up as an aside in a
memory-focused design doc and never cross-referenced into VALIDATION.md
or SPEED.md.

**Corroborating, not yet conclusive: a stray profiling trace.** Two
untracked artifact directories already existed in the working tree
before this session (`recipes/probes/cpu_profiles_v20/`,
`recipes/probes/traces_v20_b1/`, both `git status`-untracked, no
README/manifest, no cross-reference anywhere in VALIDATION.md/NOTES.md/
SPEED.md -- an orphaned capture from an earlier ad-hoc session, unclear
what context length it was captured at). Parsed the rank0 PyTorch trace
directly (gzipped Chrome-trace JSON, 632,699 events):
- Confirms the already-separately-tracked Ampere-fallback GEMM finding
  is still present and substantial: `cutlass_80_wmma_tensorop_bf16_
  s161616gemm...16x16` totals 2.74M us across 19,866 calls -- this is
  the same kernel flagged in the AEON-7 triage entry above as "~36% of
  decode GPU time, identified 2026-09-02, confirmed unchanged
  2026-09-11." Still unaddressed as of this trace. This is a constant
  per-active-token cost (MoE routed-expert GEMM), not itself obviously
  context-length-scaled, so it's a separate, compounding inefficiency,
  not the specific explanation for the *long-context-specific* delta.
- One suspicious data point: `cudaEventSynchronize` totals 7.24M us
  across only 85 calls -- an average of **~85ms per sync event**, large
  enough to be a real contributor to a per-step delta in this range. Not
  conclusive on its own without a paired short-context trace to diff
  against (this file has no companion short-context capture, and no
  metadata confirming what context length it was captured at) -- flagged
  as the most promising lead for a follow-up trace comparison, not a
  confirmed cause.

**Not investigated this session, deliberately**: did not trigger a new
profiling capture (py-spy/torch-profiler) against the live production
containers -- that requires either a longer safe window than a brief
traffic break, or explicit go-ahead, given capture overhead and the
project's standing rule not to disturb production without confirmation.
Also did not instrument the openchamber/opencode harness side (no access
from this environment) -- can't fully rule out a compounding harness-side
effect (e.g. how it paces/batches subagent calls), but the magnitude
match to an already-documented, unresolved SERVER-side ~200ms/step
long-context delta is strong enough that harness-side causes should be
considered secondary, not primary, until this is re-checked.

**Recommended next step** (not started, needs a dedicated window):
capture a fresh, labeled paired trace -- one short-context (a few K
tokens) and one long-context (100K+) decode step, same request shape
otherwise -- and diff them directly for what actually grows with context
length. Candidates worth checking first, in order: the `cudaEventSynchronize`
count/duration, any O(context) CPU-side Python work per decode step
(block-table/seq_lens/position_ids construction), and the indexer's
*scoring* pass specifically (distinct from its already-ruled-out
`cooperative_topk`/`persistent_topk` selection cost) since indexer
scoring over full KV history is inherently O(context) by construction
in this architecture family, unlike selection itself.

**Status**: root cause not found, but the search space is now much
narrower and grounded in this project's own prior (undocumented-until-
now) measurement rather than a fresh guess. Not a GPU hardware issue,
not fully explained by the already-tracked GEMM-kernel inefficiency
alone, most likely the same ~200ms/100K-KV-step delta this project
measured and set aside months ago without root-causing it. Production
untouched throughout this investigation.

## Correction: controlled short/long-context test overturns the 200ms-delta hypothesis (2026-09-16)

Direct follow-up to "Long-context decode-speed investigation, 2026-09-16"
above. With production traffic paused (a real window, not simulated),
sent three controlled, isolated requests directly against the live
`glm53-exl3-v20-upstreamsync:local` server (`c=1`, nothing else running
concurrently) and read each request's own contribution to the
`vllm:request_decode_time_seconds_sum` / `vllm:time_to_first_token_
seconds_sum` / `vllm:request_generation_tokens_sum` counters via tight
before/after deltas -- a clean, direct measurement, no profiler needed:

| Request | Prompt tokens | Completion tokens | Decode-phase (measured) | TTFT (measured) |
|---|---|---|---|---|
| Baseline | 5 | 5 | 42.4 ms/token (23.6 tok/s) | 0.80s |
| Short | 37 | 30 | 29.7 ms/token (33.6 tok/s) | 0.30s |
| Long, cold (0% cache) | 108,001 | 30 | **28.1 ms/token (35.6 tok/s)** | 69.67s |
| Long, repeated (cache hit) | 108,001 | 30 | **28.0 ms/token (35.7 tok/s)** | 2.77s |

**Decode-phase per-token cost did not degrade with context length in
this controlled test** -- 28-30 ms/token essentially flat from 37 to
108,001 tokens, cached or not. This directly contradicts reading the
old `DESIGN-indexer-workspace.md` "~200ms/step long-context delta" note
as the explanation for the current complaint. That old figure either
measured something materially different (concurrent load, a different
vLLM/patch state, real multi-turn conversational KV structure rather
than a single long synthetic prompt) or no longer reproduces on the
current build -- either way, **it does not explain today's user report**,
and citing it as the leading hypothesis in the entry above was wrong.
Retracting that as the primary lead.

**What the same test data actually explains cleanly**: TTFT. Cold (0%
cache) 108K-token prefill took 69.67s -- a real, expected cost at roughly
1,550 tok/s raw uncached prefill throughput, nothing wrong with it, just
the honest cost of genuinely new tokens. The *identical* prompt repeated
immediately after collapsed to 2.77s TTFT (25x faster) via prefix-cache
reuse, confirming caching itself works correctly. **This is the same
mechanism identified in this engagement's very first mcp-grafana probe**:
apparent "tokens/sec" as commonly computed by a client
(`output_tokens / total_wall_clock_time`, TTFT included) is dominated by
prefill/TTFT whenever output is short relative to prompt size --
median output length for this traffic class was ~40 tokens. Do the
arithmetic on a partially-cached long-context turn: even a modest
cache-miss fraction of a 100K+-token context, at ~1,550 tok/s raw prefill,
adds seconds to tens of seconds of TTFT that a client-side "tok/s" counter
will fold into the completion-token denominator, producing exactly the
kind of single-digit "tok/s" the user is seeing -- without decode itself
ever slowing down.

**Revised leading hypothesis**: the user's subagent harness (openchamber
+ opencode) most likely isn't getting the same ~95% prefix-cache hit
rate this fleet's *aggregate* traffic sees. Plausible mechanisms, neither
confirmed yet: (a) many parallel/sequential subagents with large but
mutually-diverging contexts (shared system prompt, divergent task
content) evicting each other's cached prefixes under KV-cache capacity
pressure: (b) the harness's own "tokens/sec" reporting counts TTFT in
the denominator, making a client-side measurement artifact look like a
server-side regression. Neither ruled in nor out this session -- would
need either real (not synthetic) subagent-shaped multi-branch context
traffic replayed against a monitored server, or the harness's own timing
methodology, to settle definitively.

**Status**: the ~200ms/100K-KV-step decode delta from `DESIGN-indexer-
workspace.md` stays on record as a real, previously-measured, still-
unexplained data point from past work -- but it is NOT the explanation
for this session's user report. Root cause of *that* old number remains
genuinely open; root cause of *the current complaint* is now most likely
prefix-cache-hit-rate/TTFT-related, not a decode-speed regression at all.
Production traffic resumed normally after this test (both hosts healthy,
sanity completion request verified correct output before and after).

## RoCE dual-rail verification (2026-09-16)

Resolves the follow-up flagged in "Routine upstream check, 2026-09-13"
(MiaAI-Lab measured +63% busbw dual-rail on their kit; never checked
whether our own cluster config uses both rails). Read-only host
inspection, both nodes, production untouched.

**Physical hardware: both rails already exist and are cabled/connected
on both hosts.** Each node has two separate physical ConnectX-7 cards
(`rocep1s0f0`/`enp1s0f0np0` and `roceP2p1s0f0`/`enP2p1s0f0np0`), each
with a live link-up port on the `192.168.177.0/24` RoCE subnet.
Confirmed with a direct ping across the second rail specifically
(`192.168.177.87` from the other node): sub-millisecond RTT, 0% loss,
same as the first rail. **No new cable needed** -- this is a stock
dual-CX7 DGX Spark configuration, already fully wired, just not
exploited.

**Current config: single-rail.** `start.sh` pins exactly one HCA device
per rank (`HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"`,
`WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"`), passed to NCCL as a
single `NCCL_IB_HCA=<one device>` value. The second card on each host
(`roceP2p1s0f0`/`enP2p1s0f0np0`) sits live and reachable but genuinely
idle from NCCL's point of view.

**What dual-rail would and would not help, and why**: TP=2 across two
*separate hosts* (no NVLink) needs an NCCL all-reduce over the network
at every layer, for every token, during both prefill and decode --
that's the only place RoCE bandwidth matters for this deployment.
Checkpoint loading and container/image distribution do NOT use this
fabric in our setup (weights come from HF per-host independently;
`sparkrun`'s image distribution goes over the separate management
network, `10.7.0.x` -- confirmed via the `ib:`/`mgmt:` address split
`sparkrun status` already shows per node).
- **Prefill**: processes a large batch of tokens per forward pass, so
  each all-reduce message is large -- genuinely bandwidth-bound.
  Doubling available bandwidth via a second rail should give a real,
  proportional benefit here. This lines up well with this fleet's own
  workload shape (median prompt ~122K tokens per the mcp-grafana probe
  earlier this engagement) -- unlike the dense-FP8 kernel work rejected
  earlier for favoring decode at prefill's expense, this lever favors
  the phase that actually dominates our traffic.
- **Decode**: one token (or a handful, at our typical c=1/c=2) per step
  -- the all-reduce message is tiny, and tiny-message collective time is
  dominated by per-message latency/overhead, not raw bandwidth. A
  second rail mainly helps here only if the first rail is congested
  (unlikely at low concurrency); expect little to no decode-throughput
  change from this change alone.
- MiaAI-Lab's own +63% figure (20.9 vs. 12.8 GB/s peak busbw) is a raw
  NCCL all-reduce microbenchmark, not an end-to-end inference-throughput
  measurement -- a reasonable proxy for the prefill story above, but the
  real serving-throughput gain is unmeasured and would need our own A/B.

**Known pitfall if this is ever turned on**: MiaAI-Lab's own PR #172
(referenced in the 2026-09-14 upstream-check entry above) fixed a bug
where their launcher's RoCE GID preflight only checked the FIRST HCA on
a dual-rail config (`HEAD_CX7_IB=dev1,dev2`), silently skipping
validation and failing ~60s into a real run instead of at boot. Our own
`start.sh` GID-preflight logic (around line 584-603) would need the
same fix before a dual-rail config could be trusted -- not yet checked
whether ours has this exact gap, since we've never run dual-rail.

**Status**: verified feasible (hardware ready, zero cabling cost),
config change identified (`NCCL_IB_HCA` needs both devices, comma-
separated, on both ranks), expected benefit is prefill-specific and
plausible-but-unquantified for us specifically. Not enabled this
session -- this touches core NCCL fabric behavior and should get a
deliberate test (and the GID-preflight dual-rail check ported first),
not a drive-by flip in production.

## InstantTensor checkpoint loader validation (2026-09-16)

Follow-up to the "MiaAI-Lab's instanttensor checkpoint loader" item
flagged in the 2026-09-16 upstream-check round. Read-only research +
container inspection, no scratch trial run this session (recommended as
the next step, not completed).

**Architecture is safe for our EXL3 format by design, verified not
assumed.** InstantTensor (`scitix/InstantTensor`, Apache-2.0, ScitiX AI +
Peking University, real test suite) is a generic, dtype-agnostic
`safe_open`-compatible reader -- it has zero EXL3/quant-specific code,
by design, and doesn't need any: vLLM's native support (merged upstream
via PR #36139, refined #46868/#52801 -- our base image postdates all
three, **no vLLM-side patch needed**) wires it in at
`default_loader.py`'s iterator-selection level, strictly below every
per-parameter `weight_loader` hook. Our own `overlay/exl3.py`'s
`shard_exl3_col`/`shard_exl3_row`/`_load_exl3` dispatch is completely
untouched either way -- architecturally about as low-risk as a loader
swap could be built.

**Our own guardrails would catch most corruption loudly**: `_load_exl3`
raises on any shape mismatch; `process_weights_after_loading` checks
every expert's MCG marker (`0xCBAC1FED`) and raises if it doesn't match
-- a real per-expert integrity check. **Real residual gap**: neither
check validates the trellis payload bytes themselves (the bulk of the
checkpoint) -- a silent bit-flip inside a correctly-shaped trellis
tensor would pass both checks and only show up as degraded output
quality, not a crash.

**We already have real production evidence it round-trips correctly on
this exact hardware** -- the NVFP4 lane (`glm-5.3-flash-nvfp4-vllm.yaml`,
a different non-EXL3 checkpoint format) has already run `--load-format
instanttensor` and passed a 249,951-token needle-in-haystack test, 4/4
codes retrieved, zero corruption. Doesn't cover EXL3's int16-packed
trellis format specifically, but is real, not hypothetical, evidence on
TP=2/GB10.

**Two concrete open risks, not resolved**:
1. MiaAI-Lab installs it `--no-deps` "to dodge an NCCL version
   conflict," despite upstream's own PR #52801 stating it only depends
   on Torch as of >=0.1.9 -- something doesn't match on their end, and
   pip's resolver isn't verifying it. Worth resolving before trial given
   this fleet already tracks an unrelated but real open NCCL issue
   (vLLM #51921).
2. **`scitix/InstantTensor#13` (open)**: loading a second, small model
   after the main one (their case: a speculative-decode draft model)
   fails with a host-staging-allocator OOM, even with the loader's own
   memory-budget knob turned down. This maps directly onto our own MTP
   drafter load pattern, on a fleet already flagged across half a dozen
   memory-pressure tracks as earlyoom-sensitive. `#19` (open) documents
   a related CUDA host-registration failure on some driver/kernel
   combos.
3. Silver lining: both known failure modes are LOUD (process
   termination / registration abort before reading), not silent
   corruption -- meaningfully de-risks the worst-case scenario, though
   it doesn't close the trellis-payload gap above.

**What the actual upside is, and isn't**: boot/weight-load time only --
upstream reports 10-32x load-time speedups on H200/H20; MiaAI-Lab
measured ~35s for their 164 GiB checkpoint. **Explicitly no decode/
prefill throughput change.** Boot time is not currently a pain point
this project has flagged anywhere -- we already tolerate up to a
1-hour boot timeout and it's never come up as an active complaint.

**Recommendation**: not a priority. The upside doesn't address any
current pain point, and the one open, concrete risk (#13, OOM on a
second small model load) lands exactly on this fleet's known weak spot
(MTP drafter load + earlyoom sensitivity) rather than somewhere neutral.
If ever revisited: a throwaway scratch image with the dependency
resolved WITHOUT `--no-deps` (to see what it actually wants and whether
that's a problem), then a real TP=2 boot watching specifically for the
MCG-marker check and for the drafter-load OOM pattern, then a logit-
level A/B (not just "did it boot") before any production consideration.

**Status**: architecture validated as safe-by-design; two concrete,
unresolved risk flags identified; recommendation is to deprioritize
given low payoff and risk concentration on an existing weak spot. No
scratch trial run this session.
