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
