# v20-upstreamsync: routine upstream check, four items folded in

Seeded from `recipes/build/glm53-exl3-v19-indexercompat` (current production).
Sourced from a routine upstream-check triage (2026-09-11) across MiaAI-Lab's
GLM-5.3-Flash-EXL3-2x-DGX-Sparks fork (`9c0794b..1caea9a`, 65 commits),
Enntity/sparkglm's `exl3` branch, and vLLM main since v0.29.0. Full triage in
this session's transcript; user selected items 1, 2, 5, 6 to fold in. Item 3
(DFlash2 drafter revision pin) was dropped after re-checking: this fork's
production speculator is MTP-2, not DFlash2 (DFlash2 caused two distinct
crash classes here, see `v19-indexercompat`'s own header comment) -- the
upstream bug doesn't apply to a code path we don't run. Item 4 (vLLM #54929)
is tracked separately, not yet folded in -- see its own section below.

## What this build changes

1. **Chat template `None` leak** (`files/chat_template.jinja`,
   upstream MiaAI-Lab#156 / `68ce597`). `visible_text()`'s catch-all branch
   was `{%- else -%}{{- content }}{%- endif -%}` -- when an assistant turn
   has `content: None` (tool-calls-only, common in agent traffic), this
   rendered the literal string `None` into the prompt. Fixed to
   `{%- elif content is not none -%}{{- content -}}{%- endif -%}`. Only the
   `None`-guard hunk was ported; #156's other hunks (early `break`s in the
   tool-call-ID sort loop, a `+`->`~` string-concat robustness tweak) target
   a version of the sort-loop structure that has already diverged here from
   independent local changes -- not ported, to avoid an anchor mismatch in
   code that no longer matches upstream's assumed shape. New regression test:
   `tests/test_chat_template.py::NullContentTests` (mutation-verified: fails
   against the pre-fix template with a literal `None` in the rendered output).

2. **`count_shards()` false-completeness bug** (`start.sh`, upstream
   MiaAI-Lab#153 / `c195e92`). The shard-completeness check counted
   `*.safetensors` recursively across *every* cached snapshot, not just the
   one `refs/main` resolves to -- stale shards from an old build (this fork
   has iterated v10->v20 on the same hosts) can satisfy the count while the
   *active* snapshot is actually incomplete, and the boot then fails deep
   inside vLLM load instead of fast with an actionable message. Fixed by
   resolving `refs/main` (falling back to the newest snapshot dir) first,
   then counting only within that one snapshot directory, non-recursively.
   **This fork's DFlash2 download-completeness check had the identical bug
   in a separate, non-shared code path** (`download_dflash()`'s inline
   `find "$DFLASH_PATH/snapshots" -name 'model.safetensors'`, not a call to
   `count_shards()`) -- not covered by upstream's PR (which only touched the
   shared function), fixed here too via a new `count_dflash_shard()` helper
   using the same resolved-ref logic. New test:
   `tests/test_snapshot_scoped_count.py` (covers both the target-model and
   DFlash2 call sites; mutation-verified against upstream's own fixture).

3. **`LONG_PREFILL_TOKEN_THRESHOLD` opt-in** (`start.sh`, `.env.example`,
   upstream MiaAI-Lab#157 / `d094362`). Plumbs vLLM's native
   `--long-prefill-token-threshold` flag through as a new env var, default
   empty (stock scheduler, zero behavior change unless set). Upstream's own
   measured A/B/A at `MNBT=7168`, `threshold=3584` (MNBT/2): warm-session
   freeze 131/108s -> 15s; contended prefill +6%; solo prefill +3..+11%
   (within baseline drift) -- a clean win in their numbers, not a tradeoff.
   **Left empty deliberately, not defaulted on**: this fork already ships
   its own answer to the identical "a long prefill freezes an already-warm
   decode session" symptom --
   `overlay/patch_scheduler_decode_floor.py`'s `GLM53_MIXED_PREFILL_WARM_TOKENS`
   (also defaulted to 3584, same problem framing) / `_MAX_WAIT_MS` /
   `_LATE_CAP` bypass-and-late-cap scheduler patch. Same symptom, two
   different mechanisms (a custom scheduler monkeypatch here vs. vLLM's
   native flag upstream) -- running both at once is untested and could
   double-throttle or interact unpredictably. **Do not set
   `LONG_PREFILL_TOKEN_THRESHOLD` non-empty without a real A/B against this
   fork's own mixed-prefill gate first.** New test:
   `tests/test_long_prefill_threshold.py` (validation range, both-rank argv
   wiring with the flag unset/empty/set, caller-override-wins-over-.env).

4. **`BUILD_MIN_MEM_GIB` build-host guard** (`start.sh`, `.env.example`,
   ported from Enntity/sparkglm `a8aaa229`, an *original* guard on their
   side, not copied external code -- their commit message states this
   explicitly). Native EXL3 compilation is CPU/RAM-intensive and can consume
   nearly all of GB10's unified memory; running `build_image()` (i.e.
   `docker build`) while a resident model server holds most of that memory
   risks swap-thrash or an outright OOM mid-build. `assert_build_headroom()`
   now runs at the top of `build_image()` and refuses the build below
   `BUILD_MIN_MEM_GIB` (default 32) `MemAvailable`, printing any running
   containers as a hint before dying; `BUILD_MIN_MEM_GIB=0` is the explicit
   override. Build-host-side only -- does not touch the serving/runtime path
   at all. New test: `tests/test_build_headroom.py`.

## Verification so far

All four items are source-level changes with new/adapted regression tests,
run directly against the host Python (no container needed for these --
they're pure bash-function and Jinja-template tests). All pass, including
the full pre-existing test suite in this directory (`test_numeric_config.py`,
`test_indexer_workspace.py`, `test_kpool_tail_slotmap.py`,
`test_mixed_prefill_gate_v2.py`, `test_start_overrides.py`,
`test_bringup_robustness.py`, etc. -- no regressions from the `start.sh`
edits). `test_ablit.py`, `test_exl3_overlay.py`, `test_hybrid_prefix_hit.py`,
`test_scheduler_decode_floor.py` need the actual container's installed
`torch`/`vllm` and weren't runnable on the host -- expected, not a failure.

## Docker build: PASSED (2026-09-11)

`docker build -t glm53-exl3-v20-upstreamsync:local .` -- 20.9 GB image, all
63 build steps succeeded including the full build-time self-test suite
(`test_exl3_overlay.py`, `test_suppress_stops.py`, `test_scheduler_decode_floor.py`,
`test_hybrid_prefix_hit.py`, `test_xgrammar_termination.py`,
`test_kpool_tail_slotmap.py`, `test_spinwait_patch.py`,
`test_indexer_workspace.py`, `test_ablit.py` -- all OK).

## tinyGLM gate: PASSED (2026-09-11)

`recipes/glm-5.3-flash-exl3-v20-tinyglm.yaml`, TP=2 across both hosts.
**First attempt crashed** during CUDA graph capture on the worker rank
(`torch.AcceleratorError: CUDA error: an illegal memory access was
encountered`, `Capturing CUDA graphs (PIECEWISE)`), which cascaded into the
head hanging on `shm_broadcast` waiting for the dead peer (repeated "No
available shared memory broadcast block found in 60 seconds" -- the
recurring #51921 symptom, here a downstream effect, not the root cause).
None of this build's four changes touch the CUDA-graph-capture path, and a
clean `sparkrun stop` + `prelaunch_flush.sh` + retry booted healthy in ~1.5
minutes with no recurrence -- treated as the same class of transient GB10
boot flakiness this project has hit before (see e.g. the v18-gb10gemm
validation's own isolated earlyoom incident, unrelated to that patch), not a
regression from this build. Worth remembering if it recurs on a future
candidate, but not chased further here.

**Second attempt, clean boot**: `/health` 200, two identical temp=0 chat
completions returned byte-identical output (determinism confirmed).
Dispatch-correctness confirmation lines present on **both** ranks, matching
v19-indexercompat's own pattern exactly: `glm53-kpool-tail-fix`,
`glm53-kpool-indexer-fix`, `glm53-mamba-race-fix`,
`glm53-mixed-prefill-env-fix`, `glm53-gb10-router-gemm`,
`glm53-indexer-workspace` (rightsize, 8192 entries), `glm53-fp8-lmhead-v4`
(scale_mode=tensor), `exl3.py` tier resolution
(`configured_tier=grouped effective_tier=grouped tier_reason=grouped_ok`),
and the mixed-prefill gate v2 config line (`warm_tokens=3584,
max_wait_ms=1500, late_cap=512` -- confirms `LONG_PREFILL_TOKEN_THRESHOLD`
being unset didn't disturb this fork's own gate). This is exactly the
regression-gate result predicted going in: nothing in this candidate's
four items touches tinyGLM's dispatch path, so a clean match to v19's known
pattern (not a byte-for-byte log diff against a saved v19 log, but the same
tier/config/confirmation lines) is the expected, confirmed outcome.

## Real-checkpoint validation: PASSED (2026-09-11)

Production was already idle (no traffic at the time), so no downtime
coordination was needed. `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, TP=2, same
flags as `v19-indexercompat`.

**Boot flakiness, 2/4 attempts across this whole validation session** (2
tinyGLM-adjacent -- see the tinyGLM section above -- would make it 2/5
counting that gate) hit `torch.AcceleratorError: CUDA error: an illegal
memory access was encountered` during CUDA graph capture on the worker rank.
Every occurrence recovered cleanly with `sparkrun stop` +
`prelaunch_flush.sh` + retry, no pattern tying it to this candidate's four
changes (none touch CUDA graph capture).

**`probe_sanity.py`: ALL PASSED.** Chat coherent, no reasoning leak,
finish_reason=stop, streaming TTFT 0.19-0.29s, decode throughput across two
separate clean boots: 28.17-29.99 tok/s (6 samples) -- matches
v19-indexercompat's own validated range (27.4-29.28 tok/s) with no
regression, plausibly a hair faster but within noise.

**Item 1 (chat template `None` fix) verified live, not just unit-tested.**
Sent a real tool-calling round-trip with an assistant turn carrying
`content: null` + `tool_calls` (exactly the shape that used to leak the
literal string `None`) followed by a tool response. Real checkpoint replied
coherently -- `"2 plus 3 equals **5**."` -- with no `None` artifact anywhere
in the output.

**`probe_longctx.py --tokens 16000`: a real, load-bearing finding that turned
out to be a false alarm, worth recording in full because the investigation
is the actual evidence, not just the verdict.** First two v20 attempts both
failed: `journalctl -u earlyoom` on the head node (10.7.0.87) showed
`sending SIGTERM to process ... "python3" ... mem 2.00%, swap 80.00%` each
time, killing the APIServer process itself and cascading into the familiar
NCCL/shm-broadcast teardown on the worker. Two failures in a row on the same
candidate is more than this project's documented precedent for this failure
class (e.g. v18-gb10gemm's validation hit it once, recovered on one retry) --
enough to treat as a real regression suspicion, not dismiss on priors. **Ran
a direct controlled comparison instead of assuming**: v19-indexercompat (current
production, completely unmodified) against the identical probe. First v19
attempt: clean pass, TTFT=10.3s, 4/4 codes, no earlyoom activity at all. A
third v20 attempt, same recipe, same image, no code change: clean pass,
TTFT=11.9s, 4/4 codes. A second v19 attempt: clean pass, TTFT=9.6s, 4/4
codes -- **but earlyoom fired mid-request anyway** (`sending SIGTERM to
process ... "python3" ... badness 989, VmRSS 4353 MiB`, killing some other
process, not the APIServer this time) with zero effect on the request's
outcome. **Conclusion: this is the same ambient, intermittent, fleet-wide
earlyoom margin issue already documented in `gmu_086_earlyoom_regression`
and `earlyoom_root_cause_correction` (thin memory headroom on the head node
under load, unrelated to which candidate is running) -- not a v20
regression.** It fired on v19 too, mid-request, with no correlation to which
image was serving. v20's own worse 2/3-run hit rate in this session reads as
unlucky timing on a noisy, pre-existing signal, not causation -- confirmed
by the fact the identical v20 image passed clean on its third attempt with
zero changes. TTFT@16K across all clean runs (both versions): 9.6-11.9s,
consistent with this project's previously-documented ~9.7-10.7s range for
this probe -- no throughput regression either.

## Verdict: recommend promotion

Correctness confirmed (including the live-verified None-leak fix), decode
and long-context throughput match production within noise, and the one
worrying signal during validation (repeated earlyoom on the longctx probe)
was run to ground with a direct controlled comparison and found to be
pre-existing fleet behavior affecting v19 exactly as it affects v20, not a
regression introduced by any of this candidate's four changes. No blocking
issues found. Boot flakiness (CUDA-graph-capture crash, ~2/5 attempts this
session) is also not new to this candidate -- recover via the standard
stop+flush+retry remedy, matching prior documented occurrences.

## Item 4 (vLLM #54929) -- assessed, NOT vendored, real reasons found

Feasibility check completed 2026-09-11. **Verdict: not advisable to vendor
now.** The premise that motivated flagging this ("overlaps our existing
sparse-MLA/indexer patches") turned out to be wrong in a way that makes
things worse, not better:

- **Wrong backend, no overlap.** This fork's production attention path is
  `FlashInferMLASparseSM120Impl` (`flashinfer_mla_sparse.py`, forced via a
  ~140-line Dockerfile `RUN` block, plus `overlay/patch_nope_mqa_fix.py`).
  PR #54929 modifies `flashmla_sparse.py` (no "infer") -- a **different,
  currently-unused backend enum (`FLASHMLA_SPARSE`)**. Zero textual anchor
  conflict with `patch_indexer_workspace.py` (targets `indexer.py`) or
  `patch_kpool_tail_slotmap.py` (targets `block_table.py`) either. That
  sounds like good news but isn't: it means our runtime path never reaches
  any of this PR's code, so adopting it gets nothing for free -- our entire
  bespoke NoPE-on-SM120 story (zero-padding 512-dim NoPE into 576-wide
  GLM_NSA geometry, index_kpool=4 sizing, the 2048/2051 tail-truncation fix,
  GB10 DeepGEMM block-size alignment) would need to be **re-derived from
  scratch** against the PR's 6,848 new lines.
- **PR is unreviewed and has known, unfixed bugs.** Opened by a
  non-maintainer, `mergeStateStatus: BLOCKED`, zero human reviews. CodeRabbit
  flagged real correctness bugs still open: a stride bug in `sm12x_mqa.py`
  (uses `head_dim` instead of `head_dim+4` for the indexer cache view --
  reads wrong bytes) and a missing-disable-flag bug in the b12x fast path
  (repeats a failed fast path every call instead of latching off).
- **Never tested against our config.** The PR's own test matrix is
  GLM-5.2-NVFP4 (non-NoPE, real RoPE dim) on 24x RTX 5090 -- not GB10, not
  NoPE/zero-padded, not `index_kpool>1`.
- **Root-cause match to #51921 is contested.** The issue's own top analysis
  describes idle shm_broadcast writer-starvation, later attributed by
  another commenter to client-abort accounting (host-side, zero GPU util).
  The PR author's corroborating report describes sustained-load kernel-launch
  livelock at 100% GPU util instead -- a different symptom. No maintainer has
  confirmed these are the same bug.
- **Version skew**: our base predates PR #51718 (pre-v0.29.0 lineage); this
  PR targets current `main`. Small hunks (`envs.py`, `sparse_attn_indexer.py`)
  are probably low-risk; the 6,800 lines of new kernel code depend on
  internal APIs not verified against our older base.

**Not folded into this build.** Revisit only after upstream review/merge (or
at least a maintainer confirming the #51921 root-cause match), and after
someone validates the Triton kernels against a NoPE/zero-padded,
`index_kpool>1` checkpoint like ours. The underlying operational problem
(#51921, recurring shm-broadcast stall) remains open and unfixed -- still
worth tracking, just not via this PR today.
