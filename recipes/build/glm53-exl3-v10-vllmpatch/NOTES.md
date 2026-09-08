# v10-vllmpatch: Track 1 vLLM upstream PR backports

Seeded from `recipes/build/glm53-exl3-fatfork` (2026-09-08). Carries sparkglm's
grouped-prefill kernel unchanged (`EXL3_GROUPED_PREFILL_K4`) plus one
additional backport from vLLM upstream. Part of the plan at
`/home/luser/.claude/plans/dapper-moseying-clock.md`, Track 1.

## Applied

**vllm-project/vllm#55736, commit 2/3 (`4c4c883e74b8`, "Write the absorbed
MQA query token-major and skip the empty RoPE concat")** — `overlay/
patch_nope_mqa_fix.py`, unconditional (no env flag; correctness-neutral perf
fix, verified against this exact image's installed source, both anchors
matched byte-for-byte). Two hunks:
1. `vllm/model_executor/layers/attention/mla_attention.py`: the decode
   MQA-absorption `else` branch (no head padding configured — our path) now
   writes the bmm result directly into a token-major `(B, N, L)` buffer via
   a transposed `out=` view, instead of allocating `(N, B, L)` and doing a
   separate `.transpose(0, 1)` afterward. The `if` branch (head padding)
   keeps the original shape since its buffer can't be a transposed view
   target — it just gets its own copy of the bmm+transpose instead of
   relying on code that used to run unconditionally after the if/else.
2. `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`, `forward_mqa`:
   for a NoPE model (`qk_rope_head_dim==0`, true for GLM-5.3-Flash) with a
   genuinely zero-width `q_pe` and a contiguous `ql_nope` (guaranteed by hunk
   1), skips `torch.cat` entirely instead of paying for
   `CatArrayBatchedCopy` on a zero-width concat.

## Deferred — real version drift found, not a risk-tolerance call

While preparing this candidate, checked each of the other three vLLM-PR
items against this image's actual installed vLLM source (not just the PR
diffs) and found real structural mismatches — not just "needs more care,"
genuinely different code:

- **#55736 commit 1 (KDA Triton stride-arg cleanup): deferred, target module
  doesn't exist.** The PR patches `vllm/models/glm5next/nvidia/ops/
  third_party/kda/fused_recurrent.py` and `kernels.py`. This image's
  installed vLLM has no `ops/third_party/kda` package at all — `kda.py`
  imports `fused_recurrent_kda` from `vllm.third_party.flash_linear_attention.
  ops.kda` instead, a differently-organized (and likely differently
  implemented) module. Porting the same stride-argument optimization there
  is plausible but is a real rewrite against unfamiliar code, not a diff
  cherry-pick — out of scope for this pass.
- **#55736 commit 3 (MoE router-GEMM dedup): deferred, needs a deeper trace.**
  Confirmed (parent session, before this build) that our EXL3
  `Exl3Config.apply()` takes precomputed `topk_ids`/`topk_weights`, not
  `router_logits` — so the risk lives in whether `MoERunner`'s internal
  logits computation (which this commit would rely on) produces identical
  results to the current explicit `self.gate(hidden_states)` call this
  image's installed `model.py` makes at line ~266, whose own comment reads
  "pre-computed router_logits, so compute them here unconditionally" —
  phrasing that suggests this was already a deliberate choice for this
  model, not an oversight this PR is fixing. Silently wrong here would be a
  routing-correctness bug, not just a perf regression — not worth taking on
  faith.
- **#55738 (dense/masked-MHA NoPE prefill): deferred, target code doesn't
  match and one target file doesn't even import cleanly.**
  `vllm/v1/attention/backends/mla/prefill/flash_attn.py` (one of the PR's
  three touched files) fails to import in this exact image: `ImportError:
  cannot import name 'compile_flash_attn_varlen_func_from_specs' from
  vllm.v1.attention.backends.fa_utils` — real version skew within this
  install, unrelated to anything this session touched. Separately,
  `vllm/model_executor/layers/attention/sparse_mla_attention.py`'s
  `_is_masked_mha_available` in this image is structurally different from
  what the PR's diff assumes: it starts with
  `if not current_platform.is_device_capability_family(100): return False`
  (a hard SM100-only gate, not a tuple-membership check against a list of
  supported dim-tuples) and takes different parameters. The PR's 13-line
  diff (add one more tuple to a list) doesn't apply to code shaped like
  this — a real backport would mean rewriting the gate's logic for this
  vLLM version, not adding one tuple, and is exactly the kind of
  version-specific reconstruction this project's fail-closed anchor
  convention is built to refuse rather than risk. Deferred.
- **#55737 FlashKDA: deferred, needs its own integration pass.** Confirmed
  the compiled `vllm._flashkda_C.abi3.so` extension already ships in the
  base image, but zero Python glue exists (`kda.py` has no
  `_flashkda_C` reference, no `_resolve_kda_prefill_backend`). The PR's
  actual integration is substantial — a new backend-selection function, new
  workspace buffer allocation sized from scheduler config, a new
  `_flashkda_prefill` method, and call-site wiring into the existing forward
  pass — closer in shape to a small standalone feature addition than a
  patch-file cherry-pick. Worth doing properly in its own pass rather than
  rushed here, especially since its only real benchmark numbers are
  GB300/SM100, not our SM121 — it needs our own measurement regardless.

## Track 5 spin-wait tuning check

`mmastrac/glm-5.3-flash-4x-gx10`'s reported `busy_loop_s: 0.002` win maps
directly onto this project's own existing `GLM53_SPINWAIT_MS` knob (`overlay/
patch_spinwait.py` patches the exact same `busy_loop_s` parameter in vLLM's
`shm_broadcast.py`, converting ms to seconds). `0.002` seconds = 2ms =
`GLM53_SPINWAIT_MS=2` — **already tested and rejected** on this fleet per
that patch file's own docstring: "The 2 ms candidate lost 1.68% decode in
its paired test" (16ms is what's shipped). mmastrac's win was measured on a
4-node TP4 setup, not our TP2 config — the two results aren't actually in
conflict, just different hardware/topology finding different optima. No new
work needed; this was already covered.

## Build

`docker build -t glm53-exl3-v10-vllmpatch:local recipes/build/glm53-exl3-v10-vllmpatch/`
