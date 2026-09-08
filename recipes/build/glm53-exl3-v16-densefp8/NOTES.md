# v16-densefp8: MiaAI-Lab's FP8 weight-only dense projections, ported

Seeded from `recipes/build/glm53-exl3-v15-combined` (E3 + vLLM PR backports +
OOM-observer + FlashKDA + MoE gate dedup), plus:

- **`GLM53_DENSE_FP8`** (default off, zero behavior change): FP8 e4m3
  weight-only quantization (Marlin kernel) for the BF16 dense projections --
  shared-expert MLP, dense-MLP, KDA and MLA linear-attention projections,
  group-selectable (`shared`, `dense`, `kda`, `mla`, or `all`). Ported from
  MiaAI-Lab's upstream update at `/home/luser/build-exl3-v2` commit
  `eb0da1d` (2026-09-08), analyzed and adopted as a candidate in
  `recipes/VALIDATION.md`, "MiaAI-Lab update analysis: adaptive-k (skip) and
  FP8 dense projections (real candidate, provisional)".

## What changed vs. upstream's own patch

Upstream's `overlay/patch_dense_fp8.py` stages `exl3.py` through
`/opt/glm53/exl3.py` and installs it at container-start time (their
`start.sh` re-applies all overlay patches on every launch, reading
`GLM53_DENSE_FP8` fresh each time -- this is how they get a "no rebuild
needed" runtime toggle). This fork's Dockerfile instead applies every patch
at **image build time** (`COPY`/`RUN` baked in), so:

- `overlay/exl3.py`'s graft (`Glm53DenseFp8Method`, `_glm53_dense_fp8_group`,
  the `get_quant_method` dispatch change) is applied directly to this
  fork's own `overlay/exl3.py` (already carrying E3's grouped-MoE tier and
  sparkglm's kernel-tier additions) rather than copied from upstream
  wholesale -- confirmed clean graft point: our `get_quant_method`'s
  `RoutedExperts`/`LinearBase` dispatch block and the `Exl3MoEMethod` class
  immediately after it match upstream's diff context byte-for-byte, despite
  our file's other divergence.
- `overlay/patch_dense_fp8.py` drops the redundant exl3.py install step
  (our Dockerfile's existing `COPY overlay/exl3.py ...` line already handles
  it) and keeps only the `kda.py`/`model.py` constructor patch.
- That constructor patch is applied **unconditionally at build time**
  (upstream's script gated it on `GLM53_DENSE_FP8` at the point it runs,
  which only works when applied at container start with a live env var
  already set -- gating it at build time, before any runtime knob exists,
  would have made the knob permanently inert). This matches how
  `patch_flashkda.py` already works in this build: the patch is always
  present, the actual behavior is gated by a runtime env check inside the
  patched code (`_glm53_dense_fp8_groups()` reads `GLM53_DENSE_FP8` fresh on
  every `get_quant_method` call). With the knob off (default), KDA/MLA
  layers that now reach `Exl3Config.get_quant_method` just fall through to
  `UnquantizedLinearMethod()` -- functionally identical to the un-patched
  behavior.
- Both anchor strings (`kda.py`'s `saved_quant_config` constructor lines,
  `model.py`'s `quant_config=None,  # MLA projections are BF16 in checkpoint`
  line) verified byte-for-byte against this image's actual installed vLLM
  source before grafting -- no version drift found, unlike several Track 1
  items from the earlier vLLM-PR-backport investigation.

## What was evaluated and NOT ported

`GLM53_ADAPTIVE_K` (adaptive verification length): read the full patch
(`overlay/patch_adaptive_k.py` in the upstream repo) -- it is built entirely
around DFlash2's fixed 8-token (1 anchor + 7 draft) drafter block, trimming
a per-step verified prefix from a candidate set `{2,4,7}` based on an EMA of
historical acceptance. This fleet runs **MTP-2** (2 speculative tokens per
step), not DFlash2 -- there is no 7-token block to trim a prefix of, so the
entire mechanism has no analog here. Upstream's own README even lists "MTP
k=2 baseline ~24.6 tok/s" as a comparison point, matching this fleet's own
measured 24.7-28.0 tok/s decode baseline almost exactly -- confirming we are
the config this patch has nothing to offer, not a version-drift excuse.

## Validation status (update as testing proceeds)

- [x] Symbol/import check: `Glm53DenseFp8Method`/`_glm53_dense_fp8_group`
  importable via the correct package-relative path; dispatch logic tested
  directly against representative prefixes (shared/dense/kda/mla groups,
  MTP/visual/draft exclusion, KDA-vs-MLA layer-type disambiguation) -- all
  passed. Default-off path confirmed to return `set()`/`None` throughout,
  matching pre-patch behavior.
- [x] Full existing `test_exl3_overlay.py` + sibling test suite (E2 tier
  resolution, dflash2 overlay, ablit, indexer workspace, spinwait, hybrid
  prefix-hit, kpool tail slot-map, xgrammar, scheduler decode-floor) still
  passes clean after the graft -- no interaction with E3/sparkglm/FlashKDA/
  moe-gate-dedup's own patches (different files or non-overlapping regions
  of the same file, confirmed directly for kda.py).
- [ ] tinyGLM dispatch-correctness gate (`glm-5.3-flash-exl3-v16-tinyglm.yaml`,
  `GLM53_DENSE_FP8=all`, prepped and validated as a recipe file -- NOT yet
  booted; needs the same downtime gate as a real-checkpoint boot, since
  tinyGLM still reserves `gpu_memory_utilization`'s fraction of real GPU
  memory).
- [ ] Real-checkpoint boot + memory-margin check
  (`glm-5.3-flash-exl3-v16-densefp8-vllm.yaml`, `GLM53_DENSE_FP8=dense,kda`
  -- matches upstream's own measured config, narrower than the tinyGLM
  gate's "all").
- [ ] Real-checkpoint A/B vs. v15-combined: throughput at 16K/64K minimum
  (matching upstream's own measured range), PLUS an explicit output
  coherence/accuracy check (side-by-side same-seed completions, FP8 on vs.
  off) -- not just a tok/s number, given this is a lossy quantization
  change with no full KLD panel even from its own authors.

**Deliberately not started**: any GPU-touching step (tinyGLM boot, real
boot). Production traffic is live on both nodes as of 2026-09-08 (this
build was prepped alongside a live-serving v15-combined); the user has
been told this needs a scheduled downtime window and will confirm when
ready.
