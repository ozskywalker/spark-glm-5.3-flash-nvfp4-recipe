# v19-indexercompat: forward-compat hardening for vLLM PR #51718 (defensive only)

Seeded from `recipes/build/glm53-exl3-v18-gb10gemm` (candidate 2, PASSED,
recommended for promotion in this same downtime window). Candidate 3 of 3 in
the 2026-09-09 downtime testing window (candidate 1, v17-nopemha, was NOT
promoted -- dead code; candidate 2, v18-gb10gemm, PASSED cleanly). This is
the last candidate in the window; production still needs to come back up
after this validates.

## What this build changes

**Nothing functional.** This is different in kind from the first two
candidates in this window: it is not adopting new upstream functionality, it
is hardening our own existing patch against a known future landmine. No new
overlay file, no new `COPY`/`RUN` Dockerfile step -- the two files already
shipped by every prior candidate back to v2 (`overlay/
patch_indexer_workspace.py`, `tests/test_indexer_workspace.py`) simply have
new content.

## Background

Upstream vLLM PR #51718 ("[6/N][KV-Cache Layout Refactor] Standardize KV
cache layout", merged into the vLLM lineage feeding v0.29.0, **NOT present in
our current base image** -- this build does not rebase onto it) renames, in
`vllm/v1/attention/backends/mla/indexer.py`:

- `MLAAttentionSpec.compress_ratio` -> `.tokens_per_state`, with an added
  `assert isinstance(self.kv_cache_spec.tokens_per_state, int)` immediately
  before the assignment
- `MLAAttentionSpec.storage_block_size` -> `.num_states`

Confirmed directly from `gh pr diff 51718 --repo vllm-project/vllm`
(2026-09-09), re-fetched and re-read for this hardening rather than trusted
from an earlier session summary. The `compress_ratio` rename hunk (in
`DeepseekV32IndexerMetadataBuilder.__init__`) reads:

```diff
         if isinstance(self.kv_cache_spec, MLAAttentionSpec):
-            self.compress_ratio = self.kv_cache_spec.compress_ratio
+            # MLA compression is a whole number of tokens per state (fractions
+            # are whisper block pooling and never reach MLA).
+            assert isinstance(self.kv_cache_spec.tokens_per_state, int)
+            self.compress_ratio = self.kv_cache_spec.tokens_per_state
```

`storage_block_size` -> `num_states` appears at two OTHER call sites in the
same file (`build()`'s `compressed_slot_mapping_buffer` construction and its
`get_paged_mqa_logits_metadata` call) -- neither is inside any text this
patch script pins.

Our shipped `overlay/patch_indexer_workspace.py` patches this exact file and
pins its "builder cross-check" anchor (`ANCHOR_GUARD`) on the literal old
line `self.compress_ratio = self.kv_cache_spec.compress_ratio` inside
`DeepseekV32IndexerMetadataBuilder.__init__`. The patch was already
fail-closed by design (counts anchor occurrences, raises a clear error if the
anchor doesn't match rather than silently no-op'ing or corrupting) -- so this
was not an active bug, it was a known future build-time failure whenever this
fork eventually rebases onto a vLLM base that includes #51718.

## What changed: dual-anchor matching

`overlay/patch_indexer_workspace.py`'s "builder cross-check" site (site 3 of
3) now tries the current-base anchor (`ANCHOR_GUARD_OLD`, byte-identical to
the pre-hardening single `ANCHOR_GUARD`) first, falls back to the
post-#51718 anchor (`ANCHOR_GUARD_NEW`) if the old one isn't found, and
raises the original fail-closed error -- now naming BOTH forms tried and
their individual match counts -- only if NEITHER matches. `ANCHOR_GUARD` and
`PATCHED_GUARD` are kept as backward-compat aliases pointing at the `_OLD`
forms.

**The other two pinned sites (`ANCHOR_IMPORT`/module header, `ANCHOR_SIZE`/
`get_max_prefill_buffer_size`) needed NO dual-form treatment.** Re-confirmed
by reading #51718's diff directly: neither hunk touches the module header or
`get_max_prefill_buffer_size` in this file. `storage_block_size`/
`num_states` likewise needed no treatment in THIS patch specifically --
grepped the whole script for `storage_block_size` before writing this note:
zero hits. Those two renamed call sites live inside `build()`, which this
patch never touches.

The append block injected after the guard anchor (the mode dispatch, the
compress-ratio-disagreement check, the boot-time info/warning logs) is
**identical text for both anchor forms** -- it only reads
`self.compress_ratio` (the builder's own local attribute, assigned by either
anchor form and never itself renamed by #51718) and `self.vllm_config`,
never `kv_cache_spec.compress_ratio`/`.tokens_per_state` directly. This is
why only the guard's anchor/patched PAIR needed duplicating, not any of its
downstream logic -- confirmed by a dedicated test
(`test_old_and_new_anchor_produce_identical_append_block`) that diffs the
two patched outputs' append blocks and asserts they match byte-for-byte.

## Verification of the two anchor forms

- **Old (current-base) form**: LIVE-VERIFIED, 2026-09-09, against a
  throwaway container of `glm53-exl3-v18-gb10gemm:local`
  (`docker run --rm --entrypoint python3 glm53-exl3-v18-gb10gemm:local -c
  "..."`). The installed `indexer.py` still uses `compress_ratio`/
  `storage_block_size` -- zero occurrences of `tokens_per_state`/
  `num_states` anywhere in the file (53 vs. 0, 3 vs. 0). The exact
  `ANCHOR_GUARD_OLD` text (unchanged from the pre-hardening `ANCHOR_GUARD`)
  was confirmed present exactly once in the live container's file, matching
  what every prior candidate back to v2 already relied on.
- **New (post-#51718) form**: **NOT live-verified.** We do not have a
  v0.29.0-era vLLM container to check against. `ANCHOR_GUARD_NEW` is
  transcribed directly from PR #51718's own diff hunk for this exact block.
  Confidence is high that the transcription is correct (it's a direct copy
  of upstream's diff, re-fetched for this task rather than trusted from a
  prior summary) but NOT that it will be the exact source text this fork
  eventually rebases onto -- a later PR could touch this same block again
  before that rebase happens. If that occurs, the patch fails closed with a
  message naming both forms tried and their match counts, rather than
  silently no-op'ing or corrupting the file. This is exactly the fail-closed
  behavior the original single-anchor patch already had; the hardening only
  widens what counts as a match, it does not weaken the failure path.

## Test coverage

`tests/test_indexer_workspace.py`: 16 pre-existing tests (unchanged in
substance) + 13 new ones, 29 total, all green (host-only, no GPU, no torch):

- `test_current_base_patch_is_byte_identical_to_manual_replacement` -- the
  main behavior-neutrality proof: applying `prepare()` to a current-base
  fixture produces EXACTLY the bytes a hand-substitution of each site's
  single (anchor -> patched) pair would, with no trace of the new form
  (`"tokens_per_state" not in patched`) in the output.
- `test_fixture_apply_and_idempotence_new_anchor` -- full apply / idempotent
  re-apply / `--preflight` lifecycle against a fixture built from the
  transcribed post-#51718 anchor text.
- `test_old_and_new_anchor_produce_identical_append_block` -- proves the
  injected dispatch/cross-check code is byte-identical regardless of which
  anchor form matched.
- `test_dual_anchor_prefers_old_form_when_only_old_present` -- the new
  form's anchor text and `"tokens_per_state"` never appear when only the old
  form is present.
- `test_builder_ratio_mismatch_raises_both_directions_new_anchor` /
  `test_builder_agreeing_ratios_pass_new_anchor` -- the fail-closed
  compress-ratio cross-check (Codex's original PR-8 blocker regression test)
  replayed against a spec stand-in (`_MLAAttentionSpecNew`) that has NO
  `compress_ratio` attribute at all, only `tokens_per_state` -- so a
  regression back to reading the old attribute name on a post-#51718-shaped
  spec would raise `AttributeError` immediately rather than silently
  passing.
- `test_fail_closed_when_neither_anchor_form_matches` -- garbage input (the
  guard line's RHS renamed to neither known form) still raises, with a
  message naming both forms tried and their individual match counts (0 and
  0), file left untouched.
- Plus the full pre-existing suite (sizing formula edge cases, chunk-list
  equivalence by exhaustion, knob enum, idempotence, half-patched refusal,
  launcher/Dockerfile/README wiring) unchanged, all still green against both
  the fixture and the live container's actual installed file (`GLM53_
  INDEXER_BACKEND_PY_SRC=<extracted indexer.py>`).

Also confirmed: syntax-checked (`ast.parse`), and a standalone smoke test
applying the patch to both hand-built OLD and NEW fixtures outside the test
suite, plus one applying to real garbage, before ever committing to the test
suite rewrite.

## Validation status

- [x] Re-confirmed #51718's exact rename via `gh pr diff 51718 --repo
      vllm-project/vllm` -- matches the session's prior research
      byte-for-byte, including the added `isinstance` assert.
- [x] Confirmed via the PR diff (not just re-reading a summary) that
      `get_max_prefill_buffer_size`/the `ANCHOR_SIZE` anchor is untouched by
      #51718 (zero hits for `get_max_prefill_buffer_size` or `ANCHOR_SIZE`-
      relevant lines in the diff).
- [x] Live-verified the "old" anchor against a throwaway
      `glm53-exl3-v18-gb10gemm:local` container.
- [x] Hardened `overlay/patch_indexer_workspace.py` with dual-anchor
      matching, fail-closed on neither match.
- [x] Extended `tests/test_indexer_workspace.py` with 13 new tests covering
      both anchor forms; 29/29 pass, host-only, no GPU.
- [ ] Image build (`glm53-exl3-v19-indexercompat:local`) -- pending.
- [ ] tinyGLM dispatch-correctness gate -- pending. MUST be byte-for-byte
      identical to v18-gb10gemm's tinyGLM boot (deterministic output match),
      since this patch is a pure no-op on our current base.
- [ ] Real-checkpoint sanity boot (TP2, both nodes), `probe_sanity.py` only
      -- pending. No perf delta expected; decode should land in the same
      26.9-29.6 tok/s band seen this session.
- [ ] sparkrun job(s) stopped.

(Results appended below once boots complete.)
