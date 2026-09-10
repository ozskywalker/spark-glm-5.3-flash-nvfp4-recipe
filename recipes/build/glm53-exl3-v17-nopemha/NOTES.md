# v17-nopemha: vLLM PR #55738 backport, scoped down after real verification

Seeded from `recipes/build/glm53-exl3-v15-combined` (current production).
Candidate 1 of 3 in a downtime testing window (2026-09-09/10); production
traffic stopped for the duration.

## What this build adds

**`vllm-project/vllm#55738` commit 3 of 3 only** ("skip the NoPE K concat"):
`MLACommonBaseImpl._concat_k_nope_k_pe` in `mla_attention.py` returns
`k_nope` directly when `k_pe.shape[-1] == 0` instead of allocating a fresh
tensor and copying into it -- concatenating a zero-width tensor is the
identity, so this is exact, not approximate. GLM-5.3-Flash is a NoPE model
(`qk_rope_head_dim=0`), so this fast path fires on every call to this
function for this model. New overlay patch: `overlay/
patch_nope_masked_mha.py`, unconditional, anchor verified byte-for-byte
against this image's installed source (both via a live container check
before writing the patch, and via the patch script's own fail-closed
preflight).

## What this build deliberately does NOT add -- and why

The PR's other two commits (register the GLM-5.3-Flash NoPE `(256, 0, 256)`
dims with `FlashAttnPrefillBackend.supports_mla_dimensions`; add the same
dims to `_is_masked_mha_available`'s allow-list) are the PR's actual
headline win -- switching prefill from the per-token top-k MQA kernel to
dense/masked MHA for sequences up to the masked-MHA threshold. They are
**not included here**, for real structural reasons re-confirmed live
against this exact image (base image digest
`vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...` is unchanged
since this was first investigated at v10-vllmpatch/v12-combined time --
see `recipes/build/glm53-exl3-v10-vllmpatch/NOTES.md` and
`recipes/VALIDATION.md`'s "Track 1" section):

1. `vllm/v1/attention/backends/mla/prefill/flash_attn.py` (the PR's commit 1
   target) fails to import in this image on its own:
   `ImportError: cannot import name 'compile_flash_attn_varlen_func_from_specs'
   from vllm.v1.attention.backends.fa_utils`. Real version skew, unrelated to
   this PR. Reproduced: `docker run --rm --entrypoint python3
   glm53-exl3-v15-combined:local -c "import
   vllm.v1.attention.backends.mla.prefill.flash_attn"`.
2. `vllm/model_executor/layers/attention/sparse_mla_attention.py`'s
   `_is_masked_mha_available` (the PR's commit 2 target) is structurally
   different from what the PR's diff assumes -- no dim-tuple allow-list to
   extend, just a hard-coded single-shape check for DeepSeek-V3.2's
   `(128, 512, 128, 64, 128)`, gated behind `if not
   current_platform.is_device_capability_family(100): return False`.
   That SM100 gate is the real blocker: this fleet's hardware is SM121
   (GB10), so masked MHA cannot activate here **regardless of what dims are
   added to the allow-list**. Reconstructing this function's logic for this
   vLLM version to add a dims check that can never be reached would be
   dead code, not a perf win.

Both were re-verified live in this session (not just trusted from the old
notes) via `inspect.getsource` and a direct import attempt inside a
container from `glm53-exl3-v15-combined:local`, confirming the base image
has not moved since the original investigation. This is exactly the
situation this project's fail-closed anchor convention exists to catch:
the PR's diff doesn't apply to code shaped like this, and forcing a
version-specific rewrite for a code path that's dead on our hardware isn't
worth the risk for zero measurable benefit.

**Net effect, AS ORIGINALLY EXPECTED**: this backport is real but smaller
than the PR's own headline numbers suggest for our fleet. The PR's own
ablation table shows the K-concat-alone contribution most clearly at long,
above-masked-MHA-threshold context (`2x65536: -3.5% TTFT, 0.0% noise
floor`) -- that is the regime this build's win should show up in.
Short/mid prefill, where the PR's dense/masked-MHA switch (not included
here) does the bulk of its work, should show little to no change.

**UPDATE, post-tinyGLM gate: the above expectation was wrong. The K-concat
fix is dead code on this fleet, not "smaller than expected" -- confirmed
via a fourth blocker deeper than the two already known.**

The tinyGLM boot log itself states it directly, on both TP ranks:

```
WARNING [mla_attention.py:551] Sparse MLA impl has no dense-MHA prefill
path; using the top-k MQA path only.
```

Traced the cause: `vllm/v1/attention/backends/mla/
flashinfer_mla_sparse_sm120.py`'s `FlashInferMLASparseSM120Impl` -- the
actual impl class instantiated for GLM-5.3-Flash on this fleet's SM121/
GB10 hardware -- hardcodes `supports_dense_mha_prefill = False` as a class
attribute, unconditionally:

```python
class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    is_sparse = True
    supports_dense_mha_prefill = False
```

`mla_attention.py`'s `MLAAttention.__init__` checks exactly this flag
first, before ever consulting `get_mla_prefill_backend` /
`FlashAttnPrefillBackend.supports_mla_dimensions` / `_is_masked_mha_
available`'s allow-list: `if self.impl.is_sparse and not self.impl.
supports_dense_mha_prefill: ... self.prefill_backend = None`. With
`prefill_backend` unconditionally `None` for this model on this hardware,
`forward_mha` (and every one of its four internal call sites of
`_concat_k_nope_k_pe`, across both `mla_attention.py` and `sparse_mla_
attention.py`) is structurally unreachable -- **every prefill request,
at every context length, already goes through `forward_mqa` only**, on
stock v15-combined and on this v17-nopemha build alike. This is a stronger
and more fundamental blocker than the two found originally (`flash_attn.py`'s
broken import; `_is_masked_mha_available`'s SM100-only gate) -- it sits one
layer above both of them and forecloses the entire `forward_mha` code
family regardless of whether those two were fixed.

Confirmed empirically, not just by source reading: sent chat completions
at 11, ~1,209, and ~3,009 prompt tokens (spanning tinyGLM's fresh-prompt,
mid-length, and chunked-prefill-forcing regimes, well past `max_num_
batched_tokens=2048`) against the booted tinyGLM server. The `logger.
info_once` diagnostic this patch adds (fires the first time `_concat_k_
nope_k_pe`'s NoPE fast path executes) **never fired in any of the three
requests** -- direct, positive confirmation the patched code never runs,
not just an absence of contrary evidence.

**Consequence**: this build's one included change (`_concat_k_nope_k_pe`'s
zero-width skip) is real, correct, and inert -- a no-op on this fleet's
actual serving path. It cannot regress anything (the code is simply never
reached), but it also cannot deliver any of the K-concat-alone win the PR
measured on GB300/SM100. Did not proceed to a real-checkpoint boot: the
gating property here (`supports_dense_mha_prefill`) is a static class
attribute of the SM120 backend class actually instantiated for this model,
not something that varies with model weights, so there is no reason to
expect tinyGLM's dummy weights hid a real-checkpoint-only code path --
spending a real-checkpoint boot (which two more downtime candidates are
waiting on) to reconfirm a result already demonstrated empirically was not
worth the shared GPU time. Recommend NOT promoting v17-nopemha: it is a
technically-correct but zero-benefit backport. If a real-checkpoint boot
is wanted anyway for extra confirmation, that is a real, cheap option that
remains open -- flagging it explicitly rather than deciding it unilaterally.

## Validation status

- [x] Patch preflight + correctness check (CPU, no GPU) against installed source -- pass
- [x] tinyGLM dispatch-correctness gate -- clean boot, deterministic output, BUT diagnostic
      confirms the patched code path never executes (see above)
- [ ] Real-checkpoint boot + probe_sanity.py -- **not run**, see rationale above
- [ ] Real-checkpoint prefill timing vs v15-combined baseline -- **not run**, see rationale above
- [x] sparkrun job(s) stopped
