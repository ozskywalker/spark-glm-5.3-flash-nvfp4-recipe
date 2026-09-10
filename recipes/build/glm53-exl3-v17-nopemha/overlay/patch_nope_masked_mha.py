#!/usr/bin/env python3
"""Backport of vLLM PR #55738 ("[Perf][GLM-5.3-Flash] Dense/masked-MHA sparse
prefill for the NoPE (256, 0, 256) layout + skip the NoPE K concat")
-- COMMIT 3 ONLY (the K-concat skip). The PR's other two commits (register
the NoPE dims with FlashAttnPrefillBackend; add the dims to the masked-MHA
allow-list) are NOT included here -- verified against this image's actual
installed vLLM source (not just the PR diff) and found genuinely
inapplicable, not merely risk-averse to take:

1. ``vllm/v1/attention/backends/mla/prefill/flash_attn.py`` (the file the
   PR's commit 1 patches) fails to import in this exact image on its own,
   independent of anything this PR touches:
   ``ImportError: cannot import name 'compile_flash_attn_varlen_func_from_specs'
   from vllm.v1.attention.backends.fa_utils`` -- real version skew between
   this vLLM build and the PR's tree. Confirmed live in a container built
   from this exact image (`docker run --entrypoint python3 ... -c "import
   vllm.v1.attention.backends.mla.prefill.flash_attn"`), same ImportError,
   same line. This was already found and documented once before, at
   v10-vllmpatch / v12-combined time (see recipes/build/glm53-exl3-
   v10-vllmpatch/NOTES.md and recipes/VALIDATION.md's "Track 1" section);
   re-verified here because the base image digest
   (``vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...``) is
   byte-for-byte unchanged between v10 and this build, so the installed
   source for this file has not moved.
2. ``vllm/model_executor/layers/attention/sparse_mla_attention.py``'s
   ``_is_masked_mha_available`` (the PR's commit 2 target) is structurally
   different from what the PR's diff assumes. The PR's diff adds a tuple to
   a ``model_dims not in (...)`` allow-list; this image's installed function
   instead starts with a hard ``if not
   current_platform.is_device_capability_family(100): return False`` and
   then checks each dim field against ONE fixed DeepSeek-V3.2 shape (128,
   512, 128, 64, 128) -- there is no dim-tuple allow-list to extend. Worse:
   the SM100 gate means masked MHA structurally cannot activate on this
   fleet's actual hardware (SM121 / GB10) regardless of what dims would be
   added -- so even a faithful reconstruction of the PR's allow-list logic
   for this vLLM version would be a no-op here, not a perf win. Confirmed
   live via ``inspect.getsource`` against this exact image.

Neither of those two blockers is a matter of "the anchor text drifted a
little" -- they are a broken import (unrelated to this PR) and a dead
code path on this hardware. Taking on a version-specific rewrite of either
for zero measurable benefit is exactly what this project's fail-closed
anchor convention exists to refuse. If flash_attn.py's import bug is ever
fixed upstream or independently patched here, and if this fleet moves to
SM100-family hardware, this file's scope should be revisited.

**What IS included** -- commit 3, the NoPE K-concat skip in
``vllm/model_executor/layers/attention/mla_attention.py``:
``MLACommonBaseImpl._concat_k_nope_k_pe`` unconditionally allocated a fresh
``(*k_nope.shape[:-1], nope_dim + pe_dim)`` tensor and copied k_nope and the
broadcast k_pe into it, even when ``k_pe`` is a genuinely zero-width NoPE
placeholder -- which it always is for GLM-5.3-Flash (qk_rope_head_dim=0).
Skips the allocation and copy entirely when ``k_pe.shape[-1] == 0`` and
returns ``k_nope`` directly, per the PR's own measurement: ~134 MB/layer
per 16k-token prefill chunk of allocation+copy avoided, and the PR's
isolated ablation of this hunk alone showed -3.5% TTFT at a 2x65536-token
serving point (0.0% noise floor at that point per the PR's own ablation
table) where the masked-MHA switch (commits 1/2, not included here) does
not even apply.

This function is verified byte-for-byte identical to the PR's diff context
in this image's installed ``mla_attention.py`` -- the anchor matched
without adjustment. Unconditional (no env-var gate): this project's
convention is to only ship backports it has verified, not experiments, and
this one is a strict, correctness-preserving special case of an existing
fallback path (``k_pe`` width 0 concatenated with anything is a no-op by
definition).

Emits a ``logger.info_once`` the first time the NoPE fast path fires (using
this file's existing ``logger = init_logger(__name__)``), so a tinyGLM or
real-checkpoint boot log gives a direct, unambiguous confirmation the patch
is live rather than relying only on this project's own inference from the
allocation being gone.

Verified against a live container built from this exact image before being
written here (anchor matched byte-for-byte in the installed site-packages).
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET_MLA_ATTENTION = Path(
    os.environ.get(
        "GLM53_MLA_ATTENTION_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py",
    )
)

MARK = "        # [glm53-nope-masked-mha] skip the concat entirely for a NoPE model\n"

ANCHOR = (
    "        k = torch.empty(\n"
    "            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),\n"
    "            dtype=k_nope.dtype,\n"
    "            device=k_nope.device,\n"
    "        )\n"
)

# vllm-project/vllm#55738 commit 3 (`_concat_k_nope_k_pe`): a NoPE model's
# k_pe is a genuinely zero-width tensor, so torch.cat-equivalent allocation
# and copy work is wasted -- returning k_nope directly is exact, not an
# approximation (concatenating a zero-width tensor is the identity).
PATCHED = (
    "        # [glm53-nope-masked-mha] skip the concat entirely for a NoPE model\n"
    "        # (k_pe.shape[-1] == 0, true for GLM-5.3-Flash) -- see recipe header /\n"
    "        # vllm-project/vllm#55738 commit 3. Concatenating a zero-width tensor\n"
    "        # is the identity, so returning k_nope is exact, not approximate.\n"
    "        if k_pe.shape[-1] == 0:\n"
    "            logger.info_once(\n"
    "                \"[glm53-nope-masked-mha] _concat_k_nope_k_pe: NoPE fast path \"\n"
    "                \"active (k_pe width 0) -- vllm-project/vllm#55738 commit 3 backport live\"\n"
    "            )\n"
    "            return k_nope\n"
    "\n"
    "        k = torch.empty(\n"
    "            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),\n"
    "            dtype=k_nope.dtype,\n"
    "            device=k_nope.device,\n"
    "        )\n"
)


def verified_state(text: str) -> bool:
    return (
        text.count(MARK) == 1
        and text.count(PATCHED) == 1
        and text.count(ANCHOR) == PATCHED.count(ANCHOR)
    )


def prepare(target: Path) -> tuple[str, str]:
    if not target.is_file():
        raise SystemExit(f"missing {target}")
    source = target.read_text()
    if source.count(MARK):
        if source.count(MARK) != 1 or not verified_state(source):
            raise ValueError(
                f"partial/inconsistent nope-masked-mha patch at '{target}' -- refusing to touch a half-patched file"
            )
        return source, "already present"
    n = source.count(ANCHOR)
    if n != 1:
        raise ValueError(
            f"pinned nope-masked-mha anchor drifted (found {n}, expected 1) at '{target}' -- re-derive the patch"
        )
    out = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(out):
        raise ValueError(f"nope-masked-mha post-patch verification failed at '{target}'")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-nope-masked-mha.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]
    patched_src, action = prepare(TARGET_MLA_ATTENTION)
    compile(patched_src, str(TARGET_MLA_ATTENTION), "exec")
    if preflight_only:
        print(f"{TARGET_MLA_ATTENTION.name}: nope-masked-mha preflight OK ({action})")
        return 0
    if patched_src != TARGET_MLA_ATTENTION.read_text():
        replace_file(TARGET_MLA_ATTENTION, patched_src)
        clear_pyc(TARGET_MLA_ATTENTION)
    print(f"{TARGET_MLA_ATTENTION.name}: nope-masked-mha {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
