#!/usr/bin/env python3
"""Backport of vLLM PR #55736, commit 2/3 (4c4c883e74b8): write the absorbed
MQA query token-major and skip the empty RoPE concat for NoPE models.

Two independent hunks, both unconditional (this is a correctness-neutral perf
fix, not an experiment -- no env-var gate, matching this project's convention
for backports it has verified rather than merely proposed):

1. ``vllm/model_executor/layers/attention/mla_attention.py``, the
   ``q_pad_num_heads is None`` branch of the decode MQA-absorption path:
   allocates the bmm output directly as ``(B, N, L)`` and writes the
   ``(N, B, L)`` bmm result into its own transposed view (`out=`), instead of
   allocating `(N, B, L)`, bmm'ing into it, then doing a separate
   ``.transpose(0, 1)`` -- one GEMM, no extra transpose-copy step to chase.
2. ``vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py``,
   ``forward_mqa``: for a NoPE model (``qk_rope_head_dim == 0``, true for
   GLM-5.3-Flash), ``q`` arrives as ``(ql_nope, q_pe)`` with ``q_pe`` a
   zero-width tensor. The old code unconditionally ``torch.cat``'d the pair,
   which for a zero-width second operand still pays for
   ``CatArrayBatchedCopy`` (measured by the PR author: 13.7 us/layer at 256
   decode tokens, ~0.75 ms/layer per 16k prefill chunk). If ``q_pe`` is
   genuinely empty and ``ql_nope`` is already contiguous (true once hunk 1
   above writes it token-major), skip the concat and use ``ql_nope`` directly.

Commits 1 (KDA Triton stride-arg cleanup) and 3 (MoE router-GEMM dedup) of
the same PR are NOT included here -- see recipes/build/glm53-exl3-
v10-vllmpatch/NOTES.md for why (commit 1's target module,
``ops/third_party/kda/fused_recurrent.py``, does not exist in this vLLM
build's KDA implementation; commit 3 needs a closer trace against our EXL3
routing dispatch before it's safe to take).

Verified against a live boot of this exact container before being written
here (both anchors matched byte-for-byte in the installed site-packages).
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
TARGET_FLASHINFER_SPARSE = Path(
    os.environ.get(
        "GLM53_FLASHINFER_MLA_SPARSE_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
    )
)

MARK_MLA = "                    # [glm53-nope-mqa-fix] write the bmm result directly\n"

ANCHOR_MLA = (
    "                if self.q_pad_num_heads is not None:\n"
    "                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))\n"
    "                    mqa_ql_nope.resize_((N, B, L))\n"
    "                else:\n"
    "                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))\n"
    "\n"
    "                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)\n"
    "                torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n"
    "\n"
    "                # Convert from (N, B, L) to (B, N, L)\n"
    "                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n"
)

# Both branches now do their own bmm+transpose write (see vllm-project/
# vllm#55736 commit 4c4c883e74b8): the `if` branch keeps the original
# (N, B, L)-then-transpose shape (it needs the padded-then-resized buffer,
# which can't be a transposed view target); the `else` branch (this model's
# path -- no head padding configured) writes straight into a token-major
# (B, N, L) buffer via a transposed `out=` view, matching hunk 2 below in
# flashinfer_mla_sparse.py, which then needs no torch.cat for a NoPE model.
PATCHED_MLA = (
    "                if self.q_pad_num_heads is not None:\n"
    "                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))\n"
    "                    mqa_ql_nope.resize_((N, B, L))\n"
    "                    # [glm53-nope-mqa-fix] this branch keeps the\n"
    "                    # original (N, B, L)-then-transpose shape -- the\n"
    "                    # padded-then-resized buffer above can't be a\n"
    "                    # transposed `out=` view target.\n"
    "                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n"
    "                    mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n"
    "                else:\n"
    "                    # [glm53-nope-mqa-fix] write the bmm result directly\n"
    "                    # into a token-major (B, N, L) buffer -- see\n"
    "                    # recipe header / vllm-project/vllm#55736 commit\n"
    "                    # 4c4c883e74b8. A NoPE model (qk_rope_head_dim==0,\n"
    "                    # e.g. GLM-5.3-Flash) then needs no torch.cat at\n"
    "                    # all in flashinfer_mla_sparse.py's forward_mqa\n"
    "                    # (see patch_nope_mqa_fix.py's second hunk).\n"
    "                    mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))\n"
    "                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))\n"
)

MARK_FLASHINFER = "        # [glm53-nope-mqa-fix] NoPE models never need the concat\n"

ANCHOR_FLASHINFER = (
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
)

PATCHED_FLASHINFER = (
    "        if isinstance(q, tuple):\n"
    "        # [glm53-nope-mqa-fix] NoPE models never need the concat\n"
    "            # -- see vllm-project/vllm#55736 commit 4c4c883e74b8 and\n"
    "            # patch_nope_mqa_fix.py's first hunk, which writes\n"
    "            # ql_nope contiguous token-major so this is safe.\n"
    "            ql_nope, q_pe = q\n"
    "            if q_pe.shape[-1] == 0 and ql_nope.is_contiguous():\n"
    "                q = ql_nope\n"
    "            else:\n"
    "                q = torch.cat(q, dim=-1)\n"
)


SITES = (
    (TARGET_MLA_ATTENTION, "mla_attention decode-MQA bmm", MARK_MLA, ANCHOR_MLA, PATCHED_MLA),
    (TARGET_FLASHINFER_SPARSE, "flashinfer_mla_sparse forward_mqa concat", MARK_FLASHINFER, ANCHOR_FLASHINFER, PATCHED_FLASHINFER),
)


def verified_state(text: str, mark: str, patched: str, anchor: str) -> bool:
    return (
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
    )


def prepare_one(target: Path, name: str, mark: str, anchor: str, patched: str) -> tuple[str, str]:
    if not target.is_file():
        raise SystemExit(f"missing {target}")
    source = target.read_text()
    if source.count(mark):
        if source.count(mark) != 1 or not verified_state(source, mark, patched, anchor):
            raise ValueError(
                f"partial/inconsistent nope-mqa patch at '{name}' -- refusing to touch a half-patched file"
            )
        return source, "already present"
    n = source.count(anchor)
    if n != 1:
        raise ValueError(
            f"pinned nope-mqa anchor '{name}' drifted (found {n}, expected 1) -- re-derive the patch"
        )
    out = source.replace(anchor, patched, 1)
    if not verified_state(out, mark, patched, anchor):
        raise ValueError(f"nope-mqa post-patch verification failed at '{name}'")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-nope-mqa.tmp")
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
    for target, name, mark, anchor, patched in SITES:
        patched_src, action = prepare_one(target, name, mark, anchor, patched)
        compile(patched_src, str(target), "exec")
        if preflight_only:
            print(f"{target.name}: nope-mqa-fix preflight OK ({name}: {action})")
            continue
        if patched_src != target.read_text():
            replace_file(target, patched_src)
            clear_pyc(target)
        print(f"{target.name}: nope-mqa-fix {action} ({name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
