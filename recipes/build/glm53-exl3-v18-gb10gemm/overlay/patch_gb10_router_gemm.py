#!/usr/bin/env python3
"""Backport of vLLM PR #54048, "[Bugfix][MoE] Enable cuBLAS out_dtype router
GEMM on all CUDA archs (fixes family-120/GB10)".

``vllm/model_executor/layers/fused_moe/router/gate_linear.py``'s
``GateLinear`` picks between several router/gate GEMM tiers. Tier 5 (this
patch's target) is the fused cuBLAS ``torch.mm(..., out_dtype=torch.float32)``
epilogue: one bf16 x bf16 -> fp32 GEMM instead of a plain bf16 GEMM plus a
separate bf16->fp32 cast kernel. Before this patch, eligibility for that tier
was gated on ``self.allow_specialized_router_gemm``, which is:

    can_use_specialized_kernels = (
        current_platform.is_cuda() and (is_hopper or is_blackwell) and not bias
    )

where ``is_blackwell = current_platform.is_device_capability_family(100)`` --
SM100 (B200/GB200) only. GB10 / DGX Spark is SM121a, i.e.
``is_device_capability_family(120)`` ("family-120"), which this check does
NOT recognize as either Hopper or "Blackwell" in this narrower sense -- so
GB10 fell through to Tier 6 (plain ``F.linear``, a bf16 GEMM plus a separate
bf16->fp32 cast) even though the cuBLAS out_dtype epilogue this tier uses has
nothing Hopper/SM100-specific about it: it's a plain ``torch.mm`` kwarg
(cuBLAS on CUDA, hipBLASLt on ROCm), unconditionally available on any
CUDA/ROCm device. The upstream PR author verified directly on a GB10 that
``is_device_capability_family(120)`` correctly selects this path once the
gate is decoupled from ``allow_specialized_router_gemm``.

Fix: introduce ``self._router_gemm_cublas_capable`` (``is_cuda() or
is_rocm()``, no-bias only -- ``torch.mm`` has no bias term) as its own
predicate, independent of the Hopper/SM100-only specialized-kernel gate, and
use it in both places the old compound condition appeared: the ``__init__``
eligibility computation and ``set_out_dtype``'s post-hoc recomputation (the
gate module's ``out_dtype`` is often set after construction, once the expert
quantization method is known -- true for this project's ``exl3`` path).

This is a DISTINCT bug from this project's already-shipped
``patch_moe_gate_dedup.py`` (backport of vLLM PR #55736 commit 3/3, which
removes a duplicate CALL to the gate module in
``vllm/models/glm5next/nvidia/model.py``). That patch controls how many
times the gate module runs; this patch controls which GEMM TIER the
surviving call actually executes through -- completely different file, no
interaction between the two beyond both touching MoE routing.

Correctness/precision angle, not (primarily) a throughput play: without this
fix, GB10 computed the router logits as bf16-rounded values (Tier 6's
``F.linear`` runs in the weight's own bf16 dtype, then a separate cast to
fp32) before they feed the argmax/softmax that picks routed experts. Tier 5
computes the whole bf16 x bf16 -> fp32 GEMM in one epilogue -- exact,
full-precision fp32 accumulation, not a downstream cast of an already-
bf16-rounded product. For a 288-expert router this narrows (but does not
eliminate) the chance of a near-tied logit pair flipping which expert wins
top-k purely from rounding, and removes one extra kernel launch per MoE
layer per forward pass (the separate cast Tier 6 pays for).

Verified against a live container of this exact image
(glm53-exl3-v15-combined:local) before being written here: the installed
``gate_linear.py`` matched the upstream PR's "before" context byte-for-byte
at both hunks (no reconciliation needed -- unlike several other backports in
this project's history, e.g. v10-vllmpatch/v17-nopemha, this file has NOT
drifted from stock upstream in this image).

Applied unconditionally (this project's convention for backports it has
verified rather than merely proposed, correctness-neutral-or-better, no
env-var gate) -- also adds an ``info_once`` diagnostic log line so a boot log
can directly confirm the widened path fires on this hardware (it only fires
when the new gate admits a device the OLD gate would have excluded, i.e.
family-120 CUDA or non-no-bias-carve-out ROCm -- never on Hopper/SM100,
where the specialized-kernel gate already covered this tier).
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_GATE_LINEAR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "fused_moe/router/gate_linear.py",
    )
)

MARK_INIT = "        # [glm53-gb10-router-gemm] Fused bf16 x bf16 -> fp32 GEMM eligibility.\n"

ANCHOR_INIT = (
    "        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype\n"
    "        # epilogue folds the fp32 cast into the GEMM, removing the standalone\n"
    "        # bf16->fp32 copy kernel that otherwise runs before grouped_topk.\n"
    "        # cuBLAS on CUDA (SM90+, via allow_specialized_router_gemm); hipBLASLt on\n"
    "        # ROCm, which supports the same out_dtype epilogue.\n"
    "        self._router_gemm_no_bias = not bias\n"
    "        self.allow_cublas_router_gemm = (\n"
    "            (\n"
    "                self.allow_specialized_router_gemm\n"
    "                or (current_platform.is_rocm() and self._router_gemm_no_bias)\n"
    "            )\n"
    "            and self.weight.dtype == torch.bfloat16\n"
    "            and self.out_dtype == torch.float32\n"
    "        )\n"
)

PATCHED_INIT = (
    "        # [glm53-gb10-router-gemm] Fused bf16 x bf16 -> fp32 GEMM eligibility.\n"
    "        # torch.mm's out_dtype epilogue folds the fp32 cast into the GEMM,\n"
    "        # removing the standalone bf16->fp32 copy kernel that otherwise runs\n"
    "        # before grouped_topk. This is the plain cuBLAS (CUDA) / hipBLASLt\n"
    "        # (ROCm) out_dtype epilogue, so it applies on any CUDA-alike device\n"
    "        # (no bias, since torch.mm has no bias term). The specialized-kernel\n"
    "        # gate above excludes family-120 Blackwell (GB10 / DGX Spark), which\n"
    "        # this tier still covers -- see vllm-project/vllm#54048 (and #49921,\n"
    "        # which introduced the original ROCm carve-out this generalizes).\n"
    "        self._router_gemm_no_bias = not bias\n"
    "        self._router_gemm_cublas_capable = (\n"
    "            current_platform.is_cuda() or current_platform.is_rocm()\n"
    "        ) and self._router_gemm_no_bias\n"
    "        self.allow_cublas_router_gemm = (\n"
    "            self._router_gemm_cublas_capable\n"
    "            and self.weight.dtype == torch.bfloat16\n"
    "            and self.out_dtype == torch.float32\n"
    "        )\n"
    "        # [glm53-gb10-router-gemm] diagnostic: only fires when the WIDENED\n"
    "        # gate admits a device the old allow_specialized_router_gemm-only\n"
    "        # gate would have excluded (e.g. GB10 family-120) -- never on\n"
    "        # Hopper/SM100, where the specialized-kernel gate already covered\n"
    "        # this tier. Confirms the fix actually engages on this hardware.\n"
    "        if self.allow_cublas_router_gemm and not self.allow_specialized_router_gemm:\n"
    "            logger.info_once(\n"
    "                \"[glm53-gb10-router-gemm] cuBLAS bf16xbf16->fp32 router GEMM \"\n"
    "                \"enabled on a non-specialized-kernel CUDA/ROCm device (e.g. \"\n"
    "                \"GB10 family-120) via vllm-project/vllm#54048's arch-agnostic \"\n"
    "                \"gate.\"\n"
    "            )\n"
)

MARK_SET_OUT_DTYPE = "        # [glm53-gb10-router-gemm] mirrors the __init__ gate above -- see\n"

ANCHOR_SET_OUT_DTYPE = (
    "        if (\n"
    "            not self.allow_cublas_router_gemm\n"
    "            and (\n"
    "                self.allow_specialized_router_gemm\n"
    "                or (current_platform.is_rocm() and self._router_gemm_no_bias)\n"
    "            )\n"
    "            and out_dtype == torch.float32\n"
    "        ):\n"
    "            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16\n"
)

PATCHED_SET_OUT_DTYPE = (
    "        # [glm53-gb10-router-gemm] mirrors the __init__ gate above -- see\n"
    "        # vllm-project/vllm#54048. out_dtype is frequently set AFTER\n"
    "        # construction (once the expert quantization method is known --\n"
    "        # true for this project's exl3 path), so this recomputation path\n"
    "        # matters just as much as the __init__ one.\n"
    "        if (\n"
    "            not self.allow_cublas_router_gemm\n"
    "            and self._router_gemm_cublas_capable\n"
    "            and out_dtype == torch.float32\n"
    "        ):\n"
    "            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16\n"
)

SITES = (
    (MARK_INIT, ANCHOR_INIT, PATCHED_INIT, "__init__ cuBLAS eligibility gate"),
    (MARK_SET_OUT_DTYPE, ANCHOR_SET_OUT_DTYPE, PATCHED_SET_OUT_DTYPE, "set_out_dtype recomputation gate"),
)


def verified_state(text: str) -> bool:
    for mark, _anchor, patched, _name in SITES:
        if text.count(mark) != 1 or text.count(patched) != 1:
            return False
    return True


def prepare(source: str) -> tuple[str, str]:
    any_mark_present = any(source.count(mark) for mark, _a, _p, _n in SITES)
    if any_mark_present:
        if not verified_state(source):
            raise ValueError(
                "partial/inconsistent gb10-router-gemm patch -- refusing to "
                "touch a half-patched file"
            )
        return source, "already present"

    out = source
    for mark, anchor, patched, name in SITES:
        n = out.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned gb10-router-gemm anchor '{name}' drifted (found {n}, "
                "expected 1) -- re-derive the patch (gate_linear.py may have "
                "changed in this vLLM build)"
            )
        out = out.replace(anchor, patched, 1)
    if not verified_state(out):
        raise ValueError("gb10-router-gemm post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-gb10-router-gemm.tmp")
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
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"gb10-router-gemm preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if preflight_only:
        print(f"{TARGET.name}: gb10-router-gemm preflight OK ({action})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: gb10-router-gemm {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
