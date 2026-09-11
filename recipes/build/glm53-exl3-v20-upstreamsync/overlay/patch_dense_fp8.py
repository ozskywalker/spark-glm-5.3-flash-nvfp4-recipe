#!/usr/bin/env python3
"""Ported from MiaAI-Lab's 2026-09-08 upstream update (commit eb0da1d), adapted
to this fork: the exl3.py graft (Glm53DenseFp8Method, _glm53_dense_fp8_group,
the get_quant_method dispatch change) already lives in this build's
overlay/exl3.py and is installed by the existing `COPY overlay/exl3.py ...`
Dockerfile line -- no separate install step needed here, unlike upstream's
version which stages exl3.py through /opt/glm53 first. This script does the
second half: lets the KDA and MLA constructors keep the quant config so their
projections reach Exl3Config.get_quant_method (idempotent, fail closed,
marker comment).

Applied UNCONDITIONALLY at build time, matching this fork's other
runtime-gated patches (patch_flashkda.py): the actual FP8-vs-BF16 decision
happens at runtime inside exl3.py's _glm53_dense_fp8_group(), which reads
GLM53_DENSE_FP8 fresh on every call. With the knob off (default, unset),
_glm53_dense_fp8_group() returns None for every prefix, so KDA/MLA layers
that now reach Exl3Config.get_quant_method just fall through to
UnquantizedLinearMethod() -- functionally identical to the un-patched
behavior. Upstream's own script instead gated this file mutation on
GLM53_DENSE_FP8 at the point it runs, which only works because upstream
applies it at container start (fresh env each launch); this Dockerfile
applies all patches at image build time, when a runtime knob isn't set yet,
so gating the mutation itself would have made the knob permanently inert.

PROVISIONAL, per upstream's own commit message: "changes target numerics;
needs a KLD panel before it can become a default." See recipes/VALIDATION.md
for the analysis and recipes/build/glm53-exl3-v16-densefp8/NOTES.md for this
build's provenance and validation status.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
MARK = "# [glm53-dense-fp8]"

KDA_OLD = """        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
KDA_NEW = """        saved_quant_config = vllm_config.quant_config
        if getattr(saved_quant_config, "get_name", lambda: "")() != "exl3":  # [glm53-dense-fp8]
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
MLA_OLD = """                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
"""
MLA_NEW = """                quant_config=(quant_config if getattr(quant_config, "get_name", lambda: "")() == "exl3" else None),  # [glm53-dense-fp8]
                prefix=f"{prefix}.self_attn",
"""


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: {MARK} already present — skipping")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path.name} ({label})")


def main() -> int:
    kda = SITE / "models/glm5next/nvidia/kda.py"
    if not kda.is_file():
        kda = SITE / "model_executor/models/glm5next/nvidia/kda.py"
    model = kda.parent / "model.py"
    replace_once(kda, KDA_OLD, KDA_NEW, "kda quant_config")
    replace_once(model, MLA_OLD, MLA_NEW, "mla quant_config")
    print("[glm53-dense-fp8] constructors patched; actual quantization gated at runtime by GLM53_DENSE_FP8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
