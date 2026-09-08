#!/usr/bin/env python3
"""Install the additive EXL3 fat-GEMM source and pybind entries.

exl3_fat_gemm.cu/.cuh here are Enntity/sparkglm's grouped-prefill fork
(GPU-resident task planner + 4 phase-wide kernels replacing the per-expert
host-driven loop), grafted onto this project's own known-working build tree
-- not this project's original per-expert-only kernel. Scope is deliberately
limited to grouped-prefill (EXL3_GROUPED_PREFILL_K4): sparkglm's separate
decode-side cooperative kernel (exl3_decode_moe.cu/.cuh, EXL3_DECODE_COOP_K4)
is excluded -- their own docs flag it as weaker evidence, and it is cleanly
separable (a wholly different file, no shared bindings).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one binding anchor: {old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_exl3_fat_kernel.py EXLLAMAV3_EXT SOURCE_DIR")
    ext_root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    quant = ext_root / "quant"
    bindings = ext_root / "bindings.cpp"
    if not quant.is_dir() or not bindings.is_file():
        raise RuntimeError(f"invalid extension root: {ext_root}")

    for name in ("exl3_fat_gemm.cu", "exl3_fat_gemm.cuh"):
        source = source_dir / name
        if not source.is_file():
            raise RuntimeError(f"missing additive source: {source}")
        shutil.copyfile(source, quant / name)

    text = bindings.read_text()
    text = replace_once(
        text,
        '#include "quant/exl3_moe.cuh"',
        '#include "quant/exl3_moe.cuh"\n#include "quant/exl3_fat_gemm.cuh"',
    )
    text = replace_once(
        text,
        '    m.def("exl3_moe", &exl3_moe, "exl3_moe");',
        '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
        '    m.def("exl3_fat_gemm", &exl3_fat_gemm, "exl3_fat_gemm");\n'
        '    m.def("exl3_fat_gemm_m64", &exl3_fat_gemm_m64, "exl3_fat_gemm_m64");\n'
        '    m.def("exl3_fat_gemm_pair", &exl3_fat_gemm_pair, "exl3_fat_gemm_pair");\n'
        '    m.def("exl3_fat_gemm_pair_m64", &exl3_fat_gemm_pair_m64, "exl3_fat_gemm_pair_m64");\n'
        '    m.def("exl3_fat_swiglu_had", &exl3_fat_swiglu_had, "exl3_fat_swiglu_had");\n'
        '    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");\n'
        '    m.def("exl3_fat_gemm_scatter_m64", &exl3_fat_gemm_scatter_m64, "exl3_fat_gemm_scatter_m64");\n'
        '    m.def("exl3_grouped_prefill_k4", &exl3_grouped_prefill_k4, "exl3_grouped_prefill_k4");',
    )
    bindings.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
