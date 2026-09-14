#!/usr/bin/env python3
"""Broaden vLLM's spec-decode CUDA-graph capture-size alignment to apply
regardless of decode cudagraph mode, not just FULL.

Background: with speculative decoding on (this fleet runs MTP-2,
num_speculative_tokens=2, uniform_decode_query_len=3), every uniform decode
step schedules 1+num_speculative_tokens query positions per request. vLLM's
own dispatcher (vllm/v1/cudagraph_dispatcher.py, _create_padded_batch_descriptor)
asserts that a padded uniform-decode batch size divides evenly by
uniform_decode_query_len:

    if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0

That assumption only holds if cudagraph_capture_sizes was rounded to
multiples of uniform_decode_query_len first. vllm/config/compilation.py's
resolve_cudagraph_mode_and_sizes does exactly that rounding
(adjust_cudagraph_sizes_for_spec_decode) -- but only calls it when
cudagraph_mode.decode_mode() == CUDAGraphMode.FULL. If backend support or a
future config change ever leaves decode mode at PIECEWISE (or FULL is
requested but downgraded), the alignment silently never runs, and any
uniform-decode dispatch/replay against an unaligned capture size falls into
the same class of bug as this project's own recurring, never-fully-root-
caused CUDA-graph illegal-memory-access crashes -- though direct evidence
tying THIS fleet's specific occurrences to this exact gap is still
inconclusive (the 2026-09-12 23:41 UTC incident, for example, was on an
eager (non-cudagraph) 5832-token batch, well above
max_cudagraph_capture_size, so this patch would not have prevented that
one). Applying the alignment unconditionally (whenever spec decode is on
and any cudagraph mode is active) is a no-op when decode mode is already
FULL (this fleet's normal, confirmed-at-boot case) and a real hardening for
every other case.

Mined from AEON-7/vllm-ultimate-dgx-spark's patches/patch_cudagraph_align.py
(MIT-licensed, independent GB10 vLLM build -- see recipes/VALIDATION.md,
"AEON-7 initial triage"). Their anchor text doesn't match byte-for-byte
against this image's newer vLLM commit (g487ecf187 adds a
`not use_v2_model_runner` clause AEON-7's version predates), so this is a
from-scratch port against our own installed source, not their literal diff.
See: vLLM issues #28015, #28207, #29091; upstream fixes #29102, #23679.

Applied unconditionally (no env-var gate): the broadened condition is a
strict superset of the original, so it changes nothing when decode mode
already resolves to FULL and only helps in cases the original code never
covered.

Idempotent -- safe to run multiple times.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TARGET = Path(
    os.environ.get(
        "GLM53_COMPILATION_CONFIG_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py",
    )
)

MARKER = "# cudagraph_align_spec_decode_all_modes"

OLD = (
    "        if (\n"
    "            not use_v2_model_runner\n"
    "            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)
NEW = (
    "        # cudagraph_align_spec_decode_all_modes\n"
    "        # Original: gated to cudagraph_mode.decode_mode()==FULL only (a vLLM\n"
    "        # gap -- PIECEWISE/other decode modes silently skip alignment, which\n"
    "        # can leave capture sizes not a multiple of (1 + num_speculative_\n"
    "        # tokens) and trip cudagraph_dispatcher.py's uniform-decode assert\n"
    "        # or, if assertions are stripped, dispatch/replay a mismatched graph.\n"
    "        # Apply for any non-NONE mode instead of FULL only.\n"
    "        if (\n"
    "            not use_v2_model_runner\n"
    "            and cudagraph_mode != CUDAGraphMode.NONE\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)


def main() -> int:
    if not TARGET.is_file():
        print(f"[patch_cudagraph_align] {TARGET} not found -- skipping")
        return 0

    src = TARGET.read_text()

    if MARKER in src:
        print(f"[{TARGET.name}] already applied")
        return 0

    if OLD not in src:
        print(
            f"[{TARGET.name}] anchor not found -- likely upstream merged an "
            "equivalent fix or shifted this code; skipping (idempotent)"
        )
        return 0

    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(
        f"[{TARGET.name}] applied spec-decode capture-size alignment for all "
        "cudagraph modes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
