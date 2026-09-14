#!/usr/bin/env python3
"""Apply overlay/patch_cudagraph_align.py to a copy of vllm/config/compilation.py."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_cudagraph_align.py",
        HERE.parent / "overlay" / "patch_cudagraph_align.py",
    )
    if p.is_file()
)
SRC = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py")

OLD_ANCHOR = (
    "        if (\n"
    "            not use_v2_model_runner\n"
    "            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
)


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_COMPILATION_CONFIG_PY_SRC", SRC))
    if not src.is_file():
        # Host unit test: fabricate a minimal stand-in with just the anchor,
        # since we can't assume vllm is installed on the build/test host.
        with tempfile.TemporaryDirectory() as tmp:
            fabricated = Path(tmp) / "compilation.py"
            fabricated.write_text(
                "class C:\n"
                "    def resolve_cudagraph_mode_and_sizes(self):\n"
                + OLD_ANCHOR
                + "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
                "                uniform_decode_query_len,\n"
                "                tensor_parallel_size,\n"
                "            )\n"
            )
            return _run_against(fabricated)
    return _run_against(src)


def _run_against(src: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "compilation.py"
        shutil.copyfile(src, dst)
        before = dst.read_text()
        already_applied = "# cudagraph_align_spec_decode_all_modes" in before
        if not already_applied:
            assert OLD_ANCHOR in before, (
                "anchor not present in source under test -- patch would "
                "no-op; this means the fixture or installed vllm no longer "
                "matches what patch_cudagraph_align.py expects, update "
                "both together"
            )
        env = os.environ.copy()
        env["GLM53_COMPILATION_CONFIG_PY"] = str(dst)
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        after = dst.read_text()
        assert "# cudagraph_align_spec_decode_all_modes" in after
        assert OLD_ANCHOR not in after, "original FULL-only anchor should be gone"
        assert (
            "        if (\n"
            "            not use_v2_model_runner\n"
            "            and cudagraph_mode != CUDAGraphMode.NONE\n"
            "            and uniform_decode_query_len > 1\n"
            "        ):\n"
        ) in after
        # idempotent
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text() == after
    print("cudagraph_align patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
