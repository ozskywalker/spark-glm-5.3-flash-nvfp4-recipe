#!/usr/bin/env python3
"""BUILD_MIN_MEM_GIB: refuse a native docker build under unified-memory
pressure. Ported from Enntity/sparkglm (a8aaa229) -- an original build-host
guard derived from an observed GB10 failure mode (native EXL3 compilation
competing with a resident model server for the shared host/GPU memory pool).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def function(name: str) -> str:
    text = START.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text)
    assert match, f"missing function {name}"
    return match.group(0)


def _run(mem_available_kib: int, minimum_gib: int) -> subprocess.CompletedProcess[str]:
    functions = function("build_mem_available_kib") + "\n" + function("assert_build_headroom")
    with tempfile.TemporaryDirectory() as tmp:
        meminfo = Path(tmp) / "meminfo"
        meminfo.write_text(f"MemTotal: 131072000 kB\nMemAvailable: {mem_available_kib} kB\n")
        script = (
            "warn() { printf 'WARN: %s\\n' \"$*\" >&2; }\n"
            "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
            "docker() { :; }\n"
            f"SPARKGLM_MEMINFO_PATH={str(meminfo)!r}\n"
            f"BUILD_MIN_MEM_GIB={minimum_gib}\n"
            + functions
            + "\nassert_build_headroom\n"
        )
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_enough_headroom_passes() -> None:
    result = _run(40 * 1024 * 1024, 32)
    assert result.returncode == 0, result.stderr


def test_low_headroom_refuses() -> None:
    result = _run(8 * 1024 * 1024, 32)
    assert result.returncode != 0
    assert "refusing native image build" in result.stderr


def test_explicit_zero_overrides() -> None:
    result = _run(1, 0)
    assert result.returncode == 0, result.stderr


def test_wired_into_build_image() -> None:
    src = START.read_text()
    build = src[src.index("build_image() {"):]
    assert "assert_build_headroom" in build.split("\n\n", 1)[0] or "assert_build_headroom" in build[:200]
    assert 'BUILD_MIN_MEM_GIB="${BUILD_MIN_MEM_GIB:-32}"' in src


if __name__ == "__main__":
    test_enough_headroom_passes()
    test_low_headroom_refuses()
    test_explicit_zero_overrides()
    test_wired_into_build_image()
    print("build-headroom guard OK")
