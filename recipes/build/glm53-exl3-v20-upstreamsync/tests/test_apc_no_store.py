#!/usr/bin/env python3
"""Apply overlay/patch_apc_no_store.py to copies of vLLM's sampling_params.py,
v1/request.py and v1/core/block_pool.py.

Same style as tests/test_default_max_new_tokens.py (the other 3-file source
patch in this project): copy the real (or installed) sources into a tmpdir,
run the patch via env-var overrides, and assert marker presence, idempotency
and fail-closed behavior on drift/partial state.
"""
from __future__ import annotations

import ast
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
        HERE / "patch_apc_no_store.py",
        HERE.parent / "overlay" / "patch_apc_no_store.py",
    )
    if p.is_file()
)

FILES = {
    "sampling_params.py": (
        "GLM53_SAMPLING_PARAMS_PY",
        Path("/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py"),
    ),
    "request.py": (
        "GLM53_REQUEST_PY",
        Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/request.py"),
    ),
    "block_pool.py": (
        "GLM53_BLOCK_POOL_PY",
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py"
        ),
    ),
}
MARK = "# [glm53-apc-no-store]"


def _copy_sources(root: Path) -> dict[str, Path]:
    dests = {}
    for name, (_env, default_src) in FILES.items():
        src = Path(os.environ.get(f"GLM53_APC_{name.upper().replace('.', '_')}_SRC", default_src))
        if not src.is_file():
            raise SystemExit(
                f"missing {src} -- run inside a container with vllm installed"
            )
        dst = root / name
        shutil.copyfile(src, dst)
        dests[name] = dst
    return dests


def _run_patch(dests: dict[str, Path], *, ok: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for name, (envname, _default) in FILES.items():
        env[envname] = str(dests[name])
    proc = subprocess.run(
        [sys.executable, str(PATCH)], env=env, text=True, capture_output=True
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dests = _copy_sources(root)

        _run_patch(dests)
        for name, dst in dests.items():
            text = dst.read_text()
            assert MARK in text, name
            ast.parse(text, name)
        assert "skip_writing_prefix_cache: bool | None = None" in dests["sampling_params.py"].read_text()
        assert "_glm53_resolve_no_store" in dests["sampling_params.py"].read_text()
        assert "get_skip_writing_prefix_cache" in dests["request.py"].read_text()
        assert "_glm53_log_nostore" in dests["block_pool.py"].read_text()

        # idempotent
        before = {name: dst.read_text() for name, dst in dests.items()}
        _run_patch(dests)
        for name, dst in dests.items():
            assert dst.read_text() == before[name], name

    # fail-closed: drift in one anchor refuses that file without touching others
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dests = _copy_sources(root)
        pristine = {name: dst.read_text() for name, dst in dests.items()}
        drifted = pristine["sampling_params.py"].replace(
            "    skip_reading_prefix_cache: bool | None = None\n",
            "    skip_reading_prefix_cache: bool = False\n",
            1,
        )
        dests["sampling_params.py"].write_text(drifted)
        _run_patch(dests, ok=False)
        assert dests["sampling_params.py"].read_text() == drifted
        # other files untouched by the all-or-nothing preflight
        assert dests["request.py"].read_text() == pristine["request.py"]
        assert dests["block_pool.py"].read_text() == pristine["block_pool.py"]

    # fail-closed: partially-marked file (marker present, snippet missing) is refused
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dests = _copy_sources(root)
        _run_patch(dests)
        mutilated = dests["block_pool.py"].read_text().replace(
            "_glm53_log_nostore(request, \"full\")", "pass", 1
        )
        dests["block_pool.py"].write_text(mutilated)
        _run_patch(dests, ok=False)
        assert dests["block_pool.py"].read_text() == mutilated

    print("apc-no-store patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
