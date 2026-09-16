#!/usr/bin/env python3
"""Apply overlay/patch_cache_reset.py to a copy of vLLM's api_server.py.

Same style as tests/test_cudagraph_align.py: copy the real (or installed)
source into a tmpdir, run the patch against it via the env-var override, and
assert marker presence, idempotency, flag semantics and fail-closed drift
behavior.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_cache_reset.py",
        HERE.parent / "overlay" / "patch_cache_reset.py",
    )
    if p.is_file()
)
DEFAULT_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py"
)
MARK = "# [glm53-cache-reset]"

DEV_OLD = (
    "    if envs.VLLM_SERVER_DEV_MODE:\n"
    "        from vllm.entrypoints.serve import register_vllm_dev_api_routers\n"
    "\n"
    "        register_vllm_dev_api_routers(app)\n"
)

FIXTURE = "def build_app(app):\n" + DEV_OLD


def _run_patch(target: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GLM53_API_SERVER_PY"] = str(target)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        text=True,
        capture_output=True,
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def _exec_build_app(source: str, *, dev_mode: bool, flag: str | None) -> list[str]:
    """Exec the patched build_app against fake vLLM modules; return call log."""
    calls: list[str] = []
    serve = types.ModuleType("vllm.entrypoints.serve")
    serve.register_vllm_dev_api_routers = lambda app: calls.append("dev")
    api_router = types.ModuleType("vllm.entrypoints.serve.dev.cache.api_router")
    api_router.attach_router = lambda app: calls.append("cache")

    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "vllm",
            "vllm.entrypoints",
            "vllm.entrypoints.serve",
            "vllm.entrypoints.serve.dev",
            "vllm.entrypoints.serve.dev.cache",
            "vllm.entrypoints.serve.dev.cache.api_router",
        )
    }
    saved_env = os.environ.get("GLM53_EXPOSE_CACHE_RESET")
    try:
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.entrypoints"] = types.ModuleType("vllm.entrypoints")
        sys.modules["vllm.entrypoints.serve"] = serve
        sys.modules["vllm.entrypoints.serve.dev"] = types.ModuleType(
            "vllm.entrypoints.serve.dev"
        )
        sys.modules["vllm.entrypoints.serve.dev.cache"] = types.ModuleType(
            "vllm.entrypoints.serve.dev.cache"
        )
        sys.modules["vllm.entrypoints.serve.dev.cache.api_router"] = api_router
        if flag is None:
            os.environ.pop("GLM53_EXPOSE_CACHE_RESET", None)
        else:
            os.environ["GLM53_EXPOSE_CACHE_RESET"] = flag
        namespace: dict[str, object] = {
            "os": os,
            "envs": types.SimpleNamespace(VLLM_SERVER_DEV_MODE=dev_mode),
        }
        exec(compile(source, "patched_api_server_fixture.py", "exec"), namespace)
        namespace["build_app"](object())
    finally:
        for name, mod in saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        if saved_env is None:
            os.environ.pop("GLM53_EXPOSE_CACHE_RESET", None)
        else:
            os.environ["GLM53_EXPOSE_CACHE_RESET"] = saved_env
    return calls


def test_fixture_flag_semantics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"
        target.write_text(FIXTURE)
        _run_patch(target)
        patched = target.read_text()
        assert MARK in patched
        compile(patched, "patched_fixture.py", "exec")

        assert _exec_build_app(patched, dev_mode=False, flag="1") == ["cache"]
        assert _exec_build_app(patched, dev_mode=False, flag=None) == []
        assert _exec_build_app(patched, dev_mode=False, flag="0") == []
        # dev mode keeps precedence over the cache-reset flag
        assert _exec_build_app(patched, dev_mode=True, flag="1") == ["dev"]

        # idempotent
        _run_patch(target)
        assert target.read_text() == patched


def test_fail_closed_on_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"

        drifted = FIXTURE.replace(
            "register_vllm_dev_api_routers(app)",
            "register_vllm_dev_api_routers(app, strict=True)",
            1,
        )
        target.write_text(drifted)
        _run_patch(target, ok=False)
        assert target.read_text() == drifted

        partial = FIXTURE.replace(
            "    if envs.VLLM_SERVER_DEV_MODE:",
            f"    {MARK} stray marker\n    if envs.VLLM_SERVER_DEV_MODE:",
            1,
        )
        target.write_text(partial)
        _run_patch(target, ok=False)
        assert target.read_text() == partial

        duplicated = FIXTURE + FIXTURE
        target.write_text(duplicated)
        _run_patch(target, ok=False)
        assert target.read_text() == duplicated


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_API_SERVER_PY_SRC", DEFAULT_SRC))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "api_server.py"
        shutil.copyfile(src, target)
        before = target.read_text()
        assert DEV_OLD in before, (
            "anchor not present in the installed source under test -- patch "
            "would no-op; update this test's DEV_OLD together with the "
            "installed vLLM if it has genuinely changed"
        )
        _run_patch(target)
        patched = target.read_text()
        compile(patched, str(target), "exec")
        assert MARK in patched
        _run_patch(target)
        assert target.read_text() == patched


def main() -> int:
    test_fixture_flag_semantics()
    test_fail_closed_on_drift()
    test_installed_copy_if_present()
    print("cache-reset endpoint patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
