#!/usr/bin/env python3
"""Apply overlay/patch_kv_capacity_log.py to a copy of vLLM's kv_cache_utils.py.

Same style as tests/test_cudagraph_align.py: copy the real (or installed)
source into a tmpdir, run the patch against it via the env-var override, and
assert marker presence, idempotency, and fail-closed drift behavior. A
fabricated minimal fixture is used when vLLM is not installed on the host
running this test (e.g. this test host, vs. the Docker build where the real
file is always present)."""
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
        HERE / "patch_kv_capacity_log.py",
        HERE.parent / "overlay" / "patch_kv_capacity_log.py",
    )
    if p.is_file()
)
DEFAULT_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)

HELPER_ANCHOR = (
    "\ndef get_kv_cache_capacity(\n"
    "    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig\n"
    ") -> tuple[int, float]:\n"
)
CALL_OLD = '''def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Store and log the resolved KV cache capacity."""
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
    max_model_len = vllm_config.model_config.max_model_len
    logger.info_once(
        "GPU KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )
'''
MARK = "# [glm53-kv-capacity-log]"


def _fabricate() -> str:
    """A minimal stand-in carrying both real anchors, for a host without vLLM."""
    return (
        "import os\n"
        "from vllm.config import VllmConfig\n"
        "from vllm.v1.kv_cache_interface import KVCacheConfig\n"
        "from vllm.utils.math_utils import cdiv\n"
        "from vllm.logger import init_logger\n"
        "\n"
        "logger = init_logger(__name__)\n"
        "\n\n"
        "def get_kv_cache_capacity(\n"
        "    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig\n"
        ") -> tuple[int, float]:\n"
        "    return 0, 0.0\n"
        "\n\n" + CALL_OLD
    )


def _run_patch(target: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GLM53_KV_CACHE_UTILS_PY"] = str(target)
    proc = subprocess.run(
        [sys.executable, str(PATCH)], env=env, text=True, capture_output=True
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv_cache_utils.py"
        target.write_text(_fabricate())
        assert HELPER_ANCHOR in target.read_text()
        assert CALL_OLD in target.read_text()

        _run_patch(target)
        patched = target.read_text()
        assert MARK in patched
        assert "_glm53_kv_capacity_log_enabled" in patched
        assert "_glm53_log_kv_capacity_breakdown" in patched
        assert HELPER_ANCHOR in patched, "anchor must remain, helper sits before it"
        compile(patched, str(target), "exec")

        # idempotent
        _run_patch(target)
        assert target.read_text() == patched


def test_fail_closed_on_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv_cache_utils.py"

        drifted = _fabricate().replace(
            '"GPU KV cache size: %s tokens, "', '"GPU KV cache size: %s tok, "', 1
        )
        target.write_text(drifted)
        _run_patch(target, ok=False)
        assert target.read_text() == drifted

        partial = _fabricate().replace(
            HELPER_ANCHOR, f"\n{MARK} stray marker\n" + HELPER_ANCHOR.lstrip("\n"), 1
        )
        target.write_text(partial)
        _run_patch(target, ok=False)
        assert target.read_text() == partial

        duplicated = _fabricate() + "\n" + CALL_OLD
        target.write_text(duplicated)
        _run_patch(target, ok=False)
        assert target.read_text() == duplicated


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_KV_CACHE_UTILS_PY_SRC", DEFAULT_SRC))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv_cache_utils.py"
        shutil.copyfile(src, target)
        before = target.read_text()
        assert HELPER_ANCHOR in before, "helper anchor drifted from installed vLLM"
        assert CALL_OLD in before, "update_kv_cache_capacity body drifted from installed vLLM"
        _run_patch(target)
        patched = target.read_text()
        compile(patched, str(target), "exec")
        assert MARK in patched
        _run_patch(target)
        assert target.read_text() == patched


def main() -> int:
    test_fixture()
    test_fail_closed_on_drift()
    test_installed_copy_if_present()
    print("kv-capacity-log patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
