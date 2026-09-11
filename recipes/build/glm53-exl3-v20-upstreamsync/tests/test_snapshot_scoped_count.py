#!/usr/bin/env python3
"""Regression guard: weight completeness is scoped to the active snapshot.

A repo-wide count can accept 119 stale shards in snapshots/old plus one shard
in the active snapshot as "120 present", then resolve the incomplete active
snapshot and fail deep in vLLM load instead of failing fast with an
actionable message. Ported from upstream MiaAI-Lab#153 (c195e92); this
project's own DFlash2 shard-count call sites had the identical bug
(different code, not a call to count_shards()) and are covered here too.
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


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def _make_repo(tmp: Path, active_file: str, stale_count: int) -> Path:
    repo = tmp / "repo"
    active = repo / "snapshots" / "active"
    old = repo / "snapshots" / "old"
    active.mkdir(parents=True)
    old.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("active", encoding="utf-8")
    if active_file:
        (active / active_file).touch()
    for index in range(stale_count):
        (old / f"model-{index:05d}-of-00120.safetensors").touch()
    return repo


def test_target_completeness_is_scoped_to_active_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp), "model-00120-of-00120.safetensors", 119)
        result = run_bash(
            "set -euo pipefail\n" + function("count_shards")
            + f"count_shards {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1", result.stdout


def test_empty_repo_counts_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        result = run_bash(
            "set -euo pipefail\n" + function("count_shards")
            + f"count_shards {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0", result.stdout


def test_dflash_completeness_is_scoped_to_active_snapshot() -> None:
    """Same bug class, this project's own inline DFlash2 shard-count sites
    (count_dflash_shard(), used by download_dflash())."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp), "model.safetensors", 1)
        result = run_bash(
            "set -euo pipefail\n" + function("count_dflash_shard")
            + f"count_dflash_shard {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1", result.stdout

    # Active snapshot has no model.safetensors; only the stale one does --
    # must NOT count it as present (this is the actual bug this guards).
    with tempfile.TemporaryDirectory() as tmp:
        active_dir = Path(tmp) / "repo" / "snapshots" / "active"
        repo = _make_repo(Path(tmp), None, 0)
        (Path(tmp) / "repo" / "snapshots" / "old" / "model.safetensors").touch()
        result = run_bash(
            "set -euo pipefail\n" + function("count_dflash_shard")
            + f"count_dflash_shard {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0", result.stdout


if __name__ == "__main__":
    test_target_completeness_is_scoped_to_active_snapshot()
    test_empty_repo_counts_zero()
    test_dflash_completeness_is_scoped_to_active_snapshot()
    print("snapshot-scoped count guard OK")
