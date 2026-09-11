#!/usr/bin/env python3
"""LONG_PREFILL_TOKEN_THRESHOLD: validation, both-rank argv, caller-override.

Ported from upstream MiaAI-Lab#157 (d094362) -- opt-in vLLM
--long-prefill-token-threshold flag, default empty (stock scheduler).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from test_numeric_config import guard_source
from test_start_overrides import _run_preamble

ROOT = Path(__file__).resolve().parents[1]
KEY = "LONG_PREFILL_TOKEN_THRESHOLD"
FLAG = "--long-prefill-token-threshold"


def _run_build_argv(value: str | None) -> list[str]:
    source = (ROOT / "start.sh").read_text()
    begin = source.index("write_inner_scripts() {")
    end = source.index("\n}\n", begin) + 3
    env = {
        "PATH": os.environ["PATH"], "SERVED_MODEL_NAME": "test",
        "PORT": "8888", "TP": "2", "NNODES": "2", "HEAD_IP": "127.0.0.1",
        "MASTER_PORT": "29500", "SPEC_METHOD": "none",
    }
    if value is not None:
        env[KEY] = value
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        env.update(HEAD_SCRIPT=str(tmp / "head.sh"), WORKER_SCRIPT=str(tmp / "worker.sh"))
        subprocess.run(
            ["bash", "-c", source[begin:end] + "\nwrite_inner_scripts"],
            env=env, check=True, capture_output=True,
        )
        argvs = []
        for name in ("head.sh", "worker.sh"):
            script = (tmp / name).read_text().split('[ -f "${MODEL_DIR}/config.json" ]')[0]
            script += '\nprintf "%s\\0" "${ARGS[@]}"\n'
            result = subprocess.run(["bash", "-c", script], env=env,
                                     check=True, capture_output=True)
            argvs.append(result.stdout.decode().splitlines()[-1].split("\0")[:-1])
        return argvs


def test_both_rank_argv_unset() -> None:
    for argv in _run_build_argv(None):
        assert FLAG not in argv


def test_both_rank_argv_empty() -> None:
    for argv in _run_build_argv(""):
        assert FLAG not in argv


def test_both_rank_argv_set() -> None:
    for argv in _run_build_argv("3584"):
        assert argv.count(FLAG) == 1
        assert argv[argv.index(FLAG) + 1] == "3584"


def test_validation() -> None:
    cases = [("", 0), ("3584", 0), ("003584", 0),
             ("0", 2), ("-1", 2), ("1.5", 2), (" 3584", 2), ("7169", 2)]
    for value, expected in cases:
        script = (
            guard_source() + '\nGPU_MEM_UTIL=0.85; MAX_MODEL_LEN=850000; '
            'MAX_NUM_SEQS=4; MAX_NUM_BATCHED_TOKENS=7168; GLM53_SPINWAIT_MS=stock\n'
            'validate_numeric_config || exit $?\n'
            'printf "%s" "$LONG_PREFILL_TOKEN_THRESHOLD"\n'
        )
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                 env={"PATH": os.environ["PATH"], KEY: value})
        assert result.returncode == expected, (value, result.stderr)
        if expected == 0:
            assert result.stdout == ("3584" if value else "")


def test_caller_can_restore_stock_over_env() -> None:
    probe = f'\nprintf "V=[%s]\\n" "${{{KEY}-UNSET}}"\n'
    assert _run_preamble(f"{KEY}=3584\n", {KEY: ""}, probe) == "V=[]"
    assert _run_preamble(f"{KEY}=1024\n", {KEY: "3584"}, probe) == "V=[3584]"


if __name__ == "__main__":
    test_both_rank_argv_unset()
    test_both_rank_argv_empty()
    test_both_rank_argv_set()
    test_validation()
    test_caller_can_restore_stock_over_env()
    print("long-prefill-token-threshold OK")
