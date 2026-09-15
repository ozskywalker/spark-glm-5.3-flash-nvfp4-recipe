#!/usr/bin/env python3
"""Apply overlay/patch_default_max_new_tokens.py to copies of vLLM's
api_utils.py, completion serving.py, and completion protocol.py."""
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
        HERE / "patch_default_max_new_tokens.py",
        HERE.parent / "overlay" / "patch_default_max_new_tokens.py",
    )
    if p.is_file()
)

RELATIVE_TARGETS = (
    Path("entrypoints/serve/utils/api_utils.py"),
    Path("entrypoints/openai/completion/serving.py"),
    Path("entrypoints/openai/completion/protocol.py"),
)
DEFAULT_ROOTS = {
    RELATIVE_TARGETS[0]: Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py"
    ),
    RELATIVE_TARGETS[1]: Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py"
    ),
    RELATIVE_TARGETS[2]: Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py"
    ),
}


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for relative in RELATIVE_TARGETS:
            src = Path(
                os.environ.get(
                    f"GLM53_{relative.stem.upper()}_PY_SRC", DEFAULT_ROOTS[relative]
                )
            )
            if not src.is_file():
                raise SystemExit(
                    f"missing {src} -- run inside a container with vllm installed, "
                    f"or set GLM53_{relative.stem.upper()}_PY_SRC"
                )
            dst = root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        env = os.environ.copy()
        env["DEFAULT_MAX_NEW_TOKENS"] = "65536"

        subprocess.check_call([sys.executable, str(PATCH), str(root)], env=env)
        for relative in RELATIVE_TARGETS:
            text = (root / relative).read_text()
            assert "[glm53-default-max-new-tokens]" in text, relative

        # idempotent
        subprocess.check_call([sys.executable, str(PATCH), str(root)], env=env)

        # --status reports fully applied
        status = subprocess.run(
            [sys.executable, str(PATCH), "--status", str(root)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "APPLIED" in status.stdout and "NOT APPLIED" not in status.stdout

        # invalid env value fails closed, without touching any file
        before = {rel: (root / rel).read_text() for rel in RELATIVE_TARGETS}
        bad_env = env.copy()
        bad_env["DEFAULT_MAX_NEW_TOKENS"] = "not-a-number"
        result = subprocess.run(
            [sys.executable, str(PATCH), str(root)],
            env=bad_env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result.stderr
        for rel in RELATIVE_TARGETS:
            assert (root / rel).read_text() == before[rel], rel

    print("default-max-new-tokens patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
