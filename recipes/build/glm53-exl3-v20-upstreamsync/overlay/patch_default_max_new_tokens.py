#!/usr/bin/env python3
"""Omitted-only output-token default (decode hygiene), without changing any
independent server cap.

Backport of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks PR #51 (2026-09-15
upstream-check round). Without a server-side default, a client request that
omits `max_tokens` gets vLLM's own fallback: `max_model_len - prompt_len`,
up to this fleet's full 262144-token max_model_len on a short prompt --
still large enough that one such client can decode until it preempts every
other session via KV pressure -- the same shape of problem this fork's own
`patch_scheduler_decode_floor.py` and `GLM53_MIXED_PREFILL_CHUNK` already
fight from the scheduler side, this time at the request-admission boundary
instead.

`vllm.entrypoints.serve.utils.api_utils.get_max_tokens()` keeps the model,
platform, and any explicit-override cap in its `min()` -- this overlay only
changes its *omitted-request* fallback value, read from
`DEFAULT_MAX_NEW_TOKENS` (unset/empty = stock behavior, unchanged). An
explicit client `max_tokens`/`max_completion_tokens` always wins; the two
companion hunks (completion serving's call site, and the completion
protocol's before-validator) exist solely to keep "omitted" and "explicitly
sent as null/16" distinguishable through Pydantic's field-set tracking, so
this default never masquerades as something the client actually asked for.

All three anchors verified byte-for-byte against this image's own installed
vLLM (`g487ecf187`) before being written here -- zero reconciliation needed,
unlike several other backports in this project's history. Idempotent (marker
comment gates re-application), validates every target compiles before
writing any of the three files, and validates `DEFAULT_MAX_NEW_TOKENS`
itself (must be empty or a decimal integer in 1..1000000) before touching
anything.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
    )
)
LIMITS_PATH = Path("entrypoints/serve/utils/api_utils.py")
COMPLETION_PATH = Path("entrypoints/openai/completion/serving.py")
PROTOCOL_PATH = Path("entrypoints/openai/completion/protocol.py")
MARK = "# [glm53-default-max-new-tokens]"

OLD = """    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
"""
NEW = OLD.replace(
    "    return min(\n",
    '''    # [glm53-default-max-new-tokens] Preserve explicit requests and all caps.
    if max_tokens is None:
        configured_default = os.environ.get("DEFAULT_MAX_NEW_TOKENS", "")
        if configured_default:
            if not (configured_default.isascii() and configured_default.isdecimal()):
                raise ValueError("DEFAULT_MAX_NEW_TOKENS must be a positive decimal integer")
            fallback_max_tokens = int(configured_default)
            if not 1 <= fallback_max_tokens <= 1000000:
                raise ValueError("DEFAULT_MAX_NEW_TOKENS must be in 1..1000000")

    return min(
''',
)
COMPLETION_OLD = """            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens,
                self._extract_prompt_len(engine_input),
"""
COMPLETION_NEW = """            # [glm53-default-max-new-tokens] The protocol default 16 is not an explicit request.
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens
                if "max_tokens" in request.model_fields_set
                or not os.environ.get("DEFAULT_MAX_NEW_TOKENS")
                else None,
                self._extract_prompt_len(engine_input),
"""
PROTOCOL_OLD = '''        if isinstance(data, dict) and data.get("max_tokens") is None:
'''
PROTOCOL_NEW = '''        if (
            isinstance(data, dict)
            and "max_tokens" in data
            and data["max_tokens"] is None
        ):  # [glm53-default-max-new-tokens] Do not manufacture an explicit field.
'''


def _replace_region(src: str, old: str, new: str) -> tuple[str, str]:
    if MARK in src:
        if src.count(MARK) != 1 or src.count(new) != 1:
            return src, "drifted:patched-region"
        compile(src, "token-default-target.py", "exec")
        return src, "skipped"
    if src.count(old) != 1:
        return src, "missing:unique-target"
    updated = src.replace(old, new, 1)
    compile(updated, "token-default-target.py", "exec")
    return updated, "applied"


def apply_text(src: str) -> tuple[str, str]:
    return _replace_region(src, OLD, NEW)


def apply_completion_text(src: str) -> tuple[str, str]:
    updated, status = _replace_region(src, COMPLETION_OLD, COMPLETION_NEW)
    if status not in ("applied", "skipped"):
        return src, status
    has_os = any(
        alias.name == "os" and alias.asname in (None, "os")
        for node in ast.parse(updated).body
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    if not has_os:
        if status == "skipped" or updated.count("import io\n") != 1:
            return src, "drifted:os-import"
        updated = updated.replace("import io\n", "import io\nimport os\n", 1)
    compile(updated, "completion-serving.py", "exec")
    return updated, status


def apply_protocol_text(src: str) -> tuple[str, str]:
    return _replace_region(src, PROTOCOL_OLD, PROTOCOL_NEW)


def main(argv: list[str]) -> int:
    status_only = len(argv) > 1 and argv[1] == "--status"
    root_arg = 2 if status_only else 1
    root = Path(argv[root_arg]) if len(argv) > root_arg else ROOT
    raw = os.environ.get("DEFAULT_MAX_NEW_TOKENS", "")
    if raw and not (raw.isascii() and raw.isdecimal() and 1 <= int(raw) <= 1000000):
        print(
            "DEFAULT_MAX_NEW_TOKENS must be empty or an integer in 1..1000000",
            file=sys.stderr,
        )
        return 2
    changes = []
    for relative, transform in (
        (LIMITS_PATH, apply_text),
        (COMPLETION_PATH, apply_completion_text),
        (PROTOCOL_PATH, apply_protocol_text),
    ):
        target = root / relative
        if not target.is_file():
            print(f"[glm53-default-max-new-tokens] missing {target}", file=sys.stderr)
            return 1
        original = target.read_text(encoding="utf-8")
        updated, status = transform(original)
        if status not in ("applied", "skipped"):
            print(f"[glm53-default-max-new-tokens] {status}: {target}", file=sys.stderr)
            return 1
        changes.append((target, updated, status))
    if status_only:
        complete = all(status == "skipped" for _, _, status in changes)
        print("default-max-new-tokens:", "APPLIED" if complete else "NOT APPLIED")
        return 0
    for target, updated, status in changes:
        if status == "applied":
            target.write_text(updated, encoding="utf-8")
        print(f"[glm53-default-max-new-tokens] {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
