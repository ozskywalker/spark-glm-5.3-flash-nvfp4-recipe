#!/usr/bin/env python3
"""Honest KV-capacity boot log for the hybrid model (log-only overlay).

Backport of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks PR #94 (2026-09-16
upstream-check round), scoped down to the directly-derivable part -- see
"Scoping" below.

The problem
-----------
``update_kv_cache_capacity`` (``v1/core/kv_cache_utils.py``) logs, once per
boot::

    GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x

The first number is ``int(max_concurrency * max_model_len)`` where
``max_concurrency = num_blocks / num_blocks_per_request`` and
``num_blocks_per_request`` is a SUM OVER KV-CACHE GROUPS of
``cdiv(spec.max_memory_usage_bytes(cfg), spec.page_size_bytes)``
(``get_max_concurrency_for_kv_cache_config``). For a single-group model that
is the pool's token capacity up to rounding, which is what the label
suggests. For this hybrid model (MLA + kpool tail + several mamba groups +
DFlash2 drafter SWA, one shared BlockPool with globally unique block ids) it
is a concurrency figure in token units: neither ``num_blocks`` nor
``num_blocks_per_request`` is logged, and only their ratio can be recovered
from the stock line. On boots observed upstream, the line read out to
roughly 1.5M tokens while the actual pool was under a thousand block ids --
readable as "a 1.5M-token prefix cache" by more than one reviewer before the
arithmetic was redone by hand. Upstream feature request:
vllm-project/vllm#54662.

What this overlay does
-----------------------
Immediately after the existing line (kept byte-identical) it logs, from the
SAME config objects the existing line is computed from:

* one line per KV-cache group: index, spec type (``UniformTypeKVCacheSpecs``
  unwrapped to its first member type for readability), block_size, page
  size, layer count, blocks per ``max_model_len`` request (the same
  ``cdiv`` expression as ``get_max_concurrency_for_kv_cache_config``, so the
  per-group column and the stock denominator cannot drift apart), and
  whether the group takes part in prefix caching;
* one summary line: usable block ids (``num_blocks - 1``, since
  ``BlockPool`` permanently holds back block id 0 as the null block) and the
  blocks-per-request total (which reproduces the stock ratio).

Log-only. No control flow, no allocation, no config mutation, runs once in
the engine core at boot. Knob ``GLM53_KV_CAPACITY_LOG``: unset or ``1`` =
log (default), ``0`` = one line saying it is disabled. An unrecognized value
logs one warning and defaults to enabled rather than raising -- this is a
diagnostic feature and a typo must not be able to affect boot. Any failure
inside the derivation itself is caught, logged once, and boot proceeds --
this feature must never take the server down.

Scoping (why this is a subset of upstream PR #94)
--------------------------------------------------
The upstream PR also derives an estimated "cached-conversation capacity in
tokens" by costing, per exact KV-cache-spec class, how many block ids one
aligned cached segment consumes under dense retention (including an EAGLE
detection that mirrors the coordinator's own fallback logic). That part is
tightly coupled to this fork's specific retention/EAGLE machinery and is the
riskiest part to get byte-right without matching anchors from that exact
fork lineage; a wrong number there would be worse than no number. This
overlay ports the part that is directly and unambiguously derivable from
the SAME two config objects the stock line already uses -- the per-group
breakdown and the accurate usable-block-id / blocks-per-request figures --
which already answers the upstream bug report's core complaint ("neither
num_blocks nor num_blocks_per_request is logged").

Two pinned anchors in ``kv_cache_utils.py``, both preflighted before either
is written, atomic, idempotent, one marker per site, fails closed if either
anchor is missing/duplicated -- same discipline as
``overlay/patch_default_max_new_tokens.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TARGET = Path(
    os.environ.get(
        "GLM53_KV_CACHE_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
    )
)
MARK = "# [glm53-kv-capacity-log]"

HELPER_ANCHOR = (
    "\ndef get_kv_cache_capacity(\n"
    "    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig\n"
    ") -> tuple[int, float]:\n"
)

HELPER_BLOCK = '''
def _glm53_kv_capacity_log_enabled() -> bool:  # [glm53-kv-capacity-log]
    """GLM53_KV_CAPACITY_LOG: unset/"1" -> log (default), "0" -> disabled.

    An unrecognized value logs one warning and defaults to enabled: this is
    a diagnostic feature, a typo must not be able to affect boot.
    """
    raw = os.environ.get("GLM53_KV_CAPACITY_LOG", "1")
    if raw not in ("0", "1"):
        logger.warning_once(
            "[glm53-kv-capacity-log] GLM53_KV_CAPACITY_LOG=%r is not 0 or 1 "
            "-- defaulting to enabled",
            raw,
        )
        return True
    return raw == "1"


def _glm53_unwrap_spec_type_name(spec) -> str:  # [glm53-kv-capacity-log]
    """Display name for a (possibly UniformTypeKVCacheSpecs-wrapped) spec."""
    inner = getattr(spec, "kv_cache_specs", None)
    if inner:
        spec = next(iter(inner.values()))
    return type(spec).__name__


def _glm53_log_kv_capacity_breakdown(  # [glm53-kv-capacity-log]
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    num_tokens: int,
    max_concurrency: float,
) -> None:
    """Log a per-group breakdown of the same figures the stock line uses.

    Log-only: reads kv_cache_config/vllm_config, mutates nothing. Any
    exception here is caught by the caller and never reaches the boot path.
    """
    max_model_len = vllm_config.model_config.max_model_len
    total_blocks_per_request = 0
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        blocks_per_request = cdiv(
            spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes
        )
        total_blocks_per_request += blocks_per_request
        logger.info_once(
            "[glm53-kv-capacity-log] group %s: %s layers=%s block_size=%s "
            "page_size=%s B blocks/request@%s=%s prefix_caching=%s",
            idx,
            _glm53_unwrap_spec_type_name(spec),
            len(group.layer_names),
            f"{spec.block_size:,}",
            f"{spec.page_size_bytes:,}",
            f"{max_model_len:,}",
            f"{blocks_per_request:,}",
            "yes" if spec.participates_in_prefix_caching else "no",
        )
    usable_ids = max(kv_cache_config.num_blocks - 1, 0)
    logger.info_once(
        "[glm53-kv-capacity-log] usable block ids: %s (num_blocks=%s incl. "
        "the null block); blocks/request@%s tokens totals %s across groups "
        "(matches the denominator of the 'GPU KV cache size' line above: "
        "%s / %s = %.2fx). That line is max_concurrency x max_model_len, "
        "not a cached-conversation token count.",
        f"{usable_ids:,}",
        f"{kv_cache_config.num_blocks:,}",
        f"{max_model_len:,}",
        f"{total_blocks_per_request:,}",
        f"{kv_cache_config.num_blocks:,}",
        f"{total_blocks_per_request:,}",
        max_concurrency,
    )

'''

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
CALL_NEW = CALL_OLD + '''    if _glm53_kv_capacity_log_enabled():  # [glm53-kv-capacity-log]
        try:
            _glm53_log_kv_capacity_breakdown(
                vllm_config, kv_cache_config, num_tokens, max_concurrency
            )
        except Exception as exc:  # log-only feature; must never fail boot
            logger.warning_once(
                "[glm53-kv-capacity-log] could not derive the block-level "
                "KV capacity breakdown (log-only; serving unaffected): "
                "%s: %s",
                type(exc).__name__,
                exc,
            )
    else:
        logger.info_once("[glm53-kv-capacity-log] disabled (GLM53_KV_CAPACITY_LOG=0)")
'''


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    text = TARGET.read_text()

    n_helper_mark = HELPER_BLOCK.count(MARK)
    n_call_mark = CALL_NEW.count(MARK) - CALL_OLD.count(MARK)
    want_marks = n_helper_mark + n_call_mark

    have_marks = text.count(MARK)
    if have_marks:
        have_helper = text.count(HELPER_BLOCK) == 1
        have_call = text.count(CALL_NEW) == 1 and text.count(CALL_OLD) >= 1
        if have_marks != want_marks or not (have_helper and have_call):
            raise SystemExit(
                f"{TARGET}: carries {have_marks} '{MARK}' marker(s) "
                f"(complete = {want_marks}) but does not carry every "
                "generated snippet verbatim -- refusing to patch a "
                "partially modified file (restore the pristine file first)"
            )
        compile(text, str(TARGET), "exec")
        print(f"{TARGET.name}: kv-capacity-log already present — skipping")
        return 0

    if text.count(HELPER_ANCHOR) != 1:
        raise SystemExit(
            f"{TARGET}: expected exactly one helper-insertion anchor, "
            f"found {text.count(HELPER_ANCHOR)}"
        )
    if text.count(CALL_OLD) != 1:
        raise SystemExit(
            f"{TARGET}: expected exactly one update_kv_cache_capacity "
            f"target, found {text.count(CALL_OLD)}"
        )

    patched = text.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)
    patched = patched.replace(CALL_OLD, CALL_NEW, 1)

    if patched.count(MARK) != want_marks:
        raise SystemExit(f"{TARGET}: internal error, marker count after apply")
    compile(patched, str(TARGET), "exec")
    TARGET.write_text(patched)
    print(
        f"patched {TARGET.name} (per-group KV-capacity breakdown log; "
        "GLM53_KV_CAPACITY_LOG, default enabled)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
