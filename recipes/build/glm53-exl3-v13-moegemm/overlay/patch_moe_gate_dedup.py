#!/usr/bin/env python3
"""Backport of vLLM PR #55736, commit 3/3 (MoE router-GEMM dedup): stop
computing the routed-expert gate/router logits twice per forward pass.

Deferred earlier this session pending a closer trace -- this is that trace,
done directly against the real installed source (not inferred from the PR
diff alone):

``vllm/models/glm5next/nvidia/model.py``'s MoE block constructs its expert
runner with the gate module attached::

    self.experts = FusedMoEFactory(
        ...
        gate=self.gate,
        ...
    )

and its ``forward()`` (before this patch) computed the router logits
EXTERNALLY and passed them in::

    router_logits, _ = self.gate(hidden_states)
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router_logits=router_logits
    )

Traced ``vllm.model_executor.layers.fused_moe.layer.MoERunner._forward_impl``
(the method ``self.experts(...)`` actually calls) directly in this image::

    if self.gate is not None:
        if self._fse_fuse_gate:
            self._maybe_fuse_gate_weights()
            router_logits = F.linear(hidden_states, self._combined_gate_weight)
        else:
            router_logits, _ = self.gate(hidden_states)

``self.gate`` here is the SAME ``GateLinear`` instance passed in at
construction (``FusedMoEFactory``'s ``gate=`` argument becomes the runner's
own ``self.gate`` attribute) -- so ``self.gate is not None`` is true, and
whatever ``router_logits`` value the caller passes in is unconditionally
OVERWRITTEN by this internal recomputation. ``self._fse_fuse_gate`` (set at
construction: ``gate is not None and shared_expert_gate is not None``) is
``False`` for this model, since ``FusedMoEFactory`` is never called with
``shared_expert_gate=`` here -- so the branch that actually fires is the
plain ``else``, an IDENTICAL ``self.gate(hidden_states)`` call to the one
the un-patched external code already makes.

Net effect of the un-patched code: the gate/router GEMM runs TWICE per MoE
layer per forward pass (once externally in ``model.py``, once again
internally in ``MoERunner``, silently discarding the external result) for
zero behavioral difference -- pure duplicate compute. This patch removes the
external call and passes ``hidden_states`` as a placeholder for
``router_logits`` (matches upstream's own ``DeepseekV2MoE`` pattern, and the
PR's own diff exactly), letting the internal call be the only one that runs.

This is a correctness-neutral perf fix once the above is confirmed (same
module, same input, same call) -- no env-var gate, matching this project's
convention for backports it has verified rather than merely proposed. Not
yet measured on this fleet's actual hardware; expect a small decode/prefill
win proportional to the gate GEMM's share of per-layer compute (small: gate
projects hidden_size -> n_routed_experts, i.e. 4096 -> 288 here, versus the
routed-expert compute itself), not a headline number.

Verified against a live boot of this exact container before being written
here (anchor matched byte-for-byte in the installed site-packages).
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_MODEL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py",
    )
)

MARK = "        # [glm53-moe-gate-dedup] MoERunner holds the gate (passed to\n"

ANCHOR = (
    "        # The router is always external (self.gate); main's MoERunner expects\n"
    "        # pre-computed router_logits, so compute them here unconditionally.\n"
    "        router_logits, _ = self.gate(hidden_states)\n"
    "        final_hidden_states = self.experts(\n"
    "            hidden_states=hidden_states, router_logits=router_logits\n"
    "        )\n"
)

PATCHED = (
    "        # [glm53-moe-gate-dedup] MoERunner holds the gate (passed to\n"
    "        # FusedMoEFactory as gate=self.gate) and computes the router\n"
    "        # logits itself -- MoERunner._forward_impl's `if self.gate is\n"
    "        # not None:` branch unconditionally overwrote whatever we passed\n"
    "        # here anyway, so the external call above was pure duplicate\n"
    "        # compute. See patch_moe_gate_dedup.py's module docstring for\n"
    "        # the full trace (vllm-project/vllm#55736 commit 3/3).\n"
    "        # router_logits is a placeholder (matches upstream's own\n"
    "        # DeepseekV2MoE pattern) -- MoERunner recomputes the real value.\n"
    "        final_hidden_states = self.experts(\n"
    "            hidden_states=hidden_states, router_logits=hidden_states\n"
    "        )\n"
)


def verified_state(text: str) -> bool:
    return (
        text.count(MARK) == 1
        and text.count(PATCHED) == 1
        and text.count(ANCHOR) == PATCHED.count(ANCHOR)
    )


def prepare(source: str) -> tuple[str, str]:
    if source.count(MARK):
        if source.count(MARK) != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent moe-gate-dedup patch -- refusing to touch a half-patched file"
            )
        return source, "already present"
    n = source.count(ANCHOR)
    if n != 1:
        raise ValueError(
            f"pinned moe-gate-dedup anchor drifted (found {n}, expected 1) -- "
            "re-derive the patch (model.py's MoE forward() may have changed)"
        )
    out = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(out):
        raise ValueError("moe-gate-dedup post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-moe-gate-dedup.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"moe-gate-dedup preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if preflight_only:
        print(f"{TARGET.name}: moe-gate-dedup preflight OK ({action})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: moe-gate-dedup {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
