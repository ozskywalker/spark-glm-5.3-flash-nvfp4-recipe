#!/usr/bin/env python3
"""Backport of vLLM PR #55737 ("Use FlashKDA for KDA chunked prefill,
1.7-3.8x faster than the Triton chunk path"), adapted to this image's
installed vLLM source.

FlashKDA (``vllm._flashkda_C``) is a fused CUDA kernel implementing the same
bounded-gate KDA recurrence GLM-5.3-Flash already uses
(``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, in-kernel q/k
l2norm, raw beta logits) -- a drop-in replacement for the ~15-kernel Triton
``chunk_kda_with_fused_gate`` path used during prefill. The compiled
extension already ships in this image (confirmed via
``importlib.util.find_spec('vllm._flashkda_C')`` earlier this session), but
zero Python integration existed to use it -- this patch is that integration,
ported from the PR rather than a literal patch-apply, since our installed
``kda.py`` differs cosmetically from the PR's base (import grouping, an
unrelated Config class name in unchanged context) while matching byte-for-
byte on every anchor that actually matters: the ``Glm5NextLinearAttention``
class, the end of ``__init__`` (the ``_conv_state_dim_first`` line), the
``chunk_kda_with_fused_gate(...)`` call block, and the merge-back block.

Notably, our installed file already has a *sibling* optimization for the
plain-decode path (``fused_recurrent_kda`` writing straight into ``ns_out``,
skipping the merge copy) with a comment explicitly noting "the chunked
prefill kernel cannot [do this], so this stays None there and the merge
copy below runs as before" -- this patch closes exactly that gap for the
prefill path, the same way the upstream PR does.

Six hunks, all in
``vllm/models/glm5next/nvidia/kda.py``:
  1. Import ``current_workspace_manager`` (confirmed importable and
     functional in this vLLM version -- calling it uninitialized correctly
     raises vLLM's own assertion, not an ImportError).
  2. Add ``_resolve_kda_prefill_backend()`` -- the eligibility gate
     (SM90/SM10x/SM12x, bf16, head_dim==128, a bounded gate configured).
     SM121/GB10 (capability.major == 12) passes this gate numerically, but
     the PR's own real benchmark numbers are GB300 (SM100) only -- our own
     validation is still required before trusting the claimed 1.7-3.8x here.
  3. End of ``__init__``: resolve the backend from ``GLM53_FLASHKDA_PREFILL``
     (this project's own env-var convention, not upstream's
     ``additional_config.kda_prefill_backend`` string enum) and, if eligible,
     size the FlashKDA workspace buffers.
  4. New ``_flashkda_prefill()`` method.
  5. In ``forward``'s non-spec/prefill branch: call ``_flashkda_prefill``
     instead of ``chunk_kda_with_fused_gate`` when the backend resolved to
     "flashkda", writing directly into the output buffer when there are no
     spec-decode tokens to merge (mirrors the existing decode-path pattern).
  6. Simplify the spec/non-spec merge block to two in-place ``index_copy_``
     calls instead of an intermediate ``torch.empty`` + copy (upstream's own
     cleanup, applies regardless of which prefill backend is active).

``GLM53_FLASHKDA_PREFILL=0`` (default, and the value of an unset var): the
backend resolver is forced to "triton" -- zero behavior change, matching
this project's own contract that this file, built with this patch applied,
still serves exactly stock when the flag is off.
``GLM53_FLASHKDA_PREFILL=1``: the resolver is forced to "flashkda". This is
deliberately NOT upstream's "auto" mode (which silently falls back to
Triton on ineligible hardware/dtype) -- forcing "flashkda" fails loudly
(``RuntimeError``) if this exact boot's config doesn't meet the gate,
matching this project's fail-closed convention rather than a silent
same-as-off fallback that would make a botched enable look like a no-op.
Any other value raises at boot rather than picking a mode from a typo.

NOT independently benchmarked on SM121/GB10 by this patch's author (a
fork of the parent session, running with production live and no downtime
authorized for a real-checkpoint boot) -- that validation is the
coordinator's next step once downtime is available. This patch only
establishes that the port is clean and the image builds/imports correctly.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_KDA_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py",
    )
)

ENV_NAME = "GLM53_FLASHKDA_PREFILL"


# ---------------------------------------------------------------------------
# Site 1 -- import current_workspace_manager
# ---------------------------------------------------------------------------
MARK_IMPORT = "from vllm.v1.worker.workspace import current_workspace_manager  # [glm53-flashkda]\n"

ANCHOR_IMPORT = "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"

PATCHED_IMPORT = (
    "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
    "from vllm.v1.worker.workspace import current_workspace_manager  # [glm53-flashkda]\n"
)

# ---------------------------------------------------------------------------
# Site 1b -- kda.py doesn't import `os` at all; the backend resolver
# (site 2) needs it for os.environ.get(...).
# ---------------------------------------------------------------------------
MARK_OS_IMPORT = "import os  # [glm53-flashkda] for GLM53_FLASHKDA_PREFILL\n"

ANCHOR_OS_IMPORT = (
    "import torch\n"
    "from torch import nn\n"
)

PATCHED_OS_IMPORT = (
    "import os  # [glm53-flashkda] for GLM53_FLASHKDA_PREFILL\n"
    "import torch\n"
    "from torch import nn\n"
)


# ---------------------------------------------------------------------------
# Site 2 -- the backend resolver, inserted after _cast_sigmoid
# ---------------------------------------------------------------------------
MARK_RESOLVER = "def _glm53_flashkda_backend() -> str:\n"

ANCHOR_RESOLVER = (
    "def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:\n"
    '    """Fuse the fp32 cast + sigmoid into one Inductor kernel."""\n'
    "    return x.float().sigmoid()\n"
)

PATCHED_RESOLVER = (
    ANCHOR_RESOLVER
    + '''

# [glm53-flashkda] Backport of vllm-project/vllm#55737 -- see
# patch_flashkda.py's module docstring for the full port rationale.
def _glm53_flashkda_backend() -> str:
    """"0" (unset default) forces "triton"; "1" forces "flashkda" (fails
    loudly via _resolve_kda_prefill_backend if this boot's config doesn't
    meet the eligibility gate, rather than silently falling back)."""
    raw = os.environ.get("GLM53_FLASHKDA_PREFILL")
    if raw is None or raw == "0":
        return "triton"
    if raw == "1":
        return "flashkda"
    raise ValueError(
        f"GLM53_FLASHKDA_PREFILL must be exactly one of: 0 1 (got: {raw!r})"
    )


def _resolve_kda_prefill_backend(
    backend: str, head_dim: int, dtype: torch.dtype, lower_bound
) -> str:
    """Pick the chunked-prefill kernel: FlashKDA (fused CUDA, ~1.7-3.8x
    faster upstream on SM90/SM10x/SM12x for bf16, head_dim 128 and a bounded
    gate -- NOT independently re-measured on this fleet's SM121/GB10 yet) or
    the Triton ``chunk_kda_with_fused_gate`` path."""
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    capability = current_platform.get_device_capability()
    supported = (
        current_platform.is_cuda()
        and capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, head_dim=128 "
            "and a bounded KDA gate."
        )
    return "flashkda" if supported and backend != "triton" else "triton"
'''
)


# ---------------------------------------------------------------------------
# Site 3 -- end of __init__: resolve backend + size workspace buffers
# ---------------------------------------------------------------------------
MARK_INIT = "        # [glm53-flashkda] backend resolution + workspace sizing\n"

ANCHOR_INIT = (
    "        # Process-global conv-state layout, resolved once here instead of on\n"
    "        # every _forward call (it reads an env-derived flag each time).\n"
    "        self._conv_state_dim_first = is_conv_state_dim_first()\n"
)

PATCHED_INIT = (
    ANCHOR_INIT
    + '''
        # [glm53-flashkda] backend resolution + workspace sizing
        self.kda_prefill_backend = _resolve_kda_prefill_backend(
            _glm53_flashkda_backend(),
            self.head_dim,
            vllm_config.model_config.dtype,
            self.kda_lower_bound,
        )
        self._flashkda_buffer_specs = None
        if self.kda_prefill_backend == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            workspace_size = torch.ops._flashkda_C.get_workspace_size(
                max_tokens, self.local_num_heads, max_seqs
            )
            self._flashkda_buffer_specs = (
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
                (
                    (1, max_tokens, self.local_num_heads, self.head_dim),
                    vllm_config.model_config.dtype,
                ),
            )
'''
)


# ---------------------------------------------------------------------------
# Site 4 -- new _flashkda_prefill method, inserted right before forward()
# ---------------------------------------------------------------------------
MARK_METHOD = "    def _flashkda_prefill(\n"

ANCHOR_METHOD = (
    "    def forward(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        positions: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        num_tokens = hidden_states.size(0)\n"
)

PATCHED_METHOD = (
    '''    # [glm53-flashkda] see patch_flashkda.py's module docstring
    def _flashkda_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out,
    ):
        """Fused KDA chunked prefill (FlashKDA). Mirrors
        ``chunk_kda_with_fused_gate(..., safe_gate=True)`` numerically:
        l2-normalizes q/k in-kernel and applies the bounded gate
        ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``. Writes into
        ``out`` (a workspace buffer when ``None``) and returns
        ``(out, final_state)``."""
        assert self._flashkda_buffer_specs is not None
        final_state, workspace, workspace_out = (
            current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)
        )
        final_state = final_state[: initial_state.shape[0]]
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        # [glm53-flashkda] this image's compiled _flashkda_C.fwd() takes
        # exactly 14 args (q,k,v,g,beta,scale,out,workspace,A_log,dt_bias,
        # lower_bound,initial_state,final_state,cu_seqlens) -- confirmed
        # from the real registered schema via a live tinyGLM boot's own
        # RuntimeError ("expected at most 14 argument(s) but received 16").
        # The PR's own call passes two further trailing None args our
        # bundled extension's schema has no slots for at all; both were
        # already None in the ported call (no feature was being requested
        # through them), so omitting them is behaviorally a no-op against
        # this exact compiled binary, not a semantic change.
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim**-0.5,
            out,
            workspace,
            self.A_log.view(-1),
            self.dt_bias.view(-1, self.head_dim),
            self.kda_lower_bound,
            initial_state.contiguous(),
            final_state,
            cu_seqlens.contiguous(),
        )
        return out, final_state

'''
    + ANCHOR_METHOD
)


# ---------------------------------------------------------------------------
# Site 5 -- dispatch: flashkda vs chunk_kda_with_fused_gate in the prefill branch
# ---------------------------------------------------------------------------
MARK_DISPATCH = "            if self.kda_prefill_backend == \"flashkda\":  # [glm53-flashkda]\n"

ANCHOR_DISPATCH = (
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = chunk_kda_with_fused_gate(\n"
    "                q=_rearr(q_ns),\n"
    "                k=_rearr(k_ns),\n"
    "                v=_rearr(v_ns),\n"
    "                raw_g=g1_ns,\n"
    "                # Chunk path wants the pre-sigmoided fp32 beta (its kernels\n"
    "                # don't sigmoid); beta_ns is raw bf16 from forward.\n"
    "                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),\n"
    "                A_log=self.A_log,\n"
    "                g_bias=self.dt_bias,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=True,\n"
    "                use_qk_l2norm_in_kernel=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                safe_gate=safe_gate,\n"
    "                lower_bound=lower_bound,\n"
    "            )\n"
)

_ANCHOR_DISPATCH_REINDENTED = "".join(
    "    " + line if line.strip() else line
    for line in ANCHOR_DISPATCH.splitlines(keepends=True)
)

PATCHED_DISPATCH = (
    '''            if self.kda_prefill_backend == "flashkda":  # [glm53-flashkda]
                # Non-spec-only step: write straight into the layer output
                # buffer, same pattern the plain-decode branch below already
                # uses for ns_out. A step that also carries spec tokens
                # writes to a workspace buffer instead; the merge block
                # further down scatters both halves back by index.
                ns_out = None if use_spec else core_attn_out[:, :num_actual_tokens]
                core_attn_out_non_spec, last_recurrent_state = self._flashkda_prefill(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    g=g1_ns,
                    beta=beta_ns,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    out=ns_out,
                )
            else:
'''
    + _ANCHOR_DISPATCH_REINDENTED
)


# ---------------------------------------------------------------------------
# Site 6 -- simplify the merge block (upstream's own cleanup, backend-agnostic)
# ---------------------------------------------------------------------------
MARK_MERGE = "            # [glm53-flashkda] in-place scatter, no intermediate buffer\n"

ANCHOR_MERGE = (
    "            merged = torch.empty(\n"
    "                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),\n"
    "                dtype=core_attn_out_non_spec.dtype,\n"
    "                device=core_attn_out_non_spec.device,\n"
    "            )\n"
    "            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)\n"
    "            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)\n"
    "            core_attn_out[0, :num_actual_tokens] = merged.squeeze(0)\n"
)

PATCHED_MERGE = (
    "            # [glm53-flashkda] in-place scatter, no intermediate buffer\n"
    "            # -- see vllm-project/vllm#55737, applies regardless of\n"
    "            # which prefill backend produced core_attn_out_non_spec.\n"
    "            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)\n"
    "            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)\n"
)


SITES = (
    ("import", MARK_IMPORT, ANCHOR_IMPORT, PATCHED_IMPORT),
    ("os import", MARK_OS_IMPORT, ANCHOR_OS_IMPORT, PATCHED_OS_IMPORT),
    ("backend resolver", MARK_RESOLVER, ANCHOR_RESOLVER, PATCHED_RESOLVER),
    ("__init__ workspace sizing", MARK_INIT, ANCHOR_INIT, PATCHED_INIT),
    ("_flashkda_prefill method", MARK_METHOD, ANCHOR_METHOD, PATCHED_METHOD),
    ("prefill dispatch", MARK_DISPATCH, ANCHOR_DISPATCH, PATCHED_DISPATCH),
    ("merge block", MARK_MERGE, ANCHOR_MERGE, PATCHED_MERGE),
)


def verified_state(text: str) -> bool:
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in SITES
    )


def prepare(source: str) -> tuple[str, str]:
    marks = sum(source.count(mark) for _n, mark, _a, _p in SITES)
    if marks:
        if marks != len(SITES) or not verified_state(source):
            raise ValueError(
                "partial/inconsistent flashkda patch "
                f"(marks={marks}, expected {len(SITES)}) -- refusing to touch "
                "a half-patched file"
            )
        return source, "already present"

    out = source
    for name, _mark, anchor, patched in SITES:
        n = out.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned flashkda anchor '{name}' drifted (found {n}, expected 1) "
                "-- re-derive the patch"
            )
        out = out.replace(anchor, patched, 1)

    if not verified_state(out):
        raise ValueError("flashkda post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-flashkda.tmp")
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
        raise SystemExit(f"flashkda preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")

    if preflight_only:
        print(f"{TARGET.name}: flashkda preflight OK ({action})")
        return 0

    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    mode = os.environ.get(ENV_NAME, "0 (unset)")
    print(f"{TARGET.name}: flashkda {action} ({ENV_NAME}={mode!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
