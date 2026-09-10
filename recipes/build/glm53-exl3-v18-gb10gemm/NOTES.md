# v18-gb10gemm: vLLM PR #54048 backport (cuBLAS out_dtype router GEMM on GB10)

Seeded from `recipes/build/glm53-exl3-v15-combined` (current production).
Candidate 2 of 3 in the same downtime testing window as v17-nopemha
(candidate 1, NOT promoted -- turned out to be dead code, blocked by our own
already-shipped fix for a different upstream PR, #51395). Production traffic
stopped for the duration.

## What this build adds

**`vllm-project/vllm#54048`, "[Bugfix][MoE] Enable cuBLAS out_dtype router
GEMM on all CUDA archs (fixes family-120/GB10)"**: `GateLinear` (`vllm/
model_executor/layers/fused_moe/router/gate_linear.py`) picks a GEMM tier for
the MoE router/gate computation. Tier 5 (the fused cuBLAS `torch.mm(...,
out_dtype=torch.float32)` epilogue -- one bf16xbf16->fp32 GEMM, no separate
cast kernel) was gated on `self.allow_specialized_router_gemm`, which only
recognizes Hopper (SM90) or `is_device_capability_family(100)` ("Blackwell"
in the narrow SM100 sense) -- wrongly excluding GB10's SM121a
("family-120"), even though the cuBLAS out_dtype epilogue itself is a plain
`torch.mm` kwarg with nothing Hopper/SM100-specific about it. GB10 fell
through to Tier 6 (plain `F.linear`: a bf16 GEMM plus a SEPARATE bf16->fp32
cast kernel) -- one extra kernel launch per MoE layer per forward pass, and
router logits that are bf16-rounded before the cast rather than
fp32-accumulated in a single epilogue.

Fix: decouple cuBLAS eligibility into its own arch-agnostic predicate,
`self._router_gemm_cublas_capable = (is_cuda() or is_rocm()) and
not bias`, applied in both `GateLinear.__init__` and `set_out_dtype` (the
gate's `out_dtype` is frequently set AFTER construction, once the expert
quantization method is known -- true for this project's `exl3` path, so
both sites matter). New overlay patch: `overlay/patch_gb10_router_gemm.py`,
unconditional, both anchors verified byte-for-byte against this image's
installed source before writing the patch (and again via the patch script's
own fail-closed preflight).

**Anchor reconciliation: NONE needed.** Checked a live container of
`glm53-exl3-v15-combined:local` (`docker run --rm --entrypoint python3
glm53-exl3-v15-combined:local -c "...read gate_linear.py..."`) before
writing anything -- the installed file matched vLLM PR #54048's own upstream
"before" diff context byte-for-byte at both hunks. Unlike several other
backports in this project's history (v10-vllmpatch, v17-nopemha), this file
has NOT drifted from stock upstream in this image -- the PR's diff applied
directly as-is.

**Distinct from this build's already-present `patch_moe_gate_dedup.py`**
(backport of vLLM PR #55736 commit 3/3): that patch removes a duplicate
CALL to the gate module in `vllm/models/glm5next/nvidia/model.py` (`model.py`
computed router logits externally, `MoERunner._forward_impl` unconditionally
recomputed them internally anyway, discarding the external result). This
patch controls which GEMM TIER the surviving call executes through --
completely different file (`gate_linear.py` vs. `model.py`), zero code
overlap between the two beyond both touching MoE routing. Verified they
coexist cleanly (both patches apply, both preflight-check clean, no anchor
conflicts).

## Precision angle, not (primarily) a throughput play

Router logits feed the argmax/softmax that select routed experts, for a
288-expert router on this checkpoint. Tier 6 (pre-patch, GB10's actual path)
computes the GEMM in the weight's own bf16 dtype, THEN casts the result to
fp32 -- the router logits are bf16-rounded before argmax/softmax ever sees
them. Tier 5 (post-patch) computes the entire bf16xbf16->fp32 GEMM in one
cuBLAS epilogue -- exact fp32 accumulation, no intermediate bf16 rounding.
For a 288-expert router this narrows (does not eliminate) the chance of a
near-tied logit pair flipping which expert wins top-k purely from rounding.
The secondary effect (one fewer kernel launch per MoE layer per forward
pass, since the separate cast Tier 6 pays for is gone) is a real but
secondary perf win -- not the headline reason for taking this patch.

Added a diagnostic `logger.info_once` line that fires ONLY when the widened
gate admits a device the OLD gate would have excluded (i.e.
`allow_cublas_router_gemm` true but `allow_specialized_router_gemm` false --
GB10 family-120 CUDA, in practice, on this fleet). It never fires on
Hopper/SM100, where the specialized-kernel gate already covered this tier --
so seeing it fire on a real boot is direct, positive confirmation the fix
actually engages on this hardware, not just that the patch applied cleanly.

## Validation status

- [x] Patch preflight + idempotency + fresh-process import check (CPU, no
      GPU) against `glm53-exl3-v15-combined:local`'s installed source -- pass,
      zero anchor drift
- [x] Image build (`glm53-exl3-v18-gb10gemm:local`) -- clean, all existing
      self-checks plus the new patch's own preflight passed
- [x] tinyGLM dispatch-correctness gate -- clean TP2 boot, no stall, zero
      errors/NaN in the full boot log. Diagnostic log line fired on both
      workers (`gate_linear.py:141 [glm53-gb10-router-gemm] cuBLAS
      bf16xbf16->fp32 router GEMM enabled...`), confirming the widened gate
      actually engages on this hardware, not just that the patch applied.
      Two identical chat-completion requests produced byte-identical output
      (deterministic). A third, much longer prompt produced the same
      synthetic dummy-weight output pattern -- expected for this fixture
      (weights are random/meaningless by design; the coordinator-relevant
      determinism and diagnostic checks are the ones that matter here, not
      output "coherence").
- [x] Real-checkpoint boot (TP2, both nodes) + `probe_sanity.py` -- see
      below for a real, honestly-reported complication mid-session.
- [x] One `probe_longctx.py` run at ~16K tokens -- see below.
- [x] sparkrun job(s) stopped, GPUs confirmed idle for candidate 3.

### Real-checkpoint results

First boot: clean, no NCCL/shm-broadcast stall (the known intermittent
issue flagged in the task did not occur this time). Diagnostic line fired
again on the real checkpoint. `probe_sanity.py --model glm-5.3-flash-exl3-v2`:
**ALL SHORT-CONTEXT CHECKS PASSED** -- chat coherent, no reasoning-leak,
`finish_reason=stop`, decode **27.95-29.59 tok/s** (3-run bench), at or
slightly above the v15-combined baseline (26.9-28.5 tok/s).

**Mid-session complication, real and unrelated to this patch:** while
running `probe_longctx.py` at ~16K tokens immediately after `probe_sanity.py`,
the head node's API server received SIGTERM and shut down cleanly (no
traceback, no crash signature -- an orderly shutdown sequence in the vLLM
log). Confirmed via `journalctl -u earlyoom` on the head node (10.7.0.87,
`spark-2dd4`) directly: an actual earlyoom kill, not a guess --
`low memory! at or below SIGTERM limits: mem 2.00%, swap 80.00%` followed by
`sending SIGTERM to process 2476101 uid 1000 "python3": badness 989`, at the
exact timestamp the API server's shutdown sequence began. This matches this
project's own documented "silent kill = earlyoom" signature
(`earlyoom_root_cause_correction` memory note) precisely -- checked
`journalctl -u earlyoom` first, per that note's own guidance, rather than
chasing a phantom crash.

**Not attributed to this patch.** `gpu_memory_utilization` is unchanged from
v15-combined's validated-safe 0.84; this backport touches only which GEMM
tier computes the MoE router logits (no new tensors, no larger buffers --
if anything, the fused cuBLAS epilogue should use marginally LESS transient
memory than the two-kernel bf16-GEMM-plus-cast path it replaces, since it
skips materializing the intermediate bf16 output before casting). The head
node's memory margin at this GMU is already well documented in this
project's history as thin (see `gmu_086_earlyoom_regression`,
`host_fragmentation_xid31_reboot_risk`, and the recurring earlyoom incidents
in `VALIDATION.md`) -- this reads as that same pre-existing, environment-level
fragility recurring under real serving + probe load, not a new regression
this backport introduces. Flagging it plainly rather than smoothing it over:
the coordinator should treat it as a data point about this fleet's ambient
memory risk at 0.84 GMU, not as evidence against this specific patch.

**Recovery and clean re-measurement:** ran `recipes/scripts/prelaunch_flush.sh`
on both hosts (fragmentation check passed clean on both, >21M and >14M
contiguous free blocks respectively -- well above the 2000-block warn
threshold), relaunched the identical recipe. Second boot: clean, no stall,
diagnostic fired again, `curl /health` returned 200 throughout. Re-ran
`probe_longctx.py` at ~16K tokens: **ALL PASSED** -- `prompt_tokens=15960`,
**TTFT 10.7s** (v15-combined baseline ~9.9s -- close, single n=1 run, within
normal variance for this measurement, not a rigorous A/B), `finish_reason=
stop`, all 4 planted retrieval codes found correctly (`CODE-1-BBBB`,
`CODE-2-PPPP`, `CODE-3-JJJJ`, `CODE-4-CCCC` -- confirms long-context
correctness, not just short-prompt sanity). Effective prefill throughput
~1,492 tok/s vs. baseline ~1,612 tok/s -- somewhat lower but this is a
single measurement taken shortly after an earlyoom recovery cycle, not a
controlled A/B; per the task's own guidance ("don't over-invest in prefill
benchmarking here -- primarily a correctness/precision fix"), this was not
chased further with repeat runs. Host memory checked before and after this
run (`free -h`, both nodes) -- stayed in the same thin-but-stable band seen
throughout this session (4.6-6.9 GiB available), no second kill.

Zero errors, tracebacks, NaN, or crash signatures anywhere in either boot's
full log (`grep -iE "error|traceback|nan|died"`, manually reviewed).
