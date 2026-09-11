# Experiment 01: GLM-5.3-Flash-TrellisMX-MXFP8 on 2x DGX Spark

Source: https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8

## What this checkpoint actually is

Not a self-contained quantized model. It's an **overlay**: 168 "TP4 routed-expert
sidecars" (177 GB, trellis-coded MXFP8 weights for the 288-expert MoE routed layers
only) that replace the routed-expert tensors inside a separately-hosted **carrier**
checkpoint -- `local-inference-lab/GLM-5.3-Flash-NVFP4` @ `520de24eabf5...` (~199 GB,
standard ModelOpt NVFP4, confirmed via the HF tree API). Attention, shared experts,
embeddings, output head, and MTP stay on the carrier's NVFP4 weights untouched --
only the 288-expert routed MoE tensors get swapped for the trellis-decoded ones.

Quantization lineage: "TrellisMX" is a trellis-coded scheme in the same family as
EXL3/QTIP (the paper behind our own production quantization) and QSRT, decoding a
coupled 17-K5/25-K4 stream (4.66 bits/weight incl. metadata) into E4M3 FP8 operands
with per-32-element UE8M0 block scales for `mxf8f6f4` tensor-core MMA. Runtime
kernels ship as source (`b12x` package + a vLLM `trellismx.py` quant-method patch),
license (ShapleyMcg) explicitly permits reading, modifying, and reimplementing --
this is not a black box, and it's not legally blocked. Credit line even names
ExLlamaV3/EXL3 as direct lineage, which is worth noting given that's what we already
run in production.

## Hypotheses, and what source inspection already answered

Read the actual shipped source (`runtime/vllm/vllm/model_executor/layers/
quantization/trellismx.py`, `runtime/b12x/b12x/moe/_shared/trellismx/
p8_native_kernel.py` and neighbors) rather than trusting the README's framing.
Two unconditional guards in `TrellisMXMoEMethod.__init__` / `.process_weights_after_loading`:

```python
if (parallel.tp_size != 4 or parallel.ep_size != 1 or ...):
    raise ValueError("TrellisMX GLM adapter requires TP4, EP1 and GLM Flash shapes")
...
if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
    raise ValueError("This TrellisMX runtime requires SM120 CUDA")
```

**H1 -- SM120-exact hardware gate.** `get_device_capability() != (12, 0)` is an
*exact* match, not a floor check. Our GB10 reports `(12, 1)` (SM121a). **CONFIRMED
BLOCKER**, straight from source, not inference from the README's silence on GB10.

**H2 -- our own installed vLLM already has generic MXFP8 linear-kernel scaffolding**
(`model_executor/kernels/linear/mxfp8/{flashinfer,marlin,emulation,humming,rocm_native,xpu}.py`)
but no TrellisMX-specific decode step -- moot either way, since H1/H3 block the path
that would use it.

**H3 -- TP4 is hard-baked, not just a packaging default.** Confirmed at two
independent levels: (a) `trellismx-manifest.json` names every sidecar file
`p8-layer-{N}-tp4-rank-{R}.safetensors` for R in 0..3 with no resharding metadata
(no split-axis, no byte-range, nothing to reconstruct an unsharded tensor from); (b)
the loader code above raises unconditionally on `tp_size != 4`, and separately
constructs `P8NativeTPMoE(..., world_size=4, ...)` with 4 hardcoded into the kernel
call itself, not derived from `parallel.tp_size`. We have 2 physical GPUs.
**CONFIRMED BLOCKER**, independent of H1 -- fixing H1 alone would not be enough.

**H4 -- disk/bandwidth risk.** Turned out to be a non-issue either way: the carrier
alone is ~199 GB (36 main shards + separate ~18 GB "nonexpert" shard set + MTP
shards), comfortably inside our 769 GB free. The 177 GB sidecars are **not worth
downloading** given H1+H3 -- that's 177 GB and real wall-clock time spent on an
artifact that cannot execute on this hardware as shipped. Not downloaded.

**H5 -- accuracy, soft signal only, not decisive.** The README's own KLD comparison:
TrellisMX/FP8 KV 0.0319 vs. our production TR3/EXL3 4bpw 0.0282 (lower is better) --
by the *author's own* measurement, EXL3 tracks the reference more closely than
TrellisMX here. Caveat: this specifically compares KV-cache formats (FP8 vs NVFP4
KV) in their harness, not a clean isolate of the weight codec, and they say so
themselves ("does not isolate the codec"). Not treated as conclusive either way.

## Decision: two tracks, sized to what's actually testable

**Track A (primary goal, do this for real): carrier-only fit & performance.**
`local-inference-lab/GLM-5.3-Flash-NVFP4` is a real, novel-to-us NVFP4 checkpoint
independent of the blocked TrellisMX overlay -- deploy it standalone on our TP2
fleet exactly the way we already deploy `LibertAIDAI`/`RedHatAI`'s NVFP4 builds,
and get real numbers: does it load, is context 192-256K / floor 128K reachable,
TTFT/prefill/decode/concurrency vs. our v19-indexercompat EXL3 production baseline,
and -- given `LibertAIDAI/GLM-5.3-Flash-NVFP4` (a *different* publisher, still
referenced in our own `recipes/glm-5.3-flash-nvfp4-vllm.yaml`) had a confirmed,
reproduced ModelOpt token-corruption bug that's *why* we're on EXL3/RedHatAI today
-- does *this* NVFP4 build reproduce it too? That's a genuinely open, cheap-to-test
question our existing CJK/emoji reproduction methodology already answers directly.
This is the real "can we make this fit" answer the checkpoint's carrier can give us,
even though the TrellisMX-specific routed-expert weights themselves cannot run here.

**Track B (hypothesis exploration, bounded, does not block Track A): does the P8
kernel itself even run on SM121a silicon, independent of the TP4 problem?** The
`b12x` kernels are pure Python (Triton-JIT, not precompiled CUDA/PTX for a fixed
arch) -- unlike a raw `.cu` AOT build, a Triton kernel recompiles per-GPU at
runtime, so the `!= (12, 0)` guard may be conservative rather than a true hardware
wall. Cheap, isolated test: patch just that one guard, instantiate
`P8NativeTPMoE` directly with `world_size=1, tp_rank=0` and synthetic weights on a
single GB10 (bypassing vLLM's TP launcher and the TP4 problem entirely), and see
whether it JIT-compiles and produces numerically sane output. This does NOT attempt
to solve TP4 (2 GPUs vs. required 4 is a separate, much larger problem with no
found workaround) -- it only answers "is SM121a silicon-compatible at all," which
is useful signal regardless of whether TP4 is ever solved. If this microbenchmark
fails, that's a second, independent confirmation the routed-expert path is closed
on this hardware. If it succeeds, it's a real, interesting finding worth writing up
even though it doesn't unlock production use today.

## Track B result (2026-09-10)

Real, bounded investigation completed -- see `evidence/track-b-sm121a-kernel-probe/`.

**Hypothesis correction first**: the kernel is not pure Triton as guessed. The
decode-path GEMM (`p8_h128_fc1.py`, what `P8NativeTPMoE` actually dispatches
to for small-M/decode batches -- the dominant serving path) is written in
NVIDIA's CUTLASS Python DSL, hand-selecting a specific tensor-core MMA
instruction (`mma.sync.aligned.kind::mxf8f6f4...m16n8k32`) whose own docstring
says `"""SM120 block-scaled QMMA..."""`. Triton is only used for auxiliary
scale-reformat utilities, not the main compute path. A sibling file
(`trellis_materialize.py`, the K2/K3 uniform-rate path, not the K4/K5 path
this checkpoint uses) imports genuine SM90/Hopper-only WGMMA helpers -- a
real non-portable pocket elsewhere in the package, off our tested path.

**The `!= (12, 0)` guard exists only in the vLLM adapter** (`trellismx.py`) --
zero other capability checks anywhere in the kernel or its compiler.

**Unexpected corroborating find**: this project's own production Docker image
already has `b12x` v1.2.6 installed, whose package docstring reads *"consumer-
Blackwell (SM120/SM121) kernels"* and ships a GB10-specific tuning profile
(`nvidia.gb10.48sm.json.gz`) -- GB10 is a first-class target for this kernel
family generally, independent of TrellisMX specifically.

**Real hardware test**: vendored the full `b12x` source, built a synthetic
schema-valid TP4-rank0 sidecar (shapes/metadata reverse-engineered from the
source, not the real trellis codebook -- we don't have that without the 177 GB
sidecars we decided not to download), and invoked `P8NativeTPMoE` directly on
the idle GB10, bypassing vLLM's adapter and its guard entirely (note:
`world_size=1` is rejected by the constructor -- `world_size=4, tp_rank=0` is
the correct way to exercise a single rank locally, a real fix to this
session's own test design, not just what HYPOTHESIS.md originally proposed).

**Result: it JIT-compiled and executed cleanly** -- no compile error, no
illegal-instruction fault, no launch error, confirmed `(12, 1)` device
capability inside the container, correct output shape `(1, 4096)` bf16. But
output was all-NaN with both random and maximally-tame synthetic weights --
can't distinguish "our synthetic weights never satisfy the real codebook"
from "a genuine numerical bug," and this probe can't resolve that without the
real trellis-coded weights.

**Verdict**: the SM120 guard looks like exactly the overly-conservative
author choice the hypothesis suspected, not a true ISA-level wall -- this is
not a fundamental hardware incompatibility. Numerical correctness is
genuinely unconfirmed (inconclusive, not negative). **Does not change the
primary finding**: TP4 is still hard-baked and unsolved independent of this
result -- even perfect SM121a compatibility doesn't get us from 2 physical
GPUs to the 4 the sidecar files and kernel invocation require.

## What "done" looks like

- Track A: a real sparkrun recipe (`glm-5.3-flash-nvfp4-experiment01-trellismx-carrier.yaml`),
  built, booted, tinyGLM-gated, real-checkpoint validated, with TTFT/prefill/decode/
  concurrency numbers at 128K and 192-256K against the current v19-indexercompat
  baseline, plus a clear yes/no on the corruption-bug question.
- Track B: a short, evidence-backed note in `evidence/` on whether the P8 kernel
  JIT-compiles and executes correctly on SM121a in isolation, independent of
  whether it's ever deployable end-to-end.
- This file updated with actual results, not just plans, before the experiment is
  called closed.
