"""Track B probe: does P8NativeTPMoE JIT-compile and run on real SM121a
(GB10) silicon, bypassing the vLLM adapter's `!= (12, 0)` guard entirely by
never going through trellismx.py -- we call the kernel class directly, which
has no device-capability check anywhere in its own source.
"""
import sys
sys.path.insert(0, "/workspace/vendor")

import traceback
import torch

print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(dev)
    print("device", torch.cuda.get_device_name(dev), "capability", cap)

with open("/workspace/transform_sha256.txt") as f:
    transform_sha256 = f.read().strip()

try:
    from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE
    print("import P8NativeTPMoE: OK")
except Exception:
    print("import P8NativeTPMoE: FAILED")
    traceback.print_exc()
    sys.exit(1)

EXPERTS = 288
HIDDEN = 4096
INTERMEDIATE = 512
TOPK = 8

try:
    runtime = P8NativeTPMoE(
        "/workspace/synthetic_sidecar.safetensors",
        device=torch.device("cuda:0"),
        tp_rank=0,
        world_size=4,
        layer=3,
        expected_design_sha256=None,
        expected_transform_sha256=transform_sha256,
        topk=TOPK,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        swiglu_limit=10.0,
        small_m_scheduler=True,
        fc1_tile_n=128,
        fuse_scratch_zero=True,
        prefill_chunk_tokens=0,
        grid_policy=True,
        fc1_warp_quant=False,
        fc1_broadcast_a=True,
    )
    print("P8NativeTPMoE construction: OK")
except Exception:
    print("P8NativeTPMoE construction: FAILED")
    traceback.print_exc()
    sys.exit(2)

torch.manual_seed(1)
m = 1
x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda:0")
topk_ids = torch.stack([
    torch.randperm(EXPERTS, device="cuda:0")[:TOPK] for _ in range(m)
]).to(torch.int64)
topk_weights = torch.softmax(torch.randn(m, TOPK, device="cuda:0"), dim=-1)

try:
    out = runtime(x, topk_weights, topk_ids)
    torch.cuda.synchronize()
    print("KERNEL CALL: OK (JIT compiled and executed)")
    print("output shape", tuple(out.shape), "dtype", out.dtype)
    print("isnan any:", torch.isnan(out).any().item())
    print("isinf any:", torch.isinf(out).any().item())
    print("mean", out.float().mean().item(), "std", out.float().std().item())
    print("max abs", out.float().abs().max().item())
except Exception:
    print("KERNEL CALL: FAILED")
    traceback.print_exc()
    sys.exit(3)

print("PROBE COMPLETE: SUCCESS")
