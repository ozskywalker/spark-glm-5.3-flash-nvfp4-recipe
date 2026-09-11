"""Build a synthetic, schema-valid P8 full-coupled TP4-rank0 sidecar for K4.

Shapes/metadata are reverse-engineered from:
  - b12x/moe/_shared/trellismx/p8_native_kernel.py (P8NativeTPMoE.__init__)
  - b12x/moe/_shared/trellismx/p8_coupled_scales.py (validate_coupled_component,
    validate_scale_component, rank_local_coupled_signs)

Not semantically valid trellis-coded weights (random bits) -- this is a
silicon-compatibility probe (does the kernel JIT and execute), not an
accuracy test.
"""
import sys
sys.path.insert(0, "/workspace/vendor")

import hashlib
import secrets

import torch
from safetensors.torch import save_file

from b12x.moe._shared.trellismx.p8_coupled_scales import (
    COUPLED_SCHEMA,
    COMPOSITION_TARGET,
    COUPLED_COMPONENT,
    COUPLED_CAST_ORDER,
    SIGN_GENERATOR,
    SIGN_DRAW,
    TRANSFORM_ID,
    tensor_sha256,
    rank_local_coupled_signs,
)

WORLD_SIZE = 4
RANK = 0
LAYER = 3
BITS = 4
EXPERTS = 288
HIDDEN = 4096
INTERMEDIATE = 512  # 2048 // world_size, required by small_m_scheduler check

STREAM_WORDS = 16 * BITS

torch.manual_seed(0)

w13_trellis = torch.randint(
    -(2**15), 2**15 - 1,
    (2, EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, STREAM_WORDS),
    dtype=torch.int16,
)
w2_trellis = torch.randint(
    -(2**15), 2**15 - 1,
    (EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, STREAM_WORDS),
    dtype=torch.int16,
)
# UE8M0 codes: keep in a sane exponent range (not all 0xFF) to avoid
# deliberately-reserved sentinel codes.
w13_scale_ue8m0 = torch.randint(96, 160, (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8)
w2_scale_ue8m0 = torch.randint(96, 160, (EXPERTS, HIDDEN, INTERMEDIATE // 32), dtype=torch.uint8)

gate_up_suh_fp16 = torch.empty(HIDDEN, dtype=torch.float16).uniform_(0.5, 1.5)
intermediate_scales_fp16 = torch.empty(EXPERTS, 3 * INTERMEDIATE, dtype=torch.float16).uniform_(0.5, 1.5)
down_svh_fp16 = torch.empty(HIDDEN, dtype=torch.float16).uniform_(0.5, 1.5)

signs = rank_local_coupled_signs(intermediate=INTERMEDIATE, rank=RANK, world_size=WORLD_SIZE, draw=SIGN_DRAW)

source_design_sha256 = secrets.token_hex(32)
transform_sha256 = secrets.token_hex(32)

local_atoms = INTERMEDIATE // 32

metadata = {
    "schema": COUPLED_SCHEMA,  # tp4 already, world_size=4
    "layer": str(LAYER),
    "rank": str(RANK),
    "world_size": str(WORLD_SIZE),
    "bits": str(BITS),
    "alphabet": "e4m3",
    "scale": "ue8m0-k32",
    "law": "procedural-mcg-alpha2",
    "ldlq": "false",
    "source_design_sha256": source_design_sha256,
    "encoder_transform_sha256": transform_sha256,
    "component": COUPLED_COMPONENT,
    "composition_target": COMPOSITION_TARGET,
    "boundary": COMPOSITION_TARGET,
    "cast_order": COUPLED_CAST_ORDER,
    "full_coupled": "true",
    "h512": "normalized-sylvester-512-v1",
    "h128": "normalized-sylvester-128-v1",
    "transform_id": TRANSFORM_ID,
    "fc1_interleave": "slot0-atom32-slot1-atom32-v1",
    "sign_generator": SIGN_GENERATOR,
    "sign_draw": str(SIGN_DRAW),
    "sign_pre_axis": "1",
    "sign_post_axis": "2",
    "activation": "silu-cap10",
    "global_intermediate": str(INTERMEDIATE * WORLD_SIZE),
    "local_atom_begin": str(RANK * local_atoms),
    "tp_slice": "contiguous-atom32-v1",
    "gate_up_suh_shared": "true",
    "down_svh_shared": "true",
    "coupled_signs_shared": "true",
    "signed_scales": "true",
    "sha256_gate_up_suh_fp16": tensor_sha256(gate_up_suh_fp16),
    "sha256_intermediate_scales_fp16": tensor_sha256(intermediate_scales_fp16),
    "sha256_down_svh_fp16": tensor_sha256(down_svh_fp16),
    "sha256_coupled_signs_fp16": tensor_sha256(signs),
}

tensors = {
    "w13_trellis": w13_trellis,
    "w2_trellis": w2_trellis,
    "w13_scale_ue8m0": w13_scale_ue8m0,
    "w2_scale_ue8m0": w2_scale_ue8m0,
    "gate_up_suh_fp16": gate_up_suh_fp16,
    "intermediate_scales_fp16": intermediate_scales_fp16,
    "down_svh_fp16": down_svh_fp16,
}

out_path = "/workspace/synthetic_sidecar.safetensors"
save_file(tensors, out_path, metadata=metadata)
print("wrote", out_path)
print("transform_sha256", transform_sha256)
with open("/workspace/transform_sha256.txt", "w") as f:
    f.write(transform_sha256)
