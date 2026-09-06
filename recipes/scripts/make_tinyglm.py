#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a weightless, kernel-faithful miniature GLM-5.3 checkpoint.

Adapted from Enntity/sparkglm's scripts/make_tinyglm.py (same license) -- the
idea and the tokenizer-fixture approach are theirs. The model_config() below
is NOT copied from their published example: it is built from THIS project's
real checkpoint config
(/models/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/
024db9f7e9871e8efdf21538ba55af7442be3cd5/config.json, read directly), because
that real config uses a richer nested schema (explicit
linear_attn_config.full_attn_layers/kda_layers index lists, per-layer
indexer_types, real architecture string "Glm5NextForConditionalGeneration",
not sparkglm's "Glm5NextForCausalLM") than their own published fixture
example -- copying theirs verbatim would risk silently testing a config
shape our installed vLLM doesn't actually use in production.

The checkpoint is used with vLLM's stock ``--load-format dummy``: every
weight tensor gets synthetic values from the declared shapes, no safetensors
are read. It preserves the real per-layer dimensions and kernel-selection
knobs (hidden_size, MLA/indexer geometry, EXL3 quant scheme, mHC) but
deliberately reduces layer count, routed-expert count, and vocabulary so a
boot costs seconds instead of the real ~8 minute / 164 GiB checkpoint load.
It is not a language model; never evaluate its output for quality, only for
"did it dispatch/crash/produce a well-formed response."

Deliberately dropped versus the real checkpoint (grouped-prefill is
prefill-only and doesn't need either): speculative decoding
(num_nextn_predict_layers=0 -- no --speculative-config). One dense-MLP layer
is kept (unlike sparkglm's all-sparse fixture) so both the dense and
sparse/MoE MLP code paths still get exercised.

Vision is KEPT, unlike sparkglm's own text-only fixture: this checkpoint's
real architecture (Glm5NextForConditionalGeneration) unconditionally wires
in vLLM's multimodal processor regardless of whether vision_config is
present in config.json (confirmed the hard way -- omitting it produced
`FileNotFoundError: processor_config.json` at engine startup, not a clean
text-only path). The real vision_config is small (hidden_size=1024) and
--load-format dummy makes its weight cost irrelevant, so it's kept verbatim
rather than reduced, and processor_config.json is copied byte-for-byte from
the real checkpoint. Multimodal special-token ids are placed at the top of
the (small) vocab range rather than reusing the real checkpoint's ids
(154830+), which don't fit a reduced vocabulary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_REVISION = "tinyglm-v1"

REAL_SNAPSHOT = Path(
    "/models/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/"
    "024db9f7e9871e8efdf21538ba55af7442be3cd5"
)
REAL_CONFIG = REAL_SNAPSHOT / "config.json"
REAL_PROCESSOR_CONFIG = REAL_SNAPSHOT / "processor_config.json"

# Placed at the top of the reduced vocab range (must fit vocab_size). The
# real checkpoint's own ids (154830+) don't fit a reduced vocabulary.
MM_TOKEN_IDS = {
    "image_start_token_id": -6,
    "image_end_token_id": -5,
    "image_token_id": -4,
    "video_start_token_id": -3,
    "video_end_token_id": -2,
    "video_token_id": -1,
}


def revision(experts: int, vocab_size: int, num_layers: int) -> str:
    return f"{SCHEMA_REVISION}-e{experts}-v{vocab_size}-l{num_layers}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tokenizer(vocab_size: int) -> dict:
    # Verbatim structure from sparkglm's make_tinyglm.py -- a minimal
    # WordLevel tokenizer is all vLLM's tokenizer-loading path needs for a
    # dispatch/crash fixture; its vocabulary content is never evaluated.
    special = ["<pad>", "<eos>", "<unk>", "<bos>"]
    words = [
        "the", "a", "to", "of", "and", "in", "is", "for",
        "test", "token", "prefill", "decode", "spark", "glm",
    ]
    tokens = special + words
    tokens.extend(f"t{i}" for i in range(vocab_size - len(tokens)))
    vocab = {token: index for index, token in enumerate(tokens)}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, token in enumerate(special)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def _load_real_config() -> dict:
    return json.loads(REAL_CONFIG.read_text())


def _load_real_text_config() -> dict:
    return _load_real_config()["text_config"]


def model_config(
    experts: int, vocab_size: int, num_layers: int, max_length: int
) -> dict:
    real = _load_real_text_config()

    if num_layers < 4:
        raise ValueError(
            "num_layers must be >= 4 to keep one full attention/indexer "
            "layer plus the real 3:1 KDA:full-attention ratio"
        )

    # Real layout repeats groups of 4: 3 KDA (linear_attention) layers then
    # 1 full-attention (deepseek_sparse_attention) layer. Preserve that
    # exact repeating pattern rather than inventing a different ratio.
    layer_types = []
    for i in range(num_layers):
        layer_types.append(
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
        )
    kda_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    full_attn_layers = [
        i for i, t in enumerate(layer_types) if t == "deepseek_sparse_attention"
    ]
    if not full_attn_layers:
        # num_layers < 4 already rejected above, but stay defensive if the
        # pattern above ever changes.
        full_attn_layers = [num_layers - 1]
        layer_types[-1] = "deepseek_sparse_attention"
        kda_layers = [i for i in range(num_layers - 1)]

    # Keep one dense MLP layer (unlike sparkglm's all-sparse fixture) so the
    # dense-MLP code path is exercised too, not just the fat-expert path.
    first_k_dense_replace = 1
    mlp_layer_types = [
        "dense" if i < first_k_dense_replace else "sparse" for i in range(num_layers)
    ]
    indexer_types = ["full"] * num_layers

    text_config = {
        "attention_bias": real["attention_bias"],
        "attention_dropout": real["attention_dropout"],
        "dtype": real["dtype"],
        "eos_token_id": [1],
        "first_k_dense_replace": first_k_dense_replace,
        "hc_eps": real["hc_eps"],
        "hc_mult": real["hc_mult"],
        "hc_sinkhorn_iters": real["hc_sinkhorn_iters"],
        "head_dim": real["head_dim"],
        "hidden_act": real["hidden_act"],
        "hidden_size": real["hidden_size"],
        "index_head_dim": real["index_head_dim"],
        "index_kpool": real["index_kpool"],
        "index_kpool_always_select_tail": real["index_kpool_always_select_tail"],
        "index_kpool_compress": real["index_kpool_compress"],
        "index_n_heads": real["index_n_heads"],
        "index_share_for_mtp_iteration": real["index_share_for_mtp_iteration"],
        "index_topk": real["index_topk"],
        "indexer_rope_interleave": real["indexer_rope_interleave"],
        "indexer_types": indexer_types,
        "initializer_range": real["initializer_range"],
        "intermediate_size": real["intermediate_size"],
        "kv_lora_rank": real["kv_lora_rank"],
        "layer_types": layer_types,
        "linear_attn_config": {
            "full_attn_layers": full_attn_layers,
            "gate_lower_bound": real["linear_attn_config"]["gate_lower_bound"],
            "head_dim": real["linear_attn_config"]["head_dim"],
            "kda_layers": kda_layers,
            "num_heads": real["linear_attn_config"]["num_heads"],
            "short_conv_kernel_size": real["linear_attn_config"][
                "short_conv_kernel_size"
            ],
        },
        "max_position_embeddings": max_length,
        "mhc": real["mhc"],
        "mla_use_nope": real["mla_use_nope"],
        "mlp_layer_types": mlp_layer_types,
        "model_type": real["model_type"],
        "moe_intermediate_size": real["moe_intermediate_size"],
        "moe_router_dtype": real["moe_router_dtype"],
        "n_group": real["n_group"],
        "n_routed_experts": experts,
        "n_shared_experts": real["n_shared_experts"],
        "norm_topk_prob": real["norm_topk_prob"],
        "num_attention_heads": real["num_attention_heads"],
        "num_experts_per_tok": real["num_experts_per_tok"],
        "num_hidden_layers": num_layers,
        "num_key_value_heads": real["num_key_value_heads"],
        "num_nextn_predict_layers": 0,  # no MTP draft in this fixture
        "output_router_logits": real["output_router_logits"],
        "pad_token_id": 0,
        "q_lora_rank": real["q_lora_rank"],
        "qk_head_dim": real["qk_head_dim"],
        "qk_nope_head_dim": real["qk_nope_head_dim"],
        "qk_rope_head_dim": real["qk_rope_head_dim"],
        "rms_norm_eps": real["rms_norm_eps"],
        "routed_scaling_factor": real["routed_scaling_factor"],
        "router_aux_loss_coef": real["router_aux_loss_coef"],
        "scoring_func": real["scoring_func"],
        "swiglu_limit": real["swiglu_limit"],
        "tie_word_embeddings": False,
        "topk_group": real["topk_group"],
        "topk_method": real["topk_method"],
        "use_cache": True,
        "v_head_dim": real["v_head_dim"],
        "vocab_size": vocab_size,
    }

    real_top = _load_real_config()
    # Real checkpoint's vision tower verbatim: small (hidden_size=1024) and
    # --load-format dummy makes its weight cost irrelevant, so there is no
    # reason to reduce it and introduce another guess about what vLLM's
    # multimodal processor requires.
    vision_config = dict(real_top["vision_config"])

    mm_token_ids = {name: vocab_size + offset for name, offset in MM_TOKEN_IDS.items()}
    if min(mm_token_ids.values()) < 0:
        raise ValueError("vocab_size must be large enough to hold the 6 MM token ids")

    config = {
        # Real checkpoint's architecture string -- NOT sparkglm's
        # "Glm5NextForCausalLM": ours is what our installed vLLM's model
        # registry actually maps for this checkpoint, confirmed by reading
        # the real config.json directly. This architecture unconditionally
        # wires in vLLM's multimodal processor (confirmed by omitting vision
        # entirely and hitting FileNotFoundError: processor_config.json at
        # startup) -- vision_config and processor_config.json are required,
        # not optional, for this specific architecture string.
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "tie_word_embeddings": False,
        "quantization_config": {
            "bits": 4,
            "codebook": "mcg",
            "head_bits": 16,
            "non_routed_dtype_policy": "official_source_native",
            "quant_method": "exl3",
            "scope": "glm53_routed_experts_only",
            "serving_reader_qualified": False,
            "version": "0.0.43",
        },
        "text_config": text_config,
        "vision_config": vision_config,
    }
    config.update(mm_token_ids)
    return config


def build(
    output: Path, experts: int, vocab_size: int, num_layers: int, max_length: int
) -> Path:
    if experts < 8:
        raise ValueError("experts must be at least the model's top-8 routing width")
    if vocab_size < 32:
        raise ValueError("vocab-size must be at least 32")
    if max_length < 2048:
        raise ValueError("max-length must be at least the sparse index top-k (2048)")

    snapshot_revision = revision(experts, vocab_size, num_layers)
    snapshot = output / "snapshots" / snapshot_revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (output / "refs").mkdir(parents=True, exist_ok=True)
    (output / "refs" / "main").write_text(snapshot_revision + "\n")

    config = model_config(experts, vocab_size, num_layers, max_length)
    _write_json(snapshot / "config.json", config)
    _write_json(snapshot / "quantization_config.json", config["quantization_config"])
    # Copied byte-for-byte from the real checkpoint -- required by
    # Glm5NextProcessor.from_pretrained() regardless of vocab/vision size.
    (snapshot / "processor_config.json").write_text(REAL_PROCESSOR_CONFIG.read_text())
    _write_json(snapshot / "tokenizer.json", _tokenizer(vocab_size))
    _write_json(
        snapshot / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": max_length,
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "chat_template": (
                "{% for message in messages %}{{ message['role'] + ': ' + "
                "message['content'] + '\\n' }}{% endfor %}assistant:"
            ),
        },
    )
    _write_json(
        snapshot / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
        },
    )
    _write_json(
        snapshot / "generation_config.json",
        {"do_sample": False, "pad_token_id": 0, "eos_token_id": 1},
    )
    (snapshot / "README.md").write_text(
        "# tinyGLM (grouped-prefill fixture)\n\n"
        "Synthetic, weightless GLM-5.3 kernel-integration fixture for "
        "validating EXL3_GROUPED_PREFILL_K4 dispatch before paying for the "
        "real 164 GiB checkpoint. Use only with `--load-format dummy`. "
        "Outputs are meaningless -- never evaluate them for quality.\n"
    )
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build(
        args.output, args.experts, args.vocab_size, args.num_layers, args.max_length
    )
    print(snapshot)


if __name__ == "__main__":
    main()
