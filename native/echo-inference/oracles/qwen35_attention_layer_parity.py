"""Generate a real Qwen3.5 full-attention decoder-layer oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from qwen35_decoder_layer_parity import trace_moe
from qwen35_gdn_layer_parity import (
    deterministic_inputs,
    max_array_difference,
    sha256_file,
)

SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=3)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--continuation-length", type=int, default=8)
    return parser.parse_args()


def trace_attention_continuation(
    attention: Any,
    inputs: mx.array,
    initial_keys: mx.array,
    initial_values: mx.array,
    offset: int,
) -> dict[str, mx.array]:
    batch_size, sequence_length, _ = inputs.shape

    q_projection = attention.q_proj(inputs)
    q_projection_heads = q_projection.reshape(
        batch_size,
        sequence_length,
        attention.num_attention_heads,
        -1,
    )
    queries, gate_heads = mx.split(q_projection_heads, 2, axis=-1)
    gate = gate_heads.reshape(batch_size, sequence_length, -1)
    key_projection = attention.k_proj(inputs)
    value_projection = attention.v_proj(inputs)

    normalized_queries = attention.q_norm(queries)
    normalized_keys = attention.k_norm(
        key_projection.reshape(
            batch_size,
            sequence_length,
            attention.num_key_value_heads,
            -1,
        )
    )
    queries = normalized_queries.transpose(0, 2, 1, 3)
    new_keys = normalized_keys.transpose(0, 2, 1, 3)
    new_values = value_projection.reshape(
        batch_size,
        sequence_length,
        attention.num_key_value_heads,
        -1,
    ).transpose(0, 2, 1, 3)

    rotated_queries = attention.rope(queries, offset=offset)
    rotated_new_keys = attention.rope(new_keys, offset=offset)
    keys = mx.concatenate([initial_keys, rotated_new_keys], axis=2)
    values = mx.concatenate([initial_values, new_values], axis=2)

    attention_heads = mx.fast.scaled_dot_product_attention(
        rotated_queries,
        keys,
        values,
        scale=attention.scale,
        mask="causal",
    )
    attention_flat = attention_heads.transpose(0, 2, 1, 3).reshape(
        batch_size,
        sequence_length,
        -1,
    )
    gated_attention = attention_flat * mx.sigmoid(gate)
    output = attention.o_proj(gated_attention)

    return {
        "trace.attention.attention_flat": attention_flat,
        "trace.attention.attention_heads": attention_heads,
        "trace.attention.gate": gate,
        "trace.attention.gated_attention": gated_attention,
        "trace.attention.key_projection": key_projection,
        "trace.attention.keys": keys,
        "trace.attention.new_keys": new_keys,
        "trace.attention.new_values": new_values,
        "trace.attention.normalized_keys": normalized_keys,
        "trace.attention.normalized_queries": normalized_queries,
        "trace.attention.output": output,
        "trace.attention.q_projection": q_projection,
        "trace.attention.queries": queries,
        "trace.attention.rotated_new_keys": rotated_new_keys,
        "trace.attention.rotated_queries": rotated_queries,
        "trace.attention.value_projection": value_projection,
        "trace.attention.values": values,
    }


def trace_decoder_continuation(
    decoder_layer: Any,
    inputs: mx.array,
    initial_keys: mx.array,
    initial_values: mx.array,
    offset: int,
) -> dict[str, mx.array]:
    normalized_input = decoder_layer.input_layernorm(inputs)
    attention_trace = trace_attention_continuation(
        decoder_layer.self_attn,
        normalized_input,
        initial_keys,
        initial_values,
        offset,
    )
    attention_output = attention_trace["trace.attention.output"]
    post_attention_hidden = inputs + attention_output
    normalized_hidden = decoder_layer.post_attention_layernorm(post_attention_hidden)
    moe_trace = trace_moe(decoder_layer.mlp, normalized_hidden)
    output = post_attention_hidden + moe_trace["trace.moe_output"]

    return {
        "trace.attention_output": attention_output,
        "trace.decoder_output": output,
        "trace.normalized_hidden": normalized_hidden,
        "trace.normalized_input": normalized_input,
        "trace.post_attention_hidden": post_attention_hidden,
        **attention_trace,
        **moe_trace,
    }


def run(
    model_path: Path,
    output_directory: Path,
    layer_index: int,
    prefix_length: int,
    continuation_length: int,
) -> dict[str, Any]:
    if prefix_length <= 0 or continuation_length <= 1:
        raise ValueError("prefix must be positive and continuation must exceed one")

    output_directory.mkdir(parents=True, exist_ok=True)
    fixture_path = output_directory / "attention-layer.safetensors"
    manifest_path = output_directory / "attention-layer.manifest.json"
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())

    model, _ = load(str(model_path))
    decoder_layer = model.layers[layer_index]
    if decoder_layer.is_linear:
        raise RuntimeError(f"layer {layer_index} is not a full-attention layer")
    if not hasattr(decoder_layer.mlp, "switch_mlp"):
        raise RuntimeError(f"layer {layer_index} is not a sparse MoE layer")

    prefix_input, continuation_input = deterministic_inputs(
        config["text_config"]["hidden_size"],
        prefix_length,
        continuation_length,
    )
    cache = KVCache()
    prefix_mask = create_attention_mask(prefix_input, cache)
    prefix_output = decoder_layer(prefix_input, mask=prefix_mask, cache=cache)
    initial_keys, initial_values = cache.state
    mx.eval(prefix_output, initial_keys, initial_values)

    continuation_mask = create_attention_mask(continuation_input, cache)
    direct_output = decoder_layer(
        continuation_input,
        mask=continuation_mask,
        cache=cache,
    )
    expected_keys, expected_values = cache.state
    trace = trace_decoder_continuation(
        decoder_layer,
        continuation_input,
        initial_keys,
        initial_values,
        prefix_length,
    )
    mx.eval(direct_output, expected_keys, expected_values, list(trace.values()))

    output_difference = max_array_difference(
        direct_output,
        trace["trace.decoder_output"],
    )
    keys_difference = max_array_difference(
        expected_keys,
        trace["trace.attention.keys"],
    )
    values_difference = max_array_difference(
        expected_values,
        trace["trace.attention.values"],
    )
    moe_difference = max_array_difference(
        decoder_layer.mlp(trace["trace.normalized_hidden"]),
        trace["trace.moe_output"],
    )
    if any(
        difference != 0.0
        for difference in (
            output_difference,
            keys_difference,
            values_difference,
            moe_difference,
        )
    ):
        raise RuntimeError(
            "manual trace changed official attention-layer semantics: "
            f"output={output_difference}, keys={keys_difference}, "
            f"values={values_difference}, moe={moe_difference}"
        )

    fixture = {
        "continuation_input": continuation_input,
        "expected_keys": expected_keys,
        "expected_output": direct_output,
        "expected_values": expected_values,
        "initial_keys": initial_keys,
        "initial_values": initial_values,
        "prefix_input": prefix_input,
        **trace,
    }
    mx.save_safetensors(str(fixture_path), fixture)

    text_config = config["text_config"]
    rope_parameters = text_config["rope_parameters"]
    default_quantization = config["quantization"]
    layer_prefix = f"language_model.model.layers.{layer_index}.mlp"
    router_quantization = default_quantization[layer_prefix + ".gate"]
    shared_gate_quantization = default_quantization[
        layer_prefix + ".shared_expert_gate"
    ]
    rotary_factor = float(rope_parameters["partial_rotary_factor"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(model_path.resolve()),
        "model_type": config["model_type"],
        "config_sha256": sha256_file(config_path),
        "layer_index": layer_index,
        "prefix_length": prefix_length,
        "continuation_length": continuation_length,
        "input_dtype": str(continuation_input.dtype),
        "mask_mode": str(continuation_mask),
        "sorted_expert_path": continuation_length
        * int(text_config["num_experts_per_tok"])
        >= 64,
        "dimensions": {
            "hidden_size": int(text_config["hidden_size"]),
            "attention_heads": int(text_config["num_attention_heads"]),
            "key_value_heads": int(text_config["num_key_value_heads"]),
            "head_dim": int(text_config["head_dim"]),
            "rotary_dim": int(text_config["head_dim"] * rotary_factor),
            "expert_count": int(text_config["num_experts"]),
            "experts_per_token": int(text_config["num_experts_per_tok"]),
            "moe_intermediate_size": int(text_config["moe_intermediate_size"]),
            "shared_expert_intermediate_size": int(
                text_config["shared_expert_intermediate_size"]
            ),
        },
        "rope": {
            "base": float(rope_parameters["rope_theta"]),
            "scale": 1.0,
            "traditional": False,
        },
        "quantization": {
            "default": {
                "group_size": int(default_quantization["group_size"]),
                "bits": int(default_quantization["bits"]),
                "mode": str(default_quantization["mode"]),
            },
            "router": {
                "group_size": int(router_quantization["group_size"]),
                "bits": int(router_quantization["bits"]),
                "mode": str(
                    router_quantization.get("mode", default_quantization["mode"])
                ),
            },
            "shared_expert_gate": {
                "group_size": int(shared_gate_quantization["group_size"]),
                "bits": int(shared_gate_quantization["bits"]),
                "mode": str(
                    shared_gate_quantization.get(
                        "mode",
                        default_quantization["mode"],
                    )
                ),
            },
        },
        "norm_topk_prob": bool(text_config.get("norm_topk_prob", True)),
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_tensor_count": len(fixture),
        "official_trace_output_max_absolute_difference": output_difference,
        "official_trace_keys_max_absolute_difference": keys_difference,
        "official_trace_values_max_absolute_difference": values_difference,
        "official_trace_moe_max_absolute_difference": moe_difference,
        "declared_rms_norm_eps": float(text_config["rms_norm_eps"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    arguments = parse_arguments()
    result = run(
        arguments.model.resolve(),
        arguments.output_dir.resolve(),
        arguments.layer_index,
        arguments.prefix_length,
        arguments.continuation_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
