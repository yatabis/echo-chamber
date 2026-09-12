"""Generate a real Qwen3.5 MoE decoder-layer oracle for the Rust engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache
from qwen35_gdn_layer_parity import (
    deterministic_inputs,
    max_array_difference,
    sha256_file,
    trace_continuation,
)

SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--continuation-length", type=int, default=4)
    return parser.parse_args()


def trace_moe(mlp: Any, inputs: mx.array) -> dict[str, mx.array]:
    router_logits = mlp.gate(inputs)
    router_probabilities = mx.softmax(router_logits, axis=-1, precise=True)

    top_k = mlp.top_k
    expert_indices = mx.argpartition(
        router_probabilities,
        kth=-top_k,
        axis=-1,
    )[..., -top_k:]
    expert_scores = mx.take_along_axis(
        router_probabilities,
        expert_indices,
        axis=-1,
    )
    if mlp.norm_topk_prob:
        expert_scores = expert_scores / expert_scores.sum(axis=-1, keepdims=True)

    switch_inputs = mx.expand_dims(inputs, (-2, -3))
    sorted_trace = {}
    sorted_indices = expert_indices
    sorted_inputs = switch_inputs
    inverse_order = None
    do_sort = expert_indices.size >= 64
    if do_sort:
        flattened_indices = expert_indices.flatten()
        sort_order = mx.argsort(flattened_indices)
        inverse_order = mx.argsort(sort_order)
        sorted_inputs = switch_inputs.flatten(0, -3)[sort_order // top_k]
        sorted_indices = flattened_indices[sort_order]
        sorted_trace = {
            "trace.expert_inverse_order": inverse_order,
            "trace.expert_sort_order": sort_order,
            "trace.expert_sorted_indices": sorted_indices,
            "trace.expert_sorted_inputs": sorted_inputs,
        }

    expert_up = mlp.switch_mlp.up_proj(
        sorted_inputs,
        sorted_indices,
        sorted_indices=do_sort,
    )
    expert_gate = mlp.switch_mlp.gate_proj(
        sorted_inputs,
        sorted_indices,
        sorted_indices=do_sort,
    )
    expert_activated = nn.silu(expert_gate) * expert_up
    expert_sorted_outputs = mlp.switch_mlp.down_proj(
        expert_activated,
        sorted_indices,
        sorted_indices=do_sort,
    )
    if do_sort:
        expert_outputs = mx.unflatten(
            expert_sorted_outputs[inverse_order],
            0,
            expert_indices.shape,
        )
    else:
        expert_outputs = expert_sorted_outputs
    expert_outputs = expert_outputs.squeeze(-2)
    routed_scaled = expert_outputs * expert_scores[..., None]
    routed_output = routed_scaled.sum(axis=-2)

    shared_gate_projection = mlp.shared_expert.gate_proj(inputs)
    shared_up_projection = mlp.shared_expert.up_proj(inputs)
    shared_activated = nn.silu(shared_gate_projection) * shared_up_projection
    shared_expert_output = mlp.shared_expert.down_proj(shared_activated)
    shared_gate_logits = mlp.shared_expert_gate(inputs)
    shared_gate = mx.sigmoid(shared_gate_logits)
    shared_output = shared_gate * shared_expert_output
    output = routed_output + shared_output

    return {
        "trace.expert_activated": expert_activated,
        "trace.expert_gate": expert_gate,
        "trace.expert_indices": expert_indices,
        "trace.expert_outputs": expert_outputs,
        "trace.expert_scores": expert_scores,
        "trace.expert_sorted_outputs": expert_sorted_outputs,
        "trace.expert_up": expert_up,
        "trace.moe_output": output,
        "trace.routed_output": routed_output,
        "trace.router_logits": router_logits,
        "trace.router_probabilities": router_probabilities,
        "trace.shared_activated": shared_activated,
        "trace.shared_expert_output": shared_expert_output,
        "trace.shared_gate": shared_gate,
        "trace.shared_gate_logits": shared_gate_logits,
        "trace.shared_gate_projection": shared_gate_projection,
        "trace.shared_output": shared_output,
        "trace.shared_up_projection": shared_up_projection,
        **sorted_trace,
    }


def trace_decoder_continuation(
    decoder_layer: Any,
    inputs: mx.array,
    initial_conv_state: mx.array,
    initial_recurrent_state: mx.array,
) -> dict[str, mx.array]:
    normalized_input = decoder_layer.input_layernorm(inputs)
    gdn_trace = trace_continuation(
        decoder_layer.linear_attn,
        normalized_input,
        initial_conv_state,
        initial_recurrent_state,
    )
    attention_output = gdn_trace["trace.output"]
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
        **{
            f"trace.gdn.{name.removeprefix('trace.')}": value
            for name, value in gdn_trace.items()
        },
        **moe_trace,
    }


def run(
    model_path: Path,
    output_directory: Path,
    layer_index: int,
    prefix_length: int,
    continuation_length: int,
) -> dict[str, Any]:
    if prefix_length <= 0 or continuation_length <= 0:
        raise ValueError("prefix and continuation lengths must both be positive")

    output_directory.mkdir(parents=True, exist_ok=True)
    fixture_path = output_directory / "decoder-layer.safetensors"
    manifest_path = output_directory / "decoder-layer.manifest.json"
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())

    model, _ = load(str(model_path))
    decoder_layer = model.layers[layer_index]
    if not decoder_layer.is_linear:
        raise RuntimeError(f"layer {layer_index} is not a GDN layer")
    if not hasattr(decoder_layer.mlp, "switch_mlp"):
        raise RuntimeError(f"layer {layer_index} is not a sparse MoE layer")

    prefix_input, continuation_input = deterministic_inputs(
        decoder_layer.linear_attn.hidden_size,
        prefix_length,
        continuation_length,
    )
    cache = ArraysCache(size=2)
    prefix_output = decoder_layer(prefix_input, cache=cache)
    initial_conv_state = cache[0]
    initial_recurrent_state = cache[1]
    mx.eval(prefix_output, initial_conv_state, initial_recurrent_state)

    direct_output = decoder_layer(continuation_input, cache=cache)
    expected_conv_state = cache[0]
    expected_recurrent_state = cache[1]
    trace = trace_decoder_continuation(
        decoder_layer,
        continuation_input,
        initial_conv_state,
        initial_recurrent_state,
    )
    mx.eval(
        direct_output,
        expected_conv_state,
        expected_recurrent_state,
        list(trace.values()),
    )

    output_difference = max_array_difference(
        direct_output,
        trace["trace.decoder_output"],
    )
    conv_state_difference = max_array_difference(
        expected_conv_state,
        trace["trace.gdn.conv_state"],
    )
    recurrent_state_difference = max_array_difference(
        expected_recurrent_state,
        trace["trace.gdn.recurrent_state"],
    )
    moe_difference = max_array_difference(
        decoder_layer.mlp(trace["trace.normalized_hidden"]),
        trace["trace.moe_output"],
    )
    if any(
        difference != 0.0
        for difference in (
            output_difference,
            conv_state_difference,
            recurrent_state_difference,
            moe_difference,
        )
    ):
        raise RuntimeError(
            "manual trace changed official decoder-layer semantics: "
            f"output={output_difference}, conv_state={conv_state_difference}, "
            f"recurrent_state={recurrent_state_difference}, moe={moe_difference}"
        )

    fixture = {
        "continuation_input": continuation_input,
        "expected_conv_state": expected_conv_state,
        "expected_output": direct_output,
        "expected_recurrent_state": expected_recurrent_state,
        "initial_conv_state": initial_conv_state,
        "initial_recurrent_state": initial_recurrent_state,
        "prefix_input": prefix_input,
        **trace,
    }
    mx.save_safetensors(str(fixture_path), fixture)

    text_config = config["text_config"]
    default_quantization = config["quantization"]
    layer_prefix = f"language_model.model.layers.{layer_index}.mlp"
    router_quantization = default_quantization[layer_prefix + ".gate"]
    shared_gate_quantization = default_quantization[
        layer_prefix + ".shared_expert_gate"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(model_path.resolve()),
        "model_type": config["model_type"],
        "config_sha256": sha256_file(config_path),
        "layer_index": layer_index,
        "prefix_length": prefix_length,
        "continuation_length": continuation_length,
        "input_dtype": str(continuation_input.dtype),
        "dimensions": {
            "hidden_size": int(text_config["hidden_size"]),
            "key_heads": int(text_config["linear_num_key_heads"]),
            "value_heads": int(text_config["linear_num_value_heads"]),
            "key_head_dim": int(text_config["linear_key_head_dim"]),
            "value_head_dim": int(text_config["linear_value_head_dim"]),
            "conv_kernel_size": int(text_config["linear_conv_kernel_dim"]),
            "conv_dim": int(
                2
                * text_config["linear_num_key_heads"]
                * text_config["linear_key_head_dim"]
                + text_config["linear_num_value_heads"]
                * text_config["linear_value_head_dim"]
            ),
            "expert_count": int(text_config["num_experts"]),
            "experts_per_token": int(text_config["num_experts_per_tok"]),
            "moe_intermediate_size": int(text_config["moe_intermediate_size"]),
            "shared_expert_intermediate_size": int(
                text_config["shared_expert_intermediate_size"]
            ),
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
        "sorted_expert_path": continuation_length
        * int(text_config["num_experts_per_tok"])
        >= 64,
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_tensor_count": len(fixture),
        "official_trace_output_max_absolute_difference": output_difference,
        "official_trace_conv_state_max_absolute_difference": conv_state_difference,
        "official_trace_recurrent_state_max_absolute_difference": (
            recurrent_state_difference
        ),
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
