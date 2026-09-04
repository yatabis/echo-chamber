"""Generate a four-layer Qwen3.5 hybrid-state oracle for the Rust engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from qwen35_attention_layer_parity import (
    trace_decoder_continuation as trace_attention_decoder,
)
from qwen35_decoder_layer_parity import (
    trace_decoder_continuation as trace_gdn_decoder,
)
from qwen35_gdn_layer_parity import (
    deterministic_inputs,
    max_array_difference,
    sha256_file,
)

SCHEMA_VERSION = 1
BLOCK_START_LAYER = 0
BLOCK_LAYER_COUNT = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--continuation-length", type=int, default=8)
    return parser.parse_args()


def block_caches() -> list[ArraysCache | KVCache]:
    return [
        ArraysCache(size=2),
        ArraysCache(size=2),
        ArraysCache(size=2),
        KVCache(),
    ]


def block_masks(
    inputs: mx.array,
    caches: list[ArraysCache | KVCache],
) -> tuple[Any, Any]:
    return (
        create_ssm_mask(inputs, caches[0]),
        create_attention_mask(inputs, caches[-1]),
    )


def execute_official_block(
    layers: list[Any],
    inputs: mx.array,
    caches: list[ArraysCache | KVCache],
) -> tuple[mx.array, list[mx.array]]:
    gdn_mask, attention_mask = block_masks(inputs, caches)
    hidden = inputs
    outputs = []
    for layer, cache in zip(layers, caches, strict=True):
        mask = gdn_mask if layer.is_linear else attention_mask
        hidden = layer(hidden, mask=mask, cache=cache)
        outputs.append(hidden)
    return hidden, outputs


def quantization_parameters(
    default: dict[str, Any],
    module: str,
) -> dict[str, Any]:
    override = default.get(module, {})
    return {
        "group_size": int(override.get("group_size", default["group_size"])),
        "bits": int(override.get("bits", default["bits"])),
        "mode": str(override.get("mode", default["mode"])),
    }


def run(
    model_path: Path,
    output_directory: Path,
    prefix_length: int,
    continuation_length: int,
) -> dict[str, Any]:
    if prefix_length <= 0 or continuation_length <= 1:
        raise ValueError("prefix must be positive and continuation must exceed one")

    output_directory.mkdir(parents=True, exist_ok=True)
    fixture_path = output_directory / "hybrid-block.safetensors"
    manifest_path = output_directory / "hybrid-block.manifest.json"
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config["text_config"]

    model, _ = load(str(model_path))
    layers = model.layers[BLOCK_START_LAYER : BLOCK_START_LAYER + BLOCK_LAYER_COUNT]
    layer_classes = ["gdn" if layer.is_linear else "full_attention" for layer in layers]
    expected_classes = ["gdn", "gdn", "gdn", "full_attention"]
    if layer_classes != expected_classes:
        raise RuntimeError(
            f"first hybrid block must be {expected_classes}, observed {layer_classes}"
        )
    if any(not hasattr(layer.mlp, "switch_mlp") for layer in layers):
        raise RuntimeError("every layer in the first hybrid block must use sparse MoE")

    prefix_input, continuation_input = deterministic_inputs(
        int(text_config["hidden_size"]),
        prefix_length,
        continuation_length,
    )
    caches = block_caches()
    prefix_output, _ = execute_official_block(layers, prefix_input, caches)
    initial_states = [
        tuple(cache.state) if isinstance(cache, ArraysCache) else cache.state
        for cache in caches
    ]
    mx.eval(prefix_output, [state for pair in initial_states for state in pair])

    direct_output, direct_layer_outputs = execute_official_block(
        layers,
        continuation_input,
        caches,
    )
    expected_states = [
        tuple(cache.state) if isinstance(cache, ArraysCache) else cache.state
        for cache in caches
    ]

    manual_hidden = continuation_input
    manual_layer_outputs = []
    manual_states = []
    for relative_index, layer in enumerate(layers):
        initial_first, initial_second = initial_states[relative_index]
        if layer.is_linear:
            trace = trace_gdn_decoder(
                layer,
                manual_hidden,
                initial_first,
                initial_second,
            )
            manual_states.append(
                (
                    trace["trace.gdn.conv_state"],
                    trace["trace.gdn.recurrent_state"],
                )
            )
        else:
            trace = trace_attention_decoder(
                layer,
                manual_hidden,
                initial_first,
                initial_second,
                prefix_length,
            )
            manual_states.append(
                (
                    trace["trace.attention.keys"],
                    trace["trace.attention.values"],
                )
            )
        manual_hidden = trace["trace.decoder_output"]
        manual_layer_outputs.append(manual_hidden)

    mx.eval(
        direct_output,
        direct_layer_outputs,
        [state for pair in expected_states for state in pair],
        manual_layer_outputs,
        [state for pair in manual_states for state in pair],
    )

    layer_output_differences = [
        max_array_difference(direct, manual)
        for direct, manual in zip(
            direct_layer_outputs,
            manual_layer_outputs,
            strict=True,
        )
    ]
    state_differences = []
    for expected_pair, manual_pair in zip(
        expected_states,
        manual_states,
        strict=True,
    ):
        state_differences.append(
            [
                max_array_difference(expected, manual)
                for expected, manual in zip(
                    expected_pair,
                    manual_pair,
                    strict=True,
                )
            ]
        )
    output_difference = max_array_difference(direct_output, manual_hidden)
    all_differences = [
        output_difference,
        *layer_output_differences,
        *(difference for pair in state_differences for difference in pair),
    ]
    if any(difference != 0.0 for difference in all_differences):
        raise RuntimeError(
            "manual traces changed official hybrid-block semantics: "
            f"output={output_difference}, layers={layer_output_differences}, "
            f"states={state_differences}"
        )

    fixture = {
        "continuation_input": continuation_input,
        "expected_output": direct_output,
        "prefix_input": prefix_input,
    }
    current_input = continuation_input
    for relative_index, (layer_class, direct_layer_output) in enumerate(
        zip(layer_classes, direct_layer_outputs, strict=True)
    ):
        initial_first, initial_second = initial_states[relative_index]
        expected_first, expected_second = expected_states[relative_index]
        fixture[f"layer.{relative_index}.input"] = current_input
        fixture[f"layer.{relative_index}.expected_output"] = direct_layer_output
        if layer_class == "gdn":
            fixture[f"layer.{relative_index}.initial_conv_state"] = initial_first
            fixture[f"layer.{relative_index}.initial_recurrent_state"] = initial_second
            fixture[f"layer.{relative_index}.expected_conv_state"] = expected_first
            fixture[f"layer.{relative_index}.expected_recurrent_state"] = (
                expected_second
            )
        else:
            fixture[f"layer.{relative_index}.initial_keys"] = initial_first
            fixture[f"layer.{relative_index}.initial_values"] = initial_second
            fixture[f"layer.{relative_index}.expected_keys"] = expected_first
            fixture[f"layer.{relative_index}.expected_values"] = expected_second
        current_input = direct_layer_output
    mx.save_safetensors(str(fixture_path), fixture)

    default_quantization = config["quantization"]
    router_parameters = []
    shared_gate_parameters = []
    for layer_index in range(
        BLOCK_START_LAYER,
        BLOCK_START_LAYER + BLOCK_LAYER_COUNT,
    ):
        mlp_prefix = f"language_model.model.layers.{layer_index}.mlp"
        router_parameters.append(
            quantization_parameters(default_quantization, mlp_prefix + ".gate")
        )
        shared_gate_parameters.append(
            quantization_parameters(
                default_quantization,
                mlp_prefix + ".shared_expert_gate",
            )
        )
    if len({json.dumps(value, sort_keys=True) for value in router_parameters}) != 1:
        raise RuntimeError("router quantization differs within the first hybrid block")
    if (
        len({json.dumps(value, sort_keys=True) for value in shared_gate_parameters})
        != 1
    ):
        raise RuntimeError(
            "shared-expert gate quantization differs within the first hybrid block"
        )

    rope_parameters = text_config["rope_parameters"]
    rotary_factor = float(rope_parameters["partial_rotary_factor"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(model_path.resolve()),
        "model_type": config["model_type"],
        "config_sha256": sha256_file(config_path),
        "block_start_layer": BLOCK_START_LAYER,
        "block_layer_count": BLOCK_LAYER_COUNT,
        "layer_classes": layer_classes,
        "prefix_length": prefix_length,
        "continuation_length": continuation_length,
        "input_dtype": str(continuation_input.dtype),
        "gdn_mask_mode": str(block_masks(continuation_input, caches)[0]),
        "attention_mask_mode": str(block_masks(continuation_input, caches)[1]),
        "sorted_expert_path": continuation_length
        * int(text_config["num_experts_per_tok"])
        >= 64,
        "dimensions": {
            "hidden_size": int(text_config["hidden_size"]),
            "gdn_key_heads": int(text_config["linear_num_key_heads"]),
            "gdn_value_heads": int(text_config["linear_num_value_heads"]),
            "gdn_key_head_dim": int(text_config["linear_key_head_dim"]),
            "gdn_value_head_dim": int(text_config["linear_value_head_dim"]),
            "gdn_conv_kernel_size": int(text_config["linear_conv_kernel_dim"]),
            "attention_heads": int(text_config["num_attention_heads"]),
            "key_value_heads": int(text_config["num_key_value_heads"]),
            "attention_head_dim": int(text_config["head_dim"]),
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
            "router": router_parameters[0],
            "shared_expert_gate": shared_gate_parameters[0],
        },
        "norm_topk_prob": bool(text_config.get("norm_topk_prob", True)),
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_tensor_count": len(fixture),
        "official_trace_output_max_absolute_difference": output_difference,
        "official_trace_layer_output_max_absolute_differences": (
            layer_output_differences
        ),
        "official_trace_state_max_absolute_differences": state_differences,
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
        arguments.prefix_length,
        arguments.continuation_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
