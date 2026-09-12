"""Generate a complete Qwen3.5 MoE forward/generation oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from qwen35_gdn_layer_parity import max_array_difference, sha256_file

SCHEMA_VERSION = 1
PROMPT = "1 + 1 はいくつですか？"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--continuation-length", type=int, default=2)
    parser.add_argument("--generation-steps", type=int, default=2)
    parser.add_argument("--second-prompt")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def cache_states(caches: list[Any]) -> list[tuple[mx.array, mx.array]]:
    states = []
    for cache in caches:
        first, second = cache.state
        if first is None or second is None:
            raise RuntimeError("model execution left one cache pair empty")
        states.append((first, second))
    return states


def flatten_states(
    states: list[tuple[mx.array, mx.array]],
) -> list[mx.array]:
    return [state for pair in states for state in pair]


def trace_model(
    model: Any,
    input_ids: mx.array,
    caches: list[Any],
) -> dict[str, Any]:
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(input_ids)
    embedding = hidden
    full_attention_mask = create_attention_mask(hidden, caches[text_model.fa_idx])
    gdn_mask = create_ssm_mask(hidden, caches[text_model.ssm_idx])

    layer_outputs = []
    for layer, cache in zip(text_model.layers, caches, strict=True):
        mask = gdn_mask if layer.is_linear else full_attention_mask
        hidden = layer(hidden, mask=mask, cache=cache)
        layer_outputs.append(hidden)

    normalized_hidden = text_model.norm(hidden)
    logits = model.language_model.lm_head(normalized_hidden)
    return {
        "embedding": embedding,
        "layer_outputs": layer_outputs,
        "normalized_hidden": normalized_hidden,
        "logits": logits,
        "gdn_mask": gdn_mask,
        "full_attention_mask": full_attention_mask,
    }


def state_differences(
    direct: list[tuple[mx.array, mx.array]],
    traced: list[tuple[mx.array, mx.array]],
) -> list[list[float]]:
    return [
        [
            max_array_difference(direct_value, traced_value)
            for direct_value, traced_value in zip(
                direct_pair,
                traced_pair,
                strict=True,
            )
        ]
        for direct_pair, traced_pair in zip(direct, traced, strict=True)
    ]


def add_state_fixture(
    fixture: dict[str, mx.array],
    phase: str,
    states: list[tuple[mx.array, mx.array]],
    layer_classes: list[str],
) -> None:
    for layer_index, (layer_class, (first, second)) in enumerate(
        zip(layer_classes, states, strict=True)
    ):
        if layer_class == "gdn":
            fixture[f"{phase}.layer.{layer_index}.expected_conv_state"] = first
            fixture[f"{phase}.layer.{layer_index}.expected_recurrent_state"] = second
        else:
            fixture[f"{phase}.layer.{layer_index}.expected_keys"] = first
            fixture[f"{phase}.layer.{layer_index}.expected_values"] = second


def add_trace_fixture(
    fixture: dict[str, mx.array],
    phase: str,
    trace: dict[str, Any],
) -> None:
    fixture[f"{phase}.expected_embedding"] = trace["embedding"]
    fixture[f"{phase}.expected_normalized_hidden"] = trace["normalized_hidden"]
    fixture[f"{phase}.expected_logits"] = trace["logits"]
    for layer_index, output in enumerate(trace["layer_outputs"]):
        fixture[f"{phase}.layer.{layer_index}.expected_output"] = output


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
    continuation_length: int,
    generation_steps: int,
    second_prompt: str | None,
    batch_size: int | None,
) -> dict[str, Any]:
    if continuation_length <= 1:
        raise ValueError("continuation length must exceed one")
    if generation_steps <= 1:
        raise ValueError("generation steps must exceed one")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch size must be positive")
    if batch_size is not None and second_prompt is not None:
        raise ValueError("batch size and second prompt are mutually exclusive")

    output_directory.mkdir(parents=True, exist_ok=True)
    fixture_path = output_directory / "full-model.safetensors"
    manifest_path = output_directory / "full-model.manifest.json"
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config["text_config"]

    model, tokenizer = load(str(model_path))
    prompts = (
        [PROMPT] * batch_size
        if batch_size is not None
        else [PROMPT] + ([second_prompt] if second_prompt is not None else [])
    )
    prompt_token_rows = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    prompt_lengths = {len(tokens) for tokens in prompt_token_rows}
    if len(prompt_lengths) != 1:
        raise RuntimeError(
            f"batch prompts must have equal token lengths, observed {sorted(prompt_lengths)}"
        )
    if len(prompt_token_rows[0]) <= continuation_length:
        raise RuntimeError("chat-template prompt is too short for the requested split")
    prefix_ids = mx.array(
        [tokens[:-continuation_length] for tokens in prompt_token_rows],
        dtype=mx.int32,
    )
    continuation_ids = mx.array(
        [tokens[-continuation_length:] for tokens in prompt_token_rows],
        dtype=mx.int32,
    )

    layer_classes = [
        "gdn" if layer.is_linear else "full_attention" for layer in model.layers
    ]
    expected_classes = [
        "full_attention"
        if (index + 1) % int(text_config["full_attention_interval"]) == 0
        else "gdn"
        for index in range(int(text_config["num_hidden_layers"]))
    ]
    if layer_classes != expected_classes:
        raise RuntimeError("loaded layer schedule differs from the model config")

    direct_caches = model.make_cache()
    direct_prefix_logits = model(prefix_ids, cache=direct_caches)
    direct_prefix_states = cache_states(direct_caches)
    mx.eval(direct_prefix_logits, flatten_states(direct_prefix_states))

    direct_continuation_logits = model(continuation_ids, cache=direct_caches)
    direct_continuation_states = cache_states(direct_caches)
    mx.eval(
        direct_continuation_logits,
        flatten_states(direct_continuation_states),
    )

    traced_caches = model.make_cache()
    prefix_trace = trace_model(model, prefix_ids, traced_caches)
    traced_prefix_states = cache_states(traced_caches)
    mx.eval(
        prefix_trace["logits"],
        prefix_trace["layer_outputs"],
        flatten_states(traced_prefix_states),
    )
    continuation_trace = trace_model(model, continuation_ids, traced_caches)
    traced_continuation_states = cache_states(traced_caches)
    mx.eval(
        continuation_trace["logits"],
        continuation_trace["layer_outputs"],
        flatten_states(traced_continuation_states),
    )

    prefix_logits_difference = max_array_difference(
        direct_prefix_logits,
        prefix_trace["logits"],
    )
    continuation_logits_difference = max_array_difference(
        direct_continuation_logits,
        continuation_trace["logits"],
    )
    prefix_state_differences = state_differences(
        direct_prefix_states,
        traced_prefix_states,
    )
    continuation_state_differences = state_differences(
        direct_continuation_states,
        traced_continuation_states,
    )
    all_trace_differences = [
        prefix_logits_difference,
        continuation_logits_difference,
        *(difference for pair in prefix_state_differences for difference in pair),
        *(difference for pair in continuation_state_differences for difference in pair),
    ]
    if any(difference != 0.0 for difference in all_trace_differences):
        raise RuntimeError(
            "manual full-model trace changed official semantics: "
            f"prefix_logits={prefix_logits_difference}, "
            f"continuation_logits={continuation_logits_difference}, "
            f"prefix_state_max={max(max(pair) for pair in prefix_state_differences)}, "
            "continuation_state_max="
            f"{max(max(pair) for pair in continuation_state_differences)}"
        )

    generated_tokens = []
    generation_logits = []
    current_logits = direct_continuation_logits
    for _ in range(generation_steps):
        token = mx.argmax(current_logits[:, -1, :], axis=-1)
        generated_tokens.append(token)
        current_logits = model(token[:, None], cache=direct_caches)
        generation_logits.append(current_logits)
        mx.eval(token, current_logits)
    expected_generated_tokens = mx.stack(generated_tokens, axis=1)
    final_states = cache_states(direct_caches)
    mx.eval(expected_generated_tokens, flatten_states(final_states))

    fixture = {
        "prefix.input_ids": prefix_ids,
        "continuation.input_ids": continuation_ids,
        "generation.expected_tokens": expected_generated_tokens,
    }
    add_trace_fixture(fixture, "prefix", prefix_trace)
    add_state_fixture(
        fixture,
        "prefix",
        direct_prefix_states,
        layer_classes,
    )
    add_trace_fixture(fixture, "continuation", continuation_trace)
    add_state_fixture(
        fixture,
        "continuation",
        direct_continuation_states,
        layer_classes,
    )
    for step, logits in enumerate(generation_logits):
        fixture[f"generation.step.{step}.expected_logits"] = logits
    add_state_fixture(
        fixture,
        "generation.final",
        final_states,
        layer_classes,
    )
    mx.save_safetensors(str(fixture_path), fixture)

    default_quantization = config["quantization"]
    router_parameters = [
        quantization_parameters(
            default_quantization,
            f"language_model.model.layers.{layer_index}.mlp.gate",
        )
        for layer_index in range(int(text_config["num_hidden_layers"]))
    ]
    shared_gate_parameters = [
        quantization_parameters(
            default_quantization,
            f"language_model.model.layers.{layer_index}.mlp.shared_expert_gate",
        )
        for layer_index in range(int(text_config["num_hidden_layers"]))
    ]
    if len({json.dumps(value, sort_keys=True) for value in router_parameters}) != 1:
        raise RuntimeError("router quantization differs across decoder layers")
    if (
        len({json.dumps(value, sort_keys=True) for value in shared_gate_parameters})
        != 1
    ):
        raise RuntimeError("shared-expert gate quantization differs across layers")

    rope_parameters = text_config["rope_parameters"]
    rotary_factor = float(rope_parameters["partial_rotary_factor"])
    generated_token_values = [
        int(token) for token in expected_generated_tokens[0].tolist()
    ]
    generated_token_rows = [
        [int(token) for token in row]
        for row in expected_generated_tokens.tolist()
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_type": config["model_type"],
        "config_sha256": sha256_file(config_path),
        "prompt": PROMPT,
        "prompt_token_ids": [int(token) for token in prompt_token_rows[0]],
        "batch_prompts": prompts,
        "batch_prompt_token_ids": [
            [int(token) for token in row] for row in prompt_token_rows
        ],
        "batch_size": len(prompt_token_rows),
        "prefix_length": int(prefix_ids.shape[1]),
        "continuation_length": int(continuation_ids.shape[1]),
        "generation_steps": generation_steps,
        "expected_generated_tokens": generated_token_values,
        "expected_generated_token_rows": generated_token_rows,
        "input_dtype": str(prefix_ids.dtype),
        "hidden_dtype": str(prefix_trace["embedding"].dtype),
        "logits_dtype": str(prefix_trace["logits"].dtype),
        "prefix_attention_mask_mode": str(prefix_trace["full_attention_mask"]),
        "prefix_gdn_mask_mode": str(prefix_trace["gdn_mask"]),
        "continuation_attention_mask_mode": str(
            continuation_trace["full_attention_mask"]
        ),
        "continuation_gdn_mask_mode": str(continuation_trace["gdn_mask"]),
        "layer_classes": layer_classes,
        "dimensions": {
            "hidden_size": int(text_config["hidden_size"]),
            "layer_count": int(text_config["num_hidden_layers"]),
            "full_attention_interval": int(text_config["full_attention_interval"]),
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
            "vocabulary_size": int(text_config["vocab_size"]),
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
        "tie_word_embeddings": bool(text_config["tie_word_embeddings"]),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_tensor_count": len(fixture),
        "official_trace_prefix_logits_max_absolute_difference": (
            prefix_logits_difference
        ),
        "official_trace_continuation_logits_max_absolute_difference": (
            continuation_logits_difference
        ),
        "official_trace_prefix_state_max_absolute_difference": max(
            max(pair) for pair in prefix_state_differences
        ),
        "official_trace_continuation_state_max_absolute_difference": max(
            max(pair) for pair in continuation_state_differences
        ),
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
        arguments.continuation_length,
        arguments.generation_steps,
        arguments.second_prompt,
        arguments.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
