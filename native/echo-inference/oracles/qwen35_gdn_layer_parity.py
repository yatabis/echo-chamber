"""Generate a real Qwen3.5 GDN-layer continuation oracle for the Rust engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update

SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--continuation-length", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def max_array_difference(left: mx.array, right: mx.array) -> float:
    difference = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    return float(difference.item())


def deterministic_inputs(
    hidden_size: int,
    prefix_length: int,
    continuation_length: int,
) -> tuple[mx.array, mx.array]:
    total_length = prefix_length + continuation_length
    values = mx.arange(total_length * hidden_size, dtype=mx.float32)
    values = ((values % 257) - 128) / 128
    values = values.reshape(1, total_length, hidden_size).astype(mx.bfloat16)
    return values[:, :prefix_length], values[:, prefix_length:]


def trace_continuation(
    layer: Any,
    inputs: mx.array,
    initial_conv_state: mx.array,
    initial_recurrent_state: mx.array,
) -> dict[str, mx.array]:
    batch_size, sequence_length, _ = inputs.shape
    qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(
        batch_size,
        sequence_length,
        layer.num_v_heads,
        layer.head_v_dim,
    )
    b = layer.in_proj_b(inputs)
    a = layer.in_proj_a(inputs)

    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    n_keep = layer.conv_kernel_size - 1
    conv_state = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_output = nn.silu(layer.conv1d(conv_input))

    q, k, v = [
        tensor.reshape(batch_size, sequence_length, heads, dimension)
        for tensor, heads, dimension in zip(
            mx.split(conv_output, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            strict=True,
        )
    ]
    inverse_scale = k.shape[-1] ** -0.5
    q = (inverse_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inverse_scale * mx.fast.rms_norm(k, None, 1e-6)

    recurrent_output, recurrent_state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        initial_recurrent_state,
        mask=None,
        use_kernel=True,
    )
    normalized_output = layer.norm(recurrent_output, z)
    output = layer.out_proj(normalized_output.reshape(batch_size, sequence_length, -1))
    beta = mx.sigmoid(b)
    decay = mx.exp(
        -mx.exp(layer.A_log.astype(mx.float32)) * nn.softplus(a + layer.dt_bias)
    )

    return {
        "trace.a": a,
        "trace.b": b,
        "trace.beta": beta,
        "trace.conv_input": conv_input,
        "trace.conv_output": conv_output,
        "trace.conv_state": conv_state,
        "trace.decay": decay,
        "trace.k": k,
        "trace.normalized_output": normalized_output,
        "trace.output": output,
        "trace.q": q,
        "trace.qkv": qkv,
        "trace.recurrent_output": recurrent_output,
        "trace.recurrent_state": recurrent_state,
        "trace.v": v,
        "trace.z": z,
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
    fixture_path = output_directory / "gdn-layer.safetensors"
    manifest_path = output_directory / "gdn-layer.manifest.json"
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())

    model, _ = load(str(model_path))
    decoder_layer = model.layers[layer_index]
    if not decoder_layer.is_linear:
        raise RuntimeError(f"layer {layer_index} is not a GDN layer")
    layer = decoder_layer.linear_attn

    prefix_input, continuation_input = deterministic_inputs(
        layer.hidden_size,
        prefix_length,
        continuation_length,
    )
    cache = ArraysCache(size=2)
    prefix_output = layer(prefix_input, cache=cache)
    initial_conv_state = cache[0]
    initial_recurrent_state = cache[1]
    mx.eval(prefix_output, initial_conv_state, initial_recurrent_state)

    direct_output = layer(continuation_input, cache=cache)
    expected_conv_state = cache[0]
    expected_recurrent_state = cache[1]
    trace = trace_continuation(
        layer,
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

    output_difference = max_array_difference(direct_output, trace["trace.output"])
    conv_state_difference = max_array_difference(
        expected_conv_state,
        trace["trace.conv_state"],
    )
    recurrent_state_difference = max_array_difference(
        expected_recurrent_state,
        trace["trace.recurrent_state"],
    )
    if any(
        difference != 0.0
        for difference in (
            output_difference,
            conv_state_difference,
            recurrent_state_difference,
        )
    ):
        raise RuntimeError(
            "manual trace changed official layer semantics: "
            f"output={output_difference}, conv_state={conv_state_difference}, "
            f"recurrent_state={recurrent_state_difference}"
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
            "hidden_size": layer.hidden_size,
            "key_heads": layer.num_k_heads,
            "value_heads": layer.num_v_heads,
            "key_head_dim": layer.head_k_dim,
            "value_head_dim": layer.head_v_dim,
            "conv_kernel_size": layer.conv_kernel_size,
            "conv_dim": layer.conv_dim,
        },
        "quantization": {
            "group_size": int(config["quantization"]["group_size"]),
            "bits": int(config["quantization"]["bits"]),
            "mode": str(config["quantization"]["mode"]),
        },
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_tensor_count": len(fixture),
        "official_trace_output_max_absolute_difference": output_difference,
        "official_trace_conv_state_max_absolute_difference": conv_state_difference,
        "official_trace_recurrent_state_max_absolute_difference": (
            recurrent_state_difference
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
        arguments.layer_index,
        arguments.prefix_length,
        arguments.continuation_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
