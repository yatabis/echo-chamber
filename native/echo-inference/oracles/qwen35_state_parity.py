"""Real Qwen3.5 KV+GDN checkpoint continuation oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def token_digest(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tokenize(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if not isinstance(encoded, list) or not encoded:
        raise RuntimeError("tokenizer returned no token IDs")
    return [int(token) for token in encoded]


def cache_arrays(cache: list[Any]) -> list[mx.array]:
    flattened = tree_flatten([layer.state for layer in cache])
    return [value for _, value in flattened if isinstance(value, mx.array)]


def cache_logical_nbytes(cache: list[Any]) -> int:
    return sum(int(array.nbytes) for array in cache_arrays(cache))


def max_array_difference(left: mx.array, right: mx.array) -> float:
    difference = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    return float(difference.item())


def max_cache_difference(left: list[Any], right: list[Any]) -> float:
    left_arrays = cache_arrays(left)
    right_arrays = cache_arrays(right)
    if len(left_arrays) != len(right_arrays):
        raise RuntimeError(
            f"cache arity mismatch: direct={len(left_arrays)} restored={len(right_arrays)}"
        )
    if not left_arrays:
        raise RuntimeError("checkpoint cache contains no arrays")
    differences = [
        max_array_difference(left_array, right_array)
        for left_array, right_array in zip(left_arrays, right_arrays, strict=True)
    ]
    return max(differences)


def require_hybrid_cache(cache: list[Any]) -> tuple[int, int, list[str]]:
    classes = [type(layer).__name__ for layer in cache]
    recurrent = classes.count("ArraysCache")
    full_attention = classes.count("KVCache")
    if recurrent == 0 or full_attention == 0:
        raise RuntimeError(f"expected hybrid cache, observed {classes}")
    return recurrent, full_attention, classes


def model_identity(model_path: Path, config: dict[str, Any]) -> dict[str, str]:
    weights = list(model_path.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("model directory has no safetensors weights")
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    return {
        "architecture": str(config["model_type"]),
        "config_digest": sha256_file(model_path / "config.json"),
        "weights_digest": sha256_files(weights),
        "tokenizer_digest": sha256_file(tokenizer_path),
        "template_digest": sha256_file(tokenizer_config_path),
    }


def run(model_path: Path, output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint.safetensors"
    manifest_path = output_directory / "checkpoint.manifest.json"
    result_path = output_directory / "result.json"

    config = json.loads((model_path / "config.json").read_text())
    model, tokenizer = load(str(model_path))
    prefix_tokens = tokenize(
        tokenizer,
        "E.C.H.O. inference-state checkpoint oracle prefix: "
        "Rin keeps one isolated hidden-state lineage.",
    )
    suffix_tokens = tokenize(
        tokenizer,
        " Continue from the exact committed boundary and answer with one token.",
    )

    direct_cache = model.make_cache()
    prefix_input = mx.array(prefix_tokens, dtype=mx.uint32)[None]
    model(prefix_input, cache=direct_cache)
    prefix_arrays = cache_arrays(direct_cache)
    mx.eval(prefix_arrays)
    recurrent_layers, full_attention_layers, layer_classes = require_hybrid_cache(
        direct_cache
    )

    identity = model_identity(model_path, config)
    boundary = {
        "lineage_token_count": len(prefix_tokens),
        "token_digest": token_digest(prefix_tokens),
    }
    embedded_metadata = {
        "echo_schema_version": str(SCHEMA_VERSION),
        "echo_instance_id": "oracle:qwen3.5",
        "echo_revision": "1",
        "echo_boundary_token_count": str(len(prefix_tokens)),
        "echo_boundary_token_digest": boundary["token_digest"],
        **{f"echo_model_{key}": value for key, value in identity.items()},
    }
    save_prompt_cache(str(checkpoint_path), direct_cache, embedded_metadata)

    suffix_input = mx.array(suffix_tokens, dtype=mx.uint32)[None]
    direct_logits = model(suffix_input, cache=direct_cache)
    direct_arrays = cache_arrays(direct_cache)
    mx.eval(direct_logits, direct_arrays)

    restored_cache, restored_metadata = load_prompt_cache(
        str(checkpoint_path), return_metadata=True
    )
    if restored_metadata != embedded_metadata:
        raise RuntimeError("embedded checkpoint metadata changed during reload")
    restored_logits = model(suffix_input, cache=restored_cache)
    restored_arrays = cache_arrays(restored_cache)
    mx.eval(restored_logits, restored_arrays)

    logits_max_difference = max_array_difference(direct_logits, restored_logits)
    cache_max_difference = max_cache_difference(direct_cache, restored_cache)
    direct_token = int(mx.argmax(direct_logits[0, -1]).item())
    restored_token = int(mx.argmax(restored_logits[0, -1]).item())

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": "oracle:qwen3.5",
        "revision": 1,
        "model": identity,
        "boundary": boundary,
        "cache": {
            "tensor_count": len(prefix_arrays),
            "logical_nbytes": cache_logical_nbytes(
                load_prompt_cache(str(checkpoint_path))
            ),
            "payload_digest": sha256_file(checkpoint_path),
            "recurrent_layer_count": recurrent_layers,
            "full_attention_layer_count": full_attention_layers,
            "layer_classes": layer_classes,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    result = {
        "model_path": str(model_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "prefix_token_count": len(prefix_tokens),
        "suffix_token_count": len(suffix_tokens),
        "checkpoint_sha256": manifest["cache"]["payload_digest"],
        "checkpoint_logical_nbytes": manifest["cache"]["logical_nbytes"],
        "checkpoint_tensor_count": manifest["cache"]["tensor_count"],
        "recurrent_layer_count": recurrent_layers,
        "full_attention_layer_count": full_attention_layers,
        "direct_greedy_token": direct_token,
        "restored_greedy_token": restored_token,
        "logits_max_absolute_difference": logits_max_difference,
        "final_cache_max_absolute_difference": cache_max_difference,
        "continuation_exact": (
            direct_token == restored_token
            and logits_max_difference == 0.0
            and cache_max_difference == 0.0
        ),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    arguments = parse_arguments()
    result = run(arguments.model.resolve(), arguments.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
