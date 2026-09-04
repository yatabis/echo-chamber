"""Generate an official MLX-LM oracle for unequal resident-state batching."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from qwen35_gdn_layer_parity import sha256_file

SCHEMA_VERSION = 1
PROMPT_PREFIX = (
    "これは2インスタンス同時生成の推論性能測定です。ツールは使わず、"
    "指定された固定文字列だけを独立した行へ繰り返してください。\n"
    "各行には前置き、番号、説明、後書き、省略記号を加えないでください。\n"
    "固定文字列: native-parallel-instance-"
)
EXTRA_HISTORY_TOKENS = 9


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-steps", type=int, default=32)
    return parser.parse_args()


def cache_arrays(caches: list[Any]) -> list[mx.array]:
    return [array for cache in caches for array in cache.state]


def prefill(model: Any, token_ids: list[int]) -> list[Any]:
    caches = model.make_cache()
    logits = model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    mx.eval(logits, cache_arrays(caches))
    return caches


def merge_caches(caches: list[list[Any]]) -> list[Any]:
    return [
        caches[0][layer_index].merge(
            [row[layer_index] for row in caches]
        )
        for layer_index in range(len(caches[0]))
    ]


def extract_cache_row(caches: list[Any], row: int) -> list[Any]:
    return [cache.extract(row) for cache in caches]


def named_state_tensors(
    caches: list[Any],
    layer_classes: list[str],
) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    for layer_index, (cache, layer_class) in enumerate(
        zip(caches, layer_classes, strict=True)
    ):
        first, second = cache.state
        if first is None or second is None:
            raise RuntimeError(f"layer {layer_index} cache is incomplete")
        if layer_class == "gdn":
            tensors[f"layer.{layer_index:02}.gdn.convolution"] = first
            tensors[f"layer.{layer_index:02}.gdn.recurrent"] = second
        else:
            tensors[f"layer.{layer_index:02}.attention.keys"] = first
            tensors[f"layer.{layer_index:02}.attention.values"] = second
    return tensors


def run(
    model_path: Path,
    output_directory: Path,
    generation_steps: int,
) -> dict[str, Any]:
    if generation_steps <= 0:
        raise ValueError("generation steps must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(str(model_path))
    prompt_token_rows = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{PROMPT_PREFIX}{instance}",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        for instance in "ABC"
    ]
    if len({len(tokens) for tokens in prompt_token_rows}) != 1:
        raise RuntimeError("resident oracle prompts do not have equal token lengths")
    history_token_rows = [list(tokens) for tokens in prompt_token_rows]
    for row in range(1, 3):
        source = prompt_token_rows[row]
        history_token_rows[row].extend(
            source[1 + index % (len(source) - 2)]
            for index in range(EXTRA_HISTORY_TOKENS)
        )
    history_lengths = [len(tokens) for tokens in history_token_rows]
    if not history_lengths[0] < history_lengths[1] == history_lengths[2]:
        raise RuntimeError(
            f"expected A shorter than equal-length B/C, observed {history_lengths}"
        )

    serial_caches = [prefill(model, tokens) for tokens in history_token_rows]
    batch_cache = merge_caches(serial_caches[:2])
    continuation_token_ids = [tokens[-1] for tokens in prompt_token_rows[:2]]
    logits = model(
        mx.array(continuation_token_ids, dtype=mx.int32)[:, None],
        cache=batch_cache,
    )
    generated = []
    for _ in range(generation_steps):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        generated.append(token)
        logits = model(token[:, None], cache=batch_cache)
        mx.eval(token, logits)
    expected_generated_tokens = mx.stack(generated, axis=1)
    expected_rows = [extract_cache_row(batch_cache, row) for row in range(2)]
    state_tensors = [
        named_state_tensors(
            caches,
            ["gdn" if layer.is_linear else "attention" for layer in model.layers],
        )
        for caches in expected_rows
    ]
    mx.eval(expected_generated_tokens, *[list(tensors.values()) for tensors in state_tensors])

    state_paths = [
        output_directory / "expected-row-0.safetensors",
        output_directory / "expected-row-1.safetensors",
    ]
    for path, tensors in zip(state_paths, state_tensors, strict=True):
        mx.save_safetensors(str(path), tensors)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_type": json.loads((model_path / "config.json").read_text())["model_type"],
        "config_sha256": sha256_file(model_path / "config.json"),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "history_token_rows": history_token_rows[:2],
        "continuation_token_ids": continuation_token_ids,
        "generation_steps": generation_steps,
        "expected_generated_token_rows": expected_generated_tokens.tolist(),
        "expected_state_files": [path.name for path in state_paths],
        "expected_state_sha256": [sha256_file(path) for path in state_paths],
        "expected_state_tensor_counts": [len(tensors) for tensors in state_tensors],
    }
    manifest_path = output_directory / "resident-batch.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    arguments = parse_arguments()
    print(
        json.dumps(
            run(
                arguments.model,
                arguments.output_dir,
                arguments.generation_steps,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
