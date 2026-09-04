"""Compare Native long-prefill paths with the corresponding MLX-LM paths.

This oracle pairs with
``packages/model-evaluation/src/runners/native/chunked-prefill-parity.test.ts``.
It intentionally compares each Native execution shape with the same Python
execution shape. A single 4K execution and two 2K executions are not expected
to have bit-identical BF16 GDN state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache

TARGET_CONTEXT_TOKENS = 4_096
PROMPT_BASE_TOKENS = 108
PADDING_PHRASE_TOKENS = 3
PADDING_PHRASE = " state-cache-padding"
REMAINDER_TOKEN = " x"
PREFILL_CHUNK_SIZE_TOKENS = 2_048
INITIAL_GENERATION_TOKENS = 8
NEW_SESSION_GENERATION_TOKENS = 128
BENCHMARK_PREFIX = "\n".join(
    [
        "This is a deterministic context-length performance benchmark.",
        "Ignore every padding token between <padding> and </padding>.",
        "After </padding>, write the integers from 1 through 400 on separate lines.",
        "Every line must use exactly this format: 0001: context-curve-performance",
        "Use four zero-padded digits, no preface, no explanation, and no closing text.",
        "<padding>",
    ]
)
BENCHMARK_SUFFIX = "\n".join(
    ["</padding>", "Begin the numbered output now."]
)
NEW_SESSION_PROMPT = (
    "Summarize what matters when preserving a long-lived internal state "
    "across a fresh session."
)


def parse_arguments() -> argparse.Namespace:
    """Parse paths without assuming a particular local model installation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def benchmark_prompt() -> str:
    """Construct the prompt calibrated to 4K tokens by the paired live test."""

    padding_tokens = TARGET_CONTEXT_TOKENS - PROMPT_BASE_TOKENS
    repetitions, remainder_tokens = divmod(
        padding_tokens, PADDING_PHRASE_TOKENS
    )
    return (
        f"{BENCHMARK_PREFIX}"
        f"{PADDING_PHRASE * repetitions}"
        f"{REMAINDER_TOKEN * remainder_tokens}\n"
        f"{BENCHMARK_SUFFIX}"
    )


def chat_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Render the same non-thinking Qwen chat boundary as Native."""

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return [int(token) for token in encoded]


def cache_arrays(cache: list[Any]) -> list[mx.array]:
    """Return every material state array owned by a hybrid MLX-LM cache."""

    return [
        value
        for layer in cache
        for value in layer.state
        if isinstance(value, mx.array)
    ]


def evaluate(cache: list[Any], logits: mx.array | None = None) -> None:
    """Close the same materialization boundary used between Native chunks."""

    arrays = cache_arrays(cache)
    if logits is not None:
        arrays.append(logits)
    mx.eval(*arrays)


def feed(
    model: Any,
    cache: list[Any],
    token_ids: list[int],
    chunk_size: int,
) -> mx.array:
    """Execute a token sequence through fixed-size model calls."""

    logits = None
    for start in range(0, len(token_ids), chunk_size):
        inputs = mx.array(
            token_ids[start : start + chunk_size], dtype=mx.uint32
        )[None]
        logits = model(inputs, cache=cache)
        evaluate(cache, logits)
    if logits is None:
        raise RuntimeError("oracle input unexpectedly contained no tokens")
    return logits


def feed_one(
    model: Any, cache: list[Any], token: int
) -> mx.array:
    """Advance one generated token into the hybrid state."""

    logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    evaluate(cache, logits)
    return logits


def generate(
    model: Any,
    cache: list[Any],
    logits: mx.array,
    count: int,
) -> tuple[list[int], mx.array]:
    """Run deterministic argmax generation and advance every sampled token."""

    generated: list[int] = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        generated.append(token)
        logits = feed_one(model, cache, token)
    return generated, logits


def begin_new_session(model: Any, previous: list[Any]) -> list[Any]:
    """Retain GDN arrays while replacing every full-attention KV cache."""

    current = model.make_cache()
    for old_layer, new_layer in zip(previous, current, strict=True):
        if isinstance(old_layer, ArraysCache):
            new_layer.state = list(old_layer.state)
    return current


def token_digest(tokens: list[int]) -> str:
    """Return a compact identity for one generated token sequence."""

    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def first_token_divergence(left: list[int], right: list[int]) -> int | None:
    """Return the first unequal index, including unequal-length boundaries."""

    for index, (left_token, right_token) in enumerate(
        zip(left, right, strict=False)
    ):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def require_native_variant(result: dict[str, Any], mode: str) -> dict[str, Any]:
    """Select one uniquely named Native execution-shape result."""

    variants = [variant for variant in result["variants"] if variant["mode"] == mode]
    if len(variants) != 1:
        raise RuntimeError(
            f"expected exactly one Native variant named {mode}, observed {len(variants)}"
        )
    return variants[0]


def run(model_path: Path, native_result_path: Path) -> dict[str, Any]:
    """Run both MLX-LM shapes and compare them with their Native counterparts."""

    native_result = json.loads(native_result_path.read_text())
    model, tokenizer = load(str(model_path))
    initial_input = chat_token_ids(tokenizer, benchmark_prompt())
    new_session_input = chat_token_ids(tokenizer, NEW_SESSION_PROMPT)
    if len(initial_input) != TARGET_CONTEXT_TOKENS:
        raise RuntimeError(
            "benchmark prompt calibration changed: "
            f"expected {TARGET_CONTEXT_TOKENS}, observed {len(initial_input)}"
        )
    if len(new_session_input) != 29:
        raise RuntimeError(
            "new-session prompt calibration changed: "
            f"expected 29, observed {len(new_session_input)}"
        )
    eos_token = tokenizer.eos_token_id
    if not isinstance(eos_token, int):
        raise RuntimeError("tokenizer does not expose one integer EOS token")

    python_variants: dict[str, dict[str, list[int]]] = {}
    for mode, chunk_size in [
        ("single_execution", len(initial_input)),
        ("chunked_2k", PREFILL_CHUNK_SIZE_TOKENS),
    ]:
        cache = model.make_cache()
        logits = feed(model, cache, initial_input, chunk_size)
        initial_generated, logits = generate(
            model, cache, logits, INITIAL_GENERATION_TOKENS
        )
        feed_one(model, cache, eos_token)
        cache = begin_new_session(model, cache)
        logits = feed(model, cache, new_session_input, len(new_session_input))
        new_session_generated, _ = generate(
            model, cache, logits, NEW_SESSION_GENERATION_TOKENS
        )
        python_variants[mode] = {
            "initial": initial_generated,
            "new_session": new_session_generated,
        }

    comparisons: dict[str, Any] = {}
    for mode, python_variant in python_variants.items():
        native_variant = require_native_variant(native_result, mode)
        native_initial = [int(token) for token in native_variant["initialGeneratedTokens"]]
        native_new_session = [
            int(token) for token in native_variant["newSessionGeneratedTokens"]
        ]
        comparisons[mode] = {
            "initial_matches_native": python_variant["initial"] == native_initial,
            "new_session_matches_native": (
                python_variant["new_session"] == native_new_session
            ),
            "initial_token_sha256": token_digest(python_variant["initial"]),
            "new_session_token_sha256": token_digest(
                python_variant["new_session"]
            ),
        }

    single_new_session = python_variants["single_execution"]["new_session"]
    chunked_new_session = python_variants["chunked_2k"]["new_session"]
    divergence = first_token_divergence(
        single_new_session, chunked_new_session
    )
    all_paths_match = all(
        comparison["initial_matches_native"]
        and comparison["new_session_matches_native"]
        for comparison in comparisons.values()
    )
    return {
        "model_path": str(model_path.resolve()),
        "native_result_path": str(native_result_path.resolve()),
        "conditions": {
            "initial_input_tokens": len(initial_input),
            "new_session_input_tokens": len(new_session_input),
            "chunk_size_tokens": PREFILL_CHUNK_SIZE_TOKENS,
            "initial_generation_tokens": INITIAL_GENERATION_TOKENS,
            "new_session_generation_tokens": NEW_SESSION_GENERATION_TOKENS,
            "sampling": "argmax",
        },
        "comparisons": comparisons,
        "cross_shape_observation": {
            "new_session_first_token_divergence": divergence,
            "exact_prefix_tokens": (
                len(single_new_session) if divergence is None else divergence
            ),
            "single_token_at_divergence": (
                None if divergence is None else single_new_session[divergence]
            ),
            "chunked_token_at_divergence": (
                None if divergence is None else chunked_new_session[divergence]
            ),
        },
        "all_native_paths_match_python_reference": all_paths_match,
    }


def main() -> None:
    """Run the reference comparison and optionally retain its compact result."""

    arguments = parse_arguments()
    result = run(arguments.model.resolve(), arguments.native_result.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
