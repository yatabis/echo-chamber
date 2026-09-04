"""Create a deterministic Rapid-MLX-compatible sampling parity fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.sample_utils import (
    apply_top_k,
    apply_top_p,
    make_presence_penalty,
)

VOCABULARY_SIZE = 64
PRESENCE_CONTEXT_SIZE = 4096
HISTORY_TOKENS = [
    3,
    5,
    5,
    8,
    13,
    21,
    34,
    55,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    1,
    1,
    2,
    3,
    5,
    8,
    13,
    21,
]
CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "seed": 0,
}
SEEDS = list(range(64))


def parse_args() -> argparse.Namespace:
    """Parse the output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def deterministic_logits() -> list[float]:
    """Return nontrivial float32 logits without a second random source."""

    return [
        math.sin(index * 0.73) * 2.4 + math.cos(index * 0.19) * 0.8 + index * 0.015
        for index in range(VOCABULARY_SIZE)
    ]


def build_fixture() -> dict[str, Any]:
    """Apply the installed MLX-LM operation order with explicit random keys."""

    logits_values = deterministic_logits()
    logits = mx.array([logits_values], dtype=mx.float32)
    history = mx.array(HISTORY_TOKENS, dtype=mx.int32)
    logits = make_presence_penalty(CONFIG["presence_penalty"], PRESENCE_CONTEXT_SIZE)(
        history, logits
    )
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = apply_top_p(logprobs, CONFIG["top_p"])
    logprobs = apply_top_k(logprobs, CONFIG["top_k"])

    cases = []
    for seed in SEEDS:
        token = mx.random.categorical(
            logprobs * (1 / CONFIG["temperature"]),
            key=mx.random.key(seed),
        )
        mx.eval(token)
        cases.append({"seed": seed, "expected_token": token.item()})

    if len({case["expected_token"] for case in cases}) < 4:
        raise RuntimeError("sampling fixture did not exercise enough candidates")

    return {
        "schema_version": 1,
        "oracle": "mlx_lm.sample_utils with explicit mlx random keys",
        "history_scope": "generated_output",
        "presence_context_size": PRESENCE_CONTEXT_SIZE,
        "logits": logits_values,
        "history_tokens": HISTORY_TOKENS,
        "config": CONFIG,
        "cases": cases,
    }


def main() -> None:
    """Write the deterministic fixture."""

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_fixture(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
