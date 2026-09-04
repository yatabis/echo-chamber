"""Generate exact Qwen3.5 chat-template and tokenizer fixtures for E.C.H.O."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def fixture_cases() -> list[dict[str, Any]]:
    tools = [
        {
            "name": "check_notifications",
            "description": "Check pending notifications.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "finish_thinking",
            "description": "Finish this thinking session.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "next_wake_at": {"type": ["string", "null"]},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    return [
        {
            "name": "text_only_non_thinking",
            "input": [
                {
                    "role": "user",
                    "content": "1 + 1 はいくつですか？",
                }
            ],
            "tools": [],
        },
        {
            "name": "echo_startup_tool_round_trip",
            "input": [
                {
                    "role": "developer",
                    "content": "<persona>Test persona</persona>",
                },
                {
                    "role": "developer",
                    "content": (
                        "<runtime_context>\n"
                        "Current datetime: 2026年07月31日 12:00:00\n"
                        "</runtime_context>"
                    ),
                },
                {
                    "type": "tool_call",
                    "call_id": "check_notifications",
                    "tool_name": "check_notifications",
                    "input": "{}",
                },
                {
                    "type": "tool_result",
                    "call_id": "check_notifications",
                    "output": '{"notifications":[]}',
                },
            ],
            "tools": tools,
        },
        {
            "name": "assistant_text_and_multiple_tool_results",
            "input": [
                {
                    "role": "system",
                    "content": "Reply in Japanese.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "通知を確認してください。"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "確認します。",
                },
                {
                    "type": "tool_call",
                    "call_id": "check_notifications",
                    "tool_name": "check_notifications",
                    "input": ('{"channels":["discord","email"],"limit":20}'),
                },
                {
                    "type": "tool_result",
                    "call_id": "check_notifications",
                    "output": '{"discord":1}',
                },
                {
                    "type": "tool_result",
                    "call_id": "check_notifications",
                    "output": '{"email":0}',
                },
            ],
            "tools": tools,
        },
    ]


def to_qwen_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for item in input_items:
        item_type = item.get("type")
        if item_type == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": item["tool_name"],
                                "arguments": json.loads(item["input"]),
                            },
                        }
                    ],
                }
            )
        elif item_type == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "content": item["output"],
                }
            )
        else:
            messages.append(
                {
                    **item,
                    "role": "user" if item["role"] == "developer" else item["role"],
                }
            )
    return messages


def to_qwen_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        function = {
            "name": tool["name"],
            "description": tool["description"],
        }
        if isinstance(tool["input_schema"], dict):
            function["parameters"] = tool["input_schema"]
        function["strict"] = tool.get("strict", False)
        result.append({"type": "function", "function": function})
    return result


def main() -> None:
    arguments = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model,
        local_files_only=True,
    )
    tokenizer_config = json.loads(
        (arguments.model / "tokenizer_config.json").read_text()
    )
    chat_template = tokenizer_config["chat_template"]
    cases = []
    for case in fixture_cases():
        messages = to_qwen_messages(case["input"])
        tools = to_qwen_tools(case["tools"])
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        cases.append(
            {
                **case,
                "rendered": rendered,
                "token_ids": [int(token_id) for token_id in token_ids],
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "chat_template_sha256": sha256_text(chat_template),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "cases": cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
