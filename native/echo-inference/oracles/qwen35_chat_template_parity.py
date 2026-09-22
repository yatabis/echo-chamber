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
                    "role": "system",
                    "content": (
                        "<persona>Test persona</persona>\n\n"
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
            "name": "cognitive_system_instruction_without_tools",
            "input": [
                {
                    "role": "system",
                    "content": "あなたは記憶モジュールです。共有履歴から検索クエリを返してください。\n\n現在日時: 2026年09月21日 12:00:00",
                },
            ],
            "tools": [],
        },
        {
            "name": "developer_only_system_fallback",
            "input": [{"role": "developer", "content": "Reply briefly."}],
            "tools": [],
        },
        {
            "name": "system_and_developer_with_tools",
            "input": [
                {"role": "system", "content": "Preserve the persona."},
                {"role": "developer", "content": [
                    {"type": "text", "text": "Inspect notifications. "},
                    {"type": "text", "text": "Then finish."},
                ]},
                {"role": "user", "content": "Begin."},
            ],
            "tools": tools,
        },
        {
            "name": "developer_and_system_autonomous_startup",
            "input": [
                {"role": "developer", "content": "First instruction."},
                {"role": "system", "content": "Second instruction."},
                {"role": "developer", "content": "Third instruction."},
            ],
            "tools": [],
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
            messages.append({
                **item,
                "role": "system" if item["role"] == "developer" else item["role"],
            })
    # Qwen has one instruction envelope; only the leading instruction block is
    # combined. A later instruction remains in place for the template to reject.
    prefix_length = 0
    for message in messages:
        if message["role"] != "system":
            break
        prefix_length += 1
    if prefix_length > 1:
        instructions = []
        for message in messages[:prefix_length]:
            content = message["content"]
            if not isinstance(content, str):
                if any(part["type"] != "text" for part in content):
                    raise ValueError("Native instruction content must be text")
                content = "".join(part["text"] for part in content)
            instructions.append(content)
        messages[:prefix_length] = [{
            "role": "system", "content": "\n\n".join(instructions),
        }]
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


def with_system_driven_startup(chat_template: str) -> str:
    """Extend only the official no-user guard for Core's autonomous startup.

    The system message anchors assistant-history thinking retention at index 0.
    No message, role, or rendered envelope is rewritten. Normal user conversations
    use the original guard path unchanged.
    """
    guard = "{{- raise_exception('No user query found in messages.') }}"
    if chat_template.count(guard) != 1:
        raise ValueError("Expected exactly one official no-user guard")
    return chat_template.replace(guard, """{%- if messages[0].role == 'system' %}
        {%- set ns.last_query_index = 0 %}
    {%- else %}
        {{- raise_exception('No user query found in messages.') }}
    {%- endif %}""")


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
            chat_template=with_system_driven_startup(chat_template),
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
        "template_extension": "system_driven_startup_v1",
        "input_normalization": "leading_developer_as_system_v1",
        "effective_chat_template_sha256": sha256_text(with_system_driven_startup(chat_template)),
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
