"""Check append-only Cognitive suffixes against the admitted Qwen Jinja template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from qwen35_chat_template_parity import to_qwen_messages, to_qwen_tools


def runtime_exchanges() -> list[dict[str, Any]]:
    """Represent committed Memory/Emotion observations, absent from Main's tools."""
    return [
        {"type": "tool_call", "origin": "runtime", "call_id": "memory",
         "tool_name": "search_memory", "input": '{"query":"予定の変更"}'},
        {"type": "tool_result", "call_id": "memory",
         "output": '{"success":true,"results":[{"content":"予定は中止"}]}'},
        {"type": "tool_call", "origin": "runtime", "call_id": "emotion",
         "tool_name": "update_emotion",
         "input": '{"valence":0.2,"arousal":0.3,"labels":["calm","確認"]}'},
        {"type": "tool_result", "call_id": "emotion", "output": '{"success":true}'},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "tokenizer_config.json").read_text())
    tools = [{"name": "inspect_note", "description": "Read one note.",
              "input_schema": {"type": "object", "properties": {}}, "strict": True}]
    cases = []
    for pending in [True, False]:
        for cognitive in [True, False]:
            name = f"{'tool' if pending else 'message'}_then_{'cognitive' if cognitive else 'retry'}"
            prefix_input = [{"role": "user", "content": "予定を確認してください。"}]
            if pending:
                prefix_input.append({"type": "tool_call", "call_id": "main",
                                     "tool_name": "inspect_note", "input": "{}"})
                suffix = [{"type": "tool_result", "call_id": "main", "output": "done"}]
            else:
                prefix_input.append({"role": "assistant", "content": "内容を確認します。"})
                suffix = []
            if cognitive:
                suffix.extend(runtime_exchanges())
            template_options = {"tools": to_qwen_tools(tools), "tokenize": False,
                                "enable_thinking": False}
            prefix = tokenizer.apply_chat_template(
                to_qwen_messages(prefix_input), add_generation_prompt=False, **template_options
            )
            # The engine commits EOS itself; the following newline belongs to the suffix.
            assert prefix.endswith(tokenizer.eos_token + "\n")
            committed = prefix[:-1]
            full = tokenizer.apply_chat_template(
                to_qwen_messages(prefix_input + suffix), add_generation_prompt=True,
                **template_options,
            )
            assert full.startswith(committed), name
            rendered = full[len(committed):]
            encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
            committed_tokens, suffix_tokens, full_tokens = map(encode, (committed, rendered, full))
            assert committed_tokens[-1] == tokenizer.eos_token_id
            assert committed_tokens + suffix_tokens == full_tokens, name
            cases.append({
                "name": name, "continuation": True, "input": suffix, "tools": [],
                "rendered": rendered, "token_ids": suffix_tokens,
                "prefix_input": prefix_input, "prefix_tools": tools,
                "committed_rendered": committed, "full_rendered": full,
                "committed_token_ids": committed_tokens, "full_token_ids": full_tokens,
            })
    manifest = {
        "schema_version": 2,
        "chat_template_sha256": hashlib.sha256(config["chat_template"].encode()).hexdigest(),
        "eos_token_id": tokenizer.eos_token_id,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "exact_prefix_and_token_concatenation": True}))


if __name__ == "__main__":
    main()
