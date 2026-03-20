import type {
  ModelInputItem,
  ModelMessage,
  ModelOutputItem,
  ModelToolContract,
  ModelUsage,
} from '@echo-chamber/core/ports/model';

import type {
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseUsage,
} from 'openai/resources/responses/responses';

export interface FunctionToolDefinition {
  type: 'function';
  name: string;
  description: string;
  parameters: Record<string, unknown> | null;
  strict: boolean;
}

const EMPTY_RESPONSE_USAGE: ResponseUsage = {
  input_tokens: 0,
  input_tokens_details: { cached_tokens: 0 },
  output_tokens: 0,
  output_tokens_details: { reasoning_tokens: 0 },
  total_tokens: 0,
};

/**
 * `core` の tool contract を OpenAI function tool 定義へ変換する。
 *
 * @param tool provider 非依存の tool contract
 * @returns OpenAI Responses API に渡せる function tool 定義
 */
export function toFunctionToolDefinition(
  tool: ModelToolContract
): FunctionToolDefinition {
  return {
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: toFunctionParameters(tool.inputSchema),
    strict: tool.strict ?? false,
  };
}

/**
 * 任意の input schema を OpenAI function tool の `parameters` 形式へ寄せる。
 *
 * @param inputSchema provider-neutral contract が保持する任意の schema
 * @returns object schema の場合は shallow copy、それ以外は `null`
 */
export function toFunctionParameters(
  inputSchema: unknown
): Record<string, unknown> | null {
  if (inputSchema === null) {
    return null;
  }

  if (typeof inputSchema === 'object' && !Array.isArray(inputSchema)) {
    return Object.fromEntries(Object.entries(inputSchema));
  }

  return null;
}

/**
 * provider 非依存 input item を OpenAI Responses API の input item へ変換する。
 *
 * @param item `core` が扱う input item
 * @returns OpenAI Responses API に渡せる input item
 */
export function toResponseInputItem(item: ModelInputItem): ResponseInputItem {
  if (isModelMessage(item)) {
    return {
      role: item.role,
      content: item.content,
    };
  }

  if (item.type === 'tool_call') {
    return {
      type: 'function_call',
      call_id: item.callId,
      name: item.toolName,
      arguments: item.input,
    };
  }

  return {
    type: 'function_call_output',
    call_id: item.callId,
    output: item.output,
  };
}

/**
 * OpenAI 固有の usage を `core` の `ModelUsage` に正規化する。
 *
 * @param usage OpenAI Responses API の usage。未定義なら 0 usage 扱い
 * @returns provider 非依存の usage 集計値
 */
export function toModelUsage(usage: ResponseUsage | undefined): ModelUsage {
  const safeUsage = usage ?? EMPTY_RESPONSE_USAGE;
  const cachedInputTokens = safeUsage.input_tokens_details.cached_tokens;
  const totalInputTokens = safeUsage.input_tokens;

  return {
    cachedInputTokens,
    uncachedInputTokens: totalInputTokens - cachedInputTokens,
    totalInputTokens,
    outputTokens: safeUsage.output_tokens,
    reasoningTokens: safeUsage.output_tokens_details.reasoning_tokens,
    totalTokens: safeUsage.total_tokens,
  };
}

/**
 * OpenAI の output item を `core` の output item 列へ変換する。
 *
 * @param item OpenAI Responses API の output item
 * @returns `core` の session loop が扱える output item 配列
 */
export function toModelOutputItem(item: ResponseOutputItem): ModelOutputItem[] {
  if (item.type === 'function_call') {
    return [
      {
        type: 'tool_call',
        callId: item.call_id,
        toolName: item.name,
        input: item.arguments,
      },
    ];
  }

  if (item.type === 'message') {
    return [
      {
        type: 'message',
        role: 'assistant',
        content: extractOutputMessageText(item),
      },
    ];
  }

  return [];
}

/**
 * assistant message の content 配列から表示用テキストを抽出する。
 *
 * @param message OpenAI Responses API の assistant message
 * @returns `output_text` と `refusal` を結合した文字列
 */
export function extractOutputMessageText(
  message: ResponseOutputMessage
): string {
  return message.content
    .map((content) => {
      const contentType = content.type;
      switch (contentType) {
        case 'output_text':
          return content.text;
        case 'refusal':
          return `<refusal>${content.refusal}</refusal>`;
        default:
          throw new Error(
            `Unexpected contentType: ${contentType satisfies never}`
          );
      }
    })
    .join('\n\n');
}

/**
 * input item が通常メッセージかどうかを判定する。
 *
 * @param item provider 非依存 input item
 * @returns `role` を持つ通常メッセージなら `true`
 */
function isModelMessage(item: ModelInputItem): item is ModelMessage {
  return 'role' in item;
}
