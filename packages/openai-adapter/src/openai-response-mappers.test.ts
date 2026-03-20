import { describe, expect, it } from 'vitest';

import type {
  ModelInputItem,
  ModelToolContract,
} from '@echo-chamber/core/ports/model';

import {
  extractOutputMessageText,
  toFunctionParameters,
  toFunctionToolDefinition,
  toModelOutputItem,
  toModelUsage,
  toResponseInputItem,
} from './openai-response-mappers';

import type {
  ResponseOutputItem,
  ResponseOutputMessage,
} from 'openai/resources/responses/responses';

describe('toFunctionParameters', () => {
  it('object schema を shallow copy して返す', () => {
    const schema = {
      type: 'object',
      properties: {
        thought: { type: 'string' },
      },
    };

    expect(toFunctionParameters(schema)).toEqual(schema);
    expect(toFunctionParameters(schema)).not.toBe(schema);
  });

  it('null は null のまま返す', () => {
    expect(toFunctionParameters(null)).toBeNull();
  });

  it('object 以外は null にする', () => {
    expect(toFunctionParameters('invalid')).toBeNull();
    expect(toFunctionParameters(['invalid'])).toBeNull();
  });
});

describe('toFunctionToolDefinition', () => {
  it('tool contract を OpenAI function tool 定義へ変換する', () => {
    const tool: ModelToolContract = {
      name: 'think_deeply',
      description: 'Deep thinking tool',
      inputSchema: {
        type: 'object',
      },
      strict: true,
    };

    expect(toFunctionToolDefinition(tool)).toEqual({
      type: 'function',
      name: 'think_deeply',
      description: 'Deep thinking tool',
      parameters: {
        type: 'object',
      },
      strict: true,
    });
  });

  it('strict 未指定時は false にフォールバックする', () => {
    const tool: ModelToolContract = {
      name: 'think_deeply',
      description: 'Deep thinking tool',
      inputSchema: null,
    };

    expect(toFunctionToolDefinition(tool).strict).toBe(false);
  });
});

describe('toResponseInputItem', () => {
  it('message を Responses API message に変換する', () => {
    const item: ModelInputItem = {
      role: 'user',
      content: 'hello',
    };

    expect(toResponseInputItem(item)).toEqual({
      role: 'user',
      content: 'hello',
    });
  });

  it('tool_call を function_call に変換する', () => {
    const item: ModelInputItem = {
      type: 'tool_call',
      callId: 'call_1',
      toolName: 'think_deeply',
      input: '{"thought":"test"}',
    };

    expect(toResponseInputItem(item)).toEqual({
      type: 'function_call',
      call_id: 'call_1',
      name: 'think_deeply',
      arguments: '{"thought":"test"}',
    });
  });

  it('tool_result を function_call_output に変換する', () => {
    const item: ModelInputItem = {
      type: 'tool_result',
      callId: 'call_1',
      output: '{"success":true}',
    };

    expect(toResponseInputItem(item)).toEqual({
      type: 'function_call_output',
      call_id: 'call_1',
      output: '{"success":true}',
    });
  });
});

describe('toModelUsage', () => {
  it('OpenAI usage を ModelUsage に正規化する', () => {
    expect(
      toModelUsage({
        input_tokens: 10,
        input_tokens_details: { cached_tokens: 4 },
        output_tokens: 7,
        output_tokens_details: { reasoning_tokens: 2 },
        total_tokens: 17,
      })
    ).toEqual({
      cachedInputTokens: 4,
      uncachedInputTokens: 6,
      totalInputTokens: 10,
      outputTokens: 7,
      reasoningTokens: 2,
      totalTokens: 17,
    });
  });

  it('usage undefined はゼロ usage にする', () => {
    expect(toModelUsage(undefined)).toEqual({
      cachedInputTokens: 0,
      uncachedInputTokens: 0,
      totalInputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      totalTokens: 0,
    });
  });
});

describe('extractOutputMessageText', () => {
  it('output_text と refusal を結合する', () => {
    const message: ResponseOutputMessage = {
      type: 'message',
      role: 'assistant',
      id: 'msg_1',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'Thinking complete',
          annotations: [],
        },
        {
          type: 'refusal',
          refusal: 'I cannot assist with that request.',
        },
      ],
    };

    expect(extractOutputMessageText(message)).toBe(
      'Thinking complete\n\n<refusal>I cannot assist with that request.</refusal>'
    );
  });

  it('未知の content type はエラーにする', () => {
    const message: ResponseOutputMessage = {
      type: 'message',
      role: 'assistant',
      id: 'msg_error',
      status: 'completed',
      content: [
        {
          type: 'unknown',
        },
      ] as unknown as ResponseOutputMessage['content'],
    };

    expect(() => extractOutputMessageText(message)).toThrowError(
      'Unexpected contentType: unknown'
    );
  });
});

describe('toModelOutputItem', () => {
  it('function_call を tool_call に変換する', () => {
    const item: ResponseOutputItem = {
      type: 'function_call',
      call_id: 'call_1',
      name: 'think_deeply',
      arguments: '{"thought":"test"}',
      status: 'completed',
    };

    expect(toModelOutputItem(item)).toEqual([
      {
        type: 'tool_call',
        callId: 'call_1',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
    ]);
  });

  it('message を assistant message に変換する', () => {
    const item: ResponseOutputItem = {
      type: 'message',
      role: 'assistant',
      id: 'msg_1',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'Thinking complete',
          annotations: [],
        },
      ],
    };

    expect(toModelOutputItem(item)).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: 'Thinking complete',
      },
    ]);
  });

  it('reasoning など message/function_call 以外は空配列にする', () => {
    const item = {
      type: 'reasoning',
      summary: [{ text: 'Need more context' }],
    } as unknown as ResponseOutputItem;

    expect(toModelOutputItem(item)).toEqual([]);
  });
});
