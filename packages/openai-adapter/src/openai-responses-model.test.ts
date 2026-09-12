import OpenAI from 'openai';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import type { ModelRequest } from '@echo-chamber/core/ports/model';

import {
  formatBlock,
  formatFunctionCall,
  formatFunctionCallOutput,
  formatInputItem,
  formatModelOutputContent,
  formatMessage,
  formatOutputItem,
  OpenAIResponsesModel,
} from './openai-responses-model';

import type {
  EasyInputMessage,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
} from 'openai/resources/responses/responses';

const mockCreateResponse = vi.fn();

vi.mock('openai', () => {
  // Vitest 4 requires mocks invoked with `new` to use a constructable implementation.
  function MockOpenAI(): {
    responses: { create: typeof mockCreateResponse };
  } {
    return {
      responses: {
        create: mockCreateResponse,
      },
    };
  }

  return {
    default: vi.fn(MockOpenAI),
  };
});

const mockEmit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
const mockEvents: EchoEventPort = {
  emit: mockEmit,
};

const thinkDeeplyTool = {
  name: 'think_deeply',
  description: 'Deep thinking tool',
  inputSchema: {
    type: 'object',
    properties: {
      thought: {
        type: 'string',
      },
    },
    required: ['thought'],
    additionalProperties: false,
  },
  strict: true,
};

describe('OpenAIResponsesModel', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('SDK retry設定と各HTTP attemptのadmission hookをcallerが指定できる', async () => {
    const beforeRequest = vi.fn();
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{}'));
    new OpenAIResponsesModel({
      apiKey: 'test-key',
      maxRetries: 0,
      beforeRequest,
    });

    expect(OpenAI).toHaveBeenCalledWith(
      expect.objectContaining({
        apiKey: 'test-key',
        maxRetries: 0,
      })
    );
    const constructorCalls = vi.mocked(OpenAI).mock.calls as unknown as [
      { fetch?: typeof fetch },
    ][];
    const constructorOptions = constructorCalls[0]?.[0];
    if (constructorOptions?.fetch === undefined) {
      throw new Error('Expected guarded OpenAI fetch');
    }
    expect(typeof constructorOptions.fetch).toBe('function');
    await constructorOptions.fetch('https://api.openai.test', {});

    expect(beforeRequest).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fetchMock.mockRestore();
  });

  it('createResponse は provider-neutral request を Responses API 形式へ変換する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      events: mockEvents,
    });
    const request: ModelRequest = {
      input: [
        {
          role: 'developer',
          content: 'You are helpful.',
        },
        {
          type: 'tool_call',
          callId: 'call_123',
          toolName: 'think_deeply',
          input: '{"thought":"test"}',
        },
        {
          type: 'tool_result',
          callId: 'call_123',
          output: '{"success":true}',
        },
      ],
      tools: [thinkDeeplyTool],
      previousResponseToken: 'response_prev',
    };

    mockCreateResponse.mockResolvedValue({
      output: [],
    });

    await model.createResponse(request);

    expect(mockCreateResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        input: [
          {
            role: 'developer',
            content: 'You are helpful.',
          },
          {
            type: 'function_call',
            call_id: 'call_123',
            name: 'think_deeply',
            arguments: '{"thought":"test"}',
          },
          {
            type: 'function_call_output',
            call_id: 'call_123',
            output: '{"success":true}',
          },
        ],
        model: 'gpt-5.6',
        parallel_tool_calls: true,
        previous_response_id: 'response_prev',
        reasoning: {
          effort: 'none',
        },
        store: true,
        stream: false,
        tool_choice: 'auto',
        tools: [
          {
            type: 'function',
            name: 'think_deeply',
            description: 'Deep thinking tool',
            parameters: thinkDeeplyTool.inputSchema,
            strict: true,
          },
        ],
        truncation: 'auto',
      })
    );
    expect(mockEmit).toHaveBeenCalledWith({
      type: 'model.provider.warning',
      category: 'model',
      severity: 'warn',
      streams: ['system', 'analysis'],
      summary: 'Response usage information is undefined',
      payload: {
        provider: 'openai.responses',
        model: 'gpt-5.6',
        turnIndex: undefined,
        code: 'missing_usage',
      },
    });
  });

  it('createResponse は object でない inputSchema を parameters: null に正規化する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
    });

    mockCreateResponse.mockResolvedValue({
      output: [],
      usage: {
        input_tokens: 0,
        input_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 0,
        },
        output_tokens: 0,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 0,
      },
    });

    await model.createResponse({
      input: [],
      tools: [
        {
          ...thinkDeeplyTool,
          inputSchema: 'invalid-schema',
        },
      ],
    });

    expect(mockCreateResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        tools: [
          expect.objectContaining({
            name: 'think_deeply',
            parameters: null,
          }),
        ],
      })
    );
  });

  it('constructor で指定した reasoning effort を Responses API に渡す', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      reasoningEffort: 'low',
    });

    mockCreateResponse.mockResolvedValue({
      output: [],
      usage: {
        input_tokens: 0,
        input_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 0,
        },
        output_tokens: 0,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 0,
      },
    });

    await model.createResponse({
      input: [],
      tools: [],
    });

    expect(mockCreateResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        reasoning: {
          effort: 'low',
        },
      })
    );
  });

  it('provider-neutral strict JSON Schema と output/deadline 制約を渡す', async () => {
    const model = new OpenAIResponsesModel({ apiKey: 'test-key' });
    const signal = new AbortController().signal;
    const schema = {
      type: 'object',
      properties: { answer: { type: 'string' } },
      required: ['answer'],
      additionalProperties: false,
    };

    mockCreateResponse.mockResolvedValue({
      output: [],
      usage: {
        input_tokens: 0,
        input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
        output_tokens: 0,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 0,
      },
    });

    await model.generate({
      input: [],
      tools: [],
      responseFormat: {
        type: 'json_schema',
        name: 'answer_contract',
        strict: true,
        schema,
      },
      maxOutputTokens: 512,
      signal,
    });

    expect(mockCreateResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        max_output_tokens: 512,
        text: {
          format: {
            type: 'json_schema',
            name: 'answer_contract',
            strict: true,
            schema,
          },
          verbosity: 'medium',
        },
      }),
      { signal }
    );
  });

  it('generate は OpenAI response を core model response へ変換する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      events: mockEvents,
    });

    mockCreateResponse.mockResolvedValue({
      id: 'response_123',
      output: [
        {
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
        },
        {
          type: 'function_call',
          call_id: 'call_123',
          name: 'think_deeply',
          arguments: '{"thought":"test"}',
          status: 'completed',
        },
      ],
      usage: {
        input_tokens: 10,
        input_tokens_details: {
          cached_tokens: 4,
          cache_write_tokens: 2,
        },
        output_tokens: 7,
        output_tokens_details: { reasoning_tokens: 2 },
        total_tokens: 17,
      },
    });

    const response = await model.generate({
      input: [
        {
          role: 'user',
          content: 'hello',
        },
      ],
      tools: [thinkDeeplyTool],
      turnIndex: 2,
    });

    expect(response).toEqual({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: 'Thinking complete',
        },
        {
          type: 'tool_call',
          callId: 'call_123',
          toolName: 'think_deeply',
          input: '{"thought":"test"}',
        },
      ],
      usage: {
        cachedInputTokens: 4,
        cacheWriteInputTokens: 2,
        uncachedInputTokens: 4,
        totalInputTokens: 10,
        outputTokens: 7,
        reasoningTokens: 2,
        totalTokens: 17,
      },
      responseToken: 'response_123',
    });
    const outputEvent = mockEmit.mock.calls.find(
      ([event]) => event.type === 'model.output.emitted'
    )?.[0];
    expect(outputEvent).toMatchObject({
      type: 'model.output.emitted',
      category: 'model',
      severity: 'info',
      streams: ['thought', 'analysis'],
      payload: {
        provider: 'openai.responses',
        model: 'gpt-5.6',
        turnIndex: 2,
        content: '*thinking: Thinking complete*',
      },
    });

    const exchangeEvent = mockEmit.mock.calls.find(
      ([event]) => event.type === 'model.exchange.recorded'
    )?.[0];
    expect(exchangeEvent).toMatchObject({
      type: 'model.exchange.recorded',
      category: 'model',
      severity: 'debug',
      streams: ['analysis'],
      payload: {
        provider: 'openai.responses',
        model: 'gpt-5.6',
        turnIndex: 2,
      },
    });
  });

  it('exchange eventではWeb call/resultだけを伏せ、live payloadと非Web値を保つ', async () => {
    const rawUrl = 'https://public.example/page?token=RESPONSES_WEB_ARG';
    const webResult = JSON.stringify({
      success: true,
      source: {
        requestedUrl: rawUrl,
        finalUrl: 'https://public.example/final',
        title: 'RESPONSES_PRIVATE_TITLE',
        httpStatus: 200,
        contentType: 'text/html',
        redirectCount: 1,
      },
      document: {
        text: 'RESPONSES_WEB_RESULT',
        returnedCharacters: 20,
        extractedCharacters: 40,
        truncated: false,
        links: [{ text: 'private', url: 'https://public.example/private' }],
      },
    });
    const firstProviderResponse = {
      id: 'response_web_1',
      output: [
        {
          type: 'function_call',
          call_id: 'web-call',
          name: 'read_web_page',
          arguments: JSON.stringify({ url: rawUrl }),
          status: 'completed',
        },
      ],
      usage: {
        input_tokens: 1,
        input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
        output_tokens: 1,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 2,
      },
    };
    mockCreateResponse
      .mockResolvedValueOnce(firstProviderResponse)
      .mockResolvedValueOnce({
        id: 'response_web_2',
        output: [],
        usage: {
          input_tokens: 1,
          input_tokens_details: {
            cached_tokens: 0,
            cache_write_tokens: 0,
          },
          output_tokens: 1,
          output_tokens_details: { reasoning_tokens: 0 },
          total_tokens: 2,
        },
      });
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      events: mockEvents,
    });

    await model.generate({ input: [], tools: [] });
    await model.generate({
      input: [
        { type: 'tool_result', callId: 'web-call', output: webResult },
        {
          type: 'tool_call',
          callId: 'other-call',
          toolName: 'think_deeply',
          input: 'RESPONSES_NON_WEB_CANARY',
        },
        {
          type: 'tool_result',
          callId: 'other-call',
          output: 'RESPONSES_NON_WEB_CANARY',
        },
      ],
      tools: [],
    });

    const exchangeEvents = mockEmit.mock.calls
      .map(([event]) => event)
      .filter((event) => event.type === 'model.exchange.recorded');
    const serializedEvents = JSON.stringify(exchangeEvents);
    expect(serializedEvents).not.toContain('RESPONSES_WEB_ARG');
    expect(serializedEvents).not.toContain('RESPONSES_WEB_RESULT');
    expect(serializedEvents).not.toContain('RESPONSES_PRIVATE_TITLE');
    expect(serializedEvents).toContain('RESPONSES_NON_WEB_CANARY');

    expect(JSON.stringify(mockCreateResponse.mock.calls[1]?.[0])).toContain(
      'RESPONSES_WEB_RESULT'
    );
    expect(JSON.stringify(firstProviderResponse)).toContain(
      'RESPONSES_WEB_ARG'
    );
  });

  it('usage がない response はゼロ usage として扱う', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
    });

    mockCreateResponse.mockResolvedValue({
      id: 'response_123',
      output: [],
    });

    const response = await model.generate({
      input: [
        {
          role: 'user',
          content: 'hello',
        },
      ],
      tools: [],
    });

    expect(response.usage).toEqual({
      cachedInputTokens: 0,
      cacheWriteInputTokens: 0,
      uncachedInputTokens: 0,
      totalInputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      totalTokens: 0,
    });
  });

  it('generate は refusal message を assistant message に正規化し、reasoning item は output から除外する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
    });

    mockCreateResponse.mockResolvedValue({
      id: 'response_refusal',
      output: [
        {
          type: 'reasoning',
          content: [{ text: 'Need to reject unsafe request.' }],
        } as unknown as ResponseOutputItem,
        {
          type: 'message',
          role: 'assistant',
          id: 'msg_refusal',
          status: 'completed',
          content: [
            {
              type: 'refusal',
              refusal: 'I cannot assist with that request.',
            },
          ],
        },
      ],
      usage: {
        input_tokens: 1,
        input_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 0,
        },
        output_tokens: 2,
        output_tokens_details: { reasoning_tokens: 1 },
        total_tokens: 3,
      },
    });

    const response = await model.generate({
      input: [
        {
          role: 'user',
          content: 'unsafe',
        },
      ],
      tools: [],
    });

    expect(response.output).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: '<refusal>I cannot assist with that request.</refusal>',
      },
    ]);
  });
});

describe('formatModelOutputContent', () => {
  it('message と reasoning だけを返し、function_call は tool event 側へ寄せる', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'reasoning',
        content: [{ text: 'Need a note.' }],
      } as unknown as ResponseOutputItem,
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'I will create it.',
            annotations: [],
          },
        ],
      },
      {
        type: 'function_call',
        call_id: 'call_123',
        name: 'create_note',
        arguments: '{"title":"memo"}',
        status: 'completed',
      },
    ];

    expect(formatModelOutputContent(output)).toBe(
      '*reasoning: Need a note.*\n\n*thinking: I will create it.*'
    );
  });
});

describe('formatBlock', () => {
  it('roleとcontentを正しいブロック形式でフォーマットする', () => {
    const result = formatBlock('user', 'Hello, world!');
    expect(result).toBe('[user]:\nHello, world!');
  });

  it('空のcontentでも正しく動作する', () => {
    const result = formatBlock('assistant', '');
    expect(result).toBe('[assistant]:\n');
  });

  it('複数行のcontentを正しく処理する', () => {
    const multilineContent = 'First line\nSecond line\nThird line';
    const result = formatBlock('system', multilineContent);
    expect(result).toBe('[system]:\nFirst line\nSecond line\nThird line');
  });
});

describe('formatFunctionCall', () => {
  it('Function callを正しくフォーマットする', () => {
    const functionCall: ResponseFunctionToolCall = {
      type: 'function_call',
      call_id: 'call_123',
      name: 'test_function',
      arguments: '{"param": "value"}',
      status: 'completed',
    };

    const result = formatFunctionCall(functionCall);
    expect(result).toBe(
      '[function call] call_123 (completed)\ntest_function({"param": "value"})'
    );
  });

  it('異なるstatusでも正しく動作する', () => {
    const functionCall: ResponseFunctionToolCall = {
      type: 'function_call',
      call_id: 'call_456',
      name: 'another_function',
      arguments: '{}',
      status: 'in_progress',
    };

    const result = formatFunctionCall(functionCall);
    expect(result).toBe(
      '[function call] call_456 (in_progress)\nanother_function({})'
    );
  });
});

describe('formatFunctionCallOutput', () => {
  it('文字列JSONを整形する', () => {
    expect(formatFunctionCallOutput('{"result":"success"}')).toBe(
      '{\n  "result": "success"\n}'
    );
  });

  it('配列出力を整形する', () => {
    expect(
      formatFunctionCallOutput([
        {
          type: 'input_text',
          text: 'success',
        },
      ])
    ).toBe('[\n  {\n    "type": "input_text",\n    "text": "success"\n  }\n]');
  });

  it('JSON ではない文字列はそのまま返す', () => {
    expect(formatFunctionCallOutput('plain text output')).toBe(
      'plain text output'
    );
  });
});

describe('formatMessage', () => {
  describe('文字列content', () => {
    it('文字列contentのメッセージを正しくフォーマットする', () => {
      const message: EasyInputMessage = {
        role: 'user',
        content: 'Hello, how are you?',
      };

      const result = formatMessage(message);
      expect(result).toBe('[user]:\nHello, how are you?');
    });
  });

  describe('配列content', () => {
    it('input_textタイプのcontentを正しくフォーマットする', () => {
      const message: ResponseInputItem.Message = {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Hello, world!',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe('[user]:\nHello, world!');
    });

    it('output_textタイプのcontentを正しくフォーマットする', () => {
      const message: ResponseOutputMessage = {
        type: 'message',
        role: 'assistant',
        id: 'msg_123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'Hello back!',
            annotations: [],
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe('[assistant]:\nHello back!');
    });

    it('input_imageタイプのcontentを正しくフォーマットする', () => {
      const message: ResponseInputItem.Message = {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_image',
            image_url: 'https://example.com/image.jpg',
            detail: 'auto',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe(
        '[user]:\n<image>https://example.com/image.jpg</image>'
      );
    });

    it('input_fileタイプのcontentを正しくフォーマットする（file_url使用）', () => {
      const message: ResponseInputItem.Message = {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_file',
            file_url: 'https://example.com/file.pdf',
            filename: 'document.pdf',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe('[user]:\n<file>https://example.com/file.pdf</file>');
    });

    it('input_fileタイプのcontentを正しくフォーマットする（filename使用）', () => {
      const message: ResponseInputItem.Message = {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_file',
            filename: 'document.pdf',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe('[user]:\n<file>document.pdf</file>');
    });

    it('refusalタイプのcontentを正しくフォーマットする', () => {
      const message: ResponseOutputMessage = {
        type: 'message',
        role: 'assistant',
        id: 'msg_456',
        status: 'completed',
        content: [
          {
            type: 'refusal',
            refusal: 'I cannot assist with that request.',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe(
        '[assistant]:\n<refusal>I cannot assist with that request.</refusal>'
      );
    });

    it('未知のタイプを正しく処理する', () => {
      const message: ResponseOutputMessage = {
        type: 'message',
        role: 'assistant',
        id: 'msg_456',
        status: 'completed',
        content: [
          {
            type: 'unknown',
          },
        ] as unknown as ResponseOutputMessage['content'],
      };

      expect(() => formatMessage(message)).toThrow(
        'Unexpected contentType: unknown'
      );
    });

    it('複数のcontentアイテムを正しく結合する', () => {
      const message: ResponseInputItem.Message = {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Here is an image:',
          },
          {
            type: 'input_image',
            image_url: 'https://example.com/image.jpg',
            detail: 'auto',
          },
          {
            type: 'input_text',
            text: 'What do you see?',
          },
        ],
      };

      const result = formatMessage(message);
      expect(result).toBe(
        '[user]:\nHere is an image:\n\n[user]:\n<image>https://example.com/image.jpg</image>\n\n[user]:\nWhat do you see?'
      );
    });
  });
});

describe('formatInputItem', () => {
  it('messageタイプのアイテムを正しくフォーマットする', () => {
    const item: ResponseInputItem = {
      type: 'message',
      role: 'user',
      content: 'Hello, world!',
    };

    const result = formatInputItem(item);
    expect(result).toBe('[user]:\nHello, world!');
  });

  it('function_callタイプのアイテムを正しくフォーマットする', () => {
    const item: ResponseInputItem = {
      type: 'function_call',
      call_id: 'call_123',
      name: 'test_function',
      arguments: '{"param": "value"}',
      status: 'completed',
    };

    const result = formatInputItem(item);
    expect(result).toBe(
      '[function call] call_123 (completed)\ntest_function({"param": "value"})'
    );
  });

  it('function_call_outputタイプのアイテムを正しくフォーマットする', () => {
    const item: ResponseInputItem = {
      type: 'function_call_output',
      call_id: 'call_123',
      output: '{"result": "success"}',
      status: 'completed',
    };

    const result = formatInputItem(item);
    expect(result).toBe(
      '[function call output] call_123 (completed)\n{\n  "result": "success"\n}'
    );
  });

  it('function_call_outputタイプの配列出力を正しくフォーマットする', () => {
    const item: ResponseInputItem = {
      type: 'function_call_output',
      call_id: 'call_123',
      output: [
        {
          type: 'input_text',
          text: 'success',
        },
      ],
      status: 'completed',
    };

    const result = formatInputItem(item);
    expect(result).toBe(
      '[function call output] call_123 (completed)\n[\n  {\n    "type": "input_text",\n    "text": "success"\n  }\n]'
    );
  });

  it('contentプロパティを持つアイテム（レガシー）を正しくフォーマットする', () => {
    const item = {
      role: 'user',
      content: 'Legacy message format',
    } as ResponseInputItem;

    const result = formatInputItem(item);
    expect(result).toBe('[user]:\nLegacy message format');
  });

  it('アイテム参照を正しくフォーマットする', () => {
    const item = {
      id: 'item_123',
    } as ResponseInputItem;

    const result = formatInputItem(item);
    expect(result).toBe('<item_reference>item_123</item_reference>');
  });

  it('未知のタイプを正しく処理する', () => {
    const item = {
      type: 'unknown_type',
    } as unknown as ResponseInputItem;

    const result = formatInputItem(item);
    expect(result).toBe('<unknown_type />');
  });
});

describe('formatOutputItem', () => {
  it('messageタイプのアイテムを正しくフォーマットする', () => {
    const item: ResponseOutputItem = {
      type: 'message',
      role: 'assistant',
      id: 'msg_789',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'Hello back!',
          annotations: [],
        },
      ],
    };

    const result = formatOutputItem(item);
    expect(result).toBe('[assistant]:\nHello back!');
  });

  it('function_callタイプのアイテムを正しくフォーマットする', () => {
    const item: ResponseOutputItem = {
      type: 'function_call',
      call_id: 'call_456',
      name: 'output_function',
      arguments: '{"output": "data"}',
      status: 'completed',
    };

    const result = formatOutputItem(item);
    expect(result).toBe(
      '[function call] call_456 (completed)\noutput_function({"output": "data"})'
    );
  });

  it('未知のタイプを正しく処理する', () => {
    const item = {
      type: 'unknown_output_type',
    } as unknown as ResponseOutputItem;

    const result = formatOutputItem(item);
    expect(result).toBe('<unknown_output_type />');
  });

  it('reasoningタイプはプレースホルダ表示にする', () => {
    const item = {
      type: 'reasoning',
    } as unknown as ResponseOutputItem;

    const result = formatOutputItem(item);
    expect(result).toBe('<reasoning />');
  });
});
