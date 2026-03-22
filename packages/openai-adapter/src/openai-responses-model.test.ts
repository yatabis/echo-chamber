import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ModelRequest } from '@echo-chamber/core/ports/model';

import {
  formatBlock,
  formatFunctionCall,
  formatFunctionCallOutput,
  formatInputItem,
  formatLogOutput,
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
  return {
    default: vi.fn(() => ({
      responses: {
        create: mockCreateResponse,
      },
    })),
  };
});

const mockLogger = {
  debug: vi.fn().mockResolvedValue(undefined),
  warn: vi.fn().mockResolvedValue(undefined),
};

const mockThoughtLog = {
  send: vi.fn().mockResolvedValue(undefined),
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

  it('createResponse は provider-neutral request を Responses API 形式へ変換する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
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
    expect(mockLogger.warn).toHaveBeenCalledWith(
      'Response usage information is undefined'
    );
  });

  it('createResponse は object でない inputSchema を parameters: null に正規化する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    mockCreateResponse.mockResolvedValue({
      output: [],
      usage: {
        input_tokens: 0,
        input_tokens_details: { cached_tokens: 0 },
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

  it('generate は OpenAI response を core model response へ変換する', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
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
        input_tokens_details: { cached_tokens: 4 },
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
        uncachedInputTokens: 6,
        totalInputTokens: 10,
        outputTokens: 7,
        reasoningTokens: 2,
        totalTokens: 17,
      },
      responseToken: 'response_123',
    });
    expect(mockThoughtLog.send).toHaveBeenCalledWith(
      '*thinking: Thinking complete*\n\n*think_deeply: test*'
    );
  });

  it('usage がない response はゼロ usage として扱う', async () => {
    const model = new OpenAIResponsesModel({
      apiKey: 'test-key',
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
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
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
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
        input_tokens_details: { cached_tokens: 0 },
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
    expect(mockThoughtLog.send).toHaveBeenCalledWith(
      '*reasoning: Need to reject unsafe request.*\n\n*refusal: I cannot assist with that request.*'
    );
  });
});

describe('formatLogOutput', () => {
  it('ログ出力のフォーマット', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_123',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'I am fine, thank you!',
            annotations: [],
          },
        ],
      },
    ];

    expect(formatLogOutput(output)).toBe('*thinking: I am fine, thank you!*');
  });

  it('refusal', () => {
    const output: ResponseOutputItem[] = [
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
    ];

    expect(formatLogOutput(output)).toBe(
      '*refusal: I cannot assist with that request.*'
    );
  });

  it('reasoning', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'reasoning',
        summary: [{ text: 'Need more context' }],
      } as unknown as ResponseOutputItem,
    ];

    expect(formatLogOutput(output)).toBe('*reasoning: Need more context*');
  });

  it('空のreasoningはプレースホルダを返す', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'reasoning',
        summary: [],
      } as unknown as ResponseOutputItem,
    ];

    expect(formatLogOutput(output)).toBe('*reasoning*');
  });

  it('read_chat_messages', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_read',
        name: 'read_chat_messages',
        arguments: JSON.stringify({ limit: 10 }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*read_chat_messages: 10*');
  });

  it('think_deeply', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_think',
        name: 'think_deeply',
        arguments: JSON.stringify({ thought: 'Deep philosophical question' }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*think_deeply: Deep philosophical question*'
    );
  });

  it('store_memory with type', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_store',
        name: 'store_memory',
        arguments: JSON.stringify({
          content: '今日は楽しい一日だった',
          type: 'episode',
          emotion: { valence: 0.8, arousal: 0.5, labels: ['楽しい', '満足'] },
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*store_memory [episode]: 今日は楽しい一日だった\n(0.8, 0.5) [楽しい, 満足]*'
    );
  });

  it('search_memory with type', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_search',
        name: 'search_memory',
        arguments: JSON.stringify({
          query: '楽しかった思い出',
          type: 'episode',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*search_memory [episode]: 楽しかった思い出*'
    );
  });

  it('search_memory without type', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_search_no_type',
        name: 'search_memory',
        arguments: JSON.stringify({
          query: '何か思い出',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*search_memory: 何か思い出*');
  });

  it('create_note', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_create_note',
        name: 'create_note',
        arguments: JSON.stringify({
          title: '買い物メモ',
          content: '牛乳とパン',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*create_note: 買い物メモ*');
  });

  it('list_notes', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_list_notes',
        name: 'list_notes',
        arguments: JSON.stringify({}),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*list_notes*');
  });

  it('get_note', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_get_note',
        name: 'get_note',
        arguments: JSON.stringify({
          id: 'note-1',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*get_note: note-1*');
  });

  it('search_notes', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_search_notes',
        name: 'search_notes',
        arguments: JSON.stringify({
          query: '買い物',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*search_notes: 買い物*');
  });

  it('update_note', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_update_note',
        name: 'update_note',
        arguments: JSON.stringify({
          id: 'note-1',
          title: '更新後タイトル',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*update_note: note-1*');
  });

  it('delete_note', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_delete_note',
        name: 'delete_note',
        arguments: JSON.stringify({
          id: 'note-1',
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*delete_note: note-1*');
  });

  it('finish_thinking', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_finish',
        name: 'finish_thinking',
        arguments: JSON.stringify({
          reason: 'No further action needed',
          next_wake_at: '2026-03-20T08:00:00.000Z',
          session_record: {
            content:
              'Replied to unread messages and left a concise session recap.',
            emotion: {
              valence: 0.4,
              arousal: 0.2,
              labels: ['calm', 'satisfied'],
            },
          },
        }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*finish_thinking: No further action needed(next_wake_at: 2026-03-20T08:00:00.000Z)\nsession_record: Replied to unread messages and left a concise session recap.\n(0.4, 0.2) [calm, satisfied]*'
    );
  });

  it('デフォルトの function_call', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_default',
        name: 'default_function',
        arguments: JSON.stringify({ param: 'value' }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*default_function*');
  });

  it('function の引数が不正な型でもエラーにならない', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'function_call',
        call_id: 'call_invalid_arg_type',
        name: 'read_chat_messages',
        arguments: JSON.stringify({ param: 123 }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe('*read_chat_messages: undefined*');
  });

  it('複数のoutputアイテム', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_1',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'First message',
            annotations: [],
          },
        ],
      },
      {
        type: 'function_call',
        call_id: 'call_think',
        name: 'think_deeply',
        arguments: JSON.stringify({ thought: 'Some thought' }),
        status: 'completed',
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*thinking: First message*\n\n*think_deeply: Some thought*'
    );
  });

  it('複数のcontentを持つメッセージ', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_multi',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'First part',
            annotations: [],
          },
          {
            type: 'output_text',
            text: 'Second part',
            annotations: [],
          },
        ],
      },
    ];

    expect(formatLogOutput(output)).toBe(
      '*thinking: First part*\n\n*thinking: Second part*'
    );
  });

  it('空の出力アイテム', () => {
    const output: ResponseOutputItem[] = [];
    expect(formatLogOutput(output)).toBe('');
  });

  it('空になるログ出力', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'unknown_type',
      } as unknown as ResponseOutputItem,
    ];
    expect(formatLogOutput(output)).toBe('');
  });

  it('予期しないcontentTypeでエラーが投げられる', () => {
    const output: ResponseOutputItem[] = [
      {
        type: 'message',
        role: 'assistant',
        id: 'msg_error',
        status: 'completed',
        content: [
          {
            type: 'unknown_content_type',
          },
        ] as unknown as ResponseOutputMessage['content'],
      },
    ];

    expect(() => formatLogOutput(output)).toThrowError(
      'Unexpected contentType: unknown_content_type'
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

      expect(() => formatMessage(message)).toThrowError(
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
