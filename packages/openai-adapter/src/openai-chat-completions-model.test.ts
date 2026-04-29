import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  OpenAIChatCompletionsModel,
  toChatCompletionTool,
  toChatModelUsage,
} from './openai-chat-completions-model';

const { mockChatCreate, mockOpenAIConstructor } = vi.hoisted(() => {
  const chatCreate = vi.fn();

  return {
    mockChatCreate: chatCreate,
    mockOpenAIConstructor: vi.fn(() => ({
      chat: {
        completions: {
          create: chatCreate,
        },
      },
    })),
  };
});

vi.mock('openai', () => {
  return {
    default: mockOpenAIConstructor,
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

describe('OpenAIChatCompletionsModel', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('createChatCompletion は provider-neutral request を Chat Completions 形式へ変換する', async () => {
    const model = new OpenAIChatCompletionsModel({
      apiKey: 'local-key',
      model: 'qwen3.6',
      baseURL: 'http://localhost:1234/v1',
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
      reasoningEffort: 'none',
      maxTokens: 32768,
      temperature: 0.7,
      topP: 0.95,
      presencePenalty: 0.2,
      frequencyPenalty: -0.1,
      extraBody: {
        top_k: 20,
        chat_template_kwargs: { enable_thinking: false },
      },
    });

    mockChatCreate.mockResolvedValue({
      id: 'chatcmpl_1',
      choices: [
        {
          index: 0,
          finish_reason: 'stop',
          logprobs: null,
          message: {
            role: 'assistant',
            content: 'done',
            refusal: null,
          },
        },
      ],
      created: 0,
      model: 'qwen3.6',
      object: 'chat.completion',
      usage: {
        prompt_tokens: 10,
        prompt_tokens_details: { cached_tokens: 3 },
        completion_tokens: 4,
        completion_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 14,
      },
    });

    await model.createChatCompletion({
      input: [
        {
          role: 'developer',
          content: 'You are helpful.',
        },
        {
          type: 'tool_call',
          callId: 'check_notifications',
          toolName: 'check_notifications',
          input: '{}',
        },
        {
          type: 'tool_result',
          callId: 'check_notifications',
          output: '{"success":true}',
        },
      ],
      tools: [thinkDeeplyTool],
    });

    expect(mockOpenAIConstructor).toHaveBeenCalledWith({
      apiKey: 'local-key',
      baseURL: 'http://localhost:1234/v1',
    });
    expect(mockChatCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: 'You are helpful.',
          },
          {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'check_notifications',
                type: 'function',
                function: {
                  name: 'check_notifications',
                  arguments: '{}',
                },
              },
            ],
          },
          {
            role: 'tool',
            tool_call_id: 'check_notifications',
            content: '{"success":true}',
          },
        ],
        model: 'qwen3.6',
        frequency_penalty: -0.1,
        max_tokens: 32768,
        presence_penalty: 0.2,
        reasoning_effort: 'none',
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        chat_template_kwargs: { enable_thinking: false },
        tool_choice: 'auto',
        tools: [
          {
            type: 'function',
            function: {
              name: 'think_deeply',
              description: 'Deep thinking tool',
              parameters: thinkDeeplyTool.inputSchema,
              strict: true,
            },
          },
        ],
      })
    );
  });

  it('generate は assistant tool call を履歴に積み、次 turn の tool result と一緒に送る', async () => {
    const model = new OpenAIChatCompletionsModel({
      apiKey: 'local-key',
      model: 'qwen3.6',
      logger: mockLogger,
      thoughtLog: mockThoughtLog,
    });

    mockChatCreate
      .mockResolvedValueOnce({
        id: 'chatcmpl_1',
        choices: [
          {
            index: 0,
            finish_reason: 'tool_calls',
            logprobs: null,
            message: {
              role: 'assistant',
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: 'call_123',
                  type: 'function',
                  function: {
                    name: 'think_deeply',
                    arguments: '{"thought":"test"}',
                  },
                },
              ],
            },
          },
        ],
        created: 0,
        model: 'qwen3.6',
        object: 'chat.completion',
        usage: {
          prompt_tokens: 5,
          completion_tokens: 2,
          total_tokens: 7,
        },
      })
      .mockResolvedValueOnce({
        id: 'chatcmpl_2',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            logprobs: null,
            message: {
              role: 'assistant',
              content: 'finished',
              refusal: null,
            },
          },
        ],
        created: 0,
        model: 'qwen3.6',
        object: 'chat.completion',
        usage: {
          prompt_tokens: 8,
          prompt_tokens_details: { cached_tokens: 2 },
          completion_tokens: 3,
          completion_tokens_details: { reasoning_tokens: 1 },
          total_tokens: 11,
        },
      });

    const first = await model.generate({
      input: [
        {
          role: 'user',
          content: 'start',
        },
      ],
      tools: [thinkDeeplyTool],
    });
    const second = await model.generate({
      input: [
        {
          type: 'tool_result',
          callId: 'call_123',
          output: '{"success":true}',
        },
      ],
      tools: [thinkDeeplyTool],
    });

    expect(first.output).toEqual([
      {
        type: 'tool_call',
        callId: 'call_123',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
    ]);
    expect(second).toEqual({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: 'finished',
        },
      ],
      responseToken: 'chatcmpl_2',
      usage: {
        cachedInputTokens: 2,
        uncachedInputTokens: 6,
        totalInputTokens: 8,
        outputTokens: 3,
        reasoningTokens: 1,
        totalTokens: 11,
      },
    });
    expect(mockChatCreate).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: 'start',
          },
          {
            role: 'assistant',
            content: null,
            refusal: null,
            tool_calls: [
              {
                id: 'call_123',
                type: 'function',
                function: {
                  name: 'think_deeply',
                  arguments: '{"thought":"test"}',
                },
              },
            ],
          },
          {
            role: 'tool',
            tool_call_id: 'call_123',
            content: '{"success":true}',
          },
        ],
      })
    );
  });

  it('usage がない response はゼロ usage として扱う', () => {
    expect(toChatModelUsage(undefined)).toEqual({
      cachedInputTokens: 0,
      uncachedInputTokens: 0,
      totalInputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      totalTokens: 0,
    });
  });
});

describe('toChatCompletionTool', () => {
  it('tool contract を Chat Completions function tool 定義へ変換する', () => {
    expect(toChatCompletionTool(thinkDeeplyTool)).toEqual({
      type: 'function',
      function: {
        name: 'think_deeply',
        description: 'Deep thinking tool',
        parameters: thinkDeeplyTool.inputSchema,
        strict: true,
      },
    });
  });
});
