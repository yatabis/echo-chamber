import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';

import {
  OpenAIChatCompletionsModel,
  type OpenAIChatCompletionsExtraBody,
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

describe('OpenAIChatCompletionsModel', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('SDK retry 設定と各 HTTP attempt の admission hook を caller が指定できる', async () => {
    const beforeRequest = vi.fn();
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{}'));
    new OpenAIChatCompletionsModel({
      apiKey: 'test-key',
      model: 'qwen3.6',
      maxRetries: 0,
      beforeRequest,
    });

    expect(mockOpenAIConstructor).toHaveBeenCalledWith(
      expect.objectContaining({
        apiKey: 'test-key',
        baseURL: undefined,
        maxRetries: 0,
      })
    );
    const constructorCalls = mockOpenAIConstructor.mock.calls as unknown as [
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

  it('createChatCompletion は provider-neutral request を Chat Completions 形式へ変換する', async () => {
    const model = new OpenAIChatCompletionsModel({
      apiKey: 'local-key',
      model: 'qwen3.6',
      baseURL: 'http://localhost:1234/v1',
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
        prompt_tokens_details: {
          cached_tokens: 3,
          cache_write_tokens: 2,
        },
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
      events: mockEvents,
      requestBodyExtension: ({
        hasCompletedExchange,
      }): OpenAIChatCompletionsExtraBody => ({
        runtime_extension: {
          exchange_state: hasCompletedExchange ? 'subsequent' : 'initial',
        },
      }),
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
      turnIndex: 2,
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
        cacheWriteInputTokens: 0,
        uncachedInputTokens: 6,
        totalInputTokens: 8,
        outputTokens: 3,
        reasoningTokens: 1,
        totalTokens: 11,
      },
    });
    expect(mockChatCreate).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        runtime_extension: {
          exchange_state: 'initial',
        },
      })
    );
    expect(mockChatCreate).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        runtime_extension: {
          exchange_state: 'subsequent',
        },
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
    const outputEvent = mockEmit.mock.calls.find(
      ([event]) => event.type === 'model.output.emitted'
    )?.[0];
    expect(outputEvent).toMatchObject({
      type: 'model.output.emitted',
      category: 'model',
      severity: 'info',
      streams: ['thought', 'analysis'],
      payload: {
        provider: 'openai.chat_completions',
        model: 'qwen3.6',
        turnIndex: 2,
        content: '*thinking: finished*',
      },
    });

    const exchangeEvent = mockEmit.mock.calls.find(
      ([event]) =>
        event.type === 'model.exchange.recorded' &&
        event.payload?.turnIndex === 2
    )?.[0];
    expect(exchangeEvent).toMatchObject({
      type: 'model.exchange.recorded',
      category: 'model',
      severity: 'debug',
      streams: ['analysis'],
      payload: {
        provider: 'openai.chat_completions',
        model: 'qwen3.6',
        turnIndex: 2,
      },
    });
  });

  it('exchange eventではWeb historyだけを伏せ、live payloadと非Web値を保つ', async () => {
    const rawUrl = 'https://public.example/page?token=CHAT_WEB_ARG';
    const webResult = JSON.stringify({
      success: true,
      source: {
        requestedUrl: rawUrl,
        finalUrl: 'https://public.example/final',
        title: 'CHAT_PRIVATE_TITLE',
        httpStatus: 200,
        contentType: 'text/html',
        redirectCount: 0,
      },
      document: {
        text: 'CHAT_WEB_RESULT',
        returnedCharacters: 15,
        extractedCharacters: 30,
        truncated: false,
        links: [{ text: 'private', url: 'https://public.example/private' }],
      },
    });
    const firstProviderResponse = {
      id: 'chat_web_1',
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
                id: 'web-call',
                type: 'function',
                function: {
                  name: 'read_web_page',
                  arguments: JSON.stringify({ url: rawUrl }),
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
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
      },
    };
    mockChatCreate
      .mockResolvedValueOnce(firstProviderResponse)
      .mockResolvedValueOnce({
        id: 'chat_web_2',
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
          prompt_tokens: 1,
          completion_tokens: 1,
          total_tokens: 2,
        },
      });
    const model = new OpenAIChatCompletionsModel({
      apiKey: 'local-key',
      model: 'qwen3.6',
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
          input: 'CHAT_NON_WEB_CANARY',
        },
        {
          type: 'tool_result',
          callId: 'other-call',
          output: 'CHAT_NON_WEB_CANARY',
        },
      ],
      tools: [],
    });

    const exchangeEvents = mockEmit.mock.calls
      .map(([event]) => event)
      .filter((event) => event.type === 'model.exchange.recorded');
    const serializedEvents = JSON.stringify(exchangeEvents);
    expect(serializedEvents).not.toContain('CHAT_WEB_ARG');
    expect(serializedEvents).not.toContain('CHAT_WEB_RESULT');
    expect(serializedEvents).not.toContain('CHAT_PRIVATE_TITLE');
    expect(serializedEvents).toContain('CHAT_NON_WEB_CANARY');

    expect(JSON.stringify(mockChatCreate.mock.calls[1]?.[0])).toContain(
      'CHAT_WEB_ARG'
    );
    expect(JSON.stringify(mockChatCreate.mock.calls[1]?.[0])).toContain(
      'CHAT_WEB_RESULT'
    );
    expect(JSON.stringify(firstProviderResponse)).toContain('CHAT_WEB_ARG');
  });

  it('画像付き message を Chat Completions content part に変換する', async () => {
    const model = new OpenAIChatCompletionsModel({
      apiKey: 'local-key',
      model: 'gpt-5.5',
    });

    mockChatCreate.mockResolvedValue({
      id: 'chatcmpl_vision',
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
      model: 'gpt-5.5',
      object: 'chat.completion',
      usage: {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
      },
    });

    await model.createChatCompletion({
      input: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Please inspect this image.',
            },
            {
              type: 'image',
              imageUrl: 'https://example.com/image.png',
              detail: 'original',
            },
          ],
        },
      ],
      tools: [],
    });

    expect(mockChatCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Please inspect this image.',
              },
              {
                type: 'image_url',
                image_url: {
                  url: 'https://example.com/image.png',
                  detail: 'auto',
                },
              },
            ],
          },
        ],
      })
    );
  });

  it('usage がない response はゼロ usage として扱う', () => {
    expect(toChatModelUsage(undefined)).toEqual({
      cachedInputTokens: 0,
      cacheWriteInputTokens: 0,
      uncachedInputTokens: 0,
      totalInputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      totalTokens: 0,
    });
  });

  it('cache write を uncached input から分離する', () => {
    expect(
      toChatModelUsage({
        prompt_tokens: 10,
        prompt_tokens_details: {
          cached_tokens: 3,
          cache_write_tokens: 2,
        },
        completion_tokens: 4,
        completion_tokens_details: { reasoning_tokens: 1 },
        total_tokens: 14,
      })
    ).toEqual({
      cachedInputTokens: 3,
      cacheWriteInputTokens: 2,
      uncachedInputTokens: 5,
      totalInputTokens: 10,
      outputTokens: 4,
      reasoningTokens: 1,
      totalTokens: 14,
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
