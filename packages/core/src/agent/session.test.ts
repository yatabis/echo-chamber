import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  accumulateModelUsage,
  AgentSessionExecutionError,
  executeAgentToolCall,
  runAgentSession,
  ZERO_MODEL_USAGE,
} from './session';

import type { EchoEventPort } from '../ports/echo-event';
import type { ModelPort, ModelToolContract, ModelUsage } from '../ports/model';

afterEach(() => {
  vi.useRealTimers();
});

function createFinishThinkingInput(
  reason = 'done',
  nextWakeAt?: string
): string {
  return JSON.stringify({
    reason,
    next_wake_at: nextWakeAt,
  });
}

function createUsage(overrides?: Partial<ModelUsage>): ModelUsage {
  return {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: 0,
    totalInputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens: 0,
    ...overrides,
  };
}

function createToolContract(name: string): ModelToolContract {
  return {
    name,
    description: `${name} description`,
    inputSchema: {},
    strict: true,
  };
}

describe('accumulateModelUsage', () => {
  it('usage を加算する', () => {
    expect(
      accumulateModelUsage(
        createUsage({
          cachedInputTokens: 1,
          cacheWriteInputTokens: 2,
          uncachedInputTokens: 2,
          totalInputTokens: 3,
          outputTokens: 4,
          reasoningTokens: 5,
          totalTokens: 6,
        }),
        createUsage({
          cachedInputTokens: 10,
          cacheWriteInputTokens: 20,
          uncachedInputTokens: 20,
          totalInputTokens: 30,
          outputTokens: 40,
          reasoningTokens: 50,
          totalTokens: 60,
        })
      )
    ).toEqual(
      createUsage({
        cachedInputTokens: 11,
        cacheWriteInputTokens: 22,
        uncachedInputTokens: 22,
        totalInputTokens: 33,
        outputTokens: 44,
        reasoningTokens: 55,
        totalTokens: 66,
      })
    );
  });

  it('ゼロ usage を基点にできる', () => {
    expect(accumulateModelUsage(ZERO_MODEL_USAGE, createUsage())).toEqual(
      ZERO_MODEL_USAGE
    );
  });
});

describe('executeAgentToolCall', () => {
  it('登録済みツールを実行する', async () => {
    const execute = vi.fn().mockResolvedValue('{"success":true}');

    const result = await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
      [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute,
        },
      ]
    );

    expect(execute).toHaveBeenCalledWith('{"thought":"test"}');
    expect(result).toBe('{"success":true}');
  });

  it('tool call の開始と完了をイベントに流す', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const execute = vi.fn().mockResolvedValue('{"success":true}');

    const result = await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
      [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute,
        },
      ],
      { emit },
      2
    );

    expect(result).toBe('{"success":true}');
    expect(emit).toHaveBeenCalledWith({
      type: 'tool.called',
      category: 'tool',
      severity: 'info',
      streams: ['thought', 'analysis'],
      summary: 'think_deeply called',
      payload: {
        callId: 'call-1',
        toolName: 'think_deeply',
        turnIndex: 2,
        input: '{"thought":"test"}',
      },
    });
    expect(emit).toHaveBeenCalledWith({
      type: 'tool.completed',
      category: 'tool',
      severity: 'info',
      streams: ['system', 'analysis'],
      summary: 'think_deeply completed',
      payload: {
        callId: 'call-1',
        toolName: 'think_deeply',
        turnIndex: 2,
        durationMs: 0,
        success: true,
        error: undefined,
        outputLength: 16,
      },
    });

    vi.useRealTimers();
  });

  it('note tool の entity 情報は tool event の payload に吸収する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const execute = vi.fn().mockResolvedValue(
      JSON.stringify({
        success: true,
        note: {
          id: 'note-1',
          title: '買い物メモ',
        },
      })
    );

    await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-note',
        toolName: 'create_note',
        input: '{"title":"買い物メモ","content":"牛乳"}',
      },
      [
        {
          name: 'create_note',
          contract: createToolContract('create_note'),
          execute,
        },
      ],
      { emit },
      1
    );

    const completedEvent = emit.mock.calls.find(
      ([event]) => event.type === 'tool.completed'
    )?.[0];
    expect(completedEvent).toMatchObject({
      type: 'tool.completed',
      payload: {
        toolName: 'create_note',
        operation: 'note.create',
        entityType: 'note',
        entityId: 'note-1',
      },
    });

    vi.useRealTimers();
  });

  it('tool 診断情報は event に載せ、model へ返す output からは除く', async () => {
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const execute = vi.fn().mockResolvedValue(
      JSON.stringify({
        success: false,
        error: 'Failed to read messages',
        diagnostics: {
          error: 'Discord API returned 500',
        },
      })
    );

    const result = await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-read',
        toolName: 'read_chat_messages',
        input: '{"channelKey":"main","limit":10}',
      },
      [
        {
          name: 'read_chat_messages',
          contract: createToolContract('read_chat_messages'),
          execute,
        },
      ],
      { emit },
      1
    );

    expect(result).toBe(
      JSON.stringify({
        success: false,
        error: 'Failed to read messages',
      })
    );
    const failedEvent = emit.mock.calls.find(
      ([event]) => event.type === 'tool.failed'
    )?.[0];
    expect(failedEvent).toMatchObject({
      type: 'tool.failed',
      payload: {
        error: 'Failed to read messages',
        diagnostics: {
          error: 'Discord API returned 500',
        },
      },
    });
  });

  it('read_web_pageのeventには生URLと本文を残さない', async () => {
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const rawUrl = 'https://example.com/private-path?topic=secret-canary';
    const execute = vi.fn().mockResolvedValue(
      JSON.stringify({
        success: true,
        source: {
          requestedUrl: rawUrl,
          finalUrl: rawUrl,
          retrievedAt: '2026-08-10T00:00:00.000Z',
          httpStatus: 200,
          redirectCount: 0,
          contentType: 'text/html',
          title: 'secret-title-canary',
        },
        document: {
          format: 'markdown',
          rendering: 'static',
          text: 'secret-body-canary',
          returnedCharacters: 18,
          extractedCharacters: 18,
          truncated: false,
          truncationReasons: [],
          links: [
            {
              text: 'secret-link',
              url: 'https://example.org/secret-link-canary',
            },
          ],
        },
        trust: 'untrusted_external_content',
      })
    );

    await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-web',
        toolName: 'read_web_page',
        input: JSON.stringify({ url: rawUrl, maxCharacters: 8_000 }),
      },
      [
        {
          name: 'read_web_page',
          contract: createToolContract('read_web_page'),
          execute,
        },
      ],
      { emit },
      1
    );

    const serializedEvents = JSON.stringify(
      emit.mock.calls.map(([event]) => event)
    );
    expect(serializedEvents).not.toContain(rawUrl);
    expect(serializedEvents).not.toContain('secret-title-canary');
    expect(serializedEvents).not.toContain('secret-body-canary');
    expect(serializedEvents).not.toContain('secret-link-canary');

    const calledEvent = emit.mock.calls.find(
      ([event]) => event.type === 'tool.called'
    )?.[0];
    expect(calledEvent).toMatchObject({
      payload: {
        callId: 'call-web',
        toolName: 'read_web_page',
        turnIndex: 1,
        input: {
          redacted: true,
          urlLength: rawUrl.length,
          hasQuery: true,
          maxCharacters: 8_000,
        },
      },
    });

    const completedEvent = emit.mock.calls.find(
      ([event]) => event.type === 'tool.completed'
    )?.[0];
    expect(completedEvent).toMatchObject({
      payload: {
        success: true,
        httpStatus: 200,
        contentType: 'text/html',
        redirectCount: 0,
        returnedCharacters: 18,
        extractedCharacters: 18,
        truncated: false,
        linkCount: 1,
      },
    });
  });

  it('read_web_pageのfailure eventにはraw error文字列を残さない', async () => {
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const rawError = 'private URL leaked into adapter error: FAILURE_CANARY';
    const output = JSON.stringify({
      success: false,
      code: 'http_status',
      error: rawError,
      retryable: true,
    });

    const result = await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-web-failure',
        toolName: 'read_web_page',
        input: JSON.stringify({ url: 'https://www.wikipedia.org/page' }),
      },
      [
        {
          name: 'read_web_page',
          contract: createToolContract('read_web_page'),
          execute: vi.fn().mockResolvedValue(output),
        },
      ],
      { emit },
      1
    );

    expect(result).toBe(output);
    expect(JSON.stringify(emit.mock.calls)).not.toContain('FAILURE_CANARY');
    const failedEvent = emit.mock.calls.find(
      ([event]) => event.type === 'tool.failed'
    )?.[0];
    expect(failedEvent).toMatchObject({
      payload: {
        success: false,
        code: 'http_status',
        retryable: true,
      },
    });
    expect(failedEvent?.payload).not.toHaveProperty('error');
  });

  it('未登録ツールはエラー文字列を返す', async () => {
    const result = await executeAgentToolCall(
      {
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'unknown_tool',
        input: '{}',
      },
      [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: vi.fn(),
        },
      ]
    );

    expect(result).toBe(
      JSON.stringify({
        error: "Function 'unknown_tool' is not registered",
        available_functions: ['think_deeply'],
      })
    );
  });
});

describe('runAgentSession', () => {
  it('tool call がなくても空 input で継続する', async () => {
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: 'done',
          },
        ],
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput(),
          },
        ],
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });
    const executeFinish = vi.fn().mockResolvedValue('{"success":true}');

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: executeFinish,
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(generate).toHaveBeenNthCalledWith(1, {
      input: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: undefined,
      turnIndex: 1,
    });
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(executeFinish).toHaveBeenCalledWith(createFinishThinkingInput());
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 15 }),
      responseToken: 'resp-2',
      terminationReason: 'finish_thinking',
    });
  });

  it('tool call がない model turn は warning として event payload に残す', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: 'still thinking',
          },
        ],
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput(),
          },
        ],
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });

    await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      events: { emit },
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    const completedEvent = emit.mock.calls.find(
      ([event]) =>
        event.type === 'model.turn.completed' && event.payload?.turnIndex === 1
    )?.[0];
    expect(completedEvent).toMatchObject({
      type: 'model.turn.completed',
      category: 'model',
      severity: 'warn',
      streams: ['analysis'],
      payload: {
        turnIndex: 1,
        toolCallCount: 0,
        warnings: ['no_tool_calls'],
      },
    });

    vi.useRealTimers();
  });

  it('tool call の結果を次ターンの input として渡す', async () => {
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-1',
            toolName: 'think_deeply',
            input: '{"thought":"test"}',
          },
        ],
        usage: createUsage({
          cachedInputTokens: 1,
          cacheWriteInputTokens: 0,
          uncachedInputTokens: 2,
          totalInputTokens: 3,
          outputTokens: 4,
          reasoningTokens: 5,
          totalTokens: 6,
        }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: 'done',
          },
        ],
        usage: createUsage({
          cachedInputTokens: 10,
          cacheWriteInputTokens: 0,
          uncachedInputTokens: 20,
          totalInputTokens: 30,
          outputTokens: 40,
          reasoningTokens: 50,
          totalTokens: 60,
        }),
        responseToken: 'resp-2',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput(),
          },
        ],
        usage: createUsage({
          cachedInputTokens: 100,
          cacheWriteInputTokens: 0,
          uncachedInputTokens: 200,
          totalInputTokens: 300,
          outputTokens: 400,
          reasoningTokens: 500,
          totalTokens: 600,
        }),
        responseToken: 'resp-3',
      });
    const execute = vi.fn().mockResolvedValue('{"success":true}');
    const executeFinish = vi.fn().mockResolvedValue('{"success":true}');

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute,
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: executeFinish,
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(execute).toHaveBeenCalledWith('{"thought":"test"}');
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [
        {
          type: 'tool_result',
          callId: 'call-1',
          output: '{"success":true}',
        },
      ],
      tools: [
        createToolContract('think_deeply'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(generate).toHaveBeenNthCalledWith(3, {
      input: [],
      tools: [
        createToolContract('think_deeply'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: 'resp-2',
      turnIndex: 3,
    });
    expect(executeFinish).toHaveBeenCalledWith(createFinishThinkingInput());
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({
        cachedInputTokens: 111,
        cacheWriteInputTokens: 0,
        uncachedInputTokens: 222,
        totalInputTokens: 333,
        outputTokens: 444,
        reasoningTokens: 555,
        totalTokens: 666,
      }),
      responseToken: 'resp-3',
      terminationReason: 'finish_thinking',
    });
  });

  it('read_chat_messages の画像添付を次ターンの vision input として渡す', async () => {
    const toolOutput = JSON.stringify({
      success: true,
      channelKey: 'main',
      messages: [
        {
          messageId: 'message-1',
          user: 'alice',
          message: 'Please check this.',
          created_at: '2026年03月21日 10:00:00',
          images: [
            {
              url: 'https://cdn.discordapp.com/attachments/photo.png',
              filename: 'photo.png',
              content_type: 'image/png',
              width: 640,
              height: 480,
              size: 2048,
              description: 'whiteboard photo',
            },
          ],
          reactions: [],
        },
      ],
    });
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-read',
            toolName: 'read_chat_messages',
            input: '{"channelKey":"main","limit":10}',
          },
        ],
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput(),
          },
        ],
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });

    await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'read_chat_messages',
          contract: createToolContract('read_chat_messages'),
          execute: vi.fn().mockResolvedValue(toolOutput),
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [
        {
          type: 'tool_result',
          callId: 'call-read',
          output: toolOutput,
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Discord image attachments from read_chat_messages (1/1).',
            },
            {
              type: 'text',
              text: 'Image 1, messageId=message-1, user=alice, created_at=2026年03月21日 10:00:00, filename=photo.png, content_type=image/png, size=640x480, description=whiteboard photo',
            },
            {
              type: 'image',
              imageUrl: 'https://cdn.discordapp.com/attachments/photo.png',
              detail: 'auto',
            },
          ],
        },
      ],
      tools: [
        createToolContract('read_chat_messages'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
  });

  it('turn boundary hook の追加入力を tool result の後に次ターンへ渡す', async () => {
    const firstOutput = [
      {
        type: 'tool_call' as const,
        callId: 'call-think',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
    ];
    const terminalOutput = [
      {
        type: 'tool_call' as const,
        callId: 'call-finish',
        toolName: 'finish_thinking',
        input: createFinishThinkingInput(),
      },
    ];
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: firstOutput,
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: terminalOutput,
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });
    const onTurnBoundary = vi
      .fn()
      .mockResolvedValueOnce([
        {
          role: 'developer',
          content: '<memory_module_result>ready</memory_module_result>',
        },
        {
          role: 'developer',
          content: '<emotion_module_result>ready</emotion_module_result>',
        },
      ])
      .mockResolvedValueOnce([
        {
          role: 'developer',
          content: 'terminal input must not be sent to another model turn',
        },
      ]);

    await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [{ role: 'developer', content: 'test' }],
      onTurnBoundary,
    });

    const firstResolvedInput = [
      {
        type: 'tool_result',
        callId: 'call-think',
        output: '{"success":true}',
      },
    ];
    expect(onTurnBoundary).toHaveBeenNthCalledWith(1, {
      turnIndex: 1,
      responseOutput: firstOutput,
      toolCalls: firstOutput,
      resolvedInput: firstResolvedInput,
      terminationReason: null,
    });
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [
        ...firstResolvedInput,
        {
          role: 'developer',
          content: '<memory_module_result>ready</memory_module_result>',
        },
        {
          role: 'developer',
          content: '<emotion_module_result>ready</emotion_module_result>',
        },
      ],
      tools: [
        createToolContract('think_deeply'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(onTurnBoundary).toHaveBeenNthCalledWith(2, {
      turnIndex: 2,
      responseOutput: terminalOutput,
      toolCalls: terminalOutput,
      resolvedInput: [
        {
          type: 'tool_result',
          callId: 'call-finish',
          output: '{"success":true}',
        },
      ],
      terminationReason: 'finish_thinking',
    });
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it('finish_thinking を含む場合は tool 実行後に終了する', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-think',
          toolName: 'think_deeply',
          input: '{"thought":"test"}',
        },
        {
          type: 'tool_call',
          callId: 'call-finish',
          toolName: 'finish_thinking',
          input: createFinishThinkingInput(),
        },
      ],
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
    });
    const executeThink = vi.fn().mockResolvedValue('{"success":true}');
    const executeFinish = vi.fn().mockResolvedValue('{"success":true}');

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: executeThink,
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: executeFinish,
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(executeThink).toHaveBeenCalled();
    expect(executeFinish).toHaveBeenCalled();
    expect(generate).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
      terminationReason: 'finish_thinking',
    });
  });

  it('finish_thinking の next_wake_at を返す', async () => {
    const nextWakeAt = '2026-03-23T00:00:00.000Z';
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-finish',
          toolName: 'finish_thinking',
          input: createFinishThinkingInput('done', nextWakeAt),
        },
      ],
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
    });

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(result).toEqual({
      nextWakeAt,
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
      terminationReason: 'finish_thinking',
    });
  });

  it('無効な finish_thinking は継続し、有効な finish_thinking で終了する', async () => {
    const invalidFinishInput = JSON.stringify({
      next_wake_at: '2026-08-29T12:00:00.000Z',
    });
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish-invalid',
            toolName: 'finish_thinking',
            input: invalidFinishInput,
          },
        ],
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish-valid',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput('done for real'),
          },
        ],
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });
    const executeFinish = vi
      .fn()
      .mockResolvedValueOnce('{"success":false,"error":"invalid finish"}')
      .mockResolvedValueOnce('{"success":true}');

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: executeFinish,
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [
        {
          type: 'tool_result',
          callId: 'call-finish-invalid',
          output: '{"success":false,"error":"invalid finish"}',
        },
      ],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 15 }),
      responseToken: 'resp-2',
      terminationReason: 'finish_thinking',
    });
  });

  it('入力が有効でも finish_thinking の実行に失敗した場合は継続する', async () => {
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish-failed',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput('not ready'),
          },
        ],
        usage: createUsage({ totalTokens: 10 }),
        responseToken: 'resp-1',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish-succeeded',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput('done for real'),
          },
        ],
        usage: createUsage({ totalTokens: 5 }),
        responseToken: 'resp-2',
      });
    const executeFinish = vi
      .fn()
      .mockResolvedValueOnce('{"success":false,"error":"completion rejected"}')
      .mockResolvedValueOnce('{"success":true}');

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: executeFinish,
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
    });

    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [
        {
          type: 'tool_result',
          callId: 'call-finish-failed',
          output: '{"success":false,"error":"completion rejected"}',
        },
      ],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 15 }),
      responseToken: 'resp-2',
      terminationReason: 'finish_thinking',
    });
  });

  it('boundary failure でも課金済み Main usage と response token を保持する', async () => {
    const boundaryFailure = new Error('cognitive boundary failed');
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-think',
          toolName: 'think_deeply',
          input: '{"thought":"inspect"}',
        },
      ],
      usage: createUsage({ totalTokens: 7 }),
      responseToken: 'resp-paid-turn',
    });

    const execution = runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [],
      onTurnBoundary: vi.fn().mockRejectedValue(boundaryFailure),
    });

    const error = await execution.catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(AgentSessionExecutionError);
    expect(error).toMatchObject({
      cause: boundaryFailure,
      usage: createUsage({ totalTokens: 7 }),
      responseToken: 'resp-paid-turn',
    });
  });

  it('並列read_web_pageはthinking session内で4件だけ実行する', async () => {
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: Array.from({ length: 5 }, (_, index) => ({
          type: 'tool_call' as const,
          callId: `call-web-${index + 1}`,
          toolName: 'read_web_page',
          input: JSON.stringify({ url: `https://example.com/${index + 1}` }),
        })),
        usage: createUsage(),
        responseToken: 'resp-web',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-finish',
            toolName: 'finish_thinking',
            input: createFinishThinkingInput(),
          },
        ],
        usage: createUsage(),
        responseToken: 'resp-finish',
      });
    const executeWeb = vi.fn().mockResolvedValue('{"success":true}');

    await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'read_web_page',
          contract: createToolContract('read_web_page'),
          execute: executeWeb,
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [],
    });

    expect(executeWeb).toHaveBeenCalledTimes(4);
    const secondRequest = generate.mock.calls[1]?.[0];
    expect(secondRequest).toBeDefined();
    expect(secondRequest?.input).toHaveLength(5);
    const fifthResult = secondRequest?.input[4];
    expect(fifthResult).toEqual({
      type: 'tool_result',
      callId: 'call-web-5',
      output: JSON.stringify({
        success: false,
        code: 'budget_exceeded',
        error:
          'The read_web_page call limit for this thinking session was reached.',
        retryable: false,
      }),
    });
  });

  it('read_web_pageの4件上限は新しいsessionでresetされる', async () => {
    const executeWeb = vi.fn().mockResolvedValue('{"success":true}');
    const tools = [
      {
        name: 'read_web_page',
        contract: createToolContract('read_web_page'),
        execute: executeWeb,
      },
    ];

    const runOnce = async (): Promise<void> => {
      const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
        output: Array.from({ length: 4 }, (_, index) => ({
          type: 'tool_call' as const,
          callId: `call-web-${index + 1}`,
          toolName: 'read_web_page',
          input: JSON.stringify({ url: `https://example.com/${index + 1}` }),
        })),
        usage: createUsage(),
        responseToken: 'resp-web',
      });

      await runAgentSession({
        model: { generate },
        tools,
        initialInput: [],
        maxTurns: 1,
      });
    };

    await runOnce();
    await runOnce();

    expect(executeWeb).toHaveBeenCalledTimes(8);
  });

  it('maxTurns を超えたら終了理由を返す', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-1',
          toolName: 'think_deeply',
          input: '{"thought":"test"}',
        },
      ],
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
    });

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
      maxTurns: 2,
    });

    expect(generate).toHaveBeenCalledTimes(2);
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 20 }),
      responseToken: 'resp-1',
      terminationReason: 'max_turns',
    });
  });

  it('tool call が無い状態が続いても maxTurns で終了する', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: 'still thinking',
        },
      ],
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
    });

    const result = await runAgentSession({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn(),
        },
      ],
      initialInput: [
        {
          role: 'developer',
          content: 'test',
        },
      ],
      maxTurns: 2,
    });

    expect(generate).toHaveBeenCalledTimes(2);
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
      turnIndex: 2,
    });
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 20 }),
      responseToken: 'resp-1',
      terminationReason: 'max_turns',
    });
  });
});
