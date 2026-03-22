import { describe, expect, it, vi } from 'vitest';

import {
  accumulateModelUsage,
  executeAgentToolCall,
  runAgentSession,
  ZERO_MODEL_USAGE,
} from './session';

import type { FinishThinkingSessionRecord } from './tools/thinking';
import type { ModelPort, ModelToolContract, ModelUsage } from '../ports/model';

const NO_TOOL_CALLS_CONTINUING_WARNING =
  'No tool calls returned; continuing until finish_thinking is called';

function createSessionContext(): FinishThinkingSessionRecord {
  return {
    content:
      'Responded to recent messages and left a concise recap for the next cycle.',
    emotion: {
      valence: 0.4,
      arousal: 0.2,
      labels: ['calm', 'satisfied'],
    },
  };
}

function createFinishThinkingInput(
  reason = 'done',
  nextWakeAt?: string
): string {
  return JSON.stringify({
    reason,
    next_wake_at: nextWakeAt,
    session_record: createSessionContext(),
  });
}

function createUsage(overrides?: Partial<ModelUsage>): ModelUsage {
  return {
    cachedInputTokens: 0,
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
          uncachedInputTokens: 2,
          totalInputTokens: 3,
          outputTokens: 4,
          reasoningTokens: 5,
          totalTokens: 6,
        }),
        createUsage({
          cachedInputTokens: 10,
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
    const logger = {
      warn: vi.fn().mockResolvedValue(undefined),
    };
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
      logger,
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
    });
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
    });
    expect(logger.warn).toHaveBeenCalledTimes(1);
    expect(logger.warn).toHaveBeenCalledWith(NO_TOOL_CALLS_CONTINUING_WARNING);
    expect(executeFinish).toHaveBeenCalledWith(createFinishThinkingInput());
    expect(result).toEqual({
      context: createSessionContext(),
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 15 }),
      responseToken: 'resp-2',
    });
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
    });
    expect(generate).toHaveBeenNthCalledWith(3, {
      input: [],
      tools: [
        createToolContract('think_deeply'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: 'resp-2',
    });
    expect(executeFinish).toHaveBeenCalledWith(createFinishThinkingInput());
    expect(result).toEqual({
      context: createSessionContext(),
      nextWakeAt: null,
      usage: createUsage({
        cachedInputTokens: 111,
        uncachedInputTokens: 222,
        totalInputTokens: 333,
        outputTokens: 444,
        reasoningTokens: 555,
        totalTokens: 666,
      }),
      responseToken: 'resp-3',
    });
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
      context: createSessionContext(),
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
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
      context: createSessionContext(),
      nextWakeAt,
      usage: createUsage({ totalTokens: 10 }),
      responseToken: 'resp-1',
    });
  });

  it('無効な finish_thinking は継続し、有効な finish_thinking で終了する', async () => {
    const invalidFinishInput = '{"reason":"done"}';
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
    });
    expect(result).toEqual({
      context: createSessionContext(),
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 15 }),
      responseToken: 'resp-2',
    });
  });

  it('maxTurns を超えたら warn を出して終了する', async () => {
    const logger = {
      warn: vi.fn().mockResolvedValue(undefined),
    };
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
      logger,
      maxTurns: 2,
    });

    expect(generate).toHaveBeenCalledTimes(2);
    expect(logger.warn).toHaveBeenCalledTimes(1);
    expect(logger.warn).toHaveBeenCalledWith('Maximum turns exceeded');
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 20 }),
      responseToken: 'resp-1',
    });
  });

  it('tool call が無い状態が続くと maxTurns で warn を出す', async () => {
    const logger = {
      warn: vi.fn().mockResolvedValue(undefined),
    };
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
      logger,
      maxTurns: 2,
    });

    expect(generate).toHaveBeenCalledTimes(2);
    expect(generate).toHaveBeenNthCalledWith(2, {
      input: [],
      tools: [createToolContract('finish_thinking')],
      previousResponseToken: 'resp-1',
    });
    expect(logger.warn).toHaveBeenCalledTimes(3);
    expect(logger.warn).toHaveBeenNthCalledWith(
      1,
      NO_TOOL_CALLS_CONTINUING_WARNING
    );
    expect(logger.warn).toHaveBeenNthCalledWith(
      2,
      NO_TOOL_CALLS_CONTINUING_WARNING
    );
    expect(logger.warn).toHaveBeenNthCalledWith(3, 'Maximum turns exceeded');
    expect(result).toEqual({
      nextWakeAt: null,
      usage: createUsage({ totalTokens: 20 }),
      responseToken: 'resp-1',
    });
  });
});
