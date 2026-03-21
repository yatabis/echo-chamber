import { afterEach, describe, expect, it, vi } from 'vitest';

import { buildAgentPromptMessages } from './prompt-builder';
import { ThinkingEngine } from './thinking-engine';

import type { MemoryRecord } from '../ports/memory';
import type { ModelPort, ModelToolContract, ModelUsage } from '../ports/model';

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

afterEach(() => {
  vi.useRealTimers();
});

describe('ThinkingEngine', () => {
  it('起動時 input を組み立てて session を実行する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-25T15:00:00.000Z'));

    const usage = createUsage({ totalTokens: 42 });
    const startupToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: 'done',
        },
      ],
      usage,
      responseToken: 'resp-1',
    });
    const thoughtLogSend = vi.fn().mockResolvedValue(undefined);
    const latestMemory: MemoryRecord = {
      content: 'Had a meaningful conversation.',
      type: 'episode',
      emotion: {
        valence: 0.7,
        arousal: 0.4,
        labels: ['joy', 'interest'],
      },
      createdAt: '2日前 (2025年01月23日 13:56:07)',
      updatedAt: '2日前 (2025年01月23日 13:56:07)',
    };

    const engine = new ThinkingEngine({
      model: { generate },
      thoughtLog: { send: thoughtLogSend },
      logger: {
        log: vi.fn(),
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      memory: {
        getLatest: vi.fn().mockResolvedValue(latestMemory),
      },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: startupToolExecute,
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn(),
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
    });

    const result = await engine.think();

    const promptMessages = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      latestMemory: {
        content: latestMemory.content,
        createdAt: latestMemory.createdAt,
        emotion: latestMemory.emotion,
      },
    });

    expect(startupToolExecute).toHaveBeenCalledWith('{}');
    expect(generate).toHaveBeenCalledWith({
      input: [
        ...promptMessages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
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
      tools: [
        createToolContract('check_notifications'),
        createToolContract('finish_thinking'),
      ],
      previousResponseToken: undefined,
    });
    expect(thoughtLogSend).toHaveBeenNthCalledWith(1, '*Thinking started...*');
    expect(thoughtLogSend).toHaveBeenNthCalledWith(2, '*Thinking completed.*');
    expect(result).toEqual(usage);
  });

  it('起動用 tool が未登録なら失敗する', async () => {
    const generate = vi.fn<ModelPort['generate']>();
    const thoughtLogSend = vi.fn().mockResolvedValue(undefined);

    const engine = new ThinkingEngine({
      model: { generate },
      thoughtLog: { send: thoughtLogSend },
      logger: {
        log: vi.fn(),
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      memory: {
        getLatest: vi.fn().mockResolvedValue(null),
      },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn(),
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
    });

    await expect(engine.think()).rejects.toThrow(
      "Required startup tool 'check_notifications' is not registered"
    );
    expect(thoughtLogSend).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });

  it('起動用 tool が失敗したら complete message を送らない', async () => {
    const startupToolExecute = vi
      .fn()
      .mockRejectedValue(new Error('startup failed'));
    const generate = vi.fn<ModelPort['generate']>();
    const thoughtLogSend = vi.fn().mockResolvedValue(undefined);

    const engine = new ThinkingEngine({
      model: { generate },
      thoughtLog: { send: thoughtLogSend },
      logger: {
        log: vi.fn(),
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      memory: {
        getLatest: vi.fn().mockResolvedValue(null),
      },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: startupToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
    });

    await expect(engine.think()).rejects.toThrow('startup failed');
    expect(thoughtLogSend).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });
});
