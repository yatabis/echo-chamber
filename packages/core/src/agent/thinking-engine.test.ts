import { afterEach, describe, expect, it, vi } from 'vitest';

import { buildAgentPromptMessages } from './prompt-builder';
import { ThinkingEngine } from './thinking-engine';

import type { ContextSnapshot } from '../ports/context';
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

function createSessionContext(): ContextSnapshot {
  return {
    content:
      'Replied to urgent messages and left a short recap for the next run.',
    createdAt: '2025-01-24T12:34:56.000Z',
    updatedAt: '2025-01-24T12:34:56.000Z',
    emotion: {
      valence: 0.4,
      arousal: 0.2,
      labels: ['calm', 'satisfied'],
    },
  };
}

function createFinishThinkingInput(nextWakeAt?: string): string {
  return JSON.stringify({
    reason: 'done',
    next_wake_at: nextWakeAt,
    session_record: {
      content:
        'Replied to urgent messages and left a short recap for the next run.',
      emotion: {
        valence: 0.4,
        arousal: 0.2,
        labels: ['calm', 'satisfied'],
      },
    },
  });
}

afterEach(() => {
  vi.useRealTimers();
});

describe('ThinkingEngine', () => {
  it('起動時 input を組み立てて session を実行する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-25T15:00:00.000Z'));
    const nextWakeAt = '2026-03-23T00:00:00.000Z';

    const usage = createUsage({ totalTokens: 42 });
    const startupToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const finishToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-finish',
          toolName: 'finish_thinking',
          input: createFinishThinkingInput(nextWakeAt),
        },
      ],
      usage,
      responseToken: 'resp-1',
    });
    const thoughtLogSend = vi.fn().mockResolvedValue(undefined);
    const latestContext = createSessionContext();

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
      context: {
        load: vi.fn().mockResolvedValue(latestContext),
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
          execute: finishToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
    });

    const result = await engine.think();

    const promptMessages = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      latestContext: {
        content: latestContext.content,
        createdAt: latestContext.createdAt,
        emotion: latestContext.emotion,
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
    expect(finishToolExecute).toHaveBeenCalledWith(
      createFinishThinkingInput(nextWakeAt)
    );
    expect(result).toEqual({
      context: {
        content: latestContext.content,
        createdAt: '2025-01-25T15:00:00.000Z',
        emotion: latestContext.emotion,
        updatedAt: '2025-01-25T15:00:00.000Z',
      },
      nextWakeAt,
      usage,
    });
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
      context: {
        load: vi.fn().mockResolvedValue(null),
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
      context: {
        load: vi.fn().mockResolvedValue(null),
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

  it('finish_thinking に next_wake_at が無ければ null を返す', async () => {
    const usage = createUsage({ totalTokens: 10 });
    const startupToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const finishToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-finish',
          toolName: 'finish_thinking',
          input: createFinishThinkingInput(),
        },
      ],
      usage,
      responseToken: 'resp-1',
    });

    const engine = new ThinkingEngine({
      model: { generate },
      thoughtLog: { send: vi.fn().mockResolvedValue(undefined) },
      logger: {
        log: vi.fn(),
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      context: {
        load: vi.fn().mockResolvedValue(null),
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
          execute: finishToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
    });

    const result = await engine.think();

    expect(result.nextWakeAt).toBeNull();
  });
});
