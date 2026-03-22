import { afterEach, describe, expect, it, vi } from 'vitest';

import { buildAgentPromptMessages } from './prompt-builder';
import { ThinkingEngine } from './thinking-engine';

import type { ContextSnapshot } from '../ports/context';
import type { MemorySearchResult } from '../ports/memory';
import type {
  ModelInputItem,
  ModelPort,
  ModelRequest,
  ModelToolContract,
  ModelUsage,
} from '../ports/model';

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

function createRelatedMemories(): MemorySearchResult[] {
  return [
    {
      content: 'Handled a similar high-priority thread before.',
      type: 'episode',
      createdAt: '1日前 (2025年01月25日 09:00:00)',
      updatedAt: '1日前 (2025年01月25日 09:00:00)',
      emotion: {
        valence: 0.2,
        arousal: 0.3,
        labels: ['focused'],
      },
      similarity: 0.87,
    },
  ];
}

function isDeveloperMessage(
  item: ModelInputItem
): item is { role: 'developer'; content: string } {
  return 'role' in item && item.role === 'developer';
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
    const relatedMemories = createRelatedMemories();
    const relatedMemory = relatedMemories[0];
    const searchMemory = vi.fn().mockResolvedValue(relatedMemories);

    if (relatedMemory === undefined) {
      throw new Error('Expected a related memory fixture');
    }

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
      memory: {
        search: searchMemory,
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
      relatedMemories: [
        {
          content: relatedMemory.content,
          type: relatedMemory.type,
          createdAt: relatedMemory.createdAt,
          emotion: relatedMemory.emotion,
          similarity: relatedMemory.similarity,
        },
      ],
    });

    expect(searchMemory).toHaveBeenCalledWith(latestContext.content);
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
      memory: {
        search: vi.fn(),
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
      memory: {
        search: vi.fn(),
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
      memory: {
        search: vi.fn(),
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

  it('関連メモリ検索に失敗しても warn して起動を継続する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-25T15:00:00.000Z'));
    const usage = createUsage({ totalTokens: 7 });
    const startupToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const finishToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    let capturedRequest: ModelRequest | undefined;
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockImplementation(async (request: ModelRequest) => {
        capturedRequest = request;
        return Promise.resolve({
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
      });
    const warn = vi.fn().mockResolvedValue(undefined);
    const latestContext = createSessionContext();

    const engine = new ThinkingEngine({
      model: { generate },
      thoughtLog: { send: vi.fn().mockResolvedValue(undefined) },
      logger: {
        log: vi.fn(),
        debug: vi.fn(),
        info: vi.fn(),
        warn,
        error: vi.fn(),
      },
      context: {
        load: vi.fn().mockResolvedValue(latestContext),
      },
      memory: {
        search: vi.fn().mockRejectedValue(new Error('memory search failed')),
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

    await engine.think();

    expect(warn).toHaveBeenCalledWith(
      'Failed to load related memories for startup context: memory search failed'
    );
    expect(capturedRequest).toBeDefined();
    if (capturedRequest === undefined) {
      throw new Error('Expected generate to be called');
    }

    let runtimeContextContent: string | null = null;
    for (const item of capturedRequest.input) {
      if (!isDeveloperMessage(item)) {
        continue;
      }

      if (item.content.includes('<runtime_context>')) {
        runtimeContextContent = item.content;
        break;
      }
    }

    expect(runtimeContextContent).not.toBeNull();
    expect(runtimeContextContent).toContain('Related memories:\n[]');
  });
});
