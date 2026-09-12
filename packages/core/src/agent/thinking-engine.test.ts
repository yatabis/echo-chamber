import { afterEach, describe, expect, it, vi } from 'vitest';

import { CognitiveModulePhaseError } from './cognitive-module-orchestrator';
import { buildAgentPromptMessages } from './prompt-builder';
import {
  ThinkingEngine,
  ThinkingEngineExecutionError,
} from './thinking-engine';

import type {
  CognitiveModuleActivation,
  CognitiveModuleActivationResult,
  CognitiveModuleOrchestrator,
  CognitiveModulePhaseResult,
} from './cognitive-module-orchestrator';
import type { AgentSessionTurnBoundaryHandler } from './session';
import type { EchoEventPort } from '../ports/echo-event';
import type {
  ModelInputItem,
  ModelPort,
  ModelToolContract,
  ModelUsage,
} from '../ports/model';

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

function createNoopCognitiveModules(): CognitiveModuleOrchestrator {
  const result: CognitiveModuleActivationResult = {
    activationId: 'noop-cognitive-activation',
    phases: [],
    usage: createUsage(),
  };
  return {
    beginActivation: (): CognitiveModuleActivation => ({
      // eslint-disable-next-line @typescript-eslint/require-await
      beforeMain: async (): Promise<readonly ModelInputItem[]> => [],
      // eslint-disable-next-line @typescript-eslint/require-await
      onMainTurnBoundary: async (): Promise<readonly ModelInputItem[]> => [],
      getResultSnapshot: () => result,
    }),
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

function createFinishThinkingInput(nextWakeAt?: string): string {
  return JSON.stringify({
    reason: 'done',
    next_wake_at: nextWakeAt,
  });
}

afterEach(() => {
  vi.useRealTimers();
});

describe('ThinkingEngine', () => {
  it('cognitive module activation を main session 境界へ接続して usage を集計する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-25T15:00:00.000Z'));
    const startupToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const thinkToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const finishToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [
          {
            type: 'tool_call',
            callId: 'call-think',
            toolName: 'think_deeply',
            input: '{"thought":"inspect context"}',
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
    const preMainInput: ModelInputItem = {
      role: 'developer',
      content: 'pre-main cognitive observations',
    };
    const nextTurnCognitiveInput: ModelInputItem = {
      role: 'developer',
      content: 'next-turn cognitive observations',
    };
    const beforeMain = vi
      .fn<CognitiveModuleActivation['beforeMain']>()
      .mockResolvedValue([preMainInput]);
    const onMainTurnBoundary = vi
      .fn<AgentSessionTurnBoundaryHandler>()
      .mockResolvedValueOnce([nextTurnCognitiveInput])
      .mockResolvedValueOnce([]);
    const cognitiveResult: CognitiveModuleActivationResult = {
      activationId: 'activation-1',
      phases: [],
      usage: createUsage({ totalTokens: 8 }),
    };
    const beginActivation = vi
      .fn<() => CognitiveModuleActivation>()
      .mockReturnValue({
        beforeMain,
        onMainTurnBoundary,
        getResultSnapshot: () => cognitiveResult,
      });
    const engine = new ThinkingEngine({
      model: { generate },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: startupToolExecute,
        },
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: thinkToolExecute,
        },
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: finishToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
      cognitiveModules: { beginActivation },
    });

    const result = await engine.think();

    expect(beginActivation).toHaveBeenCalledTimes(1);
    expect(beforeMain).toHaveBeenCalledTimes(1);
    const promptMessages = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      toolContracts: [
        createToolContract('check_notifications'),
        createToolContract('think_deeply'),
        createToolContract('finish_thinking'),
      ],
    });
    const sharedContext = beforeMain.mock.calls[0]?.[0];
    expect(sharedContext).toEqual([
      promptMessages.sharedRuntimeContext,
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
    ]);
    const firstRequestInput = generate.mock.calls[0]?.[0].input;
    expect(firstRequestInput?.[0]).toEqual(promptMessages.mainSystemPrompt);
    expect(firstRequestInput?.slice(1, -1)).toEqual(sharedContext);
    expect(firstRequestInput?.[firstRequestInput.length - 1]).toEqual(
      preMainInput
    );
    expect(generate.mock.calls[1]?.[0].input).toEqual([
      {
        type: 'tool_result',
        callId: 'call-think',
        output: '{"success":true}',
      },
      nextTurnCognitiveInput,
    ]);
    expect(onMainTurnBoundary).toHaveBeenCalledTimes(2);
    expect(onMainTurnBoundary.mock.calls[1]?.[0].terminationReason).toBe(
      'finish_thinking'
    );
    expect(result).toMatchObject({
      cognitiveModules: cognitiveResult,
      mainUsage: createUsage({ totalTokens: 15 }),
      usage: createUsage({ totalTokens: 23 }),
    });
  });

  it('main session の失敗を cognitive module が隠さない', async () => {
    const beforeMain = vi
      .fn<CognitiveModuleActivation['beforeMain']>()
      .mockResolvedValue([]);
    const onMainTurnBoundary = vi.fn<AgentSessionTurnBoundaryHandler>();
    const cognitiveActivation: CognitiveModuleActivation = {
      beforeMain,
      onMainTurnBoundary,
      getResultSnapshot: () => ({
        activationId: 'activation-main-failure',
        phases: [],
        usage: createUsage(),
      }),
    };
    const engine = new ThinkingEngine({
      model: {
        generate: vi.fn().mockRejectedValue(new Error('main model failed')),
      },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
      cognitiveModules: {
        beginActivation: (): CognitiveModuleActivation => cognitiveActivation,
      },
    });

    await expect(engine.think()).rejects.toThrow('main model failed');
  });

  it('次turn前のcognitive boundaryが失敗したら完了済みtoolを再実行せずMainを進めない', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'tool_call',
          callId: 'call-think',
          toolName: 'think_deeply',
          input: '{"thought":"inspect context"}',
        },
      ],
      usage: createUsage({ totalTokens: 7 }),
      responseToken: 'resp-1',
    });
    const thinkToolExecute = vi.fn().mockResolvedValue('{"success":true}');
    const phaseUsage = createUsage({ totalTokens: 3 });
    const failedPhase: CognitiveModulePhaseResult = {
      activationId: 'activation-2',
      boundaryId: 'activation-2:2:pre_main',
      sequence: 2,
      phase: 'pre_main',
      committed: {
        version: 0,
        emotion: null,
        previousSessionMemory: null,
        recalledMemories: [],
      },
      memory: {
        status: 'failed',
        reason: 'non_retryable',
        error: 'invalid memory output',
        attempts: 1,
        outputValidation: {
          code: 'schema_mismatch',
          diagnostic: {
            code: 'strict_schema',
            issues: [{ path: 'query', code: 'invalid_type' }],
          },
        },
      },
      emotion: {
        status: 'ready',
        value: { valence: 0.1, arousal: 0.2, labels: ['calm'] },
        attempts: 1,
      },
      usage: phaseUsage,
    };
    const boundaryFailure = new CognitiveModulePhaseError(failedPhase);
    const beforeMain = vi
      .fn<CognitiveModuleActivation['beforeMain']>()
      .mockResolvedValue([]);
    const onMainTurnBoundary = vi
      .fn<AgentSessionTurnBoundaryHandler>()
      .mockRejectedValue(boundaryFailure);
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const engine = new ThinkingEngine({
      model: { generate },
      events: { emit },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: vi.fn().mockResolvedValue('{"success":true}'),
        },
        {
          name: 'think_deeply',
          contract: createToolContract('think_deeply'),
          execute: thinkToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
      cognitiveModules: {
        beginActivation: (): CognitiveModuleActivation => ({
          beforeMain,
          onMainTurnBoundary,
          getResultSnapshot: () => ({
            activationId: 'activation-2',
            phases: [failedPhase],
            usage: phaseUsage,
          }),
        }),
      },
    });

    const error = await engine.think().catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ThinkingEngineExecutionError);
    expect(error).toMatchObject({
      cause: boundaryFailure,
      mainUsage: createUsage({ totalTokens: 7 }),
      cognitiveUsage: phaseUsage,
      usage: createUsage({ totalTokens: 10 }),
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(thinkToolExecute).toHaveBeenCalledTimes(1);
    expect(onMainTurnBoundary).toHaveBeenCalledTimes(1);
    const failedEvent = emit.mock.calls.find(
      ([event]) => event.type === 'session.failed'
    )?.[0];
    expect(failedEvent).toMatchObject({
      type: 'session.failed',
      payload: {
        error: boundaryFailure.message,
        failureSource: 'cognitive_module',
        activationId: 'activation-2',
        boundaryId: 'activation-2:2:pre_main',
        phase: 'pre_main',
        failedModules: [
          {
            module: 'memory',
            reason: 'non_retryable',
            error: 'invalid memory output',
            attempts: 1,
            outputValidation: {
              code: 'schema_mismatch',
              diagnostic: {
                code: 'strict_schema',
                issues: [{ path: 'query', code: 'invalid_type' }],
              },
            },
          },
        ],
        cognitiveUsage: phaseUsage,
        mainUsage: createUsage({ totalTokens: 7 }),
        usage: createUsage({ totalTokens: 10 }),
      },
    });
  });

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
    const engine = new ThinkingEngine({
      model: { generate },
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
      cognitiveModules: createNoopCognitiveModules(),
    });

    const result = await engine.think();

    const promptMessages = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      toolContracts: [
        createToolContract('check_notifications'),
        createToolContract('finish_thinking'),
      ],
    });

    expect(startupToolExecute).toHaveBeenCalledWith('{}');
    expect(generate).toHaveBeenCalledWith({
      input: [
        promptMessages.mainSystemPrompt,
        promptMessages.sharedRuntimeContext,
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
      turnIndex: 1,
    });
    expect(finishToolExecute).toHaveBeenCalledWith(
      createFinishThinkingInput(nextWakeAt)
    );
    expect(result).toEqual({
      nextWakeAt,
      mainUsage: usage,
      usage,
      cognitiveModules: {
        activationId: 'noop-cognitive-activation',
        phases: [],
        usage: createUsage(),
      },
    });
  });

  it('起動用 tool が未登録なら失敗する', async () => {
    const generate = vi.fn<ModelPort['generate']>();

    const engine = new ThinkingEngine({
      model: { generate },
      tools: [
        {
          name: 'finish_thinking',
          contract: createToolContract('finish_thinking'),
          execute: vi.fn(),
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
      cognitiveModules: createNoopCognitiveModules(),
    });

    await expect(engine.think()).rejects.toThrow(
      "Required startup tool 'check_notifications' is not registered"
    );
    expect(generate).not.toHaveBeenCalled();
  });

  it('起動用 tool が失敗したら complete message を送らない', async () => {
    const startupToolExecute = vi
      .fn()
      .mockRejectedValue(new Error('startup failed'));
    const generate = vi.fn<ModelPort['generate']>();

    const engine = new ThinkingEngine({
      model: { generate },
      tools: [
        {
          name: 'check_notifications',
          contract: createToolContract('check_notifications'),
          execute: startupToolExecute,
        },
      ],
      systemPrompt: '<persona>Test persona</persona>',
      cognitiveModules: createNoopCognitiveModules(),
    });

    await expect(engine.think()).rejects.toThrow('startup failed');
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
      cognitiveModules: createNoopCognitiveModules(),
    });

    const result = await engine.think();

    expect(result.nextWakeAt).toBeNull();
  });
});
