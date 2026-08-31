import { describe, expect, it, vi } from 'vitest';

import {
  CognitiveModuleOutputValidationError,
  ParallelCognitiveModuleOrchestrator,
} from './cognitive-module-orchestrator';

import type {
  CognitiveModuleCommittedState,
  CognitiveModuleDomainPort,
  CognitiveModulePhaseResult,
  CognitiveModuleRetryInput,
  CognitiveModuleRunResult,
  CognitiveModuleRunner,
} from './cognitive-module-orchestrator';
import type {
  EmotionCognitiveModuleOutput,
  MemoryCognitiveModuleOutput,
} from './cognitive-module-schema';
import type { AgentSessionTurnBoundary } from './session';
import type { ModelInputItem, ModelUsage } from '../ports/model';

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

type MemoryRun = CognitiveModuleRunner<MemoryCognitiveModuleOutput>['run'];
type EmotionRun = CognitiveModuleRunner<EmotionCognitiveModuleOutput>['run'];
type ShouldRetry = (input: CognitiveModuleRetryInput) => boolean;
type FormatHandoff = (
  result: CognitiveModulePhaseResult,
  committed: CognitiveModuleCommittedState
) => readonly ModelInputItem[];
type FormatInitialContext = (
  committed: CognitiveModuleCommittedState
) => readonly ModelInputItem[];
type MockCognitiveModuleDomain = Omit<
  CognitiveModuleDomainPort,
  'commitPhase' | 'failPhase'
> & {
  commitPhase: ReturnType<
    typeof vi.fn<CognitiveModuleDomainPort['commitPhase']>
  >;
  failPhase: ReturnType<typeof vi.fn<CognitiveModuleDomainPort['failPhase']>>;
};

function recallOutput(query = 'relevant context'): MemoryCognitiveModuleOutput {
  return { query };
}

function storeOutput(content = 'session memory'): MemoryCognitiveModuleOutput {
  return { content, type: 'episode' };
}

function emotionOutput(label = 'calm'): EmotionCognitiveModuleOutput {
  return { valence: 0.1, arousal: 0.2, labels: [label] };
}

function createDomain(
  initialState: CognitiveModuleCommittedState = {
    version: 0,
    emotion: null,
    previousSessionMemory: null,
    recalledMemories: [],
  }
): MockCognitiveModuleDomain {
  let state = initialState;

  return {
    beginActivation: vi.fn(async () => await Promise.resolve(state)),
    startPhase: vi.fn(async () => {
      await Promise.resolve();
    }),
    commitPhase: vi.fn<CognitiveModuleDomainPort['commitPhase']>(
      async ({ phase, emotion }) => {
        state = {
          version: state.version + 1,
          emotion: emotion.value,
          previousSessionMemory: state.previousSessionMemory,
          recalledMemories:
            phase.phase === 'pre_main'
              ? [
                  {
                    content: `recall-${phase.sequence}`,
                    type: 'semantic',
                    emotion: emotion.value,
                    createdAt: '2026-08-23T00:00:00.000Z',
                  },
                ]
              : [],
        };
        return await Promise.resolve(state);
      }
    ),
    failPhase: vi.fn<CognitiveModuleDomainPort['failPhase']>(async () => {
      await Promise.resolve();
    }),
  };
}

function deferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve: (value): void => resolvePromise?.(value),
  };
}

function createUsage(totalTokens = 0): ModelUsage {
  return {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: totalTokens,
    totalInputTokens: totalTokens,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens,
  };
}

function moduleResult<T>(
  value: T,
  totalTokens = 0
): CognitiveModuleRunResult<T> {
  return {
    value,
    usage: createUsage(totalTokens),
  };
}

function mainThinkExchange(
  turnIndex: number,
  outputIndex: number,
  thought: string
): ModelInputItem[] {
  const callId = `cognitive:main:${turnIndex}:think:${outputIndex}`;
  return [
    {
      type: 'tool_call',
      callId,
      toolName: 'think',
      input: JSON.stringify({ thought }),
    },
    {
      type: 'tool_result',
      callId,
      output: '{"success":true}',
    },
  ];
}

function boundary(
  terminationReason: AgentSessionTurnBoundary['terminationReason'] = null,
  withTool = true
): AgentSessionTurnBoundary {
  const toolCall = {
    type: 'tool_call' as const,
    callId: 'call-1',
    toolName: 'think_deeply',
    input: '{"thought":"test"}',
  };
  return {
    turnIndex: 1,
    responseOutput: withTool
      ? [toolCall]
      : [{ type: 'message', role: 'assistant', content: 'still thinking' }],
    toolCalls: withTool ? [toolCall] : [],
    resolvedInput: withTool
      ? [
          {
            type: 'tool_result',
            callId: 'call-1',
            output: '{"success":true}',
          },
        ]
      : [],
    terminationReason,
  };
}

function createOrchestrator(input: {
  memoryRun: MemoryRun;
  emotionRun: EmotionRun;
  domain?: CognitiveModuleDomainPort;
  maxAttempts?: number;
  shouldRetry?: ShouldRetry;
  formatInitialContext?: FormatInitialContext;
  formatHandoff?: FormatHandoff;
}): ParallelCognitiveModuleOrchestrator {
  return new ParallelCognitiveModuleOrchestrator({
    createActivationId: (): string => 'activation-1',
    memory: { run: input.memoryRun },
    emotion: { run: input.emotionRun },
    retryPolicy: {
      maxAttempts: input.maxAttempts ?? 1,
      shouldRetry: input.shouldRetry ?? ((): boolean => false),
    },
    domain: input.domain ?? createDomain(),
    formatInitialContext:
      input.formatInitialContext ?? ((): readonly ModelInputItem[] => []),
    formatHandoff: input.formatHandoff ?? ((): readonly ModelInputItem[] => []),
  });
}

describe('ParallelCognitiveModuleOrchestrator', () => {
  it('MemoryとEmotionの初回呼び出しを同じcontextから並列実行する', async () => {
    const memoryDeferred =
      deferred<CognitiveModuleRunResult<MemoryCognitiveModuleOutput>>();
    const emotionDeferred =
      deferred<CognitiveModuleRunResult<EmotionCognitiveModuleOutput>>();
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockReturnValue(memoryDeferred.promise);
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockReturnValue(emotionDeferred.promise);
    const formatHandoff = vi.fn(
      (
        result: CognitiveModulePhaseResult,
        committed: CognitiveModuleCommittedState
      ): ModelInputItem[] => [
        {
          role: 'developer',
          content: JSON.stringify({ result, committed }),
        },
      ]
    );
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      formatHandoff,
    }).beginActivation();

    const initialInput: ModelInputItem[] = [
      { role: 'developer', content: 'initial main input' },
    ];
    const preMain = activation.beforeMain(initialInput);
    await vi.waitFor((): void => {
      expect(memoryRun).toHaveBeenCalledTimes(1);
      expect(emotionRun).toHaveBeenCalledTimes(1);
    });

    expect(memoryRun.mock.calls[0]?.[0]).toEqual(emotionRun.mock.calls[0]?.[0]);
    expect(memoryRun.mock.calls[0]?.[0]).toMatchObject({
      boundaryId: 'activation-1:1:pre_main',
      sequence: 1,
      phase: 'pre_main',
      committed: {
        version: 0,
        emotion: null,
        previousSessionMemory: null,
        recalledMemories: [],
      },
    });
    expect(memoryRun.mock.calls[0]?.[1].sharedContext).toEqual(initialInput);
    expect(memoryRun.mock.calls[0]?.[1].sharedContext).toBe(
      emotionRun.mock.calls[0]?.[1].sharedContext
    );

    memoryDeferred.resolve(moduleResult(recallOutput()));
    emotionDeferred.resolve(moduleResult(emotionOutput()));
    await preMain;

    expect(formatHandoff).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'pre_main' }),
      expect.objectContaining({
        version: 1,
        recalledMemories: [expect.objectContaining({ content: 'recall-1' })],
      })
    );
  });

  it('前sessionのMemoryとEmotionを両moduleだけの初期contextへ載せる', async () => {
    const persistedEmotion = emotionOutput('persisted');
    const previousSessionMemory = {
      content: 'previous session memory',
      type: 'episode' as const,
      emotion: persistedEmotion,
      createdAt: '2026-08-23T00:00:00.000Z',
    };
    const domain = createDomain({
      version: 4,
      emotion: persistedEmotion,
      previousSessionMemory,
      recalledMemories: [],
    });
    const restoredSessionContext: ModelInputItem[] = [
      {
        role: 'user',
        content: [
          '前回の思考セッション終了時の状態です。',
          JSON.stringify({
            memory: previousSessionMemory,
            emotion: persistedEmotion,
          }),
        ].join('\n'),
      },
    ];
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockResolvedValue(moduleResult(recallOutput()));
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValue(moduleResult(emotionOutput()));
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      domain,
      formatInitialContext: () => restoredSessionContext,
    }).beginActivation();
    const initialInput: ModelInputItem[] = [
      { role: 'developer', content: 'initial' },
    ];

    await expect(activation.beforeMain(initialInput)).resolves.toEqual([]);

    const expectedContext = [...restoredSessionContext, ...initialInput];
    expect(memoryRun.mock.calls[0]?.[1].sharedContext).toEqual(expectedContext);
    expect(emotionRun.mock.calls[0]?.[1].sharedContext).toEqual(
      expectedContext
    );
  });

  it('Mainの自然言語をthinkへ変換し、実tool履歴と確定済みhandoffを次phaseへ渡す', async () => {
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockResolvedValueOnce(moduleResult(recallOutput('memory-1')))
      .mockResolvedValueOnce(moduleResult(recallOutput('memory-2')))
      .mockResolvedValueOnce(moduleResult(recallOutput('memory-3')));
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValueOnce(moduleResult(emotionOutput('emotion-1')))
      .mockResolvedValueOnce(moduleResult(emotionOutput('emotion-2')))
      .mockResolvedValueOnce(moduleResult(emotionOutput('emotion-3')));
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      formatHandoff: ({ boundaryId }) => [
        {
          type: 'tool_call',
          callId: boundaryId,
          toolName: 'search_memory',
          input: JSON.stringify({ query: boundaryId }),
        },
        {
          type: 'tool_result',
          callId: boundaryId,
          output: JSON.stringify({ success: true }),
        },
      ],
    }).beginActivation();

    await activation.beforeMain([{ role: 'developer', content: 'initial' }]);
    await expect(
      activation.onMainTurnBoundary(boundary(null, false))
    ).resolves.toEqual([
      {
        type: 'tool_call',
        callId: 'activation-1:2:pre_main',
        toolName: 'search_memory',
        input: '{"query":"activation-1:2:pre_main"}',
      },
      {
        type: 'tool_result',
        callId: 'activation-1:2:pre_main',
        output: '{"success":true}',
      },
    ]);
    await expect(
      activation.onMainTurnBoundary({ ...boundary(), turnIndex: 2 })
    ).resolves.toEqual([
      {
        type: 'tool_call',
        callId: 'activation-1:3:pre_main',
        toolName: 'search_memory',
        input: '{"query":"activation-1:3:pre_main"}',
      },
      {
        type: 'tool_result',
        callId: 'activation-1:3:pre_main',
        output: '{"success":true}',
      },
    ]);

    expect(memoryRun).toHaveBeenCalledTimes(3);
    expect(emotionRun).toHaveBeenCalledTimes(3);
    expect(memoryRun.mock.calls[1]?.[1].sharedContext).toEqual([
      { role: 'developer', content: 'initial' },
      {
        type: 'tool_call',
        callId: 'activation-1:1:pre_main',
        toolName: 'search_memory',
        input: '{"query":"activation-1:1:pre_main"}',
      },
      {
        type: 'tool_result',
        callId: 'activation-1:1:pre_main',
        output: '{"success":true}',
      },
      ...mainThinkExchange(1, 1, 'still thinking'),
    ]);
    expect(memoryRun.mock.calls[1]?.[1].sharedContext).toBe(
      emotionRun.mock.calls[1]?.[1].sharedContext
    );
    expect(memoryRun.mock.calls[2]?.[1].sharedContext).toEqual([
      { role: 'developer', content: 'initial' },
      {
        type: 'tool_call',
        callId: 'activation-1:1:pre_main',
        toolName: 'search_memory',
        input: '{"query":"activation-1:1:pre_main"}',
      },
      {
        type: 'tool_result',
        callId: 'activation-1:1:pre_main',
        output: '{"success":true}',
      },
      ...mainThinkExchange(1, 1, 'still thinking'),
      {
        type: 'tool_call',
        callId: 'activation-1:2:pre_main',
        toolName: 'search_memory',
        input: '{"query":"activation-1:2:pre_main"}',
      },
      {
        type: 'tool_result',
        callId: 'activation-1:2:pre_main',
        output: '{"success":true}',
      },
      {
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
      {
        type: 'tool_result',
        callId: 'call-1',
        output: '{"success":true}',
      },
    ]);
  });

  it('Main turnの出力・tool result・画像をnativeな履歴として保持する', async () => {
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockResolvedValue(moduleResult(recallOutput()));
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValue(moduleResult(emotionOutput()));
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
    }).beginActivation();
    const toolCall = {
      type: 'tool_call' as const,
      callId: 'read-1',
      toolName: 'read_chat_messages',
      input: '{"channelKey":"general","limit":10}',
    };
    const toolResult = {
      type: 'tool_result' as const,
      callId: 'read-1',
      output: '{"success":true}',
    };
    const imageContext = [
      { type: 'text' as const, text: 'チャットに添付された画像です。' },
      {
        type: 'image' as const,
        imageUrl: 'https://example.com/image.png',
        detail: 'auto' as const,
      },
    ];

    await activation.beforeMain([]);
    await activation.onMainTurnBoundary({
      turnIndex: 1,
      responseOutput: [toolCall],
      toolCalls: [toolCall],
      resolvedInput: [toolResult, { role: 'user', content: imageContext }],
      terminationReason: null,
    });

    expect(memoryRun.mock.calls[1]?.[1].sharedContext).toEqual([
      toolCall,
      toolResult,
      { role: 'user', content: imageContext },
    ]);
    expect(emotionRun.mock.calls[1]?.[1].sharedContext).toEqual(
      memoryRun.mock.calls[1]?.[1].sharedContext
    );
  });

  it('post_mainではstore入力とEmotionを一度確定しMain handoffを返さない', async () => {
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockResolvedValueOnce(moduleResult(recallOutput()))
      .mockResolvedValueOnce(moduleResult(storeOutput()));
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValueOnce(moduleResult(emotionOutput()))
      .mockResolvedValueOnce(moduleResult(emotionOutput('settled')));
    const formatHandoff = vi.fn((): ModelInputItem[] => []);
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      formatHandoff,
    }).beginActivation();

    await activation.beforeMain([{ role: 'developer', content: 'initial' }]);
    await expect(
      activation.onMainTurnBoundary(boundary('finish_thinking'))
    ).resolves.toEqual([]);

    expect(memoryRun.mock.calls[1]?.[0]).toMatchObject({
      phase: 'post_main',
    });
    expect(memoryRun.mock.calls[1]?.[1].sharedContext).toEqual([
      { role: 'developer', content: 'initial' },
      {
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'think_deeply',
        input: '{"thought":"test"}',
      },
      {
        type: 'tool_result',
        callId: 'call-1',
        output: '{"success":true}',
      },
    ]);
    expect(emotionRun.mock.calls[1]?.[1].sharedContext).toEqual(
      memoryRun.mock.calls[1]?.[1].sharedContext
    );
    expect(formatHandoff).toHaveBeenCalledTimes(1);
    expect(activation.getResultSnapshot().phases).toMatchObject([
      { phase: 'pre_main' },
      {
        phase: 'post_main',
        memory: { status: 'ready', value: storeOutput(), attempts: 1 },
        emotion: {
          status: 'ready',
          value: emotionOutput('settled'),
          attempts: 1,
        },
      },
    ]);
  });

  it('一時失敗したmoduleだけを同じ共有contextから再試行する', async () => {
    const transient = new Error('temporary memory failure');
    const memoryRun = vi
      .fn<MemoryRun>()
      .mockResolvedValueOnce(moduleResult(recallOutput(), 2))
      .mockRejectedValueOnce(transient)
      .mockResolvedValueOnce(moduleResult(storeOutput(), 4));
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValueOnce(moduleResult(emotionOutput(), 3))
      .mockResolvedValueOnce(moduleResult(emotionOutput(), 5));
    const shouldRetry = vi.fn((): boolean => true);
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      maxAttempts: 2,
      shouldRetry,
    }).beginActivation();

    await activation.beforeMain([{ role: 'developer', content: 'initial' }]);
    await activation.onMainTurnBoundary(boundary('finish_thinking'));

    expect(memoryRun).toHaveBeenCalledTimes(3);
    expect(emotionRun).toHaveBeenCalledTimes(2);
    expect(memoryRun.mock.calls[1]?.[0]).toEqual(memoryRun.mock.calls[2]?.[0]);
    expect(memoryRun.mock.calls[1]?.[1].sharedContext).toBe(
      memoryRun.mock.calls[2]?.[1].sharedContext
    );
    expect(shouldRetry).toHaveBeenCalledWith({
      module: 'memory',
      error: transient,
      failedAttempt: 1,
    });
    expect(activation.getResultSnapshot().usage.totalTokens).toBe(14);
  });

  it('non-retryable failureでは両moduleの完了後にfail closedにする', async () => {
    const invalidRequest = new Error('invalid memory request');
    const memoryRun = vi.fn<MemoryRun>().mockRejectedValue(invalidRequest);
    const emotionRun = vi
      .fn<EmotionRun>()
      .mockResolvedValue(moduleResult(emotionOutput(), 5));
    const domain = createDomain();
    const formatHandoff = vi.fn((): ModelInputItem[] => []);
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
      domain,
      maxAttempts: 2,
      shouldRetry: () => false,
      formatHandoff,
    }).beginActivation();

    await expect(
      activation.beforeMain([{ role: 'developer', content: 'initial' }])
    ).rejects.toMatchObject({
      name: 'CognitiveModulePhaseError',
      phaseResult: {
        memory: {
          status: 'failed',
          reason: 'non_retryable',
          error: 'invalid memory request',
          attempts: 1,
        },
        emotion: { status: 'ready', attempts: 1 },
      },
    });
    expect(domain.failPhase).toHaveBeenCalledTimes(1);
    expect(formatHandoff).not.toHaveBeenCalled();
    expect(activation.getResultSnapshot().usage.totalTokens).toBe(5);
  });

  it('provider後のvalidation failureでもusageとbounded診断を保持する', async () => {
    const memoryUsage = createUsage(2);
    const emotionUsage = createUsage(3);
    const memoryRun = vi.fn<MemoryRun>().mockRejectedValue(
      new CognitiveModuleOutputValidationError(
        'invalid memory output',
        memoryUsage,
        {
          code: 'schema_mismatch',
          diagnostic: {
            code: 'strict_schema',
            issues: [{ path: 'query', code: 'invalid_type' }],
          },
        }
      )
    );
    const emotionRun = vi.fn<EmotionRun>().mockResolvedValue({
      value: emotionOutput(),
      usage: emotionUsage,
    });
    const activation = createOrchestrator({
      memoryRun,
      emotionRun,
    }).beginActivation();

    await expect(
      activation.beforeMain([{ role: 'developer', content: 'initial' }])
    ).rejects.toMatchObject({
      phaseResult: {
        memory: {
          outputValidation: {
            code: 'schema_mismatch',
            diagnostic: {
              code: 'strict_schema',
              issues: [{ path: 'query', code: 'invalid_type' }],
            },
          },
        },
      },
    });
    expect(activation.getResultSnapshot().usage.totalTokens).toBe(5);
  });

  it('domain commit failureでは原因を記録してMain handoffを返さない', async () => {
    const commitFailure = new Error('durable storage unavailable');
    const domain = createDomain();
    domain.commitPhase.mockRejectedValueOnce(commitFailure);
    const formatHandoff = vi.fn((): ModelInputItem[] => []);
    const activation = createOrchestrator({
      memoryRun: vi
        .fn<MemoryRun>()
        .mockResolvedValue(moduleResult(recallOutput())),
      emotionRun: vi
        .fn<EmotionRun>()
        .mockResolvedValue(moduleResult(emotionOutput())),
      domain,
      formatHandoff,
    }).beginActivation();

    await expect(
      activation.beforeMain([{ role: 'developer', content: 'initial' }])
    ).rejects.toMatchObject({
      name: 'CognitiveModuleCommitError',
      message:
        'Cognitive module commit failed at activation-1:1:pre_main: durable storage unavailable',
    });
    expect(domain.failPhase).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'pre_main' }),
      commitFailure
    );
    expect(formatHandoff).not.toHaveBeenCalled();
  });
});
