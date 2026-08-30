import { describe, expect, it, vi } from 'vitest';

import type {
  CognitiveModuleCommittedState,
  CognitiveModuleDomainPort,
  CognitiveModuleName,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';
import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import type {
  ModelPort,
  ModelRequest,
  ModelUsage,
} from '@echo-chamber/core/ports/model';
import type { OpenAIResponsesModelOptions } from '@echo-chamber/openai-adapter/openai-responses-model';

import {
  createCognitiveModuleOrchestrator,
  isRetryableCognitiveModuleError,
} from './cognitive-modules';

type MockCognitiveModuleDomain = Omit<
  CognitiveModuleDomainPort,
  'commitPhase' | 'failPhase'
> & {
  commitPhase: ReturnType<
    typeof vi.fn<CognitiveModuleDomainPort['commitPhase']>
  >;
  failPhase: ReturnType<typeof vi.fn<CognitiveModuleDomainPort['failPhase']>>;
};

type SyntheticModuleOutput =
  | { query: string }
  | { content: string; type: 'episode' }
  | { valence: number; arousal: number; labels: string[] };

function createUsage(totalTokens: number): ModelUsage {
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

function createInstance(
  overrides: {
    model?: string;
    reasoningEffort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';
  } = {}
): EchoInstanceDefinition {
  const definition = getEchoInstanceDefinition('rin');
  return {
    ...definition,
    cognitiveModules: {
      ...definition.cognitiveModules,
      ...overrides,
    },
  };
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
                    content: 'existing related memory',
                    type: 'semantic',
                    emotion: {
                      valence: 0,
                      arousal: 0.1,
                      labels: ['calm'],
                    },
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

function getRequired<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new Error(`Expected ${label}`);
  }
  return value;
}

function createSyntheticModuleOutput(
  moduleIndex: number,
  outputName: string | undefined
): SyntheticModuleOutput {
  if (moduleIndex === 1) {
    return { valence: 0.1, arousal: 0.2, labels: ['calm'] };
  }
  if (outputName === 'cognitive_memory_store') {
    return {
      content: 'remember the completed thinking session',
      type: 'episode',
    };
  }
  return { query: 'current context and relevant memory' };
}

function createModelFactory(): {
  createModel: ReturnType<
    typeof vi.fn<(options: OpenAIResponsesModelOptions) => ModelPort>
  >;
  generateFunctions: ModelPort['generate'][];
} {
  const generateFunctions: ModelPort['generate'][] = [];
  const createModel = vi
    .fn<(options: OpenAIResponsesModelOptions) => ModelPort>()
    .mockImplementation(() => {
      const moduleIndex = generateFunctions.length;
      const generate = vi
        .fn<ModelPort['generate']>()
        .mockImplementation(async (request) => {
          const outputName = request.responseFormat?.name;
          const value = createSyntheticModuleOutput(moduleIndex, outputName);
          return await Promise.resolve({
            output: [
              {
                type: 'message',
                role: 'assistant',
                content: JSON.stringify(value),
              },
            ],
            usage: createUsage(3),
            responseToken: `${moduleIndex === 0 ? 'memory' : 'emotion'}-response-${request.turnIndex}`,
          });
        });
      generateFunctions.push(generate);
      return { generate };
    });
  return { createModel, generateFunctions };
}

describe('createCognitiveModuleOrchestrator', () => {
  it('enable flagなしで本流のHosted coordinatorを生成する', () => {
    const createModel = vi
      .fn<(options: OpenAIResponsesModelOptions) => ModelPort>()
      .mockReturnValue({ generate: vi.fn() });

    expect(
      createCognitiveModuleOrchestrator({
        env: { OPENAI_API_KEY: 'existing-key' },
        instance: createInstance(),
        domain: createDomain(),
        createModel,
      })
    ).toBeDefined();
    expect(createModel).toHaveBeenCalledTimes(2);
  });

  it('各Main turn前の出力をsearch_memoryとupdate_emotionの擬似tool exchangeにする', async () => {
    const { createModel, generateFunctions } = createModelFactory();
    const orchestrator = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance({
        model: 'instance-cognitive-model',
        reasoningEffort: 'medium',
      }),
      domain: createDomain(),
      createActivationId: () => 'rin:activation-1',
      createModel,
    });

    const sharedContext: ModelRequest['input'] = [
      { role: 'developer', content: '現在日時: 2026年08月24日' },
    ];
    const handoff = await orchestrator
      .beginActivation()
      .beforeMain(sharedContext);

    expect(createModel).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        apiKey: 'existing-key',
        model: 'instance-cognitive-model',
        reasoningEffort: 'medium',
        maxRetries: 0,
      })
    );
    expect(createModel).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        apiKey: 'existing-key',
        model: 'instance-cognitive-model',
        reasoningEffort: 'medium',
        maxRetries: 0,
      })
    );
    expect(generateFunctions).toHaveLength(2);
    const memoryGenerate = getRequired(
      generateFunctions[0],
      'Memory generate function'
    );
    const emotionGenerate = getRequired(
      generateFunctions[1],
      'Emotion generate function'
    );
    const memoryRequest = getRequired(
      vi.mocked(memoryGenerate).mock.calls[0]?.[0],
      'initial Memory request'
    );
    const emotionRequest = getRequired(
      vi.mocked(emotionGenerate).mock.calls[0]?.[0],
      'initial Emotion request'
    );
    expect(memoryRequest.input.slice(1)).toEqual(sharedContext);
    expect(emotionRequest.input.slice(1)).toEqual(sharedContext);
    expect(memoryRequest.previousResponseToken).toBeUndefined();
    expect(emotionRequest.previousResponseToken).toBeUndefined();
    expect(memoryRequest.responseFormat?.name).toBe('cognitive_memory_recall');
    expect(emotionRequest.responseFormat?.name).toBe(
      'cognitive_emotion_update'
    );
    expect(handoff).toEqual([
      {
        type: 'tool_call',
        callId: 'cognitive:1:search_memory',
        toolName: 'search_memory',
        input: '{"query":"current context and relevant memory"}',
      },
      {
        type: 'tool_result',
        callId: 'cognitive:1:search_memory',
        output: JSON.stringify({
          success: true,
          results: [
            {
              content: 'existing related memory',
              type: 'semantic',
              emotion: { valence: 0, arousal: 0.1, labels: ['calm'] },
              createdAt: '2026-08-23T00:00:00.000Z',
            },
          ],
        }),
      },
      {
        type: 'tool_call',
        callId: 'cognitive:1:update_emotion',
        toolName: 'update_emotion',
        input: '{"valence":0.1,"arousal":0.2,"labels":["calm"]}',
      },
      {
        type: 'tool_result',
        callId: 'cognitive:1:update_emotion',
        output: '{"success":true}',
      },
    ]);
    expect(
      handoff.every((item) => !('callId' in item) || item.callId.length <= 64)
    ).toBe(true);
  });

  it('前sessionのMemoryとEmotionをmoduleだけの初期状態として復元する', async () => {
    const persistedEmotion = {
      valence: -0.2,
      arousal: 0.3,
      labels: ['resting'],
    };
    const previousSessionMemory = {
      content: '前回はCognitive Moduleの責務を整理した。',
      type: 'episode' as const,
      emotion: persistedEmotion,
      createdAt: '2026-08-23T00:00:00.000Z',
    };
    const { createModel, generateFunctions } = createModelFactory();
    const activation = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain: createDomain({
        version: 3,
        emotion: persistedEmotion,
        previousSessionMemory,
        recalledMemories: [],
      }),
      createActivationId: () => 'rin:activation-restored',
      createModel,
    }).beginActivation();
    const sharedContext: ModelRequest['input'] = [
      { role: 'developer', content: '現在日時: 2026年08月24日' },
    ];

    const handoff = await activation.beforeMain(sharedContext);

    const restoredSessionContext = [
      {
        role: 'developer' as const,
        content: [
          '前回の思考セッション終了時に確定した状態です。',
          JSON.stringify({
            memory: previousSessionMemory,
            emotion: persistedEmotion,
          }),
        ].join('\n'),
      },
    ];
    expect(handoff).toHaveLength(4);
    expect(
      handoff.filter(
        (item) =>
          'type' in item &&
          item.type === 'tool_call' &&
          item.toolName === 'update_emotion'
      )
    ).toHaveLength(1);
    expect(handoff).not.toContainEqual(restoredSessionContext[0]);
    for (const generate of generateFunctions) {
      expect(vi.mocked(generate).mock.calls[0]?.[0].input.slice(1)).toEqual([
        ...sharedContext,
        ...restoredSessionContext,
      ]);
    }
  });

  it('session終了時だけMemory出力をstore_memory入力へ切り替える', async () => {
    const { createModel, generateFunctions } = createModelFactory();
    const domain = createDomain();
    const activation = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain,
      createActivationId: () => 'rin:activation-2',
      createModel,
    }).beginActivation();

    const initialContext: ModelRequest['input'] = [
      { role: 'developer', content: '現在日時: 2026年08月24日' },
    ];
    const preMainHandoff = await activation.beforeMain(initialContext);
    const finishCall = {
      type: 'tool_call' as const,
      callId: 'finish-1',
      toolName: 'finish_thinking',
      input: '{"reason":"done"}',
    };
    await expect(
      activation.onMainTurnBoundary({
        turnIndex: 1,
        responseOutput: [finishCall],
        toolCalls: [],
        resolvedInput: [],
        terminationReason: 'finish_thinking',
      })
    ).resolves.toEqual([]);

    const memoryRequest = getRequired(
      vi.mocked(getRequired(generateFunctions[0], 'Memory generate function'))
        .mock.calls[1]?.[0],
      'post_main Memory request'
    );
    const emotionRequest = getRequired(
      vi.mocked(getRequired(generateFunctions[1], 'Emotion generate function'))
        .mock.calls[1]?.[0],
      'post_main Emotion request'
    );
    expect(memoryRequest.responseFormat?.name).toBe('cognitive_memory_store');
    expect(emotionRequest.responseFormat?.name).toBe(
      'cognitive_emotion_update'
    );
    expect(memoryRequest.previousResponseToken).toBeUndefined();
    expect(emotionRequest.previousResponseToken).toBeUndefined();
    expect(memoryRequest.input.slice(1)).toEqual([
      ...initialContext,
      ...preMainHandoff,
      {
        role: 'developer',
        content: [
          '<main_output turn="1">',
          '次に続くassistant messageとtool callはMainの出力です。tool resultはその実行結果です。',
          '</main_output>',
        ].join('\n'),
      },
      finishCall,
    ]);
    expect(emotionRequest.input.slice(1)).toEqual([
      ...initialContext,
      ...preMainHandoff,
      {
        role: 'developer',
        content: [
          '<main_output turn="1">',
          '次に続くassistant messageとtool callはMainの出力です。tool resultはその実行結果です。',
          '</main_output>',
        ].join('\n'),
      },
      finishCall,
    ]);
    const committed = getRequired(
      domain.commitPhase.mock.lastCall?.[0],
      'post_main commit input'
    );
    expect(committed.phase.phase).toBe('post_main');
    expect(committed.memory.value).toEqual({
      content: 'remember the completed thinking session',
      type: 'episode',
    });
    expect(committed.emotion.value).toEqual({
      valence: 0.1,
      arousal: 0.2,
      labels: ['calm'],
    });
  });

  it('各moduleのprovider request直前に共有request budgetを消費する', async () => {
    const beforeModelRequest =
      vi.fn<(module: CognitiveModuleName, request: ModelRequest) => void>();
    const { createModel } = createModelFactory();
    const orchestrator = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain: createDomain(),
      createActivationId: () => 'rin:activation-budget',
      createModel,
      beforeModelRequest,
    });

    await orchestrator
      .beginActivation()
      .beforeMain([{ role: 'developer', content: '現在日時: 2026年08月24日' }]);

    expect(beforeModelRequest.mock.calls.map(([module]) => module)).toEqual([
      'memory',
      'emotion',
    ]);
  });

  it('request budget枯渇はretryせずphaseをfail closedにする', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [],
      usage: createUsage(0),
    });
    const budgetError = Object.assign(new Error('budget exhausted'), {
      code: 'external_request_budget_exceeded',
    });
    const orchestrator = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain: createDomain(),
      createActivationId: () => 'rin:activation-budget-failure',
      createModel: () => ({ generate }),
      beforeModelRequest: (module) => {
        if (module === 'memory') {
          throw budgetError;
        }
      },
    });

    await expect(
      orchestrator
        .beginActivation()
        .beforeMain([
          { role: 'developer', content: '現在日時: 2026年08月24日' },
        ])
    ).rejects.toThrow('Cognitive module phase failed');
    expect(generate).toHaveBeenCalledTimes(1);
  });

  it('submodule model eventをMainと同じpayloadのままmodule名付きで流す', async () => {
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const capturedOptions: OpenAIResponsesModelOptions[] = [];
    createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain: createDomain(),
      events: { emit },
      createModel: (options) => {
        capturedOptions.push(options);
        return { generate: vi.fn() };
      },
    });
    const sourceEvent: EchoEvent = {
      type: 'model.output.emitted',
      category: 'model',
      streams: ['thought', 'analysis'],
      severity: 'info',
      summary: 'model output emitted',
      payload: {
        model: 'gpt-5.6-luna',
        content: 'COGNITIVE_MODEL_OUTPUT',
      },
    };

    await capturedOptions[0]?.events?.emit(sourceEvent);

    expect(emit).toHaveBeenCalledWith({
      ...sourceEvent,
      payload: {
        model: 'gpt-5.6-luna',
        content: 'COGNITIVE_MODEL_OUTPUT',
        cognitiveModule: 'memory',
      },
    });
  });

  it('submodule exchange eventをMainと同じpayloadのまま転送する', async () => {
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const capturedOptions: OpenAIResponsesModelOptions[] = [];
    createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: 'existing-key' },
      instance: createInstance(),
      domain: createDomain(),
      events: { emit },
      createModel: (options) => {
        capturedOptions.push(options);
        return { generate: vi.fn() };
      },
    });
    const sourceEvent: EchoEvent = {
      type: 'model.exchange.recorded',
      category: 'model',
      streams: ['analysis'],
      severity: 'debug',
      summary: 'model exchange recorded',
      payload: {
        provider: 'openai.responses',
        model: 'gpt-5.6-luna',
        request: {
          input: [{ role: 'user', content: 'SECRET_BOUNDARY_TEXT' }],
        },
        response: {
          id: 'response-1',
          output: [{ type: 'message', content: 'SECRET_MODEL_OUTPUT' }],
          usage: { total_tokens: 9 },
        },
      },
    };

    await capturedOptions[0]?.events?.emit(sourceEvent);

    const forwarded = emit.mock.calls[0]?.[0];
    expect(forwarded).toEqual({
      ...sourceEvent,
      payload: {
        ...sourceEvent.payload,
        cognitiveModule: 'memory',
      },
    });
    expect(JSON.stringify(forwarded)).toContain('SECRET_BOUNDARY_TEXT');
    expect(JSON.stringify(forwarded)).toContain('SECRET_MODEL_OUTPUT');
    expect(forwarded?.payload).not.toHaveProperty('redacted');
  });
});

describe('isRetryableCognitiveModuleError', () => {
  it.each([408, 409, 429, 500, 503])(
    'status %sを一時失敗とみなす',
    (status) => {
      expect(isRetryableCognitiveModuleError({ status })).toBe(true);
    }
  );

  it.each([400, 401, 403, 404])('status %sはretryしない', (status) => {
    expect(isRetryableCognitiveModuleError({ status })).toBe(false);
  });

  it('connection timeoutは一時失敗とみなす', () => {
    expect(
      isRetryableCognitiveModuleError({ name: 'APIConnectionTimeoutError' })
    ).toBe(true);
  });

  it.each(['APIUserAbortError', 'AbortError', 'TimeoutError'])(
    '%sはrequest timeout由来の一時失敗とみなす',
    (name) => {
      expect(isRetryableCognitiveModuleError({ name })).toBe(true);
    }
  );

  it('Workers AIの一時internal codeだけをretryする', () => {
    expect(
      isRetryableCognitiveModuleError(new Error('3007: Request timeout'))
    ).toBe(true);
    expect(isRetryableCognitiveModuleError({ code: 3008 })).toBe(true);
    expect(isRetryableCognitiveModuleError({ code: '3040' })).toBe(true);
    expect(
      isRetryableCognitiveModuleError(new Error('10000: Authentication error'))
    ).toBe(false);
  });
});
