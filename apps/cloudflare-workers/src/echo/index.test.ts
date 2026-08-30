import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import {
  parseDashboardActionAnalysisResponse,
  parseDashboardInstanceSummary,
  parseDashboardSessionLogsResponse,
  parseEchoStatus,
} from '@echo-chamber/contracts/dashboard/schemas';
import type { DashboardEchoEvent } from '@echo-chamber/contracts/dashboard/types';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import {
  ThinkingEngineExecutionError,
  type ThinkingEngineResult,
} from '@echo-chamber/core/agent/thinking-engine';
import { TOKEN_LIMITS } from '@echo-chamber/core/echo/constants';
import { getEchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';
import type { Usage } from '@echo-chamber/core/echo/types';
import { calculateDynamicTokenLimit } from '@echo-chamber/core/echo/usage';
import type { ContextSnapshot } from '@echo-chamber/core/ports/context';
import type {
  EchoEvent,
  EchoEventType,
} from '@echo-chamber/core/ports/echo-event';
import type { ModelUsage } from '@echo-chamber/core/ports/model';

import { resolveEchoRuntimeBindings } from '../config/echo-runtime-bindings';
import { createEmbeddingService } from '../embedding/create-embedding-service';
import { createRerankingService } from '../reranking/create-reranking-service';
import { createCloudflareEchoEventPort } from '../utils/echo-event';

import { createToolExecutionContext } from './tool-context';

import { Echo } from './index';

import type { DashboardActionAnalysisEventRange } from './dashboard-action-analysis';

const {
  mockEmbeddingService,
  mockExecutableTools,
  mockEvents,
  mockInstanceDefinition,
  mockMemorySystem,
  mockNoteSystem,
  mockRerankingService,
  mockRuntimeBindings,
  mockToolContext,
} = vi.hoisted(() => ({
  mockEmbeddingService: {
    modelIdentifier: 'test-embedding',
    embed: vi.fn(async () => Promise.resolve([0.1])),
  },
  mockExecutableTools: [{ name: 'tool-1' }],
  mockEvents: {
    emit: vi.fn(async (_event: unknown) => Promise.resolve()),
  },
  mockInstanceDefinition: {
    id: 'rin',
    name: 'リン',
    systemPrompt: '<persona>Rin</persona>',
    mainLlm: {
      provider: 'openai' as const,
      model: 'gpt-5.5',
    },
    cognitiveModules: {
      model: 'test-cognitive-model',
      reasoningEffort: 'medium' as const,
    },
    tokenLimits: {
      dailyHardLimit: 500_000,
      dailySoftLimit: 300_000,
      hardLimitBufferFactor: 1.5,
    },
  },
  mockMemorySystem: {
    reEmbedStaleMemories: vi.fn(async () => Promise.resolve()),
  },
  mockNoteSystem: {
    getDashboardNoteSummary: vi.fn(async () =>
      Promise.resolve({
        count: 0,
        latestUpdatedAt: null,
      })
    ),
  },
  mockRerankingService: {
    modelIdentifier: 'test-reranker',
    rerank: vi.fn(async () => Promise.resolve([])),
  },
  mockRuntimeBindings: {
    discordBotToken: 'discord-token',
    chatChannels: [
      {
        key: 'main',
        displayName: 'メイン',
        description: '主な会話用チャンネル',
        discordChannelId: 'chat-channel-main',
      },
      {
        key: 'sub',
        displayName: 'サブ',
        discordChannelId: 'chat-channel-sub',
      },
    ],
    thinkingChannelId: 'thinking-channel',
    embeddingConfig: {
      provider: 'workersai' as const,
      model: '@cf/pfnet/plamo-embedding-1b',
    },
  },
  mockToolContext: { context: 'tool-context' },
}));

vi.mock('@echo-chamber/cloudflare-runtime/memory-system', () => ({
  MemorySystem: vi.fn(() => mockMemorySystem),
}));

vi.mock('@echo-chamber/cloudflare-runtime/note-system', () => ({
  NoteSystem: vi.fn(() => mockNoteSystem),
}));

vi.mock('cloudflare:workers', () => ({
  DurableObject: class {
    protected readonly ctx: DurableObjectState;
    protected readonly env: Env;

    constructor(ctx: DurableObjectState, env: Env) {
      this.ctx = ctx;
      this.env = env;
    }
  },
}));

vi.mock('@echo-chamber/core/agent/runtime-tools/catalog', () => ({
  canonicalRuntimeTools: ['runtime-tool-1', 'runtime-tool-2'],
}));

vi.mock('@echo-chamber/core/agent/runtime-tools/tool', () => ({
  bindRuntimeTools: vi.fn(() => mockExecutableTools),
}));

vi.mock('@echo-chamber/core/echo/instance-definitions', () => ({
  getEchoInstanceDefinition: vi.fn(() => mockInstanceDefinition),
}));

vi.mock('../config/echo-runtime-bindings', () => ({
  resolveEchoRuntimeBindings: vi.fn(async () =>
    Promise.resolve(mockRuntimeBindings)
  ),
}));

vi.mock('../embedding/create-embedding-service', () => ({
  createEmbeddingService: vi.fn(() => mockEmbeddingService),
}));

vi.mock('../reranking/create-reranking-service', () => ({
  createRerankingService: vi.fn(() => mockRerankingService),
}));

vi.mock('../utils/echo-event', () => ({
  createCloudflareEchoEventPort: vi.fn(() => mockEvents),
}));

vi.mock('./tool-context', () => ({
  createToolExecutionContext: vi.fn(() => mockToolContext),
}));

function createMockStorage(): {
  storage: DurableObjectStorage;
  deleteFn: ReturnType<typeof vi.fn>;
  getFn: ReturnType<typeof vi.fn>;
  putFn: ReturnType<typeof vi.fn>;
} {
  const deleteFn = vi.fn(async () => Promise.resolve(false));
  const getFn = vi.fn(async () => Promise.resolve(undefined));
  const putFn = vi.fn(async () => Promise.resolve());

  return {
    storage: {
      delete: deleteFn,
      get: getFn,
      put: putFn,
      sql: { exec: vi.fn() },
    } as unknown as DurableObjectStorage,
    deleteFn,
    getFn,
    putFn,
  };
}

function findEmittedEvent(type: EchoEventType): EchoEvent | undefined {
  return vi
    .mocked(mockEvents.emit)
    .mock.calls.map(([event]) => event as EchoEvent)
    .find((event) => event.type === type);
}

function getFirstInvocationCallOrder(callOrders: readonly number[]): number {
  const [callOrder] = callOrders;
  if (callOrder === undefined) {
    throw new Error('missing invocation call order');
  }

  return callOrder;
}

function createMockState(storage: DurableObjectStorage): DurableObjectState {
  return {
    storage,
    blockConcurrencyWhile: vi.fn(),
    waitUntil: vi.fn(),
  } as unknown as DurableObjectState;
}

function createMockEnv(): Env {
  return {
    ECHO_KV: {
      get: vi.fn(),
    },
    ENVIRONMENT: 'test',
  } as unknown as Env;
}

function createUsage(totalTokens: number): Usage {
  return {
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    uncached_input_tokens: 0,
    total_input_tokens: 0,
    output_tokens: 0,
    reasoning_tokens: 0,
    total_tokens: totalTokens,
    by_model: [
      {
        provider: 'openai',
        model: 'gpt-5.5',
        cached_input_tokens: 0,
        cache_write_input_tokens: 0,
        uncached_input_tokens: 0,
        total_input_tokens: 0,
        output_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: totalTokens,
      },
    ],
  };
}

function createModelUsage(totalTokens: number): ModelUsage {
  return {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: 0,
    totalInputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens,
  };
}

function createSuccessfulThinkingResult(
  nextWakeAt: string | null
): ThinkingEngineResult {
  const mainUsage = createModelUsage(42);
  return {
    nextWakeAt,
    usage: mainUsage,
    mainUsage,
    cognitiveModules: {
      activationId: 'rin:test-activation',
      phases: [],
      usage: createModelUsage(0),
    },
  };
}

function getEventArchive(echo: Echo): {
  getRecentActionAnalysisEventRanges(input: {
    now?: Date;
    periodDays: readonly number[];
  }): DashboardActionAnalysisEventRange[];
  getRecentActionAnalysisEvents(input: {
    days: number;
    now?: Date;
  }): DashboardActionAnalysisEventRange;
  getTodayEvents(): {
    archiveDay: string;
    events: DashboardEchoEvent[];
  };
} {
  return (
    echo as unknown as {
      eventArchive: {
        getRecentActionAnalysisEventRanges(input: {
          now?: Date;
          periodDays: readonly number[];
        }): DashboardActionAnalysisEventRange[];
        getRecentActionAnalysisEvents(input: {
          days: number;
          now?: Date;
        }): DashboardActionAnalysisEventRange;
        getTodayEvents(): {
          archiveDay: string;
          events: DashboardEchoEvent[];
        };
      };
    }
  ).eventArchive;
}

async function ensureInitialized(
  echo: Echo,
  id: 'rin' | 'marie'
): Promise<void> {
  await (
    echo as unknown as {
      ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
    }
  ).ensureInitialized(id);
}

function setInitializedDefinition(
  echo: Echo,
  id: 'rin' | 'marie' = 'rin'
): void {
  (
    echo as unknown as {
      instanceDefinition: ReturnType<typeof getEchoInstanceDefinition> | null;
    }
  ).instanceDefinition = getEchoInstanceDefinition(id);
}

async function resolveRunDecision(echo: Echo): Promise<{
  shouldRun: boolean;
  unreadCheckMs: number;
}> {
  setInitializedDefinition(echo);
  return await (
    echo as unknown as {
      resolveRunDecision(): Promise<{
        shouldRun: boolean;
        unreadCheckMs: number;
      }>;
    }
  ).resolveRunDecision();
}

describe('Echo external request budgets', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('application budget を使い切っても Discord event 通知用の10件を別枠で保つ', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const beginExternalRequestBudget = (
      echo as unknown as {
        beginExternalRequestBudget(): boolean;
      }
    ).beginExternalRequestBudget.bind(echo);
    const reserveExternalRequest = (
      echo as unknown as {
        reserveExternalRequest(): void;
      }
    ).reserveExternalRequest.bind(echo);

    expect(beginExternalRequestBudget()).toBe(true);
    for (let index = 0; index < 40; index += 1) {
      reserveExternalRequest();
    }

    const eventPortOptions = vi.mocked(createCloudflareEchoEventPort).mock
      .calls[0]?.[0];
    if (eventPortOptions?.beforeRequest === undefined) {
      throw new Error('Expected Discord event request admission hook');
    }
    const beforeNotificationRequest = async (): Promise<void> => {
      await eventPortOptions.beforeRequest?.();
    };
    await Promise.all(
      Array.from({ length: 10 }, async () => {
        await beforeNotificationRequest();
      })
    );

    await expect(
      Promise.resolve().then(async () => {
        await beforeNotificationRequest();
      })
    ).rejects.toThrow('External request budget exceeded');
  });
});

describe('Echo.ensureInitialized', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('legacy key を削除せずに初期化する', async () => {
    const env = createMockEnv();
    const { storage, deleteFn, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    await ensureInitialized(echo, 'rin');

    expect(deleteFn).not.toHaveBeenCalled();
    expect(getEchoInstanceDefinition).toHaveBeenCalledWith('rin');
    expect(resolveEchoRuntimeBindings).toHaveBeenCalledWith(
      env,
      env.ECHO_KV,
      'rin'
    );
    const embeddingServiceCalls = vi.mocked(createEmbeddingService).mock.calls;
    expect(embeddingServiceCalls).toHaveLength(1);
    expect(embeddingServiceCalls[0]?.[0]).toBe(env);
    expect(embeddingServiceCalls[0]?.[1]).toBe(
      mockRuntimeBindings.embeddingConfig
    );
    expect(embeddingServiceCalls[0]?.[2]).toBe(mockEvents);
    expect(createRerankingService).toHaveBeenCalledWith(env);
    const memorySystemInput = vi.mocked(MemorySystem).mock.calls[0]?.[0];
    if (memorySystemInput === undefined) {
      throw new Error('Expected MemorySystem construction');
    }
    expect(memorySystemInput.sql).toBe(storage.sql);
    expect(memorySystemInput.embeddingService.modelIdentifier).toBe(
      'test-embedding'
    );
    expect(memorySystemInput.embeddingService).toBe(mockEmbeddingService);
    expect(memorySystemInput.rerankingService.modelIdentifier).toBe(
      'test-reranker'
    );
    expect(memorySystemInput.rerankingService).toBe(mockRerankingService);
    expect(memorySystemInput.events).toBe(mockEvents);
    const toolContextCalls = vi.mocked(createToolExecutionContext).mock.calls;
    expect(toolContextCalls).toHaveLength(1);
    expect(toolContextCalls[0]?.[0]).toMatchObject({
      chatBindings: mockRuntimeBindings,
      memorySystem: mockMemorySystem,
      noteSystem: mockNoteSystem,
    });
    expect(bindRuntimeTools).toHaveBeenCalledWith(
      canonicalRuntimeTools,
      mockToolContext
    );
    expect(mockMemorySystem.reEmbedStaleMemories).not.toHaveBeenCalled();
    expect(putFn).toHaveBeenNthCalledWith(1, 'id', 'rin');
    expect(putFn).toHaveBeenNthCalledWith(2, 'name', 'リン');
  });

  it('同じ id で再初期化しても storage cleanup や再構築をしない', async () => {
    const env = createMockEnv();
    const { storage, deleteFn, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    await ensureInitialized(echo, 'rin');
    vi.clearAllMocks();

    await ensureInitialized(echo, 'rin');

    expect(deleteFn).not.toHaveBeenCalled();
    expect(putFn).not.toHaveBeenCalled();
    expect(resolveEchoRuntimeBindings).not.toHaveBeenCalled();
    expect(createEmbeddingService).not.toHaveBeenCalled();
    expect(createRerankingService).not.toHaveBeenCalled();
    expect(MemorySystem).not.toHaveBeenCalled();
    expect(createToolExecutionContext).not.toHaveBeenCalled();
    expect(bindRuntimeTools).not.toHaveBeenCalled();
    expect(mockMemorySystem.reEmbedStaleMemories).not.toHaveBeenCalled();
  });

  it('外部 provider の embedding だけを external request budget に接続する', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.mocked(resolveEchoRuntimeBindings).mockResolvedValueOnce({
      ...mockRuntimeBindings,
      embeddingConfig: { provider: 'openai' },
    });
    const reserveExternalRequest = vi.spyOn(
      echo as unknown as { reserveExternalRequest(): void },
      'reserveExternalRequest'
    );

    await ensureInitialized(echo, 'rin');

    const memorySystemInput = vi.mocked(MemorySystem).mock.calls[0]?.[0];
    if (memorySystemInput === undefined) {
      throw new Error('Expected MemorySystem construction');
    }
    expect(memorySystemInput.embeddingService).not.toBe(mockEmbeddingService);
    expect(memorySystemInput.rerankingService).toBe(mockRerankingService);

    await memorySystemInput.embeddingService.embed('query');
    await memorySystemInput.rerankingService.rerank('query', [], 5);

    expect(reserveExternalRequest).toHaveBeenCalledTimes(1);
    expect(mockEmbeddingService.embed).toHaveBeenCalledWith('query');
    expect(mockRerankingService.rerank).toHaveBeenCalledWith('query', [], 5);
  });
});

describe('Echo.createThinkingEngine', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('enable flag 無しで cognitive modules を ThinkingEngine の必須経路へ接続する', async () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    await ensureInitialized(echo, 'rin');

    const thinkingEngine = (
      echo as unknown as {
        createThinkingEngine(): unknown;
      }
    ).createThinkingEngine();
    const engineInput = (
      thinkingEngine as {
        input: { cognitiveModules: unknown };
      }
    ).input;

    expect(engineInput.cognitiveModules).toBeDefined();
  });

  it('main と cognitive module の usage を別 model bucket に記録する', () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    setInitializedDefinition(echo);

    const recordedUsage = (
      echo as unknown as {
        createRecordedSessionUsage(result: {
          usage: ModelUsage;
          mainUsage: ModelUsage;
          cognitiveModules: {
            activationId: string;
            phases: [];
            usage: ModelUsage;
          };
        }): Usage;
      }
    ).createRecordedSessionUsage({
      usage: createModelUsage(16),
      mainUsage: createModelUsage(10),
      cognitiveModules: {
        activationId: 'rin:activation-1',
        phases: [],
        usage: createModelUsage(6),
      },
    });

    expect(recordedUsage.total_tokens).toBe(16);
    expect(recordedUsage.by_model).toEqual([
      {
        provider: 'openai',
        model: 'gpt-5.5',
        cached_input_tokens: 0,
        cache_write_input_tokens: 0,
        uncached_input_tokens: 0,
        total_input_tokens: 0,
        output_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: 10,
      },
      {
        provider: 'openai',
        model: 'test-cognitive-model',
        cached_input_tokens: 0,
        cache_write_input_tokens: 0,
        uncached_input_tokens: 0,
        total_input_tokens: 0,
        output_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: 6,
      },
    ]);
  });
});

describe('Echo session logs route', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('raw event ではなく dashboard activity を返す', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    const getTodayEvents = vi
      .spyOn(getEventArchive(echo), 'getTodayEvents')
      .mockReturnValue({
        archiveDay: '2026-06-01',
        events: [
          {
            id: 'event-1',
            archiveDay: '2026-06-01',
            category: 'model',
            createdAt: '2026-06-01T12:00:00.000Z',
            payload: {
              content: 'I will check the notes.',
              model: 'gpt-5.5',
              provider: 'openai.responses',
              turnIndex: 1,
            },
            sessionId: 'session-1',
            severity: 'info',
            streams: ['thought', 'analysis'],
            summary: 'model output emitted',
            type: 'model.output.emitted',
          },
        ],
      });

    const response = await echo.fetch(
      new Request('http://example.com/rin/session-logs')
    );
    const cachedResponse = await echo.fetch(
      new Request('http://example.com/rin/session-logs')
    );
    const body = parseDashboardSessionLogsResponse(
      await response.json<unknown>()
    );
    await cachedResponse.json<unknown>();

    expect(response.status).toBe(200);
    expect(body).toMatchObject({
      archiveDay: '2026-06-01',
      sessionLogs: [
        {
          id: 'session:session-1',
          activities: [
            {
              body: 'I will check the notes.',
              kind: 'thought',
              title: 'Echo',
            },
          ],
        },
      ],
    });
    expect(body).not.toHaveProperty('events');
    expect(getTodayEvents).toHaveBeenCalledTimes(1);
  });
});

describe('Echo action analysis route', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('raw event ではなく period 別の行動分析を返す', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    const getRanges = vi
      .spyOn(getEventArchive(echo), 'getRecentActionAnalysisEventRanges')
      .mockReturnValue([
        {
          days: 1,
          startArchiveDay: '2026-06-01',
          endArchiveDay: '2026-06-01',
          eventCount: 2,
          events: [
            {
              archiveDay: '2026-06-01',
              createdAt: '2026-06-01T12:00:00.000Z',
              sessionId: 'session-1',
              severity: 'info',
              terminationReason: 'finish_thinking',
              totalTokens: 100,
              type: 'session.completed',
              warnings: [],
            },
            {
              archiveDay: '2026-06-01',
              createdAt: '2026-06-01T12:00:01.000Z',
              sessionId: 'session-1',
              severity: 'info',
              toolName: 'search_memory',
              type: 'tool.completed',
              warnings: [],
            },
          ],
        },
        {
          days: 7,
          startArchiveDay: '2026-05-26',
          endArchiveDay: '2026-06-01',
          eventCount: 2,
          events: [],
          metrics: {
            sessionCount: 1,
            completedSessionCount: 1,
            failedSessionCount: 0,
            warningSessionCount: 0,
            maxTurnsSessionCount: 0,
            totalTokens: 100,
            totalSessionDurationMs: 0,
            sessionDurationCount: 0,
            totalTurns: 0,
            noToolCallTurns: 0,
            toolCallCount: 0,
            toolCompletedCount: 1,
            toolFailedCount: 0,
            topTools: [
              {
                toolName: 'search_memory',
                calledCount: 0,
                completedCount: 1,
                failedCount: 0,
              },
            ],
            memorySearchCompletedCount: 0,
            memorySearchFailedCount: 0,
            memorySearchZeroResultCount: 0,
            memorySearchFinalResultTotal: 0,
            storeMemoryCompletedCount: 0,
          },
        },
        {
          days: 30,
          startArchiveDay: '2026-05-03',
          endArchiveDay: '2026-06-01',
          eventCount: 2,
          events: [],
          metrics: {
            sessionCount: 1,
            completedSessionCount: 1,
            failedSessionCount: 0,
            warningSessionCount: 0,
            maxTurnsSessionCount: 0,
            totalTokens: 100,
            totalSessionDurationMs: 0,
            sessionDurationCount: 0,
            totalTurns: 0,
            noToolCallTurns: 0,
            toolCallCount: 0,
            toolCompletedCount: 1,
            toolFailedCount: 0,
            topTools: [
              {
                toolName: 'search_memory',
                calledCount: 0,
                completedCount: 1,
                failedCount: 0,
              },
            ],
            memorySearchCompletedCount: 0,
            memorySearchFailedCount: 0,
            memorySearchZeroResultCount: 0,
            memorySearchFinalResultTotal: 0,
            storeMemoryCompletedCount: 0,
          },
        },
      ]);

    const response = await echo.fetch(
      new Request('http://example.com/rin/action-analysis')
    );
    const cachedResponse = await echo.fetch(
      new Request('http://example.com/rin/action-analysis')
    );
    const body = parseDashboardActionAnalysisResponse(
      await response.json<unknown>()
    );
    await cachedResponse.json<unknown>();

    expect(response.status).toBe(200);
    expect(body.periods[0]).toMatchObject({
      days: 1,
      completedSessionCount: 1,
      totalTokens: 100,
      topTools: [
        {
          toolName: 'search_memory',
          completedCount: 1,
        },
      ],
    });
    expect(body.periods).toHaveLength(3);
    expect(getRanges).toHaveBeenCalledTimes(1);
    expect(getRanges.mock.calls[0]?.[0]?.now).toBeInstanceOf(Date);
    expect(getRanges.mock.calls[0]?.[0]?.periodDays).toEqual([1, 7, 30]);
    expect(body).not.toHaveProperty('events');
  });
});

describe('Echo dashboard status payload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('詳細 status route は短時間 cache を使う', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    const getStatus = vi.spyOn(echo, 'getStatus').mockResolvedValue(
      parseEchoStatus({
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: null,
        nextWakeAt: null,
        context: null,
        cognitive: {
          domainVersion: 0,
          lastBoundaryId: null,
          updatedAt: null,
        },
        runtime: {
          mainLlm: {
            provider: 'openai',
            model: 'gpt-5.5',
          },
          tokenLimits: {
            dailyHardLimit: 500_000,
            dailySoftLimit: 300_000,
            hardLimitBufferFactor: 1.5,
          },
        },
        memories: [],
        notes: [],
        usage: {},
      })
    );

    const response = await echo.fetch(new Request('http://example.com/rin'));
    const cachedResponse = await echo.fetch(
      new Request('http://example.com/rin')
    );
    const body = parseEchoStatus(await response.json<unknown>());
    await cachedResponse.json<unknown>();

    expect(body.id).toBe('rin');
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it('状態変更後は dashboard read cache を破棄する', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    const getStatus = vi
      .spyOn(echo, 'getStatus')
      .mockResolvedValueOnce(
        parseEchoStatus({
          id: 'rin',
          name: 'リン',
          state: 'Idling',
          nextAlarm: null,
          nextWakeAt: null,
          context: null,
          cognitive: {
            domainVersion: 0,
            lastBoundaryId: null,
            updatedAt: null,
          },
          runtime: {
            mainLlm: {
              provider: 'openai',
              model: 'gpt-5.5',
            },
            tokenLimits: {
              dailyHardLimit: 500_000,
              dailySoftLimit: 300_000,
              hardLimitBufferFactor: 1.5,
            },
          },
          memories: [],
          notes: [],
          usage: {},
        })
      )
      .mockResolvedValueOnce(
        parseEchoStatus({
          id: 'rin',
          name: 'リン',
          state: 'Running',
          nextAlarm: null,
          nextWakeAt: null,
          context: null,
          cognitive: {
            domainVersion: 1,
            lastBoundaryId: 'rin:activation-1:1:pre_main',
            updatedAt: '2026-03-19T12:00:00.000Z',
          },
          runtime: {
            mainLlm: {
              provider: 'openai',
              model: 'gpt-5.5',
            },
            tokenLimits: {
              dailyHardLimit: 500_000,
              dailySoftLimit: 300_000,
              hardLimitBufferFactor: 1.5,
            },
          },
          memories: [],
          notes: [],
          usage: {},
        })
      );

    await echo.fetch(new Request('http://example.com/rin'));
    await echo.fetch(new Request('http://example.com/rin'));
    await echo.setState('Running', 'test');
    const response = await echo.fetch(new Request('http://example.com/rin'));
    const body = parseEchoStatus(await response.json<unknown>());

    expect(body.state).toBe('Running');
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it('summary route は短時間 cache を使う', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    const getSummary = vi.spyOn(echo, 'getSummary').mockResolvedValue(
      parseDashboardInstanceSummary({
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: null,
        nextWakeAt: null,
        noteCount: 0,
        memoryCount: 0,
        todayUsageTokens: 0,
        sevenDayUsageTokens: 0,
        thirtyDayUsageTokens: 0,
        runtime: {
          mainLlm: {
            provider: 'openai',
            model: 'gpt-5.5',
          },
          tokenLimits: {
            dailyHardLimit: 500_000,
            dailySoftLimit: 300_000,
            hardLimitBufferFactor: 1.5,
          },
        },
        latestNoteUpdatedAt: null,
        latestMemoryUpdatedAt: null,
      })
    );

    const response = await echo.fetch(
      new Request('http://example.com/rin/summary')
    );
    const cachedResponse = await echo.fetch(
      new Request('http://example.com/rin/summary')
    );
    const body = parseDashboardInstanceSummary(await response.json<unknown>());
    await cachedResponse.json<unknown>();

    expect(body.id).toBe('rin');
    expect(getSummary).toHaveBeenCalledTimes(1);
  });

  it('保存済み context と next_wake_at を詳細 status に含める', async () => {
    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const context: ContextSnapshot = {
      content: 'Latest context for dashboard display.',
      createdAt: '2026-03-22T10:00:00.000Z',
      updatedAt: '2026-03-22T11:00:00.000Z',
      emotion: {
        valence: 0.4,
        arousal: 0.3,
        labels: ['focused'],
      },
    };
    const nextWakeAt = '2026-03-22T12:00:00.000Z';
    setInitializedDefinition(echo);
    getFn.mockImplementation(async (key: unknown) => {
      await Promise.resolve();
      if (key === 'context') {
        return context;
      }
      if (key === 'next_wake_at') {
        return nextWakeAt;
      }
      return undefined;
    });
    vi.spyOn(
      echo as unknown as {
        getMemorySystemOrThrow(): { getDashboardMemories(): [] };
      },
      'getMemorySystemOrThrow'
    ).mockReturnValue({
      getDashboardMemories: () => [],
    });
    vi.spyOn(
      echo as unknown as {
        getCognitiveDomainStoreOrThrow(): {
          getDashboardState(): Promise<{
            domainVersion: number;
            lastBoundaryId: null;
            updatedAt: null;
          }>;
        };
      },
      'getCognitiveDomainStoreOrThrow'
    ).mockReturnValue({
      getDashboardState: async () =>
        await Promise.resolve({
          domainVersion: 0,
          lastBoundaryId: null,
          updatedAt: null,
        }),
    });
    vi.spyOn(echo, 'getNextAlarm').mockResolvedValue(null);
    vi.spyOn(echo, 'getNotes').mockResolvedValue([]);
    vi.spyOn(echo, 'getAllUsage').mockResolvedValue({});

    const status = await echo.getStatus();

    expect(status.context).toEqual(context);
    expect(status.nextWakeAt).toBe(nextWakeAt);
    expect(status.cognitive).toEqual({
      domainVersion: 0,
      lastBoundaryId: null,
      updatedAt: null,
    });
  });

  it('保存済み next_wake_at を一覧 summary に含める', async () => {
    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const nextWakeAt = '2026-03-22T12:00:00.000Z';
    setInitializedDefinition(echo);
    getFn.mockImplementation(async (key: unknown) => {
      await Promise.resolve();
      if (key === 'next_wake_at') {
        return nextWakeAt;
      }
      return undefined;
    });
    vi.spyOn(
      echo as unknown as {
        getMemorySystemOrThrow(): {
          getDashboardMemories(): [];
          getDashboardMemorySummary(): {
            count: number;
            latestUpdatedAt: string | null;
          };
        };
      },
      'getMemorySystemOrThrow'
    ).mockReturnValue({
      getDashboardMemories: () => [],
      getDashboardMemorySummary: () => ({
        count: 0,
        latestUpdatedAt: null,
      }),
    });
    vi.spyOn(echo, 'getNextAlarm').mockResolvedValue(null);
    vi.spyOn(echo, 'getNotes').mockResolvedValue([]);
    vi.spyOn(echo, 'getAllUsage').mockResolvedValue({});
    vi.spyOn(echo, 'getTodayUsage').mockResolvedValue(null);

    const summary = await echo.getSummary();

    expect(summary.nextWakeAt).toBe(nextWakeAt);
  });
});

describe('Echo context storage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('DO storage から context を読み出せる', async () => {
    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const context: ContextSnapshot = {
      content: 'Latest context for the next wake.',
      createdAt: '2025-01-25T15:00:00.000Z',
      updatedAt: '2025-01-25T15:00:00.000Z',
      emotion: {
        valence: 0.4,
        arousal: 0.2,
        labels: ['calm'],
      },
    };
    getFn.mockResolvedValue(context);

    const result = await (
      echo as unknown as {
        loadContext(): Promise<ContextSnapshot | null>;
      }
    ).loadContext();

    expect(getFn).toHaveBeenCalledWith('context');
    expect(result).toEqual(context);
  });

  it('run の継続状態は Cognitive commit 層へ一元化する', async () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    setInitializedDefinition(echo);
    const think = vi
      .fn()
      .mockResolvedValue(createSuccessfulThinkingResult(null));
    vi.spyOn(
      echo as unknown as { setState(state: string): Promise<void> },
      'setState'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getName(): Promise<string> },
      'getName'
    ).mockResolvedValue('リン');
    vi.spyOn(
      echo as unknown as {
        resolveRunDecision(): Promise<{
          shouldRun: boolean;
          unreadCheckMs: number;
        }>;
      },
      'resolveRunDecision'
    ).mockResolvedValue({
      shouldRun: true,
      unreadCheckMs: 12,
    });
    vi.spyOn(
      echo as unknown as {
        createThinkingEngine(): { think(): Promise<unknown> };
      },
      'createThinkingEngine'
    ).mockReturnValue({
      think,
    });
    vi.spyOn(
      echo as unknown as { updateUsage(): Promise<{ total_tokens: number }> },
      'updateUsage'
    ).mockResolvedValue({ total_tokens: 42 });
    const result = await echo.run();

    expect(think).toHaveBeenCalledTimes(1);
    expect(putFn).not.toHaveBeenCalledWith('context', expect.anything());
    expect(result).toMatchObject({
      unreadCheckMs: 12,
    });
    expect(typeof result.thinkMs).toBe('number');
  });

  it('失敗した session でも課金済み Main / Cognitive usage を保存する', async () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    setInitializedDefinition(echo);
    const executionError = new ThinkingEngineExecutionError(
      new Error('cognitive boundary failed'),
      createModelUsage(7),
      createModelUsage(3)
    );

    vi.spyOn(
      echo as unknown as { setState(state: string): Promise<void> },
      'setState'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as {
        resolveRunDecision(): Promise<{
          shouldRun: boolean;
          unreadCheckMs: number;
        }>;
      },
      'resolveRunDecision'
    ).mockResolvedValue({ shouldRun: true, unreadCheckMs: 4 });
    vi.spyOn(
      echo as unknown as {
        createThinkingEngine(): { think(): Promise<never> };
      },
      'createThinkingEngine'
    ).mockReturnValue({
      think: vi.fn().mockRejectedValue(executionError),
    });
    const updateUsage = vi
      .spyOn(
        echo as unknown as { updateUsage(usage: Usage): Promise<Usage> },
        'updateUsage'
      )
      .mockResolvedValue(createUsage(10));

    await echo.run();

    expect(updateUsage).toHaveBeenCalledWith(
      expect.objectContaining({
        total_tokens: 10,
        by_model: [
          expect.objectContaining({ model: 'gpt-5.5', total_tokens: 7 }),
          expect.objectContaining({
            model: 'test-cognitive-model',
            total_tokens: 3,
          }),
        ],
      })
    );
    expect(findEmittedEvent('usage.recorded')).toMatchObject({
      payload: {
        status: 'failed',
        usage: createModelUsage(10),
      },
    });
  });
});

describe('Echo next_wake_at storage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('DO storage から next_wake_at を読み出せる', async () => {
    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const nextWakeAt = '2026-03-23T00:00:00.000Z';
    getFn.mockResolvedValue(nextWakeAt);

    const result = await (
      echo as unknown as {
        loadNextWakeAt(): Promise<string | null>;
      }
    ).loadNextWakeAt();

    expect(getFn).toHaveBeenCalledWith('next_wake_at');
    expect(result).toBe(nextWakeAt);
  });

  it('run 時に返却された next_wake_at を DO storage へ保存する', async () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    setInitializedDefinition(echo);
    const nextWakeAt = '2026-03-23T00:00:00.000Z';
    const think = vi
      .fn()
      .mockResolvedValue(createSuccessfulThinkingResult(nextWakeAt));

    vi.spyOn(
      echo as unknown as { setState(state: string): Promise<void> },
      'setState'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getName(): Promise<string> },
      'getName'
    ).mockResolvedValue('リン');
    vi.spyOn(
      echo as unknown as {
        resolveRunDecision(): Promise<{
          shouldRun: boolean;
          unreadCheckMs: number;
        }>;
      },
      'resolveRunDecision'
    ).mockResolvedValue({
      shouldRun: true,
      unreadCheckMs: 7,
    });
    vi.spyOn(
      echo as unknown as {
        createThinkingEngine(): { think(): Promise<unknown> };
      },
      'createThinkingEngine'
    ).mockReturnValue({
      think,
    });
    vi.spyOn(
      echo as unknown as { updateUsage(): Promise<{ total_tokens: number }> },
      'updateUsage'
    ).mockResolvedValue({ total_tokens: 42 });

    const result = await echo.run();

    expect(putFn).toHaveBeenCalledWith('next_wake_at', nextWakeAt);
    expect(result).toMatchObject({
      unreadCheckMs: 7,
    });
    expect(typeof result.thinkMs).toBe('number');
  });

  it('run 時に next_wake_at が無ければ保存済み値をクリアする', async () => {
    const env = {
      ...createMockEnv(),
      OPENAI_API_KEY: 'test-openai-key',
    } as unknown as Env;
    const { storage, deleteFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    setInitializedDefinition(echo);
    const think = vi
      .fn()
      .mockResolvedValue(createSuccessfulThinkingResult(null));

    vi.spyOn(
      echo as unknown as { setState(state: string): Promise<void> },
      'setState'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getName(): Promise<string> },
      'getName'
    ).mockResolvedValue('リン');
    vi.spyOn(
      echo as unknown as {
        resolveRunDecision(): Promise<{
          shouldRun: boolean;
          unreadCheckMs: number;
        }>;
      },
      'resolveRunDecision'
    ).mockResolvedValue({
      shouldRun: true,
      unreadCheckMs: 5,
    });
    vi.spyOn(
      echo as unknown as {
        createThinkingEngine(): { think(): Promise<unknown> };
      },
      'createThinkingEngine'
    ).mockReturnValue({
      think,
    });
    vi.spyOn(
      echo as unknown as { updateUsage(): Promise<{ total_tokens: number }> },
      'updateUsage'
    ).mockResolvedValue({ total_tokens: 42 });

    const result = await echo.run();

    expect(deleteFn).toHaveBeenCalledWith('next_wake_at');
    expect(result).toMatchObject({
      unreadCheckMs: 5,
    });
    expect(typeof result.thinkMs).toBe('number');
  });
});

describe('Echo run preconditions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('未読メッセージがあれば hard limit より優先して実行する', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(true);
    const getTodayUsage = vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    );
    const loadNextWakeAt = vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    );
    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(true);
    expect(getTodayUsage).not.toHaveBeenCalled();
    expect(loadNextWakeAt).not.toHaveBeenCalled();
  });

  it('hard limit 超過時は到達済み next_wake_at より優先して起動しない', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    const hardLimit = calculateDynamicTokenLimit(
      TOKEN_LIMITS.DAILY_HARD_LIMIT,
      TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR
    );
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(hardLimit));
    const loadNextWakeAt = vi
      .spyOn(
        echo as unknown as { loadNextWakeAt(): Promise<string | null> },
        'loadNextWakeAt'
      )
      .mockResolvedValue('2026-03-22T00:59:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(false);
    expect(loadNextWakeAt).toHaveBeenCalled();
    expect(putFn).toHaveBeenCalledWith(
      'next_wake_at',
      '2026-03-22T01:01:00.000Z'
    );
    expect(
      findEmittedEvent('system.schedule.next_wake_at_updated')
    ).toMatchObject({
      payload: {
        previousValue: '2026-03-22T00:59:00.000Z',
        nextValue: '2026-03-22T01:01:00.000Z',
        reason: 'hard_token_limit_defer',
        totalTokens: hardLimit,
        hardLimit,
      },
    });
  });

  it('hard limit 超過でも next_wake_at 未到達なら warn しない', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    const hardLimit = calculateDynamicTokenLimit(
      TOKEN_LIMITS.DAILY_HARD_LIMIT,
      TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR
    );
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(hardLimit));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('2026-03-22T01:15:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(false);
    expect(
      findEmittedEvent('system.schedule.next_wake_at_updated')
    ).toBeUndefined();
  });

  it('hard limit が当日中に回復しない場合は次の usage reset へ next_wake_at をフォールバックする', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T14:30:00.000Z'));

    const env = createMockEnv();
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(TOKEN_LIMITS.DAILY_HARD_LIMIT));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('2026-03-22T14:29:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(false);
    expect(putFn).toHaveBeenCalledWith(
      'next_wake_at',
      '2026-03-22T22:00:00.000Z'
    );
    expect(
      findEmittedEvent('system.schedule.next_wake_at_updated')
    ).toMatchObject({
      payload: {
        previousValue: '2026-03-22T14:29:00.000Z',
        nextValue: '2026-03-22T22:00:00.000Z',
        reason: 'hard_token_limit_defer',
      },
    });
  });

  it('到達済みの next_wake_at なら soft limit を超えていても実行する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(100_000));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('2026-03-22T00:59:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(true);
    expect(findEmittedEvent('system.run_decision.evaluated')).toMatchObject({
      payload: {
        reason: 'next_wake_at_reached',
        nextWakeAt: '2026-03-22T00:59:00.000Z',
      },
    });
  });

  it('soft limit 未満でも next_wake_at が10分以内なら起動しない', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(0));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('2026-03-22T01:05:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(false);
    expect(findEmittedEvent('system.run_decision.evaluated')).toMatchObject({
      payload: {
        reason: 'next_wake_at_suppression',
        nextWakeAt: '2026-03-22T01:05:00.000Z',
      },
    });
  });

  it('soft limit 未満で next_wake_at が10分より先なら通常起動する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(0));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('2026-03-22T01:15:00.000Z');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(true);
    expect(findEmittedEvent('system.run_decision.evaluated')).toMatchObject({
      payload: {
        reason: 'soft_limit_allows_run',
        softLimit: 45_000,
      },
    });
  });

  it('不正な next_wake_at は warn して破棄し、他条件が通れば起動する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-22T01:00:00.000Z'));

    const env = createMockEnv();
    const { storage, deleteFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as { validateEchoState(): Promise<boolean> },
      'validateEchoState'
    ).mockResolvedValue(true);
    vi.spyOn(
      echo as unknown as { validateChatMessage(): Promise<boolean> },
      'validateChatMessage'
    ).mockResolvedValue(false);
    vi.spyOn(
      echo as unknown as { getTodayUsage(): Promise<Usage | null> },
      'getTodayUsage'
    ).mockResolvedValue(createUsage(0));
    vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    ).mockResolvedValue('not-a-date');

    const result = await resolveRunDecision(echo);

    expect(result.shouldRun).toBe(true);
    expect(deleteFn).toHaveBeenCalledWith('next_wake_at');
    expect(
      findEmittedEvent('system.schedule.next_wake_at_invalidated')
    ).toMatchObject({
      payload: {
        storedValue: 'not-a-date',
        reason: 'invalid_date',
      },
    });
  });
});

describe('Echo run metrics', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('実行しない場合でも unread_check_ms を返す', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    vi.spyOn(
      echo as unknown as {
        resolveRunDecision(): Promise<{
          shouldRun: boolean;
          unreadCheckMs: number;
        }>;
      },
      'resolveRunDecision'
    ).mockResolvedValue({
      shouldRun: false,
      unreadCheckMs: 18,
    });

    const result = await echo.run();

    expect(result).toEqual({
      unreadCheckMs: 18,
      thinkMs: 0,
    });
  });

  it('alarm 終了時に実行メトリクスを構造化ログへ出す', async () => {
    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    getFn.mockResolvedValue('rin');

    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getState(): Promise<string> },
      'getState'
    ).mockResolvedValue('Idling');
    vi.spyOn(
      echo as unknown as {
        run(): Promise<{
          unreadCheckMs: number;
          thinkMs: number;
        }>;
      },
      'run'
    ).mockResolvedValue({
      unreadCheckMs: 14,
      thinkMs: 0,
    });
    vi.spyOn(
      echo as unknown as { setNextAlarm(nextAlarm?: Date): Promise<void> },
      'setNextAlarm'
    ).mockResolvedValue(undefined);

    await echo.alarm();

    const completedEvent = findEmittedEvent('system.schedule.alarm_completed');
    expect(completedEvent).toMatchObject({
      severity: 'debug',
      payload: {
        status: 'completed',
        unreadCheckMs: 14,
        thinkMs: 0,
      },
    });
    expect(typeof completedEvent?.payload?.alarmTotalMs).toBe('number');
  });

  it('JST 03:00 の daily sleep alarm で event retention cleanup を実行する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T18:00:00.000Z'));

    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    getFn.mockResolvedValue('rin');

    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getState(): Promise<string> },
      'getState'
    ).mockResolvedValue('Idling');
    const sleep = vi
      .spyOn(echo as unknown as { sleep(): Promise<void> }, 'sleep')
      .mockResolvedValue(undefined);
    const cleanUpExpiredEvents = vi
      .spyOn(
        echo as unknown as { cleanUpExpiredEvents(now: Date): Promise<void> },
        'cleanUpExpiredEvents'
      )
      .mockResolvedValue(undefined);
    const reEmbedStaleMemoriesForDailyMaintenance = vi
      .spyOn(
        echo as unknown as {
          reEmbedStaleMemoriesForDailyMaintenance(): Promise<{
            status: 'completed';
          }>;
        },
        'reEmbedStaleMemoriesForDailyMaintenance'
      )
      .mockResolvedValue({
        status: 'completed',
      });
    const setNextAlarm = vi
      .spyOn(
        echo as unknown as {
          setNextAlarm(nextAlarm?: Date, reason?: string): Promise<void>;
        },
        'setNextAlarm'
      )
      .mockResolvedValue(undefined);
    const run = vi.spyOn(echo, 'run').mockResolvedValue({
      unreadCheckMs: 0,
      thinkMs: 0,
    });

    await echo.alarm();

    expect(sleep).toHaveBeenCalledTimes(1);
    expect(reEmbedStaleMemoriesForDailyMaintenance).toHaveBeenCalledTimes(1);
    expect(cleanUpExpiredEvents).toHaveBeenCalledWith(
      new Date('2026-03-21T18:00:00.000Z')
    );
    expect(setNextAlarm).toHaveBeenCalledWith(
      new Date('2026-03-21T22:00:00.000Z'),
      'daily_sleep_wake'
    );
    expect(run).not.toHaveBeenCalled();

    vi.useRealTimers();
  });

  it('event retention cleanup が失敗しても daily wake alarm を維持する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T18:00:00.000Z'));

    const env = createMockEnv();
    const { storage, getFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    getFn.mockResolvedValue('rin');

    vi.spyOn(
      echo as unknown as {
        ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
      },
      'ensureInitialized'
    ).mockResolvedValue(undefined);
    vi.spyOn(
      echo as unknown as { getState(): Promise<string> },
      'getState'
    ).mockResolvedValue('Idling');
    const sleep = vi
      .spyOn(echo as unknown as { sleep(): Promise<void> }, 'sleep')
      .mockResolvedValue(undefined);
    const cleanUpExpiredEvents = vi
      .spyOn(
        echo as unknown as { cleanUpExpiredEvents(now: Date): Promise<void> },
        'cleanUpExpiredEvents'
      )
      .mockRejectedValue(new Error('SQLite cleanup failed'));
    const reEmbedStaleMemoriesForDailyMaintenance = vi
      .spyOn(
        echo as unknown as {
          reEmbedStaleMemoriesForDailyMaintenance(): Promise<{
            status: 'completed';
          }>;
        },
        'reEmbedStaleMemoriesForDailyMaintenance'
      )
      .mockResolvedValue({
        status: 'completed',
      });
    const setNextAlarm = vi
      .spyOn(
        echo as unknown as {
          setNextAlarm(nextAlarm?: Date, reason?: string): Promise<void>;
        },
        'setNextAlarm'
      )
      .mockResolvedValue(undefined);
    const run = vi.spyOn(echo, 'run').mockResolvedValue({
      unreadCheckMs: 0,
      thinkMs: 0,
    });

    await expect(echo.alarm()).resolves.toBeUndefined();

    expect(sleep).toHaveBeenCalledTimes(1);
    expect(reEmbedStaleMemoriesForDailyMaintenance).toHaveBeenCalledTimes(1);
    expect(setNextAlarm).toHaveBeenCalledWith(
      new Date('2026-03-21T22:00:00.000Z'),
      'daily_sleep_wake'
    );
    expect(cleanUpExpiredEvents).toHaveBeenCalledWith(
      new Date('2026-03-21T18:00:00.000Z')
    );
    expect(
      getFirstInvocationCallOrder(setNextAlarm.mock.invocationCallOrder)
    ).toBeLessThan(
      getFirstInvocationCallOrder(cleanUpExpiredEvents.mock.invocationCallOrder)
    );
    expect(run).not.toHaveBeenCalled();

    const completedEvent = findEmittedEvent('system.schedule.alarm_completed');
    expect(completedEvent).toMatchObject({
      severity: 'warn',
      summary: 'alarm sleep_scheduled: event retention cleanup failed',
      payload: {
        status: 'sleep_scheduled',
        reason: 'daily_sleep_window',
        eventRetentionCleanup: {
          status: 'failed',
          error: 'SQLite cleanup failed',
        },
      },
    });

    vi.useRealTimers();
  });

  it('日次 memory maintenance で stale memory の再 embedding を実行する', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    (
      echo as unknown as {
        memorySystem: typeof mockMemorySystem | null;
      }
    ).memorySystem = mockMemorySystem;

    const result = await (
      echo as unknown as {
        reEmbedStaleMemoriesForDailyMaintenance(): Promise<{
          status: 'completed' | 'failed';
          error?: string;
        }>;
      }
    ).reEmbedStaleMemoriesForDailyMaintenance();

    expect(mockMemorySystem.reEmbedStaleMemories).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      status: 'completed',
    });
  });

  it('日次 memory maintenance が失敗しても結果として返す', async () => {
    const env = createMockEnv();
    const { storage } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    (
      echo as unknown as {
        memorySystem: typeof mockMemorySystem | null;
      }
    ).memorySystem = mockMemorySystem;
    vi.mocked(mockMemorySystem.reEmbedStaleMemories).mockRejectedValueOnce(
      new Error('embedding maintenance failed')
    );

    const result = await (
      echo as unknown as {
        reEmbedStaleMemoriesForDailyMaintenance(): Promise<{
          status: 'completed' | 'failed';
          error?: string;
        }>;
      }
    ).reEmbedStaleMemoriesForDailyMaintenance();

    expect(result).toEqual({
      status: 'failed',
      error: 'embedding maintenance failed',
    });
  });
});
