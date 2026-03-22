import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import {
  SCHEDULING_CONFIG,
  TOKEN_LIMITS,
} from '@echo-chamber/core/echo/constants';
import { getEchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';
import type { Usage } from '@echo-chamber/core/echo/types';
import { calculateDynamicTokenLimit } from '@echo-chamber/core/echo/usage';
import type { ContextSnapshot } from '@echo-chamber/core/ports/context';

import { resolveEchoRuntimeBindings } from '../config/echo-runtime-bindings';
import { createEmbeddingService } from '../embedding/create-embedding-service';
import { createRerankingService } from '../reranking/create-reranking-service';
import { createLogger } from '../utils/logger';

import { createToolExecutionContext } from './tool-context';

import { Echo } from './index';

const {
  mockEmbeddingService,
  mockExecutableTools,
  mockInstanceDefinition,
  mockLogger,
  mockMemorySystem,
  mockNoteSystem,
  mockRerankingService,
  mockRuntimeBindings,
  mockToolContext,
} = vi.hoisted(() => ({
  mockEmbeddingService: { provider: 'test-embedding' },
  mockExecutableTools: [{ name: 'tool-1' }],
  mockInstanceDefinition: {
    id: 'rin',
    name: 'リン',
    systemPrompt: '<persona>Rin</persona>',
  },
  mockLogger: {
    debug: vi.fn(async () => Promise.resolve()),
    info: vi.fn(async () => Promise.resolve()),
    warn: vi.fn(async () => Promise.resolve()),
    error: vi.fn(async () => Promise.resolve()),
  },
  mockMemorySystem: {
    reEmbedStaleMemories: vi.fn(async () => Promise.resolve()),
  },
  mockNoteSystem: {},
  mockRerankingService: { provider: 'test-reranker' },
  mockRuntimeBindings: {
    discordBotToken: 'discord-token',
    chatChannelId: 'chat-channel',
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

vi.mock('../utils/logger', () => ({
  createLogger: vi.fn(() => mockLogger),
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
    uncached_input_tokens: 0,
    total_input_tokens: 0,
    output_tokens: 0,
    reasoning_tokens: 0,
    total_tokens: totalTokens,
    total_cost: 0,
  };
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

async function validateRunPreconditions(echo: Echo): Promise<boolean> {
  return await (
    echo as unknown as {
      validateRunPreconditions(): Promise<boolean>;
    }
  ).validateRunPreconditions();
}

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
    expect(createLogger).toHaveBeenCalledWith(env);
    expect(getEchoInstanceDefinition).toHaveBeenCalledWith('rin');
    expect(resolveEchoRuntimeBindings).toHaveBeenCalledWith(
      env,
      env.ECHO_KV,
      'rin'
    );
    expect(createEmbeddingService).toHaveBeenCalledWith(
      env,
      mockRuntimeBindings.embeddingConfig
    );
    expect(createRerankingService).toHaveBeenCalledWith(env);
    expect(MemorySystem).toHaveBeenCalledWith({
      sql: storage.sql,
      embeddingService: mockEmbeddingService,
      rerankingService: mockRerankingService,
      logger: mockLogger,
    });
    expect(createToolExecutionContext).toHaveBeenCalledWith({
      chatBindings: mockRuntimeBindings,
      memorySystem: mockMemorySystem,
      noteSystem: mockNoteSystem,
      logger: mockLogger,
    });
    expect(bindRuntimeTools).toHaveBeenCalledWith(
      canonicalRuntimeTools,
      mockToolContext
    );
    expect(mockMemorySystem.reEmbedStaleMemories).toHaveBeenCalledTimes(1);
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

  it('run 時に返却された context を DO storage へ保存する', async () => {
    const env = createMockEnv();
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const context: ContextSnapshot = {
      content: 'Summarized the session for the next cycle.',
      createdAt: '2025-01-25T15:00:00.000Z',
      updatedAt: '2025-01-25T15:00:00.000Z',
      emotion: {
        valence: 0.4,
        arousal: 0.2,
        labels: ['calm', 'satisfied'],
      },
    };
    const think = vi.fn().mockResolvedValue({
      context,
      usage: {
        cachedInputTokens: 0,
        uncachedInputTokens: 0,
        totalInputTokens: 0,
        outputTokens: 0,
        reasoningTokens: 0,
        totalTokens: 42,
      },
    });
    const thoughtLogSend = vi.fn().mockResolvedValue(undefined);

    vi.spyOn(
      echo as unknown as { validateRunPreconditions(): Promise<boolean> },
      'validateRunPreconditions'
    ).mockResolvedValue(true);
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
    vi.spyOn(
      echo as unknown as {
        createThoughtLog(): { send(message: string): Promise<void> };
      },
      'createThoughtLog'
    ).mockReturnValue({
      send: thoughtLogSend,
    });

    await echo.run();

    expect(think).toHaveBeenCalledTimes(1);
    expect(putFn).toHaveBeenCalledWith('context', context);
    expect(thoughtLogSend).toHaveBeenCalledWith(
      'Usage: 42 tokens (Total: 42 tokens)'
    );
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
    const env = createMockEnv();
    const { storage, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const nextWakeAt = '2026-03-23T00:00:00.000Z';
    const think = vi.fn().mockResolvedValue({
      context: null,
      nextWakeAt,
      usage: {
        cachedInputTokens: 0,
        uncachedInputTokens: 0,
        totalInputTokens: 0,
        outputTokens: 0,
        reasoningTokens: 0,
        totalTokens: 42,
      },
    });

    vi.spyOn(
      echo as unknown as { validateRunPreconditions(): Promise<boolean> },
      'validateRunPreconditions'
    ).mockResolvedValue(true);
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
    vi.spyOn(
      echo as unknown as {
        createThoughtLog(): { send(message: string): Promise<void> };
      },
      'createThoughtLog'
    ).mockReturnValue({
      send: vi.fn().mockResolvedValue(undefined),
    });

    await echo.run();

    expect(putFn).toHaveBeenCalledWith('next_wake_at', nextWakeAt);
  });

  it('run 時に next_wake_at が無ければ保存済み値をクリアする', async () => {
    const env = createMockEnv();
    const { storage, deleteFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);
    const think = vi.fn().mockResolvedValue({
      context: null,
      nextWakeAt: null,
      usage: {
        cachedInputTokens: 0,
        uncachedInputTokens: 0,
        totalInputTokens: 0,
        outputTokens: 0,
        reasoningTokens: 0,
        totalTokens: 42,
      },
    });

    vi.spyOn(
      echo as unknown as { validateRunPreconditions(): Promise<boolean> },
      'validateRunPreconditions'
    ).mockResolvedValue(true);
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
    vi.spyOn(
      echo as unknown as {
        createThoughtLog(): { send(message: string): Promise<void> };
      },
      'createThoughtLog'
    ).mockReturnValue({
      send: vi.fn().mockResolvedValue(undefined),
    });

    await echo.run();

    expect(deleteFn).toHaveBeenCalledWith('next_wake_at');
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

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(true);
    expect(getTodayUsage).not.toHaveBeenCalled();
    expect(loadNextWakeAt).not.toHaveBeenCalled();
  });

  it('hard limit 超過時は到達済み next_wake_at より優先して起動しない', async () => {
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
    const loadNextWakeAt = vi.spyOn(
      echo as unknown as { loadNextWakeAt(): Promise<string | null> },
      'loadNextWakeAt'
    );

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(false);
    expect(loadNextWakeAt).not.toHaveBeenCalled();
    expect(mockLogger.warn).toHaveBeenCalledWith(
      `Usage hard limit reached: ${hardLimit}  (Hard limit: ${hardLimit})`
    );
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

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(true);
    expect(mockLogger.info).toHaveBeenCalledWith(
      'next_wake_at reached: 2026-03-22T00:59:00.000Z'
    );
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

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(false);
    expect(mockLogger.info).toHaveBeenCalledWith(
      `Skipping soft-limit run because next_wake_at is within ${SCHEDULING_CONFIG.SOFT_LIMIT_NEXT_WAKE_AT_WINDOW_MINUTES} minutes: 2026-03-22T01:05:00.000Z`
    );
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

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(true);
    expect(mockLogger.info).toHaveBeenCalledWith(
      'Usage: 0  (Soft limit: 45000)'
    );
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

    const result = await validateRunPreconditions(echo);

    expect(result).toBe(true);
    expect(mockLogger.warn).toHaveBeenCalledWith(
      'Stored next_wake_at is invalid and will be ignored: not-a-date'
    );
    expect(deleteFn).toHaveBeenCalledWith('next_wake_at');
  });
});
