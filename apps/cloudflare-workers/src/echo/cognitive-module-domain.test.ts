import { describe, expect, it, type Mock, vi } from 'vitest';

import type { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import type {
  CognitiveModuleDomainCommitInput,
  CognitiveModulePhase,
  CognitiveModulePhaseInput,
  CognitiveModulePhaseResult,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import type { EmotionCognitiveModuleOutput } from '@echo-chamber/core/agent/cognitive-module-schema';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import type { ModelUsage } from '@echo-chamber/core/ports/model';

import { CognitiveModuleDomainStore } from './cognitive-module-domain';

interface MemoryRuntime {
  searchMemory: Mock<MemorySystem['searchMemory']>;
  prepareMemoryWrite: Mock<MemorySystem['prepareMemoryWrite']>;
  commitPreparedMemoryWrites: Mock<MemorySystem['commitPreparedMemoryWrites']>;
  emitMemoryCommitEvents: Mock<MemorySystem['emitMemoryCommitEvents']>;
}

interface MemoryStorage {
  data: Map<string, unknown>;
  storage: DurableObjectStorage;
  transaction: ReturnType<typeof vi.fn>;
}

function createStorage(
  entries: Readonly<Record<string, unknown>> = {}
): MemoryStorage {
  const data = new Map(Object.entries(entries));
  const transactionApi = {
    get: vi.fn(
      async <T>(key: string): Promise<T | undefined> =>
        await Promise.resolve(data.get(key) as T | undefined)
    ),
    put: vi.fn(async (key: string, value: unknown): Promise<void> => {
      data.set(key, value);
      await Promise.resolve();
    }),
    delete: vi.fn(
      async (key: string | string[]): Promise<boolean | number> =>
        await Promise.resolve(
          Array.isArray(key)
            ? key.reduce(
                (deleted, item) => deleted + (data.delete(item) ? 1 : 0),
                0
              )
            : data.delete(key)
        )
    ),
  };
  const transaction = vi.fn(
    async <T>(
      callback: (transaction: DurableObjectTransaction) => Promise<T>
    ): Promise<T> =>
      await callback(transactionApi as unknown as DurableObjectTransaction)
  );
  const storage = {
    get: transactionApi.get,
    put: transactionApi.put,
    delete: transactionApi.delete,
    transaction,
  } as unknown as DurableObjectStorage;
  return { data, storage, transaction };
}

function createMemoryRuntime(): MemoryRuntime {
  return {
    searchMemory: vi.fn<MemorySystem['searchMemory']>().mockResolvedValue([]),
    prepareMemoryWrite: vi.fn<MemorySystem['prepareMemoryWrite']>(
      async (id, content, emotion, type) =>
        await Promise.resolve({
          status: 'prepared',
          id,
          content,
          emotion,
          type,
          embedding: new ArrayBuffer(4),
          embeddingModel: 'test/embedding',
          createdAt: '2026-08-23T00:00:00.000Z',
        })
    ),
    commitPreparedMemoryWrites: vi.fn<
      MemorySystem['commitPreparedMemoryWrites']
    >((writes) => ({
      storedIds: writes.map((write) => write.id),
      existingIds: [],
      evicted: [],
    })),
    emitMemoryCommitEvents: vi
      .fn<MemorySystem['emitMemoryCommitEvents']>()
      .mockResolvedValue(undefined),
  };
}

function emotionOutput(label = 'deliberate'): EmotionCognitiveModuleOutput {
  return {
    valence: 0.1,
    arousal: 0.2,
    labels: [label],
  };
}

function phaseInput(
  phase: CognitiveModulePhase,
  committedVersion = 0,
  sequence = phase === 'pre_main' ? 1 : 2
): CognitiveModulePhaseInput {
  return {
    activationId: 'activation-1',
    boundaryId: `activation-1:${sequence}:${phase}`,
    sequence,
    phase,
    committed: {
      version: committedVersion,
      emotion: committedVersion === 0 ? null : emotionOutput('previous'),
      previousSessionMemory: null,
      recalledMemories: [],
    },
  };
}

function commitInput(
  phase: CognitiveModulePhase,
  committedVersion = 0
): CognitiveModuleDomainCommitInput {
  return {
    phase: phaseInput(phase, committedVersion),
    memory: {
      status: 'ready',
      value:
        phase === 'pre_main'
          ? { query: 'review requirements and current implementation' }
          : {
              content: 'The session established phase-specific interfaces.',
              type: 'semantic',
            },
      attempts: 1,
    },
    emotion: { status: 'ready', value: emotionOutput(), attempts: 1 },
  };
}

function createUsage(): ModelUsage {
  return {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: 1,
    totalInputTokens: 1,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens: 1,
  };
}

describe('CognitiveModuleDomainStore', () => {
  it('activation開始時は前sessionのMemoryとEmotionを読み、検索結果は引き継がない', async () => {
    const storedEmotion = emotionOutput('persisted');
    const previousSessionMemory = {
      content: 'The previous session settled the implementation boundary.',
      type: 'episode' as const,
      emotion: storedEmotion,
      createdAt: '2026-08-22T00:00:00.000Z',
    };
    const { storage } = createStorage({
      'cognitive:domain-state': {
        version: 2,
        emotion: storedEmotion,
        previousSessionMemory,
        recalledMemories: [
          {
            content: 'stale recall',
            type: 'semantic',
            emotion: storedEmotion,
            createdAt: '2026-08-22T00:00:00.000Z',
          },
        ],
        lastBoundaryId: 'previous:post_main',
        updatedAt: '2026-08-22T00:00:00.000Z',
      },
    });
    const memory = createMemoryRuntime();
    const domain = new CognitiveModuleDomainStore({ storage, memory });

    await expect(domain.beginActivation('activation-1')).resolves.toEqual({
      version: 2,
      emotion: storedEmotion,
      previousSessionMemory,
      recalledMemories: [],
    });
    expect(memory.searchMemory).not.toHaveBeenCalled();
  });

  it('pre_mainではMemory Moduleのqueryで検索し、Emotionと検索結果を一度確定する', async () => {
    const previousSessionMemory = {
      content: 'The previous session established the initial context.',
      type: 'episode' as const,
      emotion: emotionOutput('previous'),
      createdAt: '2026-08-21T00:00:00.000Z',
    };
    const { data, storage, transaction } = createStorage({
      'cognitive:domain-state': {
        version: 1,
        emotion: emotionOutput('previous'),
        previousSessionMemory,
        recalledMemories: [],
        lastBoundaryId: 'previous:post_main',
        updatedAt: '2026-08-21T00:00:00.000Z',
      },
    });
    const memory = createMemoryRuntime();
    memory.searchMemory.mockResolvedValue([
      {
        content: 'Existing memory',
        type: 'semantic',
        emotion: emotionOutput('remembered'),
        createdAt: '2026-08-22T00:00:00.000Z',
        updatedAt: '2026-08-22T00:00:00.000Z',
        similarity: 0.9,
      },
    ]);
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory,
      now: (): Date => new Date('2026-08-23T00:00:00.000Z'),
    });
    const input = commitInput('pre_main', 1);
    input.phase.committed.previousSessionMemory = previousSessionMemory;

    await expect(domain.commitPhase(input)).resolves.toEqual({
      version: 2,
      emotion: emotionOutput(),
      previousSessionMemory,
      recalledMemories: [
        {
          content: 'Existing memory',
          type: 'semantic',
          emotion: emotionOutput('remembered'),
          createdAt: '2026-08-22T00:00:00.000Z',
        },
      ],
    });
    await domain.commitPhase(input);

    expect(memory.searchMemory).toHaveBeenCalledTimes(1);
    expect(memory.searchMemory).toHaveBeenCalledWith(
      'review requirements and current implementation'
    );
    expect(memory.prepareMemoryWrite).not.toHaveBeenCalled();
    expect(memory.commitPreparedMemoryWrites).not.toHaveBeenCalled();
    expect(memory.emitMemoryCommitEvents).not.toHaveBeenCalled();
    expect(transaction).toHaveBeenCalledTimes(1);
    expect(data.get('cognitive:domain-state')).toMatchObject({
      version: 2,
      emotion: emotionOutput(),
      previousSessionMemory,
      lastBoundaryId: 'activation-1:1:pre_main',
    });
  });

  it('Memory検索の一時失敗だけを同じqueryで1回再試行する', async () => {
    const { storage } = createStorage();
    const memory = createMemoryRuntime();
    memory.searchMemory
      .mockRejectedValueOnce(
        Object.assign(new Error('temporary'), { status: 503 })
      )
      .mockResolvedValueOnce([]);
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory,
      isRetryable: (error: unknown): boolean =>
        (error as { status?: number }).status === 503,
    });

    await domain.commitPhase(commitInput('pre_main'));

    expect(memory.searchMemory).toHaveBeenCalledTimes(2);
    expect(memory.searchMemory.mock.calls[0]?.[0]).toBe(
      memory.searchMemory.mock.calls[1]?.[0]
    );
  });

  it('Memory検索の確定失敗ではEmotionも永続化しない', async () => {
    const { data, storage } = createStorage();
    const memory = createMemoryRuntime();
    memory.searchMemory.mockRejectedValue(
      new Error('invalid embedding request')
    );
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory,
      isRetryable: (): boolean => false,
    });

    await expect(domain.commitPhase(commitInput('pre_main'))).rejects.toThrow(
      'invalid embedding request'
    );
    expect(data.has('cognitive:domain-state')).toBe(false);
    expect(memory.commitPreparedMemoryWrites).not.toHaveBeenCalled();
  });

  it('post_mainではcontentとtypeを同phaseのEmotion付きで一度保存する', async () => {
    const previous = {
      version: 1,
      emotion: emotionOutput('previous'),
      previousSessionMemory: {
        content: 'Older session memory',
        type: 'episode' as const,
        emotion: emotionOutput('previous'),
        createdAt: '2026-08-22T00:00:00.000Z',
      },
      recalledMemories: [],
      lastBoundaryId: 'activation-1:1:pre_main',
      updatedAt: '2026-08-23T00:00:00.000Z',
    };
    const { data, storage, transaction } = createStorage({
      'cognitive:domain-state': previous,
    });
    const memory = createMemoryRuntime();
    const domain = new CognitiveModuleDomainStore({ storage, memory });
    const input = commitInput('post_main', 1);

    await expect(domain.commitPhase(input)).resolves.toMatchObject({
      version: 2,
      emotion: emotionOutput(),
      previousSessionMemory: {
        content: 'The session established phase-specific interfaces.',
        type: 'semantic',
        emotion: emotionOutput(),
        createdAt: '2026-08-23T00:00:00.000Z',
      },
      recalledMemories: [],
    });
    await domain.commitPhase(input);

    expect(memory.prepareMemoryWrite).toHaveBeenCalledTimes(1);
    expect(memory.prepareMemoryWrite).toHaveBeenCalledWith(
      'activation-1:2:post_main:store_memory',
      'The session established phase-specific interfaces.',
      emotionOutput(),
      'semantic'
    );
    expect(memory.commitPreparedMemoryWrites).toHaveBeenCalledTimes(1);
    expect(transaction).toHaveBeenCalledTimes(1);
    expect(data.get('cognitive:domain-state')).toMatchObject({
      version: 2,
      previousSessionMemory: {
        content: 'The session established phase-specific interfaces.',
        type: 'semantic',
        emotion: emotionOutput(),
        createdAt: '2026-08-23T00:00:00.000Z',
      },
      lastBoundaryId: 'activation-1:2:post_main',
    });
  });

  it('Memory保存の一時失敗を同じdeterministic IDで1回再試行する', async () => {
    const { storage } = createStorage({
      'cognitive:domain-state': {
        version: 1,
        emotion: emotionOutput('previous'),
        lastBoundaryId: 'activation-1:1:pre_main',
        updatedAt: '2026-08-23T00:00:00.000Z',
      },
    });
    const memory = createMemoryRuntime();
    memory.prepareMemoryWrite
      .mockRejectedValueOnce(
        Object.assign(new Error('temporary'), { status: 503 })
      )
      .mockResolvedValueOnce({
        status: 'prepared',
        id: 'activation-1:2:post_main:store_memory',
        content: 'The session established phase-specific interfaces.',
        emotion: emotionOutput(),
        type: 'semantic',
        embedding: new ArrayBuffer(4),
        embeddingModel: 'test/embedding',
        createdAt: '2026-08-23T00:00:00.000Z',
      });
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory,
      isRetryable: (error: unknown): boolean =>
        (error as { status?: number }).status === 503,
    });

    await domain.commitPhase(commitInput('post_main', 1));

    expect(memory.prepareMemoryWrite).toHaveBeenCalledTimes(2);
    expect(memory.prepareMemoryWrite.mock.calls[0]?.[0]).toBe(
      memory.prepareMemoryWrite.mock.calls[1]?.[0]
    );
  });

  it('version conflictではMemory side effectを開始しない', async () => {
    const { storage } = createStorage({
      'cognitive:domain-state': {
        version: 2,
        emotion: emotionOutput(),
        lastBoundaryId: 'another-boundary',
        updatedAt: '2026-08-23T00:00:00.000Z',
      },
    });
    const memory = createMemoryRuntime();
    const domain = new CognitiveModuleDomainStore({ storage, memory });

    await expect(
      domain.commitPhase(commitInput('post_main', 1))
    ).rejects.toThrow('Cognitive domain state version conflict');
    expect(memory.searchMemory).not.toHaveBeenCalled();
    expect(memory.prepareMemoryWrite).not.toHaveBeenCalled();
  });

  it('確定済みEmotionをMainの明示的store_memory用に返す', async () => {
    const { storage } = createStorage({
      'cognitive:domain-state': {
        version: 1,
        emotion: emotionOutput('current'),
        lastBoundaryId: 'activation-1:1:pre_main',
        updatedAt: '2026-08-23T00:00:00.000Z',
      },
    });
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory: createMemoryRuntime(),
    });

    await expect(domain.getCurrentEmotion()).resolves.toEqual(
      emotionOutput('current')
    );
  });

  it('phase failureは状態を変更せず、bounded errorをeventへ残す', async () => {
    const { data, storage } = createStorage();
    const emit = vi.fn<EchoEventPort['emit']>().mockResolvedValue(undefined);
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory: createMemoryRuntime(),
      events: { emit },
    });
    const phase = phaseInput('pre_main');
    const result: CognitiveModulePhaseResult = {
      ...phase,
      memory: {
        status: 'failed',
        reason: 'non_retryable',
        error: 'schema mismatch',
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
        value: emotionOutput(),
        attempts: 1,
      },
      usage: createUsage(),
    };

    await domain.failPhase(result, new Error('sqlite disk full'));

    expect(emit.mock.calls[0]?.[0]).toMatchObject({
      type: 'cognitive.phase.failed',
      payload: {
        commitError: 'sqlite disk full',
        memory: {
          status: 'failed',
          error: 'schema mismatch',
          outputValidation: {
            diagnostic: {
              issues: [{ path: 'query', code: 'invalid_type' }],
            },
          },
        },
        emotion: { status: 'ready' },
      },
    });
    expect(data.size).toBe(0);
  });

  it('dashboard stateは確定済みのMemoryとEmotionを返す', async () => {
    const { storage } = createStorage();
    const onStateChanged = vi.fn();
    const domain = new CognitiveModuleDomainStore({
      storage,
      memory: createMemoryRuntime(),
      now: (): Date => new Date('2026-08-23T00:00:00.000Z'),
      onStateChanged,
    });

    await expect(domain.getDashboardState()).resolves.toEqual({
      domainVersion: 0,
      emotion: null,
      previousSessionMemory: null,
      recalledMemories: [],
      lastBoundaryId: null,
      updatedAt: null,
    });
    await domain.commitPhase(commitInput('pre_main'));
    await expect(domain.getDashboardState()).resolves.toEqual({
      domainVersion: 1,
      emotion: emotionOutput(),
      previousSessionMemory: null,
      recalledMemories: [],
      lastBoundaryId: 'activation-1:1:pre_main',
      updatedAt: '2026-08-23T00:00:00.000Z',
    });
    await domain.commitPhase(commitInput('post_main', 1));
    await expect(domain.getDashboardState()).resolves.toEqual({
      domainVersion: 2,
      emotion: emotionOutput(),
      previousSessionMemory: {
        content: 'The session established phase-specific interfaces.',
        type: 'semantic',
        emotion: emotionOutput(),
        createdAt: '2026-08-23T00:00:00.000Z',
      },
      recalledMemories: [],
      lastBoundaryId: 'activation-1:2:post_main',
      updatedAt: '2026-08-23T00:00:00.000Z',
    });
    expect(onStateChanged).toHaveBeenCalledTimes(2);
  });
});
