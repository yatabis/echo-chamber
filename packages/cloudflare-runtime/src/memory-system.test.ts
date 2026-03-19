import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Emotion } from '@echo-chamber/core/echo/types';
import type { LoggerPort } from '@echo-chamber/core/ports/logger';

import { MemorySystem } from './memory-system';

import type { EmbeddingService } from './embedding-service';

interface MockMemoryRow {
  id: string;
  content: string;
  type: 'semantic' | 'episode';
  embedding: ArrayBuffer;
  embedding_model: string;
  emotion_valence: number;
  emotion_arousal: number;
  emotion_labels: string;
  created_at: string;
  updated_at: string;
}

interface MockSqlResult {
  toArray(): MockMemoryRow[];
  one(): MockMemoryRow | { count: number } | undefined;
}

interface MockSqlStorage {
  exec: ReturnType<typeof vi.fn>;
  _tables: { memories: MockMemoryRow[] };
  _reset(): void;
}

/**
 * SqlStorageのモック作成用ヘルパー
 */
function createMockSqlStorage(): MockSqlStorage {
  const tables: { memories: MockMemoryRow[] } = {
    memories: [],
  };

  function handleSelect(
    queryLower: string,
    args: unknown[]
  ): MockSqlResult | null {
    if (queryLower.includes('count(*)')) {
      return {
        toArray: () => [],
        one: () => ({ count: tables.memories.length }),
      };
    }
    if (queryLower.includes('order by created_at desc limit 1')) {
      const sorted = [...tables.memories].sort((a, b) =>
        b.created_at.localeCompare(a.created_at)
      );
      return { toArray: () => sorted.slice(0, 1), one: () => sorted[0] };
    }
    if (queryLower.includes('order by updated_at asc limit 1')) {
      const sorted = [...tables.memories].sort((a, b) =>
        a.updated_at.localeCompare(b.updated_at)
      );
      return { toArray: () => sorted.slice(0, 1), one: () => sorted[0] };
    }
    if (queryLower.includes('from memories where embedding_model')) {
      const targetModel = args[0] as string;
      const stale = tables.memories.filter(
        (m) => m.embedding_model !== targetModel
      );
      return {
        toArray: () => stale.map((m) => ({ ...m })),
        one: () => (stale[0] ? { ...stale[0] } : undefined),
      };
    }
    if (
      queryLower.includes('from memories') &&
      !queryLower.includes('order by')
    ) {
      return {
        toArray: () => tables.memories.map((m) => ({ ...m })),
        one: () => (tables.memories[0] ? { ...tables.memories[0] } : undefined),
      };
    }
    return null;
  }

  function handleMutation(
    queryLower: string,
    args: unknown[]
  ): MockSqlResult | null {
    if (queryLower.startsWith('update memories set embedding')) {
      const [newEmbedding, newModel, targetId] = args as [
        ArrayBuffer,
        string,
        string,
        string,
      ];
      const idx = tables.memories.findIndex((m) => m.id === targetId);
      const existing = tables.memories[idx];
      if (idx !== -1 && existing !== undefined) {
        tables.memories[idx] = {
          ...existing,
          embedding: newEmbedding,
          embedding_model: newModel,
        };
      }
      return { toArray: () => [], one: () => undefined };
    }
    if (queryLower.startsWith('insert into memories')) {
      const [
        id,
        content,
        type,
        embedding,
        embedding_model,
        valence,
        arousal,
        labels,
        created_at,
        updated_at,
      ] = args as [
        string,
        string,
        'semantic' | 'episode',
        ArrayBuffer,
        string,
        number,
        number,
        string,
        string,
        string,
      ];
      tables.memories.push({
        id,
        content,
        type,
        embedding,
        embedding_model,
        emotion_valence: valence,
        emotion_arousal: arousal,
        emotion_labels: labels,
        created_at,
        updated_at,
      });
      return { toArray: () => [], one: () => undefined };
    }
    if (queryLower.startsWith('delete from memories')) {
      const idToDelete = args[0] as string;
      tables.memories = tables.memories.filter((m) => m.id !== idToDelete);
      return { toArray: () => [], one: () => undefined };
    }
    return null;
  }

  const mockExec = vi
    .fn()
    .mockImplementation((query: string, ...args: unknown[]): MockSqlResult => {
      const queryLower = query.toLowerCase().trim();
      if (
        queryLower.startsWith('create') ||
        queryLower.startsWith('pragma') ||
        queryLower.startsWith('alter')
      ) {
        return { toArray: () => [], one: () => undefined };
      }
      if (queryLower.includes('select')) {
        const result = handleSelect(queryLower, args);
        if (result) return result;
      }
      const mutResult = handleMutation(queryLower, args);
      if (mutResult) return mutResult;
      return { toArray: () => [], one: () => undefined };
    });

  return {
    exec: mockExec,
    _tables: tables,
    _reset: (): void => {
      tables.memories = [];
    },
  };
}

const mockEmbeddingService: EmbeddingService = {
  embed: vi.fn(),
  modelIdentifier: 'test/mock-embedding-model',
};

const mockLogger: Pick<LoggerPort, 'debug' | 'info' | 'error'> = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  log: vi.fn(),
} as unknown as LoggerPort;

/**
 * Float32ArrayをArrayBufferに変換（テスト用）
 */
function float32ArrayToBuffer(arr: number[]): ArrayBuffer {
  const float32 = new Float32Array(arr);
  return float32.buffer;
}

/**
 * テスト用のメモリ行を作成
 */
function createMockMemoryRow(
  overrides?: Partial<MockMemoryRow>
): MockMemoryRow {
  return {
    id: crypto.randomUUID(),
    content: 'Test memory content',
    type: 'episode',
    embedding: float32ArrayToBuffer(new Array<number>(1536).fill(0)),
    embedding_model: 'openai/text-embedding-3-small',
    emotion_valence: 0.5,
    emotion_arousal: 0.3,
    emotion_labels: JSON.stringify(['neutral']),
    created_at: '2025-01-25T10:00:00.000Z',
    updated_at: '2025-01-25T10:00:00.000Z',
    ...overrides,
  };
}

describe('MemorySystem', () => {
  let memorySystem: MemorySystem;
  let mockSql: MockSqlStorage;

  // テストデータのタイムスタンプ: 2025-01-25T15:00:00.000Z (UTC)
  // 基準時刻を2025-01-27に固定して「2日前」を期待
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-27T23:59:59.999Z'));
    vi.resetAllMocks();
    // デフォルトのembedding実装を設定
    (mockEmbeddingService.embed as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Array<number>(1536).fill(0)
    );
    mockSql = createMockSqlStorage();
    memorySystem = new MemorySystem({
      sql: mockSql as unknown as SqlStorage,
      embeddingService: mockEmbeddingService,
      logger: mockLogger,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('getLatestMemory', () => {
    it('メモリが存在しない場合はnullを返す', () => {
      const result = memorySystem.getLatestMemory();

      expect(result).toBeNull();
    });

    it('最も新しいメモリを返す（created_atで判定）', () => {
      // データをモックに追加
      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Older memory',
          created_at: '2025-01-24T10:00:00.000Z',
          updated_at: '2025-01-24T10:00:00.000Z',
        }),
        createMockMemoryRow({
          content: 'Newer memory',
          created_at: '2025-01-25T15:00:00.000Z',
          updated_at: '2025-01-25T15:00:00.000Z',
        }),
        createMockMemoryRow({
          content: 'Middle memory',
          created_at: '2025-01-25T10:00:00.000Z',
          updated_at: '2025-01-25T10:00:00.000Z',
        }),
      ];

      const result = memorySystem.getLatestMemory();

      expect(result).toEqual({
        content: 'Newer memory',
        type: 'episode',
        emotion: {
          valence: 0.5,
          arousal: 0.3,
          labels: ['neutral'],
        },
        createdAt: '2日前 (2025年01月26日 00:00:00)',
        updatedAt: '2日前 (2025年01月26日 00:00:00)',
      });
    });

    it('embeddingを含まない結果を返す', () => {
      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Test memory',
          embedding: float32ArrayToBuffer(new Array<number>(1536).fill(0.1)),
        }),
      ];

      const result = memorySystem.getLatestMemory();

      expect(result).not.toHaveProperty('embedding');
      expect(result).toHaveProperty('content');
      expect(result).toHaveProperty('type');
      expect(result).toHaveProperty('emotion');
      expect(result).toHaveProperty('createdAt');
    });
  });

  describe('storeMemory', () => {
    it('新しいメモリを保存できる', async () => {
      const emotion: Emotion = {
        valence: 0.8,
        arousal: 0.5,
        labels: ['happy'],
      };

      await memorySystem.storeMemory('New memory content', emotion, 'episode');

      expect(mockSql._tables.memories).toHaveLength(1);
      expect(mockSql._tables.memories[0]).toMatchObject({
        content: 'New memory content',
        type: 'episode',
        emotion_valence: 0.8,
        emotion_arousal: 0.5,
        emotion_labels: JSON.stringify(['happy']),
      });
    });

    it('semanticタイプのメモリを保存できる', async () => {
      const emotion: Emotion = {
        valence: 0.5,
        arousal: 0.3,
        labels: ['neutral'],
      };

      await memorySystem.storeMemory('Semantic memory', emotion, 'semantic');

      expect(mockSql._tables.memories).toHaveLength(1);
      expect(mockSql._tables.memories[0]).toMatchObject({
        content: 'Semantic memory',
        type: 'semantic',
      });
    });

    it('embeddingServiceが呼ばれる', async () => {
      const emotion: Emotion = {
        valence: 0.5,
        arousal: 0.3,
        labels: ['neutral'],
      };

      await memorySystem.storeMemory('Test content', emotion, 'episode');

      expect(mockEmbeddingService.embed).toHaveBeenCalledWith('Test content'); // eslint-disable-line @typescript-eslint/unbound-method
    });
  });

  describe('getMemoryCount', () => {
    it('メモリ件数を返す', () => {
      mockSql._tables.memories = [
        createMockMemoryRow(),
        createMockMemoryRow(),
        createMockMemoryRow(),
      ];

      const count = memorySystem.getMemoryCount();

      expect(count).toBe(3);
    });

    it('メモリが空の場合は0を返す', () => {
      const count = memorySystem.getMemoryCount();

      expect(count).toBe(0);
    });
  });

  describe('searchMemory', () => {
    it('メモリが空の場合は空配列を返す', async () => {
      const results = await memorySystem.searchMemory('test query');

      expect(results).toEqual([]);
    });

    it('検索クエリに対してembeddingを生成する', async () => {
      // embeddingを正しくArrayBufferとして設定
      const embedding = new Float32Array(1536).fill(0.5);
      mockSql._tables.memories = [
        createMockMemoryRow({
          embedding: embedding.buffer,
        }),
      ];

      await memorySystem.searchMemory('search query');

      expect(mockEmbeddingService.embed).toHaveBeenCalledWith('search query'); // eslint-disable-line @typescript-eslint/unbound-method
    });

    it('類似度に基づいて結果を返す', async () => {
      // 同じembeddingを使うことで類似度1.0を期待
      const embedding = new Float32Array(1536).fill(0.5);
      (
        mockEmbeddingService.embed as ReturnType<typeof vi.fn>
      ).mockResolvedValue(Array.from(embedding));

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Similar memory',
          embedding: embedding.buffer,
        }),
      ];

      const results = await memorySystem.searchMemory('test query');

      expect(results).toHaveLength(1);
      const firstResult = results[0];
      expect(firstResult).toBeDefined();
      expect(firstResult).toMatchObject({
        content: 'Similar memory',
      });
      expect(firstResult?.similarity).toBeCloseTo(1.0, 5);
    });

    it('type未指定時は全タイプのメモリを検索する', async () => {
      const embedding = new Float32Array(1536).fill(0.5);
      (
        mockEmbeddingService.embed as ReturnType<typeof vi.fn>
      ).mockResolvedValue(Array.from(embedding));

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Episode memory',
          type: 'episode',
          embedding: embedding.buffer,
        }),
        createMockMemoryRow({
          content: 'Semantic memory',
          type: 'semantic',
          embedding: embedding.buffer,
        }),
      ];

      const results = await memorySystem.searchMemory('test query');

      expect(results).toHaveLength(2);
    });

    it('type指定時はそのタイプのメモリのみ返す（episode）', async () => {
      const embedding = new Float32Array(1536).fill(0.5);
      (
        mockEmbeddingService.embed as ReturnType<typeof vi.fn>
      ).mockResolvedValue(Array.from(embedding));

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Episode memory',
          type: 'episode',
          embedding: embedding.buffer,
        }),
        createMockMemoryRow({
          content: 'Semantic memory',
          type: 'semantic',
          embedding: embedding.buffer,
        }),
      ];

      const results = await memorySystem.searchMemory('test query', 'episode');

      expect(results).toHaveLength(1);
      expect(results[0]?.content).toBe('Episode memory');
      expect(results[0]?.type).toBe('episode');
    });

    it('type指定時はそのタイプのメモリのみ返す（semantic）', async () => {
      const embedding = new Float32Array(1536).fill(0.5);
      (
        mockEmbeddingService.embed as ReturnType<typeof vi.fn>
      ).mockResolvedValue(Array.from(embedding));

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Episode memory',
          type: 'episode',
          embedding: embedding.buffer,
        }),
        createMockMemoryRow({
          content: 'Semantic memory',
          type: 'semantic',
          embedding: embedding.buffer,
        }),
      ];

      const results = await memorySystem.searchMemory('test query', 'semantic');

      expect(results).toHaveLength(1);
      expect(results[0]?.content).toBe('Semantic memory');
      expect(results[0]?.type).toBe('semantic');
    });
  });

  describe('reEmbedStaleMemories', () => {
    it('現在のモデルと同じ場合は何もしない', async () => {
      mockSql._tables.memories = [
        createMockMemoryRow({
          embedding_model: 'test/mock-embedding-model',
        }),
      ];

      await memorySystem.reEmbedStaleMemories();

      expect(mockEmbeddingService.embed).not.toHaveBeenCalled(); // eslint-disable-line @typescript-eslint/unbound-method
    });

    it('古いモデルの memory を再 embedding する', async () => {
      const newEmbedding = new Array<number>(2048).fill(0.5);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      vi.mocked(mockEmbeddingService.embed).mockResolvedValue(newEmbedding);

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Old memory',
          embedding_model: 'openai/text-embedding-3-small',
        }),
      ];

      await memorySystem.reEmbedStaleMemories();

      expect(mockEmbeddingService.embed).toHaveBeenCalledWith('Old memory'); // eslint-disable-line @typescript-eslint/unbound-method
      expect(mockSql._tables.memories[0]?.embedding_model).toBe(
        'test/mock-embedding-model'
      );
    });

    it('複数の stale memory を順に再 embedding する', async () => {
      const newEmbedding = new Array<number>(2048).fill(0.1);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      vi.mocked(mockEmbeddingService.embed).mockResolvedValue(newEmbedding);

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Memory A',
          embedding_model: 'openai/text-embedding-3-small',
        }),
        createMockMemoryRow({
          content: 'Memory B',
          embedding_model: 'openai/text-embedding-3-small',
        }),
      ];

      await memorySystem.reEmbedStaleMemories();

      expect(mockEmbeddingService.embed).toHaveBeenCalledTimes(2); // eslint-disable-line @typescript-eslint/unbound-method
      expect(mockSql._tables.memories[0]?.embedding_model).toBe(
        'test/mock-embedding-model'
      );
      expect(mockSql._tables.memories[1]?.embedding_model).toBe(
        'test/mock-embedding-model'
      );
    });

    it('embed に失敗した memory はスキップして続行する', async () => {
      // eslint-disable-next-line @typescript-eslint/unbound-method
      vi.mocked(mockEmbeddingService.embed)
        .mockRejectedValueOnce(new Error('API error'))
        .mockResolvedValueOnce(new Array<number>(2048).fill(0.1));

      mockSql._tables.memories = [
        createMockMemoryRow({
          content: 'Memory A',
          embedding_model: 'openai/text-embedding-3-small',
        }),
        createMockMemoryRow({
          content: 'Memory B',
          embedding_model: 'openai/text-embedding-3-small',
        }),
      ];

      await memorySystem.reEmbedStaleMemories();

      expect(mockEmbeddingService.embed).toHaveBeenCalledTimes(2); // eslint-disable-line @typescript-eslint/unbound-method
      // Memory A は失敗したので embedding_model は更新されていない
      expect(mockSql._tables.memories[0]?.embedding_model).toBe(
        'openai/text-embedding-3-small'
      );
      // Memory B は成功したので更新されている
      expect(mockSql._tables.memories[1]?.embedding_model).toBe(
        'test/mock-embedding-model'
      );
    });
  });
});
