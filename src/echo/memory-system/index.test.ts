import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MemorySystem } from './index';

import type { EmbeddingService } from '../../llm/openai/embedding';
import type { Logger } from '../../utils/logger';
import type { Emotion } from '../types';

interface MockMemoryRow {
  id: string;
  content: string;
  embedding: ArrayBuffer;
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

  const mockExec = vi
    .fn()
    .mockImplementation((query: string, ...args: unknown[]): MockSqlResult => {
      const queryLower = query.toLowerCase().trim();

      // CREATE TABLE / CREATE INDEX は無視
      if (queryLower.startsWith('create')) {
        return { toArray: () => [], one: () => undefined };
      }

      // COUNT クエリ
      if (queryLower.includes('count(*)')) {
        return {
          toArray: () => [],
          one: () => ({ count: tables.memories.length }),
        };
      }

      // SELECT クエリ（最新取得）
      if (queryLower.includes('order by created_at desc limit 1')) {
        const sorted = [...tables.memories].sort((a, b) =>
          b.created_at.localeCompare(a.created_at)
        );
        return {
          toArray: () => sorted.slice(0, 1),
          one: () => sorted[0],
        };
      }

      // SELECT クエリ（最古取得）
      if (queryLower.includes('order by updated_at asc limit 1')) {
        const sorted = [...tables.memories].sort((a, b) =>
          a.updated_at.localeCompare(b.updated_at)
        );
        return {
          toArray: () => sorted.slice(0, 1),
          one: () => sorted[0],
        };
      }

      // SELECT 全件（order byなし）
      if (
        queryLower.includes('select') &&
        queryLower.includes('from memories') &&
        !queryLower.includes('order by')
      ) {
        return {
          toArray: () => tables.memories.map((m) => ({ ...m })),
          one: () =>
            tables.memories[0] ? { ...tables.memories[0] } : undefined,
        };
      }

      // INSERT
      if (queryLower.startsWith('insert into memories')) {
        const [
          id,
          content,
          embedding,
          valence,
          arousal,
          labels,
          created_at,
          updated_at,
        ] = args as [
          string,
          string,
          ArrayBuffer,
          number,
          number,
          string,
          string,
          string,
        ];
        tables.memories.push({
          id,
          content,
          embedding,
          emotion_valence: valence,
          emotion_arousal: arousal,
          emotion_labels: labels,
          created_at,
          updated_at,
        });
        return { toArray: () => [], one: () => undefined };
      }

      // DELETE
      if (queryLower.startsWith('delete from memories')) {
        const idToDelete = args[0] as string;
        tables.memories = tables.memories.filter((m) => m.id !== idToDelete);
        return { toArray: () => [], one: () => undefined };
      }

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
};

const mockLogger: Logger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
} as unknown as Logger;

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
    embedding: float32ArrayToBuffer(new Array<number>(1536).fill(0)),
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

  beforeEach(() => {
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
        emotion: {
          valence: 0.5,
          arousal: 0.3,
          labels: ['neutral'],
        },
        createdAt: '2025/01/26 00:00:00',
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

      await memorySystem.storeMemory('New memory content', emotion);

      expect(mockSql._tables.memories).toHaveLength(1);
      expect(mockSql._tables.memories[0]).toMatchObject({
        content: 'New memory content',
        emotion_valence: 0.8,
        emotion_arousal: 0.5,
        emotion_labels: JSON.stringify(['happy']),
      });
    });

    it('embeddingServiceが呼ばれる', async () => {
      const emotion: Emotion = {
        valence: 0.5,
        arousal: 0.3,
        labels: ['neutral'],
      };

      await memorySystem.storeMemory('Test content', emotion);

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
  });
});
