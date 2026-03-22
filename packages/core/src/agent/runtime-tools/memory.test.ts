import { beforeEach, describe, expect, it, vi } from 'vitest';

import { searchMemoryTool, storeMemoryTool } from './memory';
import { mockToolContext } from './mock-tool-context';

import type { Emotion } from '../../echo/types';

const mockedStoreMemory = vi.mocked(mockToolContext.memory.store);
const mockedSearchMemory = vi.mocked(mockToolContext.memory.search);

beforeEach(() => {
  vi.resetAllMocks();
});

const createMockEmotion = (overrides?: Partial<Emotion>): Emotion => {
  return {
    valence: 0.5,
    arousal: 0.3,
    labels: ['neutral'],
    ...overrides,
  };
};

describe('Memory Functions', () => {
  describe('storeMemoryTool', () => {
    it('name', () => {
      expect(storeMemoryTool.name).toBe('store_memory');
    });

    it('description', () => {
      expect(storeMemoryTool.description).toBeDefined();
    });

    it('parameters', () => {
      const { parameters } = storeMemoryTool;
      expect(parameters).toBeDefined();

      expect(parameters).toHaveProperty('content');
      expect(parameters).toHaveProperty('type');
      expect(parameters).toHaveProperty('emotion');
    });

    describe('handler', () => {
      it('MemorySystem.storeMemoryを呼び出す', async () => {
        const args = {
          content: 'Had a great conversation about AI',
          type: 'episode' as const,
          emotion: createMockEmotion({
            valence: 0.7,
            arousal: 0.5,
            labels: ['joy', 'interest'],
          }),
        };

        const result = await storeMemoryTool.handler(args, mockToolContext);

        expect(mockedStoreMemory).toHaveBeenCalledWith(
          'Had a great conversation about AI',
          {
            valence: 0.7,
            arousal: 0.5,
            labels: ['joy', 'interest'],
          },
          'episode'
        );
        expect(result).toEqual({ success: true });
      });

      it('semanticタイプで呼び出すことができる', async () => {
        const args = {
          content: 'General knowledge about AI',
          type: 'semantic' as const,
          emotion: createMockEmotion(),
        };

        const result = await storeMemoryTool.handler(args, mockToolContext);

        expect(mockedStoreMemory).toHaveBeenCalledWith(
          'General knowledge about AI',
          createMockEmotion(),
          'semantic'
        );
        expect(result).toEqual({ success: true });
      });

      it('MemorySystemエラー時はエラーを返す', async () => {
        mockedStoreMemory.mockRejectedValue(new Error('Memory System Error'));

        const args = {
          content: 'Memory that will fail',
          type: 'episode' as const,
          emotion: createMockEmotion(),
        };

        const result = await storeMemoryTool.handler(args, mockToolContext);

        expect(result).toEqual({
          success: false,
          error: 'Failed to store memory',
        });
      });
    });
  });

  describe('searchMemoryTool', () => {
    it('name', () => {
      expect(searchMemoryTool.name).toBe('search_memory');
    });

    it('description', () => {
      expect(searchMemoryTool.description).toBeDefined();
    });

    it('parameters', () => {
      const { parameters } = searchMemoryTool;
      expect(parameters).toBeDefined();

      expect(parameters).toHaveProperty('query');
    });

    describe('handler', () => {
      it('MemorySystem.searchMemoryを呼び出して結果を返す', async () => {
        const mockResults = [
          {
            content: 'High similarity memory',
            type: 'episode' as const,
            emotion: createMockEmotion({ valence: 0.8 }),
            createdAt: '2025-08-04T08:00:00.000Z',
            updatedAt: '2025-08-04T08:00:00.000Z',
            similarity: 0.95,
          },
          {
            content: 'Medium similarity memory',
            type: 'semantic' as const,
            emotion: createMockEmotion({ valence: 0.5 }),
            createdAt: '2025-08-03T08:00:00.000Z',
            updatedAt: '2025-08-03T08:00:00.000Z',
            similarity: 0.75,
          },
        ];
        mockedSearchMemory.mockResolvedValue(mockResults);

        const args = { query: 'test query' };
        const result = await searchMemoryTool.handler(args, mockToolContext);

        expect(mockedSearchMemory).toHaveBeenCalledWith(
          'test query',
          undefined
        );
        expect(result).toEqual({
          success: true,
          results: [
            {
              content: 'High similarity memory',
              type: 'episode',
              emotion: { valence: 0.8, arousal: 0.3, labels: ['neutral'] },
              createdAt: '2025-08-04T08:00:00.000Z',
            },
            {
              content: 'Medium similarity memory',
              type: 'semantic',
              emotion: { valence: 0.5, arousal: 0.3, labels: ['neutral'] },
              createdAt: '2025-08-03T08:00:00.000Z',
            },
          ],
        });
      });

      it('メモリが存在しない場合は空配列を返す', async () => {
        mockedSearchMemory.mockResolvedValue([]);

        const args = { query: 'test query' };
        const result = await searchMemoryTool.handler(args, mockToolContext);

        expect(result).toEqual({
          success: true,
          results: [],
        });
      });

      it('MemorySystemエラー時はエラーを返す', async () => {
        mockedSearchMemory.mockRejectedValue(new Error('Memory System Error'));

        const args = { query: 'test query' };
        const result = await searchMemoryTool.handler(args, mockToolContext);

        expect(result).toEqual({
          success: false,
          error: 'Failed to search memory',
        });
      });

      it('type指定時はMemorySystem.searchMemoryにtypeを渡す', async () => {
        mockedSearchMemory.mockResolvedValue([]);

        const args = { query: 'test query', type: 'episode' as const };
        await searchMemoryTool.handler(args, mockToolContext);

        expect(mockedSearchMemory).toHaveBeenCalledWith(
          'test query',
          'episode'
        );
      });

      it('type未指定時はMemorySystem.searchMemoryにundefinedを渡す', async () => {
        mockedSearchMemory.mockResolvedValue([]);

        const args = { query: 'test query' };
        await searchMemoryTool.handler(args, mockToolContext);

        expect(mockedSearchMemory).toHaveBeenCalledWith(
          'test query',
          undefined
        );
      });
    });
  });
});
