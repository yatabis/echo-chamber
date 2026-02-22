import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Emotion } from '@echo-chamber/core';

import { mockToolContext } from '../../../../test/mocks/tool';

import { storeMemoryFunction, searchMemoryFunction } from './memory';

import type { MemorySearchResult } from '../../../echo/memory-system';

const mockedMemorySystem = vi.mocked(mockToolContext.memorySystem);

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
  describe('storeMemoryFunction', () => {
    it('name', () => {
      expect(storeMemoryFunction.name).toBe('store_memory');
    });

    it('description', () => {
      expect(storeMemoryFunction.description).toBeDefined();
    });

    it('parameters', () => {
      const { parameters } = storeMemoryFunction;
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

        const result = await storeMemoryFunction.handler(args, mockToolContext);

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockedMemorySystem.storeMemory).toHaveBeenCalledWith(
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

        const result = await storeMemoryFunction.handler(args, mockToolContext);

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockedMemorySystem.storeMemory).toHaveBeenCalledWith(
          'General knowledge about AI',
          createMockEmotion(),
          'semantic'
        );
        expect(result).toEqual({ success: true });
      });

      it('MemorySystemエラー時はエラーを返す', async () => {
        mockedMemorySystem.storeMemory.mockRejectedValue(
          new Error('Memory System Error')
        );

        const args = {
          content: 'Memory that will fail',
          type: 'episode' as const,
          emotion: createMockEmotion(),
        };

        const result = await storeMemoryFunction.handler(args, mockToolContext);

        expect(result).toEqual({
          success: false,
          error: 'Failed to store memory',
        });
      });
    });
  });

  describe('searchMemoryFunction', () => {
    it('name', () => {
      expect(searchMemoryFunction.name).toBe('search_memory');
    });

    it('description', () => {
      expect(searchMemoryFunction.description).toBeDefined();
    });

    it('parameters', () => {
      const { parameters } = searchMemoryFunction;
      expect(parameters).toBeDefined();

      expect(parameters).toHaveProperty('query');
    });

    describe('handler', () => {
      it('MemorySystem.searchMemoryを呼び出して結果を返す', async () => {
        const mockResults: MemorySearchResult[] = [
          {
            content: 'High similarity memory',
            type: 'episode',
            emotion: createMockEmotion({ valence: 0.8 }),
            createdAt: '2025-08-04T08:00:00.000Z',
            similarity: 0.95,
          },
          {
            content: 'Medium similarity memory',
            type: 'semantic',
            emotion: createMockEmotion({ valence: 0.5 }),
            createdAt: '2025-08-03T08:00:00.000Z',
            similarity: 0.75,
          },
        ];
        mockedMemorySystem.searchMemory.mockResolvedValue(mockResults);

        const args = { query: 'test query' };
        const result = await searchMemoryFunction.handler(
          args,
          mockToolContext
        );

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockedMemorySystem.searchMemory).toHaveBeenCalledWith(
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
        mockedMemorySystem.searchMemory.mockResolvedValue([]);

        const args = { query: 'test query' };
        const result = await searchMemoryFunction.handler(
          args,
          mockToolContext
        );

        expect(result).toEqual({
          success: true,
          results: [],
        });
      });

      it('MemorySystemエラー時はエラーを返す', async () => {
        mockedMemorySystem.searchMemory.mockRejectedValue(
          new Error('Memory System Error')
        );

        const args = { query: 'test query' };
        const result = await searchMemoryFunction.handler(
          args,
          mockToolContext
        );

        expect(result).toEqual({
          success: false,
          error: 'Failed to search memory',
        });
      });

      it('type指定時はMemorySystem.searchMemoryにtypeを渡す', async () => {
        mockedMemorySystem.searchMemory.mockResolvedValue([]);

        const args = { query: 'test query', type: 'episode' as const };
        await searchMemoryFunction.handler(args, mockToolContext);

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockedMemorySystem.searchMemory).toHaveBeenCalledWith(
          'test query',
          'episode'
        );
      });

      it('type未指定時はMemorySystem.searchMemoryにundefinedを渡す', async () => {
        mockedMemorySystem.searchMemory.mockResolvedValue([]);

        const args = { query: 'test query' };
        await searchMemoryFunction.handler(args, mockToolContext);

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockedMemorySystem.searchMemory).toHaveBeenCalledWith(
          'test query',
          undefined
        );
      });
    });
  });
});
