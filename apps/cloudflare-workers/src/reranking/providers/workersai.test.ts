import { beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkersAIRerankingService } from './workersai';

const mockAiRun = vi.fn();

const mockEnv = {
  AI: { run: mockAiRun },
} as unknown as Env;

describe('WorkersAIRerankingService', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('デフォルトモデルで rerank を実行する', async () => {
    mockAiRun.mockResolvedValue({
      response: [
        { id: 1, score: 0.92 },
        { id: 0, score: 0.75 },
      ],
    });

    const service = new WorkersAIRerankingService(mockEnv);
    const result = await service.rerank(
      'best pet',
      ['a cyberpunk cat', 'a cyberpunk dog'],
      2
    );

    expect(mockAiRun).toHaveBeenCalledWith('@cf/baai/bge-reranker-base', {
      query: 'best pet',
      contexts: [{ text: 'a cyberpunk cat' }, { text: 'a cyberpunk dog' }],
      top_k: 2,
    });
    expect(result).toEqual([
      { id: 1, score: 0.92 },
      { id: 0, score: 0.75 },
    ]);
  });

  it('カスタムモデルを使用できる', async () => {
    mockAiRun.mockResolvedValue({ response: [{ id: 0, score: 0.81 }] });

    const service = new WorkersAIRerankingService(
      mockEnv,
      '@cf/baai/bge-reranker-base'
    );
    await service.rerank('query', ['context']);

    expect(mockAiRun).toHaveBeenCalledWith('@cf/baai/bge-reranker-base', {
      query: 'query',
      contexts: [{ text: 'context' }],
      top_k: undefined,
    });
  });

  it('response が配列でない場合はエラーを投げる', async () => {
    mockAiRun.mockResolvedValue({});

    const service = new WorkersAIRerankingService(mockEnv);

    await expect(service.rerank('query', ['context'])).rejects.toThrow(
      'Failed to rerank contexts with Workers AI'
    );
  });

  it('response 要素が不正ならエラーを投げる', async () => {
    mockAiRun.mockResolvedValue({ response: [{ id: '0', score: 0.5 }] });

    const service = new WorkersAIRerankingService(mockEnv);

    await expect(service.rerank('query', ['context'])).rejects.toThrow(
      'Workers AI reranker returned an invalid result'
    );
  });

  it('modelIdentifier を返す', () => {
    const service = new WorkersAIRerankingService(mockEnv);

    expect(service.modelIdentifier).toBe(
      'workersai/@cf/baai/bge-reranker-base'
    );
  });
});
