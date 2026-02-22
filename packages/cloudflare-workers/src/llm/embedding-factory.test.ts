import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createEmbeddingService } from './embedding-factory';
import { OpenAIEmbeddingService } from './openai/embedding';
import { WorkersAIEmbeddingService } from './workersai/embedding';

vi.mock('./openai/embedding', () => ({
  OpenAIEmbeddingService: vi.fn(),
}));

vi.mock('./workersai/embedding', () => ({
  WorkersAIEmbeddingService: vi.fn(),
}));

const mockEnv = {} as Env;

describe('createEmbeddingService', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('config 省略時は OpenAIEmbeddingService を返す', () => {
    createEmbeddingService(mockEnv);
    expect(OpenAIEmbeddingService).toHaveBeenCalledWith(mockEnv);
    expect(WorkersAIEmbeddingService).not.toHaveBeenCalled();
  });

  it("provider: 'openai' 指定時は OpenAIEmbeddingService を返す", () => {
    createEmbeddingService(mockEnv, { provider: 'openai' });
    expect(OpenAIEmbeddingService).toHaveBeenCalledWith(mockEnv);
    expect(WorkersAIEmbeddingService).not.toHaveBeenCalled();
  });

  it("provider: 'workersai' 指定時は WorkersAIEmbeddingService を返す", () => {
    createEmbeddingService(mockEnv, { provider: 'workersai' });
    expect(WorkersAIEmbeddingService).toHaveBeenCalledWith(mockEnv, undefined);
    expect(OpenAIEmbeddingService).not.toHaveBeenCalled();
  });

  it("provider: 'workersai' + model 指定時はモデルが渡される", () => {
    createEmbeddingService(mockEnv, {
      provider: 'workersai',
      model: '@cf/baai/bge-large-en-v1.5',
    });
    expect(WorkersAIEmbeddingService).toHaveBeenCalledWith(
      mockEnv,
      '@cf/baai/bge-large-en-v1.5'
    );
  });
});
