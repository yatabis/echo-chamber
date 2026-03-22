import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createEmbeddingService } from './create-embedding-service';
import { OpenAIEmbeddingService } from './providers/openai';
import { WorkersAIEmbeddingService } from './providers/workersai';

vi.mock('./providers/openai', () => ({
  OpenAIEmbeddingService: vi.fn(),
}));

vi.mock('./providers/workersai', () => ({
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
