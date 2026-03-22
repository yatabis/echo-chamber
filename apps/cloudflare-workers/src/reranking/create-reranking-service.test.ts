import { describe, expect, it, vi } from 'vitest';

import { createRerankingService } from './create-reranking-service';
import { WorkersAIRerankingService } from './providers/workersai';

vi.mock('./providers/workersai', () => ({
  WorkersAIRerankingService: vi.fn(),
}));

describe('createRerankingService', () => {
  const mockEnv = {} as Env;

  it('WorkersAIRerankingService を返す', () => {
    createRerankingService(mockEnv);

    expect(WorkersAIRerankingService).toHaveBeenCalledWith(mockEnv);
  });
});
