import { beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkersAIEmbeddingService } from './embedding';

const mockAiRun = vi.fn();

vi.mock('cloudflare:test', () => ({
  env: {
    AI: { run: mockAiRun },
    ENVIRONMENT: 'test',
    DISCORD_BOT_TOKEN: 'test-token',
    DISCORD_BOT_TOKEN_RIN: 'test-token-rin',
    DISCORD_BOT_TOKEN_MARIE: 'test-token-marie',
    OPENAI_API_KEY: 'test-key',
    LOG_CHANNEL_ID: 'log-channel',
  },
}));

const mockEnv = {
  AI: { run: mockAiRun },
  ENVIRONMENT: 'test',
  DISCORD_BOT_TOKEN: 'test-token',
  DISCORD_BOT_TOKEN_RIN: 'test-token-rin',
  DISCORD_BOT_TOKEN_MARIE: 'test-token-marie',
  OPENAI_API_KEY: 'test-key',
  LOG_CHANNEL_ID: 'log-channel',
} as unknown as Env;

describe('WorkersAIEmbeddingService', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('デフォルトモデル (@cf/pfnet/plamo-embedding-1b) で embedding を生成する', async () => {
    const mockEmbedding = new Array<number>(2048).fill(0.1);
    mockAiRun.mockResolvedValue({ data: [mockEmbedding] });

    const service = new WorkersAIEmbeddingService(mockEnv);
    const result = await service.embed('test text');

    expect(mockAiRun).toHaveBeenCalledWith('@cf/pfnet/plamo-embedding-1b', {
      text: 'test text',
    });
    expect(result).toEqual(mockEmbedding);
    expect(result).toHaveLength(2048);
  });

  it('カスタムモデルを使用できる', async () => {
    const mockEmbedding = new Array<number>(384).fill(0.2);
    mockAiRun.mockResolvedValue({ data: [mockEmbedding] });

    const service = new WorkersAIEmbeddingService(
      mockEnv,
      '@cf/baai/bge-small-en-v1.5'
    );
    const result = await service.embed('test text');

    expect(mockAiRun).toHaveBeenCalledWith('@cf/baai/bge-small-en-v1.5', {
      text: 'test text',
    });
    expect(result).toHaveLength(384);
  });

  it('AI レスポンスが空の場合はエラーをスローする', async () => {
    mockAiRun.mockResolvedValue({ data: [] });

    const service = new WorkersAIEmbeddingService(mockEnv);

    await expect(service.embed('test text')).rejects.toThrow(
      'Failed to generate embedding from Workers AI'
    );
  });

  it('EmbeddingService インターフェースを満たす', () => {
    const service = new WorkersAIEmbeddingService(mockEnv);
    expect(typeof service.embed).toBe('function');
  });

  it('modelIdentifier はデフォルトモデルの識別子を返す', () => {
    const service = new WorkersAIEmbeddingService(mockEnv);
    expect(service.modelIdentifier).toBe(
      'workersai/@cf/pfnet/plamo-embedding-1b'
    );
  });

  it('modelIdentifier はカスタムモデルの識別子を返す', () => {
    const service = new WorkersAIEmbeddingService(
      mockEnv,
      '@cf/baai/bge-large-en-v1.5'
    );
    expect(service.modelIdentifier).toBe(
      'workersai/@cf/baai/bge-large-en-v1.5'
    );
  });
});
