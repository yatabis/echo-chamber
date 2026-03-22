import type { EmbeddingService } from '@echo-chamber/cloudflare-runtime/embedding-service';
import type { EmbeddingConfig } from '@echo-chamber/core/types/echo-config';

import { OpenAIEmbeddingService } from './providers/openai';
import { WorkersAIEmbeddingService } from './providers/workersai';

/**
 * EmbeddingConfig に基づいて適切な EmbeddingService を生成するファクトリー関数
 *
 * @param env - Cloudflare Workers 環境変数
 * @param config - Embedding プロバイダー設定（省略時は OpenAI を使用）
 * @returns EmbeddingService の実装
 */
export function createEmbeddingService(
  env: Env,
  config?: EmbeddingConfig
): EmbeddingService {
  if (!config || config.provider === 'openai') {
    return new OpenAIEmbeddingService(env);
  }

  return new WorkersAIEmbeddingService(env, config.model);
}
