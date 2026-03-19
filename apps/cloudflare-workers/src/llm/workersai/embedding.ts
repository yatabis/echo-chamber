import { createLogger } from '../../utils/logger';

import type { EmbeddingService } from '../../runtime/embedding-service';
import type { Logger } from '../../utils/logger';

const DEFAULT_MODEL = '@cf/pfnet/plamo-embedding-1b';

/**
 * Workers AI Embedding APIを使用したEmbedding生成サービス
 *
 * モデルごとの次元数:
 * - @cf/pfnet/plamo-embedding-1b → 2048次元 (デフォルト、日本語特化)
 * - @cf/baai/bge-large-en-v1.5  → 1024次元
 * - @cf/baai/bge-base-en-v1.5   → 768次元
 * - @cf/baai/bge-small-en-v1.5  → 384次元
 *
 * 注意: 同一 Durable Object インスタンス内の全 embedding は同じ次元数が必要。
 * プロバイダーを変更した場合は既存の memory データを再生成すること。
 */
export class WorkersAIEmbeddingService implements EmbeddingService {
  private readonly ai: Ai;
  private readonly model: string;
  private readonly logger: Logger;

  constructor(env: Env, model: string = DEFAULT_MODEL) {
    this.ai = env.AI;
    this.model = model;
    this.logger = createLogger(env);
  }

  get modelIdentifier(): string {
    return `workersai/${this.model}`;
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.ai.run(this.model as Parameters<Ai['run']>[0], {
      text,
    });

    await this.logger.info(
      `Workers AI Embedding generated (model: ${this.model})`
    );

    const output = response as { data: number[][] };
    const embedding = output.data[0];
    if (!embedding) {
      throw new Error('Failed to generate embedding from Workers AI');
    }
    return embedding;
  }
}
