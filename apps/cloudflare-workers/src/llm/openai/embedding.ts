import OpenAI from 'openai';

import { createLogger } from '../../utils/logger';

import type { EmbeddingService } from '../../runtime/embedding-service';
import type { Logger } from '../../utils/logger';

const EMBEDDING_DIMENSIONS = 1536;

/**
 * OpenAI Embedding APIを使用したEmbedding生成サービス
 */
export class OpenAIEmbeddingService implements EmbeddingService {
  private readonly client: OpenAI;
  private readonly logger: Logger;
  readonly modelIdentifier = 'openai/text-embedding-3-small';

  constructor(env: Env) {
    this.client = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    this.logger = createLogger(env);
  }

  /**
   * テキストの埋め込みベクトルを生成
   * @param text - 埋め込むテキスト
   * @returns 1536次元の埋め込みベクトル
   */
  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      dimensions: EMBEDDING_DIMENSIONS,
    });

    await this.logger.info(
      `Embedding usage: ${response.usage.total_tokens} tokens`
    );

    const embedding = response.data[0]?.embedding;
    if (!embedding) {
      throw new Error('Failed to generate embedding');
    }
    return embedding;
  }
}
