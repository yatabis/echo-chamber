import OpenAI from 'openai';

import type { EmbeddingService } from '@echo-chamber/cloudflare-runtime/embedding-service';
import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';

const EMBEDDING_DIMENSIONS = 1536;

/**
 * OpenAI Embedding APIを使用したEmbedding生成サービス
 */
export class OpenAIEmbeddingService implements EmbeddingService {
  private readonly client: OpenAI;
  private readonly events: EchoEventPort | undefined;
  readonly modelIdentifier = 'openai/text-embedding-3-small';

  /**
   * @param env Cloudflare Workers environment
   * @param events embedding 生成イベントの送信先
   */
  constructor(env: Env, events?: EchoEventPort) {
    this.client = new OpenAI({
      apiKey: env.OPENAI_API_KEY,
      maxRetries: 0,
    });
    this.events = events;
  }

  /**
   * テキストの埋め込みベクトルを生成
   * @param text - 埋め込むテキスト
   * @returns 1536次元の埋め込みベクトル
   */
  async embed(text: string): Promise<number[]> {
    const startedAt = Date.now();
    const response = await this.client.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      dimensions: EMBEDDING_DIMENSIONS,
    });

    const embedding = response.data[0]?.embedding;
    if (!embedding) {
      throw new Error('Failed to generate embedding');
    }
    await emitEchoEvent(this.events, {
      type: 'memory.embedding.generated',
      severity: 'debug',
      summary: 'embedding generated',
      payload: {
        provider: 'openai',
        model: this.modelIdentifier,
        inputLength: text.length,
        dimensions: embedding.length,
        totalTokens: response.usage.total_tokens,
        durationMs: Date.now() - startedAt,
      },
    });
    return embedding;
  }
}
