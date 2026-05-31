import type { EmbeddingService } from '@echo-chamber/cloudflare-runtime/embedding-service';
import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';

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
  private readonly events: EchoEventPort | undefined;

  /**
   * @param env Cloudflare Workers environment
   * @param model Workers AI embedding model
   * @param events embedding 生成イベントの送信先
   */
  constructor(env: Env, model: string = DEFAULT_MODEL, events?: EchoEventPort) {
    this.ai = env.AI;
    this.model = model;
    this.events = events;
  }

  get modelIdentifier(): string {
    return `workersai/${this.model}`;
  }

  async embed(text: string): Promise<number[]> {
    const startedAt = Date.now();
    const response = await this.ai.run(this.model as Parameters<Ai['run']>[0], {
      text,
    });

    const output = response as { data: number[][] };
    const embedding = output.data[0];
    if (!embedding) {
      throw new Error('Failed to generate embedding from Workers AI');
    }
    await emitEchoEvent(this.events, {
      type: 'memory.embedding.generated',
      severity: 'debug',
      summary: 'embedding generated',
      payload: {
        provider: 'workersai',
        model: this.modelIdentifier,
        inputLength: text.length,
        dimensions: embedding.length,
        durationMs: Date.now() - startedAt,
      },
    });
    return embedding;
  }
}
