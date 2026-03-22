import type {
  RerankResult,
  RerankingService,
} from '@echo-chamber/cloudflare-runtime/reranking-service';

const DEFAULT_MODEL = '@cf/baai/bge-reranker-base';

/**
 * Workers AI reranker を使って context 候補を再順位付けする。
 */
export class WorkersAIRerankingService implements RerankingService {
  private readonly ai: Ai;
  private readonly model: string;

  constructor(env: Env, model: string = DEFAULT_MODEL) {
    this.ai = env.AI;
    this.model = model;
  }

  get modelIdentifier(): string {
    return `workersai/${this.model}`;
  }

  async rerank(
    query: string,
    contexts: string[],
    topK?: number
  ): Promise<RerankResult[]> {
    const response = await this.ai.run(this.model as Parameters<Ai['run']>[0], {
      query,
      contexts: contexts.map((text) => ({ text })),
      top_k: topK,
    });
    const output = response as {
      response?: { id?: number; score?: number }[];
    };

    if (!Array.isArray(output.response)) {
      throw new Error('Failed to rerank contexts with Workers AI');
    }

    return output.response.map((item) => {
      if (typeof item.id !== 'number' || typeof item.score !== 'number') {
        throw new Error('Workers AI reranker returned an invalid result');
      }

      return {
        id: item.id,
        score: item.score,
      };
    });
  }
}
