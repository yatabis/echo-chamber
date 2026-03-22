import type { RerankingService } from '@echo-chamber/cloudflare-runtime/reranking-service';

import { WorkersAIRerankingService } from './providers/workersai';

/**
 * memory search の後段で使う reranking service を生成する。
 */
export function createRerankingService(env: Env): RerankingService {
  return new WorkersAIRerankingService(env);
}
