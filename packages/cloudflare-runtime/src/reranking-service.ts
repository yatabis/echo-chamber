/**
 * 二段階検索の後段で使う reranker 抽象。
 *
 * provider 固有の API 形状は worker 側に閉じ込め、
 * runtime package には query / context / score だけを持ち込む。
 */
export interface RerankResult {
  id: number;
  score: number;
}

export interface RerankingService {
  rerank(
    query: string,
    contexts: string[],
    topK?: number
  ): Promise<RerankResult[]>;
  readonly modelIdentifier: string;
}
