/**
 * Cloudflare runtime で永続化と検索に使う embedding 生成サービス。
 *
 * OpenAI / Workers AI など具体プロバイダーは worker 側で組み立て、
 * runtime package にはこの抽象だけを渡す。
 */
export interface EmbeddingService {
  embed(text: string): Promise<number[]>;
  readonly modelIdentifier: string;
}
