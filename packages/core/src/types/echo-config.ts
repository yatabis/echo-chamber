/**
 * Echo インスタンス ID と embedding 設定の型定義
 */

export const ECHO_INSTANCE_IDS = ['rin', 'marie'] as const;

/**
 * Echo インスタンス ID
 */
export type EchoInstanceId = (typeof ECHO_INSTANCE_IDS)[number];

/**
 * 文字列が有効な EchoInstanceId かどうかを判定する型ガード
 */
export function isValidInstanceId(id: string): id is EchoInstanceId {
  return (ECHO_INSTANCE_IDS as readonly string[]).includes(id);
}

/**
 * Embedding プロバイダー設定
 */
export interface OpenAIEmbeddingConfig {
  provider: 'openai';
}

export interface WorkersAIEmbeddingConfig {
  provider: 'workersai';
  /** Workers AI モデル名。 */
  model?: string;
}

export type EmbeddingConfig = OpenAIEmbeddingConfig | WorkersAIEmbeddingConfig;
