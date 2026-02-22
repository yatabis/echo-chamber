/**
 * Echo インスタンス設定の型定義
 *
 * 複数の Echo インスタンスがそれぞれ固有の Discord Bot トークンやチャンネル設定を持つためのレジストリシステム
 */

/**
 * Echo インスタンス ID
 */
export type EchoInstanceId = 'rin' | 'marie';

/**
 * 文字列が有効な EchoInstanceId かどうかを判定する型ガード
 */
export function isValidInstanceId(id: string): id is EchoInstanceId {
  return id === 'rin' || id === 'marie';
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

/**
 * Echo インスタンスごとの設定
 */
export interface EchoInstanceConfig {
  /** インスタンス ID */
  id: EchoInstanceId;

  /** インスタンス名（表示用） */
  name: string;

  /** システムプロンプト */
  systemPrompt: string;

  /** Discord Bot トークン（インスタンス固有） */
  discordBotToken: string;

  /** チャットチャンネル ID */
  chatChannelId: string;

  /** Thinking ログを送信する Discord チャンネル ID */
  thinkingChannelId: string;

  /** Embedding プロバイダー設定。省略時は OpenAI を使用 */
  embeddingConfig?: EmbeddingConfig;
}
