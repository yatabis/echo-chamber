import type { Emotion } from '../echo/types';

/**
 * 次回起動時の runtime context として再注入する永続スナップショット。
 * 最新セッションの要点だけを保持し、長期記憶とは分けて扱う。
 */
export interface ContextSnapshot {
  content: string;
  emotion: Emotion;
  createdAt: string;
  updatedAt: string;
}

/**
 * 一時的な再開コンテキストの永続化 port。
 * Durable Object storage などの実装詳細は隠蔽し、load/save/clear だけを公開する。
 */
export interface ContextPort {
  load(): Promise<ContextSnapshot | null>;
  save(snapshot: ContextSnapshot): Promise<void>;
  clear(): Promise<void>;
}
