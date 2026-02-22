import type { Logger } from '../../utils/logger';

/**
 * 感情エンジン（スケルトン）
 * 将来的にテキストやイベントから感情状態を推定・更新する。
 */
export class EmotionEngine {
  private readonly _env: Env;
  private readonly _storage: DurableObjectStorage;
  private readonly _store: KVNamespace;
  private readonly _logger: Logger;
  private readonly _echoId: string;

  constructor(options: {
    env: Env;
    storage: DurableObjectStorage;
    store: KVNamespace;
    logger: Logger;
    echoId: string;
  }) {
    this._env = options.env;
    this._storage = options.storage;
    this._store = options.store;
    this._logger = options.logger;
    this._echoId = options.echoId;
  }

  /**
   * テキストから感情を推定（暫定: 未実装）。
   */
  // eslint-disable-next-line @typescript-eslint/require-await
  async analyze(_text: string): Promise<{ label: string; score: number }> {
    throw new Error('EmotionEngine.analyze is not implemented yet');
  }

  /**
   * 内部のムード状態を設定（暫定: 未実装）。
   */
  // eslint-disable-next-line @typescript-eslint/require-await
  async setMood(_mood: string): Promise<void> {
    throw new Error('EmotionEngine.setMood is not implemented yet');
  }
}
