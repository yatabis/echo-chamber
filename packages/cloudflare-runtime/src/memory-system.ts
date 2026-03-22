import type { Emotion, MemoryType } from '@echo-chamber/core/echo/types';
import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';
import { getErrorMessage } from '@echo-chamber/core/utils/error';
import { cosineSimilarity } from '@echo-chamber/core/utils/vector';

import { bufferToNumberArray, float32ArrayToBuffer } from './memory-codec';

import type { EmbeddingService } from './embedding-service';

const MAX_MEMORY_COUNT = 500;
const SEARCH_RESULT_LIMIT = 5;
const SIMILARITY_THRESHOLD = 0.001;

/**
 * メモリのスナップショット（embeddingを除いた情報）
 *
 * 永続化層の生データを agent で扱いやすい形へ正規化したもの。
 */
export interface MemorySnapshot {
  content: string;
  type: MemoryType;
  emotion: Emotion;
  createdAt: string;
  updatedAt: string;
}

/**
 * メモリ検索結果
 *
 * snapshot に類似度を付与した検索専用の戻り値。
 */
export interface MemorySearchResult extends MemorySnapshot {
  similarity: number;
}

/**
 * SQLiteに保存されるメモリ行の型
 *
 * Durable Object SQLite から取得する raw row 形状。
 */
export interface StoredMemoryRow extends Record<string, SqlStorageValue> {
  id: string;
  content: string;
  type: MemoryType;
  embedding: ArrayBuffer;
  embedding_model: string;
  emotion_valence: number;
  emotion_arousal: number;
  emotion_labels: string;
  created_at: string;
  updated_at: string;
}

/**
 * 記憶システム
 * SQLiteベースのエピソード記憶の保存とセマンティック検索を提供する。
 */
export class MemorySystem {
  private readonly sql: SqlStorage;
  private readonly embeddingService: EmbeddingService;
  private readonly logger: Pick<LoggerPort, 'debug' | 'info' | 'error'>;
  private initialized = false;

  /**
   * SQLite と embedding service を使う memory runtime を構築する。
   *
   * @param options SQLite storage、embedding service、logger
   */
  constructor(options: {
    sql: SqlStorage;
    embeddingService: EmbeddingService;
    logger: Pick<LoggerPort, 'debug' | 'info' | 'error'>;
  }) {
    this.sql = options.sql;
    this.embeddingService = options.embeddingService;
    this.logger = options.logger;
  }

  /**
   * SQLiteスキーマを初期化する
   */
  private ensureSchema(): void {
    if (this.initialized) return;

    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        type TEXT NOT NULL,
        embedding BLOB NOT NULL,
        embedding_model TEXT NOT NULL,
        emotion_valence REAL NOT NULL,
        emotion_arousal REAL NOT NULL,
        emotion_labels TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)
    `);

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at)
    `);

    // マイグレーション: 既存テーブルにtypeカラム, embedding_modelカラムがない場合は追加
    this.migrateColumn();

    this.initialized = true;
  }

  /**
   * typeカラム, embedding_modelカラムが存在しない既存テーブルにカラムを追加するマイグレーション
   */
  private migrateColumn(): void {
    // PRAGMA table_infoでカラム存在確認
    const columns = this.sql
      .exec<{ name: string }>('PRAGMA table_info(memories)')
      .toArray();

    const hasTypeColumn = columns.some((col) => col.name === 'type');

    if (!hasTypeColumn) {
      // typeカラムを追加（既存データは'episode'をデフォルト値とする）
      this.sql.exec(
        "ALTER TABLE memories ADD COLUMN type TEXT NOT NULL DEFAULT 'episode'"
      );
    }

    const hasEmbeddingModelColumn = columns.some(
      (col) => col.name === 'embedding_model'
    );

    if (!hasEmbeddingModelColumn) {
      this.sql.exec(
        "ALTER TABLE memories ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'openai/text-embedding-3-small'"
      );
    }
  }

  /**
   * 記憶を保存する
   * 容量超過時は最古のメモリを自動削除する
   *
   * @param content 保存する本文
   * @param emotion 感情メタデータ
   * @param type 記憶タイプ
   * @returns 保存完了
   */
  async storeMemory(
    content: string,
    emotion: Emotion,
    type: MemoryType
  ): Promise<void> {
    this.ensureSchema();

    const embedding = await this.embeddingService.embed(content);

    // 容量超過時は最古のメモリを削除
    const memoryCount = this.getMemoryCount();
    if (memoryCount >= MAX_MEMORY_COUNT) {
      // 500件以上存在する場合は必ず1行返るのでone()で取得可能
      const oldest = this.sql
        .exec<{
          id: string;
          content: string;
        }>('SELECT id, content FROM memories ORDER BY updated_at ASC LIMIT 1')
        .one();

      this.sql.exec('DELETE FROM memories WHERE id = ?', oldest.id);
      await this.logger.info(
        `Memory capacity reached. Removed oldest memory: ${oldest.content}`
      );
    }

    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const embeddingBuffer = float32ArrayToBuffer(embedding);
    const embeddingModel = this.embeddingService.modelIdentifier;

    this.sql.exec(
      `INSERT INTO memories (id, content, type, embedding, embedding_model, emotion_valence, emotion_arousal, emotion_labels, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      content,
      type,
      embeddingBuffer,
      embeddingModel,
      emotion.valence,
      emotion.arousal,
      JSON.stringify(emotion.labels),
      now,
      now
    );
  }

  /**
   * セマンティック検索でメモリを取得する
   * @param query 検索クエリ
   * @param type 検索対象のメモリタイプ（省略時は全タイプ）
   * @returns 類似度順にソートされた検索結果（最大5件）
   */
  async searchMemory(
    query: string,
    type?: MemoryType
  ): Promise<MemorySearchResult[]> {
    this.ensureSchema();

    let rows = this.getAllMemories();

    // タイプが指定された場合はフィルタ
    if (type !== undefined) {
      rows = rows.filter((row) => row.type === type);
    }

    if (rows.length === 0) {
      return [];
    }

    const queryEmbedding = await this.embeddingService.embed(query);

    // 類似度計算
    const memoriesWithSimilarity = rows.map((row) => ({
      row,
      similarity: cosineSimilarity(
        queryEmbedding,
        bufferToNumberArray(row.embedding)
      ),
    }));

    // 閾値でフィルタ、類似度降順でソート、上位N件を取得
    const filteredMemories = memoriesWithSimilarity
      .filter((m) => m.similarity >= SIMILARITY_THRESHOLD)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, SEARCH_RESULT_LIMIT);

    await this.logger.info(
      `Search Memory with query:\n[${type}] ${query}\nResults:\n${filteredMemories
        .map(
          ({ row, similarity }) =>
            `[${row.type}] ${row.content} (${similarity.toFixed(4)})`
        )
        .join('\n')}`
    );

    return filteredMemories.map(({ row, similarity }) => ({
      ...this.rowToSnapshot(row),
      similarity,
    }));
  }

  /**
   * メモリの件数を取得する
   *
   * @returns 現在保存されている memory 件数
   */
  getMemoryCount(): number {
    this.ensureSchema();

    // COUNT(*)は常に1行を返すのでone()で取得可能
    const { count } = this.sql
      .exec<{ count: number }>('SELECT COUNT(*) as count FROM memories')
      .one();

    return count;
  }

  /**
   * 全メモリを取得する
   *
   * @returns SQLite に保存された全 memory row
   */
  getAllMemories(): StoredMemoryRow[] {
    this.ensureSchema();

    return this.sql.exec<StoredMemoryRow>('SELECT * FROM memories').toArray();
  }

  /**
   * 現在の embedding モデルと異なるモデルで生成された memory を再 embedding する
   *
   * @returns 再 embedding 完了
   */
  async reEmbedStaleMemories(): Promise<void> {
    this.ensureSchema();

    const currentModel = this.embeddingService.modelIdentifier;

    const staleRows = this.sql
      .exec<{
        id: string;
        content: string;
      }>(
        'SELECT id, content FROM memories WHERE embedding_model != ?',
        currentModel
      )
      .toArray();

    if (staleRows.length === 0) {
      await this.logger.debug(
        `No stale memories to re-embed (model: ${currentModel})`
      );
      return;
    }

    await this.logger.info(
      `Re-embedding ${staleRows.length} memories with ${currentModel}...`
    );

    for (const row of staleRows) {
      try {
        // eslint-disable-next-line no-await-in-loop
        const embedding = await this.embeddingService.embed(row.content);
        const buffer = float32ArrayToBuffer(embedding);
        this.sql.exec(
          'UPDATE memories SET embedding = ?, embedding_model = ? WHERE id = ?',
          buffer,
          currentModel,
          row.id
        );
      } catch (error) {
        // eslint-disable-next-line no-await-in-loop
        await this.logger.error(
          `Failed to re-embed memory ${row.id}: ${getErrorMessage(error)}`
        );
      }
    }

    await this.logger.info(`Re-embedding complete.`);
  }

  /**
   * SQLite行をMemorySnapshotに変換
   */
  private rowToSnapshot(row: StoredMemoryRow): MemorySnapshot {
    return {
      content: row.content,
      type: row.type,
      emotion: {
        valence: row.emotion_valence,
        arousal: row.emotion_arousal,
        labels: JSON.parse(row.emotion_labels) as string[],
      },
      createdAt: formatDatetimeForAgent(new Date(row.created_at)),
      updatedAt: formatDatetimeForAgent(new Date(row.updated_at)),
    };
  }
}
