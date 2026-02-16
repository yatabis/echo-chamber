import { formatDatetimeForAgent } from '../../utils/datetime';
import { cosineSimilarity } from '../../utils/vector';

import type { EmbeddingService } from '../../llm/openai/embedding';
import type { Logger } from '../../utils/logger';
import type { Emotion, MemoryType } from '../types';

const MAX_MEMORY_COUNT = 500;
const SEARCH_RESULT_LIMIT = 5;
const SIMILARITY_THRESHOLD = 0.001;

/**
 * メモリのスナップショット（embeddingを除いた情報）
 */
export interface MemorySnapshot {
  content: string;
  type: MemoryType;
  emotion: Emotion;
  createdAt: string;
}

/**
 * メモリ検索結果
 */
export interface MemorySearchResult extends MemorySnapshot {
  similarity: number;
}

/**
 * SQLiteに保存されるメモリ行の型
 */
interface MemoryRow extends Record<string, SqlStorageValue> {
  id: string;
  content: string;
  type: MemoryType;
  embedding: ArrayBuffer;
  emotion_valence: number;
  emotion_arousal: number;
  emotion_labels: string;
  created_at: string;
  updated_at: string;
}

/**
 * Float32ArrayをArrayBufferに変換（SQLite BLOB用）
 */
function float32ArrayToBuffer(arr: number[]): ArrayBuffer {
  const float32 = new Float32Array(arr);
  return float32.buffer;
}

/**
 * ArrayBufferをnumber[]に変換
 */
function bufferToNumberArray(buffer: ArrayBuffer): number[] {
  const float32 = new Float32Array(buffer);
  return Array.from(float32);
}

/**
 * 記憶システム
 * SQLiteベースのエピソード記憶の保存とセマンティック検索を提供する。
 */
export class MemorySystem {
  private readonly sql: SqlStorage;
  private readonly embeddingService: EmbeddingService;
  private readonly logger: Logger;
  private initialized = false;

  constructor(options: {
    sql: SqlStorage;
    embeddingService: EmbeddingService;
    logger: Logger;
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

    // マイグレーション: 既存テーブルにtypeカラムがない場合は追加
    this.migrateAddTypeColumn();

    this.initialized = true;
  }

  /**
   * typeカラムが存在しない既存テーブルにカラムを追加するマイグレーション
   */
  private migrateAddTypeColumn(): void {
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
  }

  /**
   * 記憶を保存する
   * 容量超過時は最古のメモリを自動削除する
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

    this.sql.exec(
      `INSERT INTO memories (id, content, type, embedding, emotion_valence, emotion_arousal, emotion_labels, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      content,
      type,
      embeddingBuffer,
      emotion.valence,
      emotion.arousal,
      JSON.stringify(emotion.labels),
      now,
      now
    );
  }

  /**
   * 最新のメモリを取得する
   * @returns 最新のメモリ、存在しない場合はnull
   */
  getLatestMemory(): MemorySnapshot | null {
    this.ensureSchema();

    // 0行の可能性があるのでtoArray()を使用（one()は0行で例外をスロー）
    const rows = this.sql
      .exec<MemoryRow>(
        'SELECT * FROM memories ORDER BY created_at DESC LIMIT 1'
      )
      .toArray();

    const row = rows[0];
    if (row === undefined) {
      return null;
    }

    return this.rowToSnapshot(row);
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
      `Search Memory with query:\n${query}\nResults:\n${filteredMemories
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
   */
  getAllMemories(): MemoryRow[] {
    this.ensureSchema();

    return this.sql.exec<MemoryRow>('SELECT * FROM memories').toArray();
  }

  /**
   * SQLite行をMemorySnapshotに変換
   */
  private rowToSnapshot(row: MemoryRow): MemorySnapshot {
    return {
      content: row.content,
      type: row.type,
      emotion: {
        valence: row.emotion_valence,
        arousal: row.emotion_arousal,
        labels: JSON.parse(row.emotion_labels) as string[],
      },
      createdAt: formatDatetimeForAgent(new Date(row.created_at)),
    };
  }
}
