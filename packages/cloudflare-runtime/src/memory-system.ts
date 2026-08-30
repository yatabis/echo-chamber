import type { Emotion, MemoryType } from '@echo-chamber/core/echo/types';
import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';
import { getErrorMessage } from '@echo-chamber/core/utils/error';
import { cosineSimilarity } from '@echo-chamber/core/utils/vector';

import { bufferToNumberArray, float32ArrayToBuffer } from './memory-codec';

import type { EmbeddingService } from './embedding-service';
import type { RerankingService } from './reranking-service';

const MAX_MEMORY_COUNT = 500;
const MEMORY_SEARCH_SOURCE_LIMIT = MAX_MEMORY_COUNT;
const DASHBOARD_MEMORY_ROW_LIMIT = MAX_MEMORY_COUNT;
const VECTOR_CANDIDATE_LIMIT = 20;
const SEARCH_RESULT_LIMIT = 5;
const SIMILARITY_THRESHOLD = 0.001;

/** application全体のexternal request枯渇はlocal fallbackで隠さない。 */
function isExternalRequestBudgetError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    error.code === 'external_request_budget_exceeded'
  );
}

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
 * Dashboard 表示に必要な embedding 以外の memory row。
 */
export interface StoredMemoryDashboardRow
  extends Record<string, SqlStorageValue> {
  id: string;
  content: string;
  type: MemoryType;
  embedding_model: string;
  emotion_valence: number;
  emotion_arousal: number;
  emotion_labels: string;
  created_at: string;
  updated_at: string;
}

interface StoredMemoryDashboardSummaryRow
  extends Record<string, SqlStorageValue> {
  memory_count: number;
  latest_updated_at: string | null;
}

interface ReEmbedStaleMemoriesInput {
  limit?: number;
}

export interface MemoryDashboardSummary {
  count: number;
  latestUpdatedAt: string | null;
}

/** 外部 embedding 済みで、SQLite commit だけを残した memory write。 */
export type PreparedMemoryWrite =
  | {
      status: 'existing';
      id: string;
    }
  | {
      status: 'prepared';
      id: string;
      content: string;
      emotion: Emotion;
      type: MemoryType;
      embedding: ArrayBuffer;
      embeddingModel: string;
      createdAt: string;
    };

/** 同期 memory commit が確定した行と capacity eviction の記録。 */
export interface MemoryCommitReceipt {
  storedIds: string[];
  existingIds: string[];
  evicted: { id: string; content: string }[];
}

/**
 * 記憶システム
 * SQLiteベースのエピソード記憶の保存とセマンティック検索を提供する。
 */
export class MemorySystem {
  private readonly sql: SqlStorage;
  private readonly embeddingService: EmbeddingService;
  private readonly rerankingService: RerankingService;
  private readonly events: EchoEventPort | undefined;
  private searchableMemoryRows: StoredMemoryRow[] | null = null;
  private initialized = false;

  /**
   * SQLite と embedding / reranking service を使う memory runtime を構築する。
   *
   * @param options SQLite storage、embedding service、reranking service、event port
   */
  constructor(options: {
    sql: SqlStorage;
    embeddingService: EmbeddingService;
    rerankingService: RerankingService;
    events?: EchoEventPort;
  }) {
    this.sql = options.sql;
    this.embeddingService = options.embeddingService;
    this.rerankingService = options.rerankingService;
    this.events = options.events;
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

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_memories_embedding_model ON memories(embedding_model)
    `);

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
    await this.storeMemoryIdempotently(
      crypto.randomUUID(),
      content,
      emotion,
      type
    );
  }

  /**
   * Commit key 由来の deterministic id で memory write を一度だけ適用する。
   *
   * 同じ id の row がすでに確定している再実行では、embedding・保存・evictionを
   * 繰り返さない。
   *
   * @param id Cognitive Module commit 層が決定する一意 id
   * @param content 保存する本文
   * @param emotion 感情メタデータ
   * @param type 記憶タイプ
   * @returns 新規保存か既存 commit の再確認か
   */
  async storeMemoryIdempotently(
    id: string,
    content: string,
    emotion: Emotion,
    type: MemoryType
  ): Promise<'stored' | 'existing'> {
    const prepared = await this.prepareMemoryWrite(id, content, emotion, type);
    const receipt = this.commitPreparedMemoryWrites([prepared]);
    await this.emitMemoryCommitEvents(receipt);
    return receipt.storedIds.includes(id) ? 'stored' : 'existing';
  }

  /**
   * 外部 embedding を完了し、永続化前の値として返す。
   *
   * この段階では row も eviction も変更しないため、呼び出し側は返り値を
   * Durable Object の state transaction 内で同期 commit できる。
   *
   * @param id Cognitive boundary から導出した deterministic id
   * @param content 保存する本文
   * @param emotion 感情メタデータ
   * @param type 記憶タイプ
   * @returns 既存 row token または embedding 済み write
   */
  async prepareMemoryWrite(
    id: string,
    content: string,
    emotion: Emotion,
    type: MemoryType
  ): Promise<PreparedMemoryWrite> {
    this.ensureSchema();

    const existing = this.sql
      .exec<{ id: string }>('SELECT id FROM memories WHERE id = ? LIMIT 1', id)
      .toArray();
    if (existing.length > 0) {
      return { status: 'existing', id };
    }

    const embedding = await this.embeddingService.embed(content);
    return {
      status: 'prepared',
      id,
      content,
      emotion: {
        valence: emotion.valence,
        arousal: emotion.arousal,
        labels: [...emotion.labels],
      },
      type,
      embedding: float32ArrayToBuffer(embedding),
      embeddingModel: this.embeddingService.modelIdentifier,
      createdAt: new Date().toISOString(),
    };
  }

  /**
   * Prepared write と capacity eviction を同期 SQL だけで確定する。
   *
   * Echo は SQLite-backed Durable Object のため、このメソッドを
   * `storage.transaction()` 内で呼ぶと hidden KV state と同じ transaction に入る。
   * 外部 I/O は `prepareMemoryWrite()` 側で完了済みなので transaction を跨がない。
   *
   * @param writes embedding 済みの bounded write 群
   * @returns commit 後に配送する event を含む receipt
   */
  commitPreparedMemoryWrites(
    writes: readonly PreparedMemoryWrite[]
  ): MemoryCommitReceipt {
    this.ensureSchema();
    const storedIds: string[] = [];
    const existingIds: string[] = [];
    const evicted: { id: string; content: string }[] = [];
    const pending: Extract<PreparedMemoryWrite, { status: 'prepared' }>[] = [];
    const seenIds = new Set<string>();

    for (const write of writes) {
      if (seenIds.has(write.id)) {
        throw new Error(`Duplicate prepared memory id: ${write.id}`);
      }
      seenIds.add(write.id);

      const existing = this.sql
        .exec<{
          id: string;
        }>('SELECT id FROM memories WHERE id = ? LIMIT 1', write.id)
        .toArray();
      if (existing.length > 0) {
        existingIds.push(write.id);
        continue;
      }
      if (write.status === 'existing') {
        throw new Error(`Prepared existing memory disappeared: ${write.id}`);
      }
      pending.push(write);
    }

    let rowsToEvict = Math.max(
      0,
      this.getMemoryCount() + pending.length - MAX_MEMORY_COUNT
    );
    while (rowsToEvict > 0) {
      const [oldest] = this.sql
        .exec<{
          id: string;
          content: string;
        }>('SELECT id, content FROM memories ORDER BY updated_at ASC LIMIT 1')
        .toArray();
      if (oldest === undefined) {
        throw new Error('Memory capacity count did not match stored rows');
      }
      this.sql.exec('DELETE FROM memories WHERE id = ?', oldest.id);
      evicted.push({ id: oldest.id, content: oldest.content });
      rowsToEvict -= 1;
    }

    for (const write of pending) {
      this.sql.exec(
        `INSERT INTO memories (id, content, type, embedding, embedding_model, emotion_valence, emotion_arousal, emotion_labels, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        write.id,
        write.content,
        write.type,
        write.embedding,
        write.embeddingModel,
        write.emotion.valence,
        write.emotion.arousal,
        JSON.stringify(write.emotion.labels),
        write.createdAt,
        write.createdAt
      );
      storedIds.push(write.id);
    }

    if (storedIds.length > 0 || evicted.length > 0) {
      this.searchableMemoryRows = null;
    }
    return { storedIds, existingIds, evicted };
  }

  /** Transaction 確定後に capacity eviction event を配送する。 */
  async emitMemoryCommitEvents(receipt: MemoryCommitReceipt): Promise<void> {
    await Promise.all(
      receipt.evicted.map(async (memory) =>
        emitEchoEvent(this.events, {
          type: 'memory.evicted',
          severity: 'warn',
          summary: 'memory capacity reached; removed oldest memory',
          payload: {
            id: memory.id,
            content: memory.content,
            maxMemoryCount: MAX_MEMORY_COUNT,
          },
        })
      )
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
    const startedAt = Date.now();
    await emitEchoEvent(this.events, {
      type: 'memory.search.started',
      severity: 'debug',
      summary: `memory search started: ${query}`,
      payload: {
        query,
        type: type ?? 'all',
      },
    });

    let rows = this.getSearchableMemories();

    // タイプが指定された場合はフィルタ
    if (type !== undefined) {
      rows = rows.filter((row) => row.type === type);
    }

    if (rows.length === 0) {
      await this.emitMemorySearchCompleted({
        query,
        type,
        durationMs: Date.now() - startedAt,
        sourceCount: 0,
        vectorCandidates: [],
        finalResults: [],
      });
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

    // 閾値でフィルタ、類似度降順でソートした上位候補だけを rerank へ渡す
    const vectorCandidates = memoriesWithSimilarity
      .filter((m) => m.similarity >= SIMILARITY_THRESHOLD)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, VECTOR_CANDIDATE_LIMIT);

    if (vectorCandidates.length === 0) {
      await this.emitMemorySearchCompleted({
        query,
        type,
        durationMs: Date.now() - startedAt,
        sourceCount: rows.length,
        vectorCandidates: [],
        finalResults: [],
      });
      return [];
    }

    const rerankedMemories = await this.rerankCandidates(
      query,
      vectorCandidates
    );

    const results = rerankedMemories.map(({ row, rerankScore }) => ({
      ...this.rowToSnapshot(row),
      similarity: rerankScore,
    }));
    await this.emitMemorySearchCompleted({
      query,
      type,
      durationMs: Date.now() - startedAt,
      sourceCount: rows.length,
      vectorCandidates: vectorCandidates.map(({ row, similarity }) => ({
        id: row.id,
        content: row.content,
        type: row.type,
        vectorScore: similarity,
      })),
      finalResults: rerankedMemories.map(
        ({ row, similarity, rerankScore }) => ({
          id: row.id,
          content: row.content,
          type: row.type,
          vectorScore: similarity,
          rerankScore,
        })
      ),
    });

    return results;
  }

  /**
   * memory search の分析用イベントを送る。
   */
  private async emitMemorySearchCompleted(input: {
    query: string;
    type?: MemoryType;
    durationMs: number;
    sourceCount: number;
    vectorCandidates: Record<string, unknown>[];
    finalResults: Record<string, unknown>[];
  }): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'memory.search.completed',
      severity: 'info',
      summary: `memory search completed: ${input.finalResults.length} results`,
      payload: {
        query: input.query,
        type: input.type ?? 'all',
        durationMs: input.durationMs,
        sourceCount: input.sourceCount,
        vectorCandidateCount: input.vectorCandidates.length,
        finalResultCount: input.finalResults.length,
        vectorCandidates: input.vectorCandidates,
        finalResults: input.finalResults,
      },
    });
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
   * Dashboard 表示用に embedding BLOB を除いた全 memory row を取得する。
   *
   * @returns SQLite に保存された memory metadata rows
   */
  getDashboardMemories(
    input: { limit?: number } = {}
  ): StoredMemoryDashboardRow[] {
    this.ensureSchema();
    const limit = Math.max(
      1,
      Math.floor(input.limit ?? DASHBOARD_MEMORY_ROW_LIMIT)
    );

    return this.sql
      .exec<StoredMemoryDashboardRow>(
        `SELECT
           id,
           content,
           type,
           embedding_model,
           emotion_valence,
           emotion_arousal,
           emotion_labels,
           created_at,
           updated_at
         FROM memories
         ORDER BY updated_at DESC
         LIMIT ?`,
        limit
      )
      .toArray();
  }

  /**
   * Dashboard summary 表示に必要な memory 件数と最新更新時刻だけを返す。
   *
   * @returns memory summary
   */
  getDashboardMemorySummary(): MemoryDashboardSummary {
    this.ensureSchema();

    const row = this.sql
      .exec<StoredMemoryDashboardSummaryRow>(
        `SELECT
           COUNT(*) AS memory_count,
           MAX(updated_at) AS latest_updated_at
         FROM memories`
      )
      .one();

    return {
      count: row.memory_count,
      latestUpdatedAt: row.latest_updated_at,
    };
  }

  /**
   * 検索用に current embedding model の memory row を読み込む。
   *
   * Durable Objects SQLite の rows read 制限を避けるため、1 リクエスト内では
   * 読み込み済み rows を再利用する。異なる embedding model の row はベクトル空間や
   * 次元が一致しない可能性があるため、日次再 embedding で current model へ戻す。
   *
   * @returns semantic search の候補にできる memory rows
   */
  private getSearchableMemories(): StoredMemoryRow[] {
    this.ensureSchema();

    if (this.searchableMemoryRows !== null) {
      return this.searchableMemoryRows;
    }

    this.searchableMemoryRows = this.sql
      .exec<StoredMemoryRow>(
        `SELECT *
         FROM memories
         WHERE embedding_model = ?
         ORDER BY updated_at DESC
         LIMIT ?`,
        this.embeddingService.modelIdentifier,
        MEMORY_SEARCH_SOURCE_LIMIT
      )
      .toArray();

    return this.searchableMemoryRows;
  }

  /**
   * 現在の embedding モデルと異なるモデルで生成された memory を再 embedding する
   *
   * @param input 1 回で処理する最大件数
   * @returns 再 embedding 完了
   */
  async reEmbedStaleMemories(
    input: ReEmbedStaleMemoriesInput = {}
  ): Promise<void> {
    this.ensureSchema();

    const currentModel = this.embeddingService.modelIdentifier;
    const limit = Math.max(1, Math.floor(input.limit ?? MAX_MEMORY_COUNT));

    const staleRows = this.sql
      .exec<{
        id: string;
        content: string;
      }>(
        `SELECT id, content
         FROM memories
         WHERE embedding_model != ?
         ORDER BY updated_at ASC
         LIMIT ?`,
        currentModel,
        limit
      )
      .toArray();

    if (staleRows.length === 0) {
      await emitEchoEvent(this.events, {
        type: 'memory.reembedding.skipped',
        severity: 'debug',
        summary: 'no stale memories to re-embed',
        payload: {
          currentModel,
        },
      });
      return;
    }

    await emitEchoEvent(this.events, {
      type: 'memory.reembedding.started',
      severity: 'info',
      summary: `re-embedding ${staleRows.length} memories`,
      payload: {
        currentModel,
        staleCount: staleRows.length,
      },
    });

    let failedCount = 0;
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
        this.searchableMemoryRows = null;
      } catch (error) {
        if (isExternalRequestBudgetError(error)) {
          throw error;
        }
        failedCount += 1;
        // eslint-disable-next-line no-await-in-loop
        await emitEchoEvent(this.events, {
          type: 'memory.reembedding.item_failed',
          severity: 'warn',
          summary: `failed to re-embed memory ${row.id}`,
          payload: {
            id: row.id,
            currentModel,
            error: getErrorMessage(error),
          },
        });
      }
    }
    await emitEchoEvent(this.events, {
      type: 'memory.reembedding.completed',
      severity: failedCount > 0 ? 'warn' : 'info',
      summary: 're-embedding completed',
      payload: {
        currentModel,
        staleCount: staleRows.length,
        successCount: staleRows.length - failedCount,
        failedCount,
      },
    });
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

  /**
   * ベクトル検索で絞った候補を rerank し、最終順位を返す。
   * reranker が失敗した場合はベクトル順位へフォールバックする。
   */
  private async rerankCandidates(
    query: string,
    candidates: { row: StoredMemoryRow; similarity: number }[]
  ): Promise<
    { row: StoredMemoryRow; similarity: number; rerankScore: number }[]
  > {
    try {
      const rerankedResults = await this.rerankingService.rerank(
        query,
        candidates.map(({ row }) => row.content),
        SEARCH_RESULT_LIMIT
      );
      const seenIds = new Set<number>();
      const rerankedCandidates = rerankedResults.flatMap(({ id, score }) => {
        const candidate = candidates[id];
        if (candidate === undefined || seenIds.has(id)) {
          return [];
        }
        seenIds.add(id);
        return [{ ...candidate, rerankScore: score }];
      });

      if (rerankedCandidates.length > 0) {
        return rerankedCandidates;
      }

      await emitEchoEvent(this.events, {
        type: 'memory.rerank.fallback',
        severity: 'warn',
        summary:
          'reranker returned no usable results; falling back to vector ranking',
        payload: {
          query,
          candidateCount: candidates.length,
        },
      });
    } catch (error) {
      if (isExternalRequestBudgetError(error)) {
        throw error;
      }
      await emitEchoEvent(this.events, {
        type: 'memory.rerank.failed',
        severity: 'warn',
        summary: `failed to rerank memory search results: ${getErrorMessage(error)}`,
        payload: {
          query,
          candidateCount: candidates.length,
          error: getErrorMessage(error),
        },
      });
    }

    return candidates.slice(0, SEARCH_RESULT_LIMIT).map((candidate) => ({
      ...candidate,
      rerankScore: candidate.similarity,
    }));
  }
}
