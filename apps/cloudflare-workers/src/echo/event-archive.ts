import type { DashboardEchoEvent } from '@echo-chamber/contracts/dashboard/types';
import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';
import { formatDate } from '@echo-chamber/core/utils/datetime';

import { DASHBOARD_ACTIVITY_EVENT_TYPES } from './dashboard-activity-events';

import type {
  DashboardActionAnalysisEventRange,
  DashboardActionAnalysisMetrics,
  DashboardActionAnalysisToolMetrics,
} from './dashboard-action-analysis';
import type { EchoEventArchive } from '../utils/echo-event';

const EVENT_ARCHIVE_DAY_BOUNDARY_HOUR_JST = 3;
const EVENT_RETENTION_DAYS = 90;
const DAY_MS = 24 * 60 * 60 * 1000;
const ECHO_EVENT_SEVERITIES = ['debug', 'info', 'warn', 'error'] as const;
const ECHO_EVENT_STREAMS = ['thought', 'system', 'analysis'] as const;

interface StoredEchoEventRow extends Record<string, SqlStorageValue> {
  id: string;
  created_at_ms: number;
  archive_day: string;
  session_id: string | null;
  type: string;
  category: string;
  severity: string;
  streams_json: string;
  summary: string;
  payload_json: string;
}

interface StoredActionAnalysisDailyStatsRow
  extends Record<string, SqlStorageValue> {
  archive_day: string;
  event_count: number;
  session_count: number;
  completed_session_count: number;
  failed_session_count: number;
  warning_session_count: number;
  max_turns_session_count: number;
  total_tokens: number;
  total_session_duration_ms: number;
  session_duration_count: number;
  total_turns: number;
  no_tool_call_turns: number;
  tool_call_count: number;
  tool_completed_count: number;
  tool_failed_count: number;
  memory_search_completed_count: number;
  memory_search_failed_count: number;
  memory_search_zero_result_count: number;
  memory_search_final_result_total: number;
  store_memory_completed_count: number;
}

interface StoredActionAnalysisToolStatsRow
  extends Record<string, SqlStorageValue> {
  archive_day: string;
  tool_name: string;
  called_count: number;
  completed_count: number;
  failed_count: number;
}

interface StoredActionAnalysisSessionRow
  extends Record<string, SqlStorageValue> {
  started_at_ms: number;
}

interface EchoEventArchiveDateRange {
  days: number;
  startArchiveDay: string;
  endArchiveDay: string;
}

export interface EchoEventArchiveDay {
  archiveDay: string;
  events: DashboardEchoEvent[];
}

export interface EchoEventArchiveRange {
  days: number;
  startArchiveDay: string;
  endArchiveDay: string;
  events: DashboardEchoEvent[];
}

const ACTION_ANALYSIS_EVENT_TYPES = [
  'session.started',
  'session.completed',
  'session.failed',
  'model.turn.completed',
  'tool.called',
  'tool.completed',
  'tool.failed',
  'memory.search.completed',
  'memory.search.failed',
] as const satisfies readonly EchoEvent['type'][];
const DASHBOARD_SESSION_LOG_EVENT_LIMIT = 200;
// SQLite needs the same literal predicate in the query and partial index to select the bounded index path.
const DASHBOARD_ACTIVITY_EVENT_TYPE_SQL_LIST =
  DASHBOARD_ACTIVITY_EVENT_TYPES.map((type) => `'${type}'`).join(', ');

type ActionAnalysisEventType = (typeof ACTION_ANALYSIS_EVENT_TYPES)[number];

interface ActionAnalysisEventMeta {
  archiveDay: string;
  createdAtMs: number;
}

interface ActionAnalysisDailyStatsIncrements {
  event_count?: number;
  session_count?: number;
  completed_session_count?: number;
  failed_session_count?: number;
  warning_session_count?: number;
  max_turns_session_count?: number;
  total_tokens?: number;
  total_session_duration_ms?: number;
  session_duration_count?: number;
  total_turns?: number;
  no_tool_call_turns?: number;
  tool_call_count?: number;
  tool_completed_count?: number;
  tool_failed_count?: number;
  memory_search_completed_count?: number;
  memory_search_failed_count?: number;
  memory_search_zero_result_count?: number;
  memory_search_final_result_total?: number;
  store_memory_completed_count?: number;
}

interface ActionAnalysisToolStatsIncrements {
  called_count?: number;
  completed_count?: number;
  failed_count?: number;
}

interface ActionAnalysisMetricsRangeResult {
  eventCount: number;
  metrics: DashboardActionAnalysisMetrics;
}

/**
 * JST 03:00 を境界にした event archive day を返す。
 *
 * @param date 判定対象時刻
 * @returns `YYYY-MM-DD` 形式の archive day
 */
export function getEventArchiveDay(date: Date): string {
  return formatDate(
    new Date(
      date.getTime() - EVENT_ARCHIVE_DAY_BOUNDARY_HOUR_JST * 60 * 60_000
    ),
    'Asia/Tokyo'
  );
}

/**
 * DO SQLite を Echo event の永続保存層として扱う archive。
 */
export class SqliteEchoEventArchive implements EchoEventArchive {
  private readonly sql: SqlStorage;
  private initialized = false;

  /**
   * @param options DO SQLite
   */
  constructor(options: { sql: SqlStorage }) {
    this.sql = options.sql;
  }

  /**
   * Echo event を DO SQLite に保存する。
   *
   * @param event 保存する event
   * @param context 保存時の実行 context
   */
  async recordEvent(
    event: EchoEvent,
    context: {
      sessionId: string | null;
    }
  ): Promise<void> {
    this.ensureSchema();

    const now = new Date();
    const createdAtMs = now.getTime();
    const archiveDay = getEventArchiveDay(now);
    this.sql.exec(
      `INSERT INTO echo_events (
        id,
        created_at_ms,
        archive_day,
        session_id,
        type,
        category,
        severity,
        streams_json,
        summary,
        payload_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      crypto.randomUUID(),
      createdAtMs,
      archiveDay,
      context.sessionId,
      event.type,
      event.category,
      event.severity,
      safeJsonStringify(event.streams),
      event.summary,
      safeJsonStringify(event.payload ?? null)
    );
    this.recordActionAnalysisStats(event, context, {
      archiveDay,
      createdAtMs,
    });
    await Promise.resolve();
  }

  /**
   * 現在 archive day の raw event 一覧を返す。
   *
   * @param input archive day 判定の基準時刻と取得上限
   * @returns archive day と raw event list
   */
  getTodayEvents(
    input: { now?: Date; limit?: number } = {}
  ): EchoEventArchiveDay {
    const archiveDay = getEventArchiveDay(input.now ?? new Date());

    return {
      archiveDay,
      events: this.getDashboardActivityEventsByArchiveDay(
        archiveDay,
        input.limit ?? DASHBOARD_SESSION_LOG_EVENT_LIMIT
      ),
    };
  }

  /**
   * 指定日数分の archive day に含まれる raw event 一覧を返す。
   *
   * @param input 基準時刻と集計対象日数
   * @returns archive day range と raw event list
   */
  getRecentEvents(input: {
    now?: Date;
    days: number;
    limit?: number;
  }): EchoEventArchiveRange {
    const range = getRecentArchiveDateRange(input);

    return {
      ...range,
      events: this.getEventsByArchiveDayRange(
        range.startArchiveDay,
        range.endArchiveDay,
        input.limit ?? DASHBOARD_SESSION_LOG_EVENT_LIMIT
      ),
    };
  }

  /**
   * 指定日数分の archive day から action analysis の集計済み metrics を返す。
   *
   * @param input 基準時刻と集計対象日数
   * @returns archive day range と action analysis metrics
   */
  getRecentActionAnalysisEvents(input: {
    now?: Date;
    days: number;
  }): DashboardActionAnalysisEventRange {
    const [range] = this.getRecentActionAnalysisEventRanges({
      now: input.now,
      periodDays: [input.days],
    });
    if (range === undefined) {
      throw new Error('failed to create action analysis event range');
    }

    return range;
  }

  /**
   * 指定 period 群の action analysis metrics を日次集計 read model から作る。
   *
   * @param input 基準時刻、集計対象 period 群
   * @returns period ごとの action analysis metrics range
   */
  getRecentActionAnalysisEventRanges(input: {
    now?: Date;
    periodDays: readonly number[];
  }): DashboardActionAnalysisEventRange[] {
    const now = input.now ?? new Date();
    const maxDays = Math.max(
      1,
      ...input.periodDays.map((days) => Math.max(1, Math.floor(days)))
    );
    const maxRange = getRecentArchiveDateRange({
      now,
      days: maxDays,
    });
    const dailyRows = this.getActionAnalysisDailyStatsByArchiveDayRange(
      maxRange.startArchiveDay,
      maxRange.endArchiveDay
    );
    const toolRows = this.getActionAnalysisToolStatsByArchiveDayRange(
      maxRange.startArchiveDay,
      maxRange.endArchiveDay
    );

    return input.periodDays.map((days) => {
      const normalizedDays = Math.max(1, Math.floor(days));
      const range = getRecentArchiveDateRange({
        now,
        days: normalizedDays,
      });
      const metrics = buildActionAnalysisMetricsForRange(
        dailyRows,
        toolRows,
        range
      );

      return {
        ...range,
        eventCount: metrics.eventCount,
        events: [],
        metrics: metrics.metrics,
      };
    });
  }

  /**
   * 保持期間を超えた Echo event を SQLite から削除する。
   *
   * @param input 基準時刻
   */
  async deleteExpiredEvents(input: { now?: Date } = {}): Promise<void> {
    this.ensureSchema();

    const now = input.now ?? new Date();
    const cutoffDay = getEventArchiveDay(
      new Date(now.getTime() - EVENT_RETENTION_DAYS * DAY_MS)
    );
    this.sql.exec(
      `DELETE FROM echo_events
       WHERE archive_day < ?`,
      cutoffDay
    );
    this.sql.exec(
      `DELETE FROM echo_action_analysis_daily_stats
       WHERE archive_day < ?`,
      cutoffDay
    );
    this.sql.exec(
      `DELETE FROM echo_action_analysis_tool_daily_stats
       WHERE archive_day < ?`,
      cutoffDay
    );
    this.sql.exec(
      `DELETE FROM echo_action_analysis_sessions
       WHERE archive_day < ?`,
      cutoffDay
    );
    await Promise.resolve();
  }

  /**
   * SQLite schema を初期化する。
   */
  private ensureSchema(): void {
    if (this.initialized) {
      return;
    }

    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS echo_events (
        id TEXT PRIMARY KEY,
        created_at_ms INTEGER NOT NULL,
        archive_day TEXT NOT NULL,
        session_id TEXT,
        type TEXT NOT NULL,
        category TEXT NOT NULL,
        severity TEXT NOT NULL,
        streams_json TEXT NOT NULL,
        summary TEXT NOT NULL,
        payload_json TEXT NOT NULL
      )
    `);

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_echo_events_archive_day_created
      ON echo_events(archive_day, created_at_ms)
    `);

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_echo_events_archive_day_dashboard_activity_created
      ON echo_events(archive_day, created_at_ms)
      WHERE session_id IS NOT NULL
        AND type IN (${DASHBOARD_ACTIVITY_EVENT_TYPE_SQL_LIST})
    `);

    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS echo_action_analysis_daily_stats (
        archive_day TEXT PRIMARY KEY,
        event_count INTEGER NOT NULL DEFAULT 0,
        session_count INTEGER NOT NULL DEFAULT 0,
        completed_session_count INTEGER NOT NULL DEFAULT 0,
        failed_session_count INTEGER NOT NULL DEFAULT 0,
        warning_session_count INTEGER NOT NULL DEFAULT 0,
        max_turns_session_count INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        total_session_duration_ms INTEGER NOT NULL DEFAULT 0,
        session_duration_count INTEGER NOT NULL DEFAULT 0,
        total_turns INTEGER NOT NULL DEFAULT 0,
        no_tool_call_turns INTEGER NOT NULL DEFAULT 0,
        tool_call_count INTEGER NOT NULL DEFAULT 0,
        tool_completed_count INTEGER NOT NULL DEFAULT 0,
        tool_failed_count INTEGER NOT NULL DEFAULT 0,
        memory_search_completed_count INTEGER NOT NULL DEFAULT 0,
        memory_search_failed_count INTEGER NOT NULL DEFAULT 0,
        memory_search_zero_result_count INTEGER NOT NULL DEFAULT 0,
        memory_search_final_result_total INTEGER NOT NULL DEFAULT 0,
        store_memory_completed_count INTEGER NOT NULL DEFAULT 0
      )
    `);

    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS echo_action_analysis_tool_daily_stats (
        archive_day TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        called_count INTEGER NOT NULL DEFAULT 0,
        completed_count INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (archive_day, tool_name)
      )
    `);

    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS echo_action_analysis_sessions (
        session_id TEXT PRIMARY KEY,
        archive_day TEXT NOT NULL,
        started_at_ms INTEGER NOT NULL
      )
    `);

    this.sql.exec(`
      CREATE INDEX IF NOT EXISTS idx_echo_action_analysis_sessions_archive_day
      ON echo_action_analysis_sessions(archive_day)
    `);

    this.initialized = true;
  }

  /**
   * action-analysis 用 event だけを日次集計へ反映する。
   */
  private recordActionAnalysisStats(
    event: EchoEvent,
    context: {
      sessionId: string | null;
    },
    meta: ActionAnalysisEventMeta
  ): void {
    if (!isActionAnalysisEventType(event.type)) {
      return;
    }

    const payload = getEventPayload(event);
    const increments: ActionAnalysisDailyStatsIncrements = {
      event_count: 1,
    };
    const recorders: Record<ActionAnalysisEventType, () => void> = {
      'session.started': () => {
        this.recordSessionStartedStats(context.sessionId, meta, increments);
      },
      'session.completed': () => {
        this.recordSessionCompletedStats(event, context.sessionId, meta, {
          payload,
          increments,
        });
      },
      'session.failed': () => {
        this.recordSessionFailedStats(context.sessionId, meta, increments);
      },
      'model.turn.completed': () => {
        this.recordModelTurnCompletedStats(payload, increments);
      },
      'tool.called': () => {
        this.recordToolStats(meta.archiveDay, payload, increments, {
          called_count: 1,
        });
      },
      'tool.completed': () => {
        this.recordToolCompletedStats(meta.archiveDay, payload, increments);
      },
      'tool.failed': () => {
        this.recordToolStats(meta.archiveDay, payload, increments, {
          failed_count: 1,
        });
      },
      'memory.search.completed': () => {
        this.recordMemorySearchCompletedStats(payload, increments);
      },
      'memory.search.failed': () => {
        increments.memory_search_failed_count = 1;
      },
    };

    recorders[event.type]();
    this.incrementActionAnalysisDailyStats(meta.archiveDay, increments);
  }

  /**
   * session started event を action-analysis 集計へ反映する。
   */
  private recordSessionStartedStats(
    sessionId: string | null,
    meta: ActionAnalysisEventMeta,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    if (sessionId === null) {
      return;
    }

    increments.session_count = 1;
    this.sql.exec(
      `INSERT OR REPLACE INTO echo_action_analysis_sessions (
        session_id,
        archive_day,
        started_at_ms
      ) VALUES (?, ?, ?)`,
      sessionId,
      meta.archiveDay,
      meta.createdAtMs
    );
  }

  /**
   * session completed event を action-analysis 集計へ反映する。
   */
  private recordSessionCompletedStats(
    event: EchoEvent,
    sessionId: string | null,
    meta: ActionAnalysisEventMeta,
    input: {
      payload: Record<string, unknown>;
      increments: ActionAnalysisDailyStatsIncrements;
    }
  ): void {
    input.increments.completed_session_count = 1;
    input.increments.total_tokens = getPayloadNumber(
      input.payload,
      'totalTokens'
    );
    if (event.severity === 'warn') {
      input.increments.warning_session_count = 1;
    }
    if (getPayloadString(input.payload, 'terminationReason') === 'max_turns') {
      input.increments.max_turns_session_count = 1;
    }

    this.addSessionDurationStats(sessionId, meta, input.increments);
  }

  /**
   * session failed event を action-analysis 集計へ反映する。
   */
  private recordSessionFailedStats(
    sessionId: string | null,
    meta: ActionAnalysisEventMeta,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    increments.failed_session_count = 1;
    this.addSessionDurationStats(sessionId, meta, increments);
  }

  /**
   * session start と finish の差分を duration metrics へ加算する。
   */
  private addSessionDurationStats(
    sessionId: string | null,
    meta: ActionAnalysisEventMeta,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    if (sessionId === null) {
      return;
    }

    const [row] = this.sql
      .exec<StoredActionAnalysisSessionRow>(
        `SELECT started_at_ms
         FROM echo_action_analysis_sessions
         WHERE session_id = ?
         LIMIT 1`,
        sessionId
      )
      .toArray();
    if (row === undefined) {
      return;
    }

    increments.total_session_duration_ms = Math.max(
      0,
      meta.createdAtMs - row.started_at_ms
    );
    increments.session_duration_count = 1;
  }

  /**
   * model turn completed event を action-analysis 集計へ反映する。
   */
  private recordModelTurnCompletedStats(
    payload: Record<string, unknown>,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    increments.total_turns = 1;
    if (getPayloadStringArray(payload, 'warnings').includes('no_tool_calls')) {
      increments.no_tool_call_turns = 1;
    }
  }

  /**
   * tool completed event を action-analysis 集計へ反映する。
   */
  private recordToolCompletedStats(
    archiveDay: string,
    payload: Record<string, unknown>,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    this.recordToolStats(archiveDay, payload, increments, {
      completed_count: 1,
    });

    if (
      normalizeToolName(getPayloadString(payload, 'toolName')) ===
      'store_memory'
    ) {
      increments.store_memory_completed_count = 1;
    }
  }

  /**
   * tool event を全体・tool 別集計へ反映する。
   */
  private recordToolStats(
    archiveDay: string,
    payload: Record<string, unknown>,
    increments: ActionAnalysisDailyStatsIncrements,
    toolIncrements: ActionAnalysisToolStatsIncrements
  ): void {
    if (toolIncrements.called_count !== undefined) {
      increments.tool_call_count = 1;
    }
    if (toolIncrements.completed_count !== undefined) {
      increments.tool_completed_count = 1;
    }
    if (toolIncrements.failed_count !== undefined) {
      increments.tool_failed_count = 1;
    }

    this.incrementActionAnalysisToolStats(
      archiveDay,
      normalizeToolName(getPayloadString(payload, 'toolName')),
      toolIncrements
    );
  }

  /**
   * memory search completed event を action-analysis 集計へ反映する。
   */
  private recordMemorySearchCompletedStats(
    payload: Record<string, unknown>,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    const finalResultCount = getPayloadNumber(payload, 'finalResultCount');
    increments.memory_search_completed_count = 1;
    increments.memory_search_final_result_total = finalResultCount;
    if (finalResultCount === 0) {
      increments.memory_search_zero_result_count = 1;
    }
  }

  /**
   * 日次 action-analysis stats row を作る。
   */
  private ensureActionAnalysisDailyStatsRow(archiveDay: string): void {
    this.sql.exec(
      `INSERT INTO echo_action_analysis_daily_stats (archive_day)
       VALUES (?)
       ON CONFLICT(archive_day) DO NOTHING`,
      archiveDay
    );
  }

  /**
   * 日次 action-analysis stats を差分更新する。
   */
  private incrementActionAnalysisDailyStats(
    archiveDay: string,
    increments: ActionAnalysisDailyStatsIncrements
  ): void {
    const entries = getDefinedIncrementEntries(increments);
    if (entries.length === 0) {
      return;
    }

    this.ensureActionAnalysisDailyStatsRow(archiveDay);
    this.sql.exec(
      `UPDATE echo_action_analysis_daily_stats
       SET ${entries.map(([column]) => `${column} = ${column} + ?`).join(', ')}
       WHERE archive_day = ?`,
      ...entries.map(([, value]) => value),
      archiveDay
    );
  }

  /**
   * tool 別 action-analysis stats row を作る。
   */
  private ensureActionAnalysisToolStatsRow(
    archiveDay: string,
    toolName: string
  ): void {
    this.sql.exec(
      `INSERT INTO echo_action_analysis_tool_daily_stats (
        archive_day,
        tool_name
      ) VALUES (?, ?)
       ON CONFLICT(archive_day, tool_name) DO NOTHING`,
      archiveDay,
      toolName
    );
  }

  /**
   * tool 別 action-analysis stats を差分更新する。
   */
  private incrementActionAnalysisToolStats(
    archiveDay: string,
    toolName: string,
    increments: ActionAnalysisToolStatsIncrements
  ): void {
    const entries = getDefinedIncrementEntries(increments);
    if (entries.length === 0) {
      return;
    }

    this.ensureActionAnalysisToolStatsRow(archiveDay, toolName);
    this.sql.exec(
      `UPDATE echo_action_analysis_tool_daily_stats
       SET ${entries.map(([column]) => `${column} = ${column} + ?`).join(', ')}
       WHERE archive_day = ?
         AND tool_name = ?`,
      ...entries.map(([, value]) => value),
      archiveDay,
      toolName
    );
  }

  /**
   * 指定 archive day の Dashboard Activity 候補 event を取得する。
   */
  private getDashboardActivityEventsByArchiveDay(
    archiveDay: string,
    limit: number
  ): DashboardEchoEvent[] {
    this.ensureSchema();
    const normalizedLimit = Math.max(1, Math.floor(limit));

    return this.sql
      .exec<StoredEchoEventRow>(
        `SELECT *
         FROM echo_events
         WHERE archive_day = ?
           AND session_id IS NOT NULL
           AND type IN (${DASHBOARD_ACTIVITY_EVENT_TYPE_SQL_LIST})
         ORDER BY created_at_ms DESC
         LIMIT ?`,
        archiveDay,
        normalizedLimit
      )
      .toArray()
      .map(rowToDashboardEvent);
  }

  /**
   * 指定 archive day 範囲の event を dashboard 用に取得する。
   */
  private getEventsByArchiveDayRange(
    startArchiveDay: string,
    endArchiveDay: string,
    limit: number
  ): DashboardEchoEvent[] {
    this.ensureSchema();
    const normalizedLimit = Math.max(1, Math.floor(limit));

    return this.sql
      .exec<StoredEchoEventRow>(
        `SELECT *
         FROM echo_events
         WHERE archive_day >= ?
           AND archive_day <= ?
         ORDER BY created_at_ms DESC
         LIMIT ?`,
        startArchiveDay,
        endArchiveDay,
        normalizedLimit
      )
      .toArray()
      .map(rowToDashboardEvent);
  }

  /**
   * 指定 archive day 範囲の日次 action-analysis stats を返す。
   */
  private getActionAnalysisDailyStatsByArchiveDayRange(
    startArchiveDay: string,
    endArchiveDay: string
  ): StoredActionAnalysisDailyStatsRow[] {
    this.ensureSchema();

    return this.sql
      .exec<StoredActionAnalysisDailyStatsRow>(
        `SELECT
           archive_day,
           event_count,
           session_count,
           completed_session_count,
           failed_session_count,
           warning_session_count,
           max_turns_session_count,
           total_tokens,
           total_session_duration_ms,
           session_duration_count,
           total_turns,
           no_tool_call_turns,
           tool_call_count,
           tool_completed_count,
           tool_failed_count,
           memory_search_completed_count,
           memory_search_failed_count,
           memory_search_zero_result_count,
           memory_search_final_result_total,
           store_memory_completed_count
         FROM echo_action_analysis_daily_stats
         WHERE archive_day >= ?
           AND archive_day <= ?
         ORDER BY archive_day ASC`,
        startArchiveDay,
        endArchiveDay
      )
      .toArray();
  }

  /**
   * 指定 archive day 範囲の tool 別 action-analysis stats を返す。
   */
  private getActionAnalysisToolStatsByArchiveDayRange(
    startArchiveDay: string,
    endArchiveDay: string
  ): StoredActionAnalysisToolStatsRow[] {
    this.ensureSchema();

    return this.sql
      .exec<StoredActionAnalysisToolStatsRow>(
        `SELECT
           archive_day,
           tool_name,
           called_count,
           completed_count,
           failed_count
         FROM echo_action_analysis_tool_daily_stats
         WHERE archive_day >= ?
           AND archive_day <= ?
         ORDER BY archive_day ASC, tool_name ASC`,
        startArchiveDay,
        endArchiveDay
      )
      .toArray();
  }
}

/**
 * 指定日数分の archive day 範囲を返す。
 */
function getRecentArchiveDateRange(input: {
  now?: Date;
  days: number;
}): EchoEventArchiveDateRange {
  const now = input.now ?? new Date();
  const days = Math.max(1, Math.floor(input.days));
  const endArchiveDay = getEventArchiveDay(now);
  const startArchiveDay = getEventArchiveDay(
    new Date(now.getTime() - (days - 1) * DAY_MS)
  );

  return {
    days,
    startArchiveDay,
    endArchiveDay,
  };
}

/**
 * SQLite row を dashboard event へ変換する。
 */
function rowToDashboardEvent(row: StoredEchoEventRow): DashboardEchoEvent {
  return {
    id: row.id,
    createdAt: new Date(row.created_at_ms).toISOString(),
    archiveDay: row.archive_day,
    sessionId: row.session_id,
    type: row.type,
    category: row.category,
    severity: parseDashboardEventSeverity(row.severity),
    streams: parseDashboardEventStreams(row.streams_json),
    summary: row.summary,
    payload: parseJsonObject(row.payload_json),
  };
}

/**
 * JSON へ変換できない payload でも event 保存を失敗させない。
 */
function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return JSON.stringify({
      error: 'json_serialization_failed',
    });
  }
}

/**
 * 保存済み JSON array を読み出す。
 */
function parseDashboardEventStreams(
  value: string
): DashboardEchoEvent['streams'] {
  try {
    const parsed: unknown = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.filter(
        (item): item is DashboardEchoEvent['streams'][number] => {
          return (
            typeof item === 'string' &&
            ECHO_EVENT_STREAMS.includes(
              item as DashboardEchoEvent['streams'][number]
            )
          );
        }
      );
    }
  } catch {
    return [];
  }

  return [];
}

/**
 * 保存済み severity を dashboard 用の既知値へ丸める。
 */
function parseDashboardEventSeverity(
  value: string
): DashboardEchoEvent['severity'] {
  if (ECHO_EVENT_SEVERITIES.includes(value as DashboardEchoEvent['severity'])) {
    return value as DashboardEchoEvent['severity'];
  }

  return 'debug';
}

/**
 * 保存済み JSON object/null を読み出す。
 */
function parseJsonObject(value: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (parsed === null) {
      return null;
    }
    if (typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return {
      error: 'json_parse_failed',
    };
  }

  return {
    value,
  };
}

/**
 * event type が action-analysis 集計対象かを判定する。
 */
function isActionAnalysisEventType(
  type: EchoEvent['type']
): type is ActionAnalysisEventType {
  return (ACTION_ANALYSIS_EVENT_TYPES as readonly string[]).includes(type);
}

/**
 * event payload を record として取り出す。
 */
function getEventPayload(event: EchoEvent): Record<string, unknown> {
  return event.payload ?? {};
}

/**
 * payload の string property を取り出す。
 */
function getPayloadString(
  payload: Record<string, unknown>,
  key: string
): string | undefined {
  const value = payload[key];
  return typeof value === 'string' ? value : undefined;
}

/**
 * payload の number property を取り出す。欠損時は 0 として扱う。
 */
function getPayloadNumber(
  payload: Record<string, unknown>,
  key: string
): number {
  const value = payload[key];
  return typeof value === 'number' ? value : 0;
}

/**
 * payload の string array property を取り出す。
 */
function getPayloadStringArray(
  payload: Record<string, unknown>,
  key: string
): string[] {
  const value = payload[key];
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === 'string');
}

/**
 * undefined / zero を除いた SQL increment entry を返す。
 */
function getDefinedIncrementEntries(increments: object): [string, number][] {
  return Object.entries(
    increments as Record<string, number | undefined>
  ).flatMap(([column, value]) => {
    if (value === undefined || value === 0) {
      return [];
    }

    return [[column, value]];
  });
}

/**
 * tool name を集計用の非空文字列へ丸める。
 */
function normalizeToolName(toolName: string | undefined): string {
  const normalized = toolName?.trim();
  return normalized === undefined || normalized.length === 0
    ? 'tool'
    : normalized;
}

/**
 * 空の action-analysis metrics を作る。
 */
function createEmptyActionAnalysisMetrics(): DashboardActionAnalysisMetrics {
  return {
    sessionCount: 0,
    completedSessionCount: 0,
    failedSessionCount: 0,
    warningSessionCount: 0,
    maxTurnsSessionCount: 0,
    totalTokens: 0,
    totalSessionDurationMs: 0,
    sessionDurationCount: 0,
    totalTurns: 0,
    noToolCallTurns: 0,
    toolCallCount: 0,
    toolCompletedCount: 0,
    toolFailedCount: 0,
    topTools: [],
    memorySearchCompletedCount: 0,
    memorySearchFailedCount: 0,
    memorySearchZeroResultCount: 0,
    memorySearchFinalResultTotal: 0,
    storeMemoryCompletedCount: 0,
  };
}

/**
 * 取得済み日次 stats から指定 range の metrics を合成する。
 */
function buildActionAnalysisMetricsForRange(
  dailyRows: readonly StoredActionAnalysisDailyStatsRow[],
  toolRows: readonly StoredActionAnalysisToolStatsRow[],
  range: EchoEventArchiveDateRange
): ActionAnalysisMetricsRangeResult {
  const metrics = createEmptyActionAnalysisMetrics();
  let eventCount = 0;

  for (const row of dailyRows) {
    if (!isArchiveDayInRange(row.archive_day, range)) {
      continue;
    }

    eventCount += row.event_count;
    metrics.sessionCount += row.session_count;
    metrics.completedSessionCount += row.completed_session_count;
    metrics.failedSessionCount += row.failed_session_count;
    metrics.warningSessionCount += row.warning_session_count;
    metrics.maxTurnsSessionCount += row.max_turns_session_count;
    metrics.totalTokens += row.total_tokens;
    metrics.totalSessionDurationMs += row.total_session_duration_ms;
    metrics.sessionDurationCount += row.session_duration_count;
    metrics.totalTurns += row.total_turns;
    metrics.noToolCallTurns += row.no_tool_call_turns;
    metrics.toolCallCount += row.tool_call_count;
    metrics.toolCompletedCount += row.tool_completed_count;
    metrics.toolFailedCount += row.tool_failed_count;
    metrics.memorySearchCompletedCount += row.memory_search_completed_count;
    metrics.memorySearchFailedCount += row.memory_search_failed_count;
    metrics.memorySearchZeroResultCount += row.memory_search_zero_result_count;
    metrics.memorySearchFinalResultTotal +=
      row.memory_search_final_result_total;
    metrics.storeMemoryCompletedCount += row.store_memory_completed_count;
  }

  return {
    eventCount,
    metrics: {
      ...metrics,
      topTools: buildActionAnalysisToolMetricsForRange(toolRows, range),
    },
  };
}

/**
 * tool stats rows から指定 range の tool metrics を合成する。
 */
function buildActionAnalysisToolMetricsForRange(
  rows: readonly StoredActionAnalysisToolStatsRow[],
  range: EchoEventArchiveDateRange
): DashboardActionAnalysisToolMetrics[] {
  const metricsByTool = new Map<string, DashboardActionAnalysisToolMetrics>();

  for (const row of rows) {
    if (!isArchiveDayInRange(row.archive_day, range)) {
      continue;
    }

    const existing = metricsByTool.get(row.tool_name) ?? {
      toolName: row.tool_name,
      calledCount: 0,
      completedCount: 0,
      failedCount: 0,
    };
    existing.calledCount += row.called_count;
    existing.completedCount += row.completed_count;
    existing.failedCount += row.failed_count;
    metricsByTool.set(row.tool_name, existing);
  }

  return [...metricsByTool.values()];
}

/**
 * archive day が対象 range に含まれるかを判定する。
 */
function isArchiveDayInRange(
  archiveDay: string,
  range: EchoEventArchiveDateRange
): boolean {
  return (
    archiveDay >= range.startArchiveDay && archiveDay <= range.endArchiveDay
  );
}
