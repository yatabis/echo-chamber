import type {
  DashboardEchoEvent,
  DashboardEchoEventsResponse,
} from '@echo-chamber/contracts/dashboard/types';
import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';
import { formatDate } from '@echo-chamber/core/utils/datetime';

import type { EchoEventArchive } from '../utils/echo-event';

const EVENT_ARCHIVE_DAY_BOUNDARY_HOUR_JST = 3;
const ARCHIVE_RETENTION_DAYS = 90;
const DAY_MS = 24 * 60 * 60 * 1000;
const ECHO_EVENT_SEVERITIES = ['debug', 'info', 'warn', 'error'] as const;
const ECHO_EVENT_STREAMS = ['thought', 'system', 'analysis'] as const;

type ArchiveRunStatus = 'uploaded' | 'deleted';

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

interface ArchiveRunRow extends Record<string, SqlStorageValue> {
  archive_day: string;
  status: ArchiveRunStatus;
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
 * R2 上の event archive object key を決定的に作る。
 *
 * @param input instance と archive day
 * @returns R2 object key
 */
export function buildEventArchiveObjectKey(input: {
  instanceId: string;
  archiveDay: string;
}): string {
  return `echo-events/instance=${input.instanceId}/day=${input.archiveDay}/events.ndjson`;
}

/**
 * DO SQLite を event の staging buffer として扱う archive。
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
      now.getTime(),
      getEventArchiveDay(now),
      context.sessionId,
      event.type,
      event.category,
      event.severity,
      safeJsonStringify(event.streams),
      event.summary,
      safeJsonStringify(event.payload ?? null)
    );
    await Promise.resolve();
  }

  /**
   * dashboard に表示する現在 archive day の event 一覧を返す。
   *
   * @param now archive day 判定の基準時刻
   * @returns dashboard event response
   */
  getTodayEvents(now = new Date()): DashboardEchoEventsResponse {
    const archiveDay = getEventArchiveDay(now);

    return {
      archiveDay,
      events: this.getEventsByArchiveDay(archiveDay),
    };
  }

  /**
   * 完了済み archive day を R2 へ退避し、退避済み event を削除する。
   *
   * @param input R2 bucket、instance id、基準時刻
   */
  async rotateCompletedDays(input: {
    bucket: R2Bucket;
    instanceId: string;
    now?: Date;
  }): Promise<void> {
    this.ensureSchema();

    const currentArchiveDay = getEventArchiveDay(input.now ?? new Date());
    const days = this.sql
      .exec<{ archive_day: string }>(
        `SELECT DISTINCT archive_day
         FROM echo_events
         WHERE archive_day < ?
         ORDER BY archive_day ASC`,
        currentArchiveDay
      )
      .toArray();

    for (const { archive_day: archiveDay } of days) {
      // Daily rotation must keep day order to preserve predictable retry behavior.
      // eslint-disable-next-line no-await-in-loop
      await this.rotateArchiveDay({
        bucket: input.bucket,
        instanceId: input.instanceId,
        archiveDay,
      });
    }

    await this.deleteExpiredArchives({
      bucket: input.bucket,
      instanceId: input.instanceId,
      now: input.now ?? new Date(),
    });
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
      CREATE TABLE IF NOT EXISTS event_archive_runs (
        archive_day TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK (status IN ('uploaded', 'deleted'))
      )
    `);

    this.initialized = true;
  }

  /**
   * 指定 archive day の event を dashboard 用に取得する。
   */
  private getEventsByArchiveDay(archiveDay: string): DashboardEchoEvent[] {
    this.ensureSchema();

    return this.sql
      .exec<StoredEchoEventRow>(
        `SELECT *
         FROM echo_events
         WHERE archive_day = ?
         ORDER BY created_at_ms DESC`,
        archiveDay
      )
      .toArray()
      .map(rowToDashboardEvent);
  }

  /**
   * 指定 archive day を R2 へ退避し、SQLite から削除する。
   */
  private async rotateArchiveDay(input: {
    bucket: R2Bucket;
    instanceId: string;
    archiveDay: string;
  }): Promise<void> {
    const currentRun = this.sql
      .exec<ArchiveRunRow>(
        `SELECT archive_day, status
         FROM event_archive_runs
         WHERE archive_day = ?`,
        input.archiveDay
      )
      .toArray()[0];

    if (currentRun?.status === 'deleted') {
      return;
    }

    if (currentRun?.status !== 'uploaded') {
      const events = this.getEventsByArchiveDay(input.archiveDay);
      if (events.length === 0) {
        return;
      }

      await input.bucket.put(
        buildEventArchiveObjectKey(input),
        `${events.map((event) => JSON.stringify(event)).join('\n')}\n`,
        {
          httpMetadata: {
            contentType: 'application/x-ndjson; charset=utf-8',
          },
          customMetadata: {
            archiveDay: input.archiveDay,
            instanceId: input.instanceId,
          },
        }
      );
      this.upsertArchiveRun(input.archiveDay, 'uploaded');
    }

    this.sql.exec(
      'DELETE FROM echo_events WHERE archive_day = ?',
      input.archiveDay
    );
    this.upsertArchiveRun(input.archiveDay, 'deleted');
  }

  /**
   * archive run の status を保存する。
   */
  private upsertArchiveRun(archiveDay: string, status: ArchiveRunStatus): void {
    this.sql.exec(
      `INSERT INTO event_archive_runs (archive_day, status)
       VALUES (?, ?)
       ON CONFLICT(archive_day) DO UPDATE SET status = excluded.status`,
      archiveDay,
      status
    );
  }

  /**
   * 90日を超えた R2 object と完了済み archive run を削除する。
   */
  private async deleteExpiredArchives(input: {
    bucket: R2Bucket;
    instanceId: string;
    now: Date;
  }): Promise<void> {
    const cutoffDay = getEventArchiveDay(
      new Date(input.now.getTime() - ARCHIVE_RETENTION_DAYS * DAY_MS)
    );
    const expiredRuns = this.sql
      .exec<ArchiveRunRow>(
        `SELECT archive_day, status
         FROM event_archive_runs
         WHERE status = 'deleted'
           AND archive_day < ?
         ORDER BY archive_day ASC`,
        cutoffDay
      )
      .toArray();

    for (const run of expiredRuns) {
      // R2 cleanup is intentionally serialized so retry checkpoints stay simple.
      // eslint-disable-next-line no-await-in-loop
      await input.bucket.delete(
        buildEventArchiveObjectKey({
          instanceId: input.instanceId,
          archiveDay: run.archive_day,
        })
      );
    }

    this.sql.exec(
      `DELETE FROM event_archive_runs
       WHERE status = 'deleted'
         AND archive_day < ?`,
      cutoffDay
    );
  }
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
