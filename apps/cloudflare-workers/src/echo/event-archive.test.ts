import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';

import {
  SqliteEchoEventArchive,
  buildEventArchiveObjectKey,
  getEventArchiveDay,
} from './event-archive';

function createCursor<T extends Record<string, SqlStorageValue>>(
  rows: T[]
): SqlStorageCursor<T> {
  return {
    next: vi.fn(() => {
      const value = rows.shift();
      return value === undefined ? { done: true } : { done: false, value };
    }),
    toArray: vi.fn(() => rows),
    one: vi.fn(() => {
      const first = rows[0];
      if (first === undefined) {
        throw new Error('empty cursor');
      }

      return first;
    }),
    raw: vi.fn(function* raw() {
      yield* [];
    }),
    columnNames: [],
    rowsRead: rows.length,
    rowsWritten: 0,
    [Symbol.iterator]: function* iterator() {
      yield* rows;
    },
  } as unknown as SqlStorageCursor<T>;
}

function createMockSql(
  exec: (
    query: string,
    ...bindings: unknown[]
  ) => SqlStorageCursor<Record<string, SqlStorageValue>> = () =>
    createCursor([])
): {
  exec: ReturnType<typeof vi.fn>;
  sql: SqlStorage;
} {
  const execMock = vi.fn(exec);

  return {
    exec: execMock,
    sql: {
      exec: execMock,
    } as unknown as SqlStorage,
  };
}

describe('event archive helpers', () => {
  it('JST 03:00 を archive day 境界にする', () => {
    expect(getEventArchiveDay(new Date('2026-06-01T17:59:00.000Z'))).toBe(
      '2026-06-01'
    );
    expect(getEventArchiveDay(new Date('2026-06-01T18:00:00.000Z'))).toBe(
      '2026-06-02'
    );
  });

  it('instance と archive day から R2 key を作る', () => {
    expect(
      buildEventArchiveObjectKey({
        instanceId: 'rin',
        archiveDay: '2026-06-01',
      })
    ).toBe('echo-events/instance=rin/day=2026-06-01/events.ndjson');
  });
});

describe('SqliteEchoEventArchive', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('Echo event を archive day 付きで SQLite に保存する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-01T17:59:00.000Z'));

    const { exec, sql } = createMockSql();
    const archive = new SqliteEchoEventArchive({
      sql,
    });
    const event: EchoEvent = {
      type: 'tool.completed',
      category: 'tool',
      severity: 'info',
      streams: ['system', 'analysis'],
      summary: 'tool completed',
      payload: {
        toolName: 'search_memory',
      },
    };

    await archive.recordEvent(event, {
      sessionId: 'session-1',
    });

    const insertCall = exec.mock.calls.find(([query]) => {
      return String(query).includes('INSERT INTO echo_events');
    });
    expect(insertCall).toEqual([
      expect.stringContaining('INSERT INTO echo_events'),
      expect.any(String),
      new Date('2026-06-01T17:59:00.000Z').getTime(),
      '2026-06-01',
      'session-1',
      'tool.completed',
      'tool',
      'info',
      '["system","analysis"]',
      'tool completed',
      '{"toolName":"search_memory"}',
    ]);
  });

  it('完了済み archive day を R2 に退避してから SQLite から削除する', async () => {
    const archivedRow = {
      id: 'event-1',
      created_at_ms: new Date('2026-06-01T12:00:00.000Z').getTime(),
      archive_day: '2026-06-01',
      session_id: null,
      type: 'session.completed',
      category: 'session',
      severity: 'info',
      streams_json: '["system","analysis"]',
      summary: 'thinking session completed',
      payload_json: 'null',
    };
    const { exec, sql } = createMockSql((query) => {
      if (query.includes('SELECT DISTINCT archive_day')) {
        return createCursor([{ archive_day: '2026-06-01' }]);
      }
      if (query.includes('FROM event_archive_runs')) {
        return createCursor([]);
      }
      if (query.includes('FROM echo_events')) {
        return createCursor([archivedRow]);
      }

      return createCursor([]);
    });
    const put = vi.fn(async () => Promise.resolve({}));
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    await archive.rotateCompletedDays({
      bucket: {
        put,
      } as unknown as R2Bucket,
      instanceId: 'rin',
      now: new Date('2026-06-02T18:00:00.000Z'),
    });

    expect(put).toHaveBeenCalledWith(
      'echo-events/instance=rin/day=2026-06-01/events.ndjson',
      expect.stringContaining('"type":"session.completed"'),
      expect.objectContaining({
        httpMetadata: {
          contentType: 'application/x-ndjson; charset=utf-8',
        },
      })
    );
    expect(exec.mock.calls).toContainEqual([
      'DELETE FROM echo_events WHERE archive_day = ?',
      '2026-06-01',
    ]);
  });

  it('90日を超えた R2 archive と archive run を削除する', async () => {
    const { exec, sql } = createMockSql((query) => {
      if (query.includes('SELECT DISTINCT archive_day')) {
        return createCursor([]);
      }
      if (query.includes("WHERE status = 'deleted'")) {
        return createCursor([
          {
            archive_day: '2026-02-01',
            status: 'deleted',
          },
        ]);
      }

      return createCursor([]);
    });
    const deleteObject = vi.fn(async () => Promise.resolve());
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    await archive.rotateCompletedDays({
      bucket: {
        delete: deleteObject,
      } as unknown as R2Bucket,
      instanceId: 'rin',
      now: new Date('2026-06-02T18:00:00.000Z'),
    });

    expect(deleteObject).toHaveBeenCalledWith(
      'echo-events/instance=rin/day=2026-02-01/events.ndjson'
    );
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('DELETE FROM event_archive_runs'),
      '2026-03-05',
    ]);
  });
});
