import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';

import { SqliteEchoEventArchive, getEventArchiveDay } from './event-archive';

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

  it('90日を超えた Echo event を SQLite から削除する', async () => {
    const { exec, sql } = createMockSql();
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    await archive.deleteExpiredEvents({
      now: new Date('2026-06-02T18:00:00.000Z'),
    });

    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('DELETE FROM echo_events'),
      '2026-03-05',
    ]);
  });

  it('指定日数分の archive day 範囲で Echo event を取得する', () => {
    const { exec, sql } = createMockSql();
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    const result = archive.getRecentEvents({
      now: new Date('2026-06-02T18:00:00.000Z'),
      days: 7,
    });

    expect(result).toEqual({
      days: 7,
      startArchiveDay: '2026-05-28',
      endArchiveDay: '2026-06-03',
      events: [],
    });
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('WHERE archive_day >= ?'),
      '2026-05-28',
      '2026-06-03',
    ]);
  });
});
