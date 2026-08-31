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

function createDailyStatsRow(
  overrides: Partial<Record<string, SqlStorageValue>>
): Record<string, SqlStorageValue> {
  return {
    archive_day: '2026-06-03',
    event_count: 0,
    session_count: 0,
    completed_session_count: 0,
    failed_session_count: 0,
    warning_session_count: 0,
    max_turns_session_count: 0,
    total_tokens: 0,
    total_session_duration_ms: 0,
    session_duration_count: 0,
    total_turns: 0,
    no_tool_call_turns: 0,
    tool_call_count: 0,
    tool_completed_count: 0,
    tool_failed_count: 0,
    memory_search_completed_count: 0,
    memory_search_failed_count: 0,
    memory_search_zero_result_count: 0,
    memory_search_final_result_total: 0,
    store_memory_completed_count: 0,
    ...overrides,
  };
}

function createToolStatsRow(
  overrides: Partial<Record<string, SqlStorageValue>>
): Record<string, SqlStorageValue> {
  return {
    archive_day: '2026-06-03',
    tool_name: 'search_memory',
    called_count: 0,
    completed_count: 0,
    failed_count: 0,
    ...overrides,
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

  it('action analysis 対象 event を日次 stats と tool stats に反映する', async () => {
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
      summary: 'store memory completed',
      payload: {
        toolName: 'store_memory',
      },
    };

    await archive.recordEvent(event, {
      sessionId: 'session-1',
    });

    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining(
        'INSERT INTO echo_action_analysis_tool_daily_stats'
      ),
      '2026-06-01',
      'store_memory',
    ]);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('UPDATE echo_action_analysis_tool_daily_stats'),
      1,
      '2026-06-01',
      'store_memory',
    ]);
    const dailyStatsUpdate = exec.mock.calls.find(([query]) => {
      return String(query).includes('UPDATE echo_action_analysis_daily_stats');
    });
    expect(dailyStatsUpdate).toEqual([
      expect.stringContaining('event_count = event_count + ?'),
      1,
      1,
      1,
      '2026-06-01',
    ]);
    expect(String(dailyStatsUpdate?.[0])).toContain(
      'tool_completed_count = tool_completed_count + ?'
    );
    expect(String(dailyStatsUpdate?.[0])).toContain(
      'store_memory_completed_count = store_memory_completed_count + ?'
    );
  });

  it('session duration を session start から completed day の stats に加算する', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-01T12:00:10.000Z'));

    const startedAtMs = new Date('2026-06-01T12:00:00.000Z').getTime();
    const { exec, sql } = createMockSql((query) => {
      if (query.includes('SELECT started_at_ms')) {
        return createCursor([
          {
            started_at_ms: startedAtMs,
          },
        ]);
      }

      return createCursor([]);
    });
    const archive = new SqliteEchoEventArchive({
      sql,
    });
    const event: EchoEvent = {
      type: 'session.completed',
      category: 'session',
      severity: 'warn',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session completed',
      payload: {
        terminationReason: 'max_turns',
        totalTokens: 900,
      },
    };

    await archive.recordEvent(event, {
      sessionId: 'session-1',
    });

    const dailyStatsUpdate = exec.mock.calls.find(([query]) => {
      return String(query).includes('UPDATE echo_action_analysis_daily_stats');
    });
    expect(dailyStatsUpdate).toEqual([
      expect.stringContaining('event_count = event_count + ?'),
      1,
      1,
      900,
      1,
      1,
      10_000,
      1,
      '2026-06-01',
    ]);
    expect(String(dailyStatsUpdate?.[0])).toContain(
      'completed_session_count = completed_session_count + ?'
    );
    expect(String(dailyStatsUpdate?.[0])).toContain(
      'total_session_duration_ms = total_session_duration_ms + ?'
    );
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
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('DELETE FROM echo_action_analysis_daily_stats'),
      '2026-03-05',
    ]);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining(
        'DELETE FROM echo_action_analysis_tool_daily_stats'
      ),
      '2026-03-05',
    ]);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('DELETE FROM echo_action_analysis_sessions'),
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
      200,
    ]);
  });

  it('action analysis は raw event ではなく日次 stats から取得する', () => {
    const { exec, sql } = createMockSql((query) => {
      const sqlText = query;
      if (sqlText.includes('FROM echo_action_analysis_daily_stats')) {
        return createCursor([
          createDailyStatsRow({
            archive_day: '2026-06-01',
            event_count: 8,
            session_count: 1,
            completed_session_count: 1,
            total_tokens: 900,
            total_session_duration_ms: 10_000,
            session_duration_count: 1,
            total_turns: 1,
            no_tool_call_turns: 1,
            tool_call_count: 1,
            tool_completed_count: 2,
            tool_failed_count: 1,
            memory_search_completed_count: 1,
            memory_search_zero_result_count: 1,
            store_memory_completed_count: 1,
          }),
        ]);
      }
      if (sqlText.includes('FROM echo_action_analysis_tool_daily_stats')) {
        return createCursor([
          createToolStatsRow({
            archive_day: '2026-06-01',
            tool_name: 'search_memory',
            called_count: 1,
            completed_count: 1,
          }),
          createToolStatsRow({
            archive_day: '2026-06-01',
            tool_name: 'read_chat_messages',
            failed_count: 1,
          }),
        ]);
      }

      return createCursor([]);
    });
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    const result = archive.getRecentActionAnalysisEvents({
      now: new Date('2026-06-02T18:00:00.000Z'),
      days: 7,
    });

    expect(result).toEqual({
      days: 7,
      startArchiveDay: '2026-05-28',
      endArchiveDay: '2026-06-03',
      eventCount: 8,
      events: [],
      metrics: {
        sessionCount: 1,
        completedSessionCount: 1,
        failedSessionCount: 0,
        warningSessionCount: 0,
        maxTurnsSessionCount: 0,
        totalTokens: 900,
        totalSessionDurationMs: 10_000,
        sessionDurationCount: 1,
        totalTurns: 1,
        noToolCallTurns: 1,
        toolCallCount: 1,
        toolCompletedCount: 2,
        toolFailedCount: 1,
        topTools: [
          {
            toolName: 'search_memory',
            calledCount: 1,
            completedCount: 1,
            failedCount: 0,
          },
          {
            toolName: 'read_chat_messages',
            calledCount: 0,
            completedCount: 0,
            failedCount: 1,
          },
        ],
        memorySearchCompletedCount: 1,
        memorySearchFailedCount: 0,
        memorySearchZeroResultCount: 1,
        memorySearchFinalResultTotal: 0,
        storeMemoryCompletedCount: 1,
      },
    });
    expect(
      exec.mock.calls.some(
        ([query]) =>
          String(query).includes('FROM echo_events') &&
          String(query).includes('json_extract')
      )
    ).toBe(false);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('FROM echo_action_analysis_daily_stats'),
      '2026-05-28',
      '2026-06-03',
    ]);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('FROM echo_action_analysis_tool_daily_stats'),
      '2026-05-28',
      '2026-06-03',
    ]);
  });

  it('action analysis の複数 period を 1 回の stats 取得から作る', () => {
    const { exec, sql } = createMockSql((query) => {
      const sqlText = query;
      if (sqlText.includes('FROM echo_action_analysis_daily_stats')) {
        return createCursor([
          createDailyStatsRow({
            archive_day: '2026-06-04',
            event_count: 2,
            session_count: 1,
            completed_session_count: 1,
            total_tokens: 100,
          }),
          createDailyStatsRow({
            archive_day: '2026-05-06',
            event_count: 3,
            session_count: 2,
            failed_session_count: 1,
          }),
        ]);
      }
      if (sqlText.includes('FROM echo_action_analysis_tool_daily_stats')) {
        return createCursor([
          createToolStatsRow({
            archive_day: '2026-06-04',
            tool_name: 'search_memory',
            completed_count: 1,
          }),
          createToolStatsRow({
            archive_day: '2026-05-06',
            tool_name: 'read_chat_messages',
            failed_count: 1,
          }),
        ]);
      }

      return createCursor([]);
    });
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    const ranges = archive.getRecentActionAnalysisEventRanges({
      now: new Date('2026-06-03T18:00:00.000Z'),
      periodDays: [1, 30],
    });

    expect(ranges[0]).toEqual({
      days: 1,
      startArchiveDay: '2026-06-04',
      endArchiveDay: '2026-06-04',
      eventCount: 2,
      events: [],
      metrics: {
        sessionCount: 1,
        completedSessionCount: 1,
        failedSessionCount: 0,
        warningSessionCount: 0,
        maxTurnsSessionCount: 0,
        totalTokens: 100,
        totalSessionDurationMs: 0,
        sessionDurationCount: 0,
        totalTurns: 0,
        noToolCallTurns: 0,
        toolCallCount: 0,
        toolCompletedCount: 0,
        toolFailedCount: 0,
        topTools: [
          {
            toolName: 'search_memory',
            calledCount: 0,
            completedCount: 1,
            failedCount: 0,
          },
        ],
        memorySearchCompletedCount: 0,
        memorySearchFailedCount: 0,
        memorySearchZeroResultCount: 0,
        memorySearchFinalResultTotal: 0,
        storeMemoryCompletedCount: 0,
      },
    });
    expect(ranges[1]).toMatchObject({
      days: 30,
      startArchiveDay: '2026-05-06',
      endArchiveDay: '2026-06-04',
      eventCount: 5,
      metrics: {
        sessionCount: 3,
        completedSessionCount: 1,
        failedSessionCount: 1,
        totalTokens: 100,
        topTools: [
          {
            toolName: 'search_memory',
            calledCount: 0,
            completedCount: 1,
            failedCount: 0,
          },
          {
            toolName: 'read_chat_messages',
            calledCount: 0,
            completedCount: 0,
            failedCount: 1,
          },
        ],
      },
    });
    expect(ranges[1]?.events).toEqual([]);
    expect(
      exec.mock.calls.filter(([query]) =>
        String(query).includes('FROM echo_action_analysis_daily_stats')
      )
    ).toHaveLength(1);
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining('FROM echo_action_analysis_daily_stats'),
      '2026-05-06',
      '2026-06-04',
    ]);
  });

  it('dashboard session logs の件数上限を表示候補 event に適用する', () => {
    const activityEventTypes = [
      'session.started',
      'session.completed',
      'session.failed',
      'model.turn.completed',
      'model.output.emitted',
      'model.provider.warning',
      'tool.called',
      'tool.completed',
      'tool.failed',
      'memory.search.completed',
      'memory.search.failed',
      'cognitive.phase.committed',
      'cognitive.phase.failed',
    ];
    const activityEventTypeSqlList = activityEventTypes
      .map((type) => `'${type}'`)
      .join(', ');
    const { exec, sql } = createMockSql();
    const archive = new SqliteEchoEventArchive({
      sql,
    });

    const result = archive.getTodayEvents({
      now: new Date('2026-06-02T18:00:00.000Z'),
      limit: 50,
    });

    expect(result).toEqual({
      archiveDay: '2026-06-03',
      events: [],
    });
    const eventQuery = exec.mock.calls.find(([query]) => {
      return String(query).includes('FROM echo_events');
    });
    expect(eventQuery).toEqual([
      expect.stringContaining(`AND type IN (${activityEventTypeSqlList})`),
      '2026-06-03',
      50,
    ]);
    expect(String(eventQuery?.[0])).toContain('AND session_id IS NOT NULL');
    expect(String(eventQuery?.[0])).toContain(
      'ORDER BY created_at_ms DESC\n         LIMIT ?'
    );
    expect(exec.mock.calls).toContainEqual([
      expect.stringContaining(
        'CREATE INDEX IF NOT EXISTS idx_echo_events_archive_day_dashboard_activity_created'
      ),
    ]);
    const activityEventIndexQuery = exec.mock.calls.find(([query]) => {
      return String(query).includes(
        'idx_echo_events_archive_day_dashboard_activity_created'
      );
    });
    expect(String(activityEventIndexQuery?.[0])).toContain(
      `AND type IN (${activityEventTypeSqlList})`
    );
    expect(String(activityEventIndexQuery?.[0])).toContain(
      'WHERE session_id IS NOT NULL'
    );
  });
});
