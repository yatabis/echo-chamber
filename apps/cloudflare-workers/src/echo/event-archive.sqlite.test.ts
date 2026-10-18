import { runInDurableObject } from 'cloudflare:test';
import { env } from 'cloudflare:workers';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';

import { SqliteEchoEventArchive } from './event-archive';

import type {
  EchoEventArchiveExportCursor,
  EchoEventArchiveExportLine,
} from './event-archive-export';

const TARGET_DAY_TIME = '2026-08-15T03:00:00.000Z';
const TARGET_DAY_LATER_TIME = '2026-08-15T03:00:00.001Z';
const PREVIOUS_DAY_TIME = '2026-08-14T03:00:00.000Z';
const NEXT_DAY_TIME = '2026-08-16T03:00:00.000Z';
const TARGET_ARCHIVE_DAY = '2026-08-15';

interface StoredEventFixture {
  at: string;
  summary: string;
  payload?: Record<string, unknown>;
}

/**
 * 実 DO SQLite 上の archive で検証を実行する。
 * test ごとに別 DO を使い、保存済み row を共有しない。
 */
async function withArchive<T>(
  callback: (archive: SqliteEchoEventArchive) => Promise<T> | T
): Promise<T> {
  const stub = env.ECHO.get(env.ECHO.newUniqueId());

  return await runInDurableObject(stub, async (_instance, state) => {
    return await callback(
      new SqliteEchoEventArchive({ sql: state.storage.sql })
    );
  });
}

/**
 * production と同じ保存経路で event を記録する。
 */
async function recordEvents(
  archive: SqliteEchoEventArchive,
  fixtures: readonly StoredEventFixture[]
): Promise<void> {
  for (const fixture of fixtures) {
    vi.setSystemTime(new Date(fixture.at));
    const event: EchoEvent = {
      type: 'tool.called',
      category: 'tool',
      severity: 'info',
      streams: ['thought', 'analysis'],
      summary: fixture.summary,
      payload: fixture.payload ?? { order: fixture.summary },
    };
    // eslint-disable-next-line no-await-in-loop -- 保存順を rowid の順序として固定する。
    await archive.recordEvent(event, { sessionId: 'session-1' });
  }
}

/**
 * archive day の全ページを読み、行と各ページの rowsRead を返す。
 */
function readAllPages(
  archive: SqliteEchoEventArchive,
  input: { limit: number; maxBodyChars?: number }
): { lines: EchoEventArchiveExportLine[]; rowsReadPerPage: number[] } {
  const lines: EchoEventArchiveExportLine[] = [];
  const rowsReadPerPage: number[] = [];
  let after: EchoEventArchiveExportCursor | null = null;

  for (let page = 0; page < 1000; page += 1) {
    const result = archive.exportEventsByArchiveDay({
      archiveDay: TARGET_ARCHIVE_DAY,
      after,
      ...input,
    });
    rowsReadPerPage.push(result.rowsRead);
    lines.push(
      ...result.lines.map(
        (line) => JSON.parse(line) as EchoEventArchiveExportLine
      )
    );
    if (result.nextCursor === null) {
      return { lines, rowsReadPerPage };
    }
    after = result.nextCursor;
  }

  throw new Error('export did not finish');
}

describe('SqliteEchoEventArchive export', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('指定 archive day の event だけを、保存順に漏れなく重複なく返す', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });

    const lines = await withArchive(async (archive) => {
      await recordEvents(archive, [
        { at: PREVIOUS_DAY_TIME, summary: 'previous-day' },
        ...Array.from({ length: 7 }, (_, index) => ({
          // 同じ作成時刻の event を複数含め、ページ境界をまたがせる。
          at: TARGET_DAY_TIME,
          summary: `target-${index}`,
        })),
        { at: TARGET_DAY_LATER_TIME, summary: 'target-7' },
        { at: TARGET_DAY_LATER_TIME, summary: 'target-8' },
        { at: NEXT_DAY_TIME, summary: 'next-day' },
      ]);

      return readAllPages(archive, { limit: 3 }).lines;
    });

    expect(lines.map((line) => line.summary)).toEqual(
      Array.from({ length: 9 }, (_, index) => `target-${index}`)
    );
    expect(lines.every((line) => line.archiveDay === TARGET_ARCHIVE_DAY)).toBe(
      true
    );
  });

  it('保存した streams と payload を JSON として読み戻せる形で返す', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });

    const [line] = await withArchive(async (archive) => {
      await recordEvents(archive, [
        {
          at: TARGET_DAY_TIME,
          summary: '改行\nと "引用符" を含む要約',
          payload: { content: '思考\n本文', nested: { values: [1, null] } },
        },
      ]);

      return readAllPages(archive, { limit: 10 }).lines;
    });

    expect(line).toEqual({
      id: expect.any(String) as string,
      createdAtMs: new Date(TARGET_DAY_TIME).getTime(),
      archiveDay: TARGET_ARCHIVE_DAY,
      sessionId: 'session-1',
      type: 'tool.called',
      category: 'tool',
      severity: 'info',
      streams: ['thought', 'analysis'],
      summary: '改行\nと "引用符" を含む要約',
      payload: { content: '思考\n本文', nested: { values: [1, null] } },
    });
  });

  it('他の日の保存量や同じ作成時刻の件数に関係なく、1ページの読み取り行数をページ件数程度に抑える', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });

    const result = await withArchive(async (archive) => {
      await recordEvents(archive, [
        ...Array.from({ length: 60 }, (_, index) => ({
          at: PREVIOUS_DAY_TIME,
          summary: `previous-${index}`,
        })),
        ...Array.from({ length: 30 }, (_, index) => ({
          at: TARGET_DAY_TIME,
          summary: `target-${index}`,
        })),
        ...Array.from({ length: 60 }, (_, index) => ({
          at: NEXT_DAY_TIME,
          summary: `next-${index}`,
        })),
      ]);

      return readAllPages(archive, { limit: 10 });
    });

    expect(result.lines).toHaveLength(30);
    // 続きの有無を確かめる 1 行と、同じ時刻の残りを読み切ったときの範囲終端の確認 1 行だけを許す。
    // 他の日の 120 行や、同じ作成時刻に並ぶ 30 行には比例しない。
    expect(Math.max(...result.rowsReadPerPage)).toBeLessThanOrEqual(10 + 2);
  });

  it('本文量の上限に達したら件数上限より手前でページを区切る', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    const largeText = 'x'.repeat(1_000);

    const result = await withArchive(async (archive) => {
      await recordEvents(
        archive,
        Array.from({ length: 5 }, (_, index) => ({
          at: TARGET_DAY_TIME,
          summary: `large-${index}`,
          payload: { content: largeText },
        }))
      );
      const readFirstPage = (
        maxBodyChars: number
      ): ReturnType<SqliteEchoEventArchive['exportEventsByArchiveDay']> =>
        archive.exportEventsByArchiveDay({
          archiveDay: TARGET_ARCHIVE_DAY,
          after: null,
          limit: 10,
          maxBodyChars,
        });
      // 各行は同じ長さなので、1 行と改行の長さから 2 行ちょうどの本文量を求める。
      const lineLength = readFirstPage(1).lines[0]?.length ?? 0;
      const twoLinesChars = (lineLength + 1) * 2;

      return {
        exactlyTwoLines: readFirstPage(twoLinesChars),
        belowTwoLines: readFirstPage(twoLinesChars - 1),
        allPages: readAllPages(archive, {
          limit: 10,
          maxBodyChars: twoLinesChars,
        }),
      };
    });

    expect(result.exactlyTwoLines.lines).toHaveLength(2);
    expect(result.exactlyTwoLines.nextCursor).not.toBeNull();
    expect(result.belowTwoLines.lines).toHaveLength(1);
    expect(result.allPages.lines.map((line) => line.summary)).toEqual(
      Array.from({ length: 5 }, (_, index) => `large-${index}`)
    );
  });

  it('1件だけで本文量の上限を超える event も、読み出しを止めずに1件ずつ返す', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });

    const result = await withArchive(async (archive) => {
      await recordEvents(
        archive,
        Array.from({ length: 2 }, (_, index) => ({
          at: TARGET_DAY_TIME,
          summary: `oversized-${index}`,
          payload: { content: 'y'.repeat(5_000) },
        }))
      );

      return readAllPages(archive, { limit: 10, maxBodyChars: 1_000 });
    });

    expect(result.lines.map((line) => line.summary)).toEqual([
      'oversized-0',
      'oversized-1',
    ]);
  });

  it('event の無い archive day は空ページで終わる', async () => {
    const page = await withArchive((archive) => {
      return archive.exportEventsByArchiveDay({
        archiveDay: TARGET_ARCHIVE_DAY,
        after: null,
        limit: 10,
      });
    });

    expect(page).toEqual({ lines: [], nextCursor: null, rowsRead: 0 });
  });
});
