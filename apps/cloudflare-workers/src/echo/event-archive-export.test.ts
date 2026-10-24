import { describe, expect, it } from 'vitest';

import {
  ECHO_EVENT_ARCHIVE_EXPORT_DEFAULT_LIMIT,
  ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT,
  formatEchoEventArchiveExportCursor,
  parseEchoEventArchiveExportCursor,
  parseEchoEventArchiveExportQuery,
} from './event-archive-export';

/**
 * query string から export 条件を解釈する。
 */
function parseQuery(
  query: string
): ReturnType<typeof parseEchoEventArchiveExportQuery> {
  return parseEchoEventArchiveExportQuery(new URLSearchParams(query));
}

describe('parseEchoEventArchiveExportQuery', () => {
  it('archive day だけを指定すると先頭ページを既定件数で読む', () => {
    expect(parseQuery('day=2026-08-15')).toEqual({
      ok: true,
      query: {
        archiveDay: '2026-08-15',
        after: null,
        limit: ECHO_EVENT_ARCHIVE_EXPORT_DEFAULT_LIMIT,
      },
    });
  });

  it('続きの cursor と件数を受け取る', () => {
    expect(
      parseQuery('day=2026-08-15&after=1786000000000:42&limit=50')
    ).toEqual({
      ok: true,
      query: {
        archiveDay: '2026-08-15',
        after: { createdAtMs: 1786000000000, rowId: 42 },
        limit: 50,
      },
    });
  });

  it('上限を超える件数は上限に丸める', () => {
    expect(parseQuery('day=2026-08-15&limit=100000')).toMatchObject({
      ok: true,
      query: { limit: ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT },
    });
  });

  it.each([
    ['', 'day'],
    ['day=2026-8-15', 'day'],
    ['day=2026-02-30', 'day'],
    ['day=2026-08-15&limit=0', 'limit'],
    ['day=2026-08-15&limit=1.5', 'limit'],
    ['day=2026-08-15&limit=abc', 'limit'],
    ['day=2026-08-15&after=', 'after'],
    ['day=2026-08-15&after=123', 'after'],
    ['day=2026-08-15&after=-1:2', 'after'],
    ['day=2026-08-15&after=99999999999999999:1', 'after'],
  ])('不正な条件 %s は %s を理由に拒否する', (query, field) => {
    const result = parseQuery(query);

    expect(result.ok).toBe(false);
    expect(result.ok ? '' : result.error).toContain(field);
  });
});

describe('export cursor', () => {
  it('format した cursor を同じ位置として読み戻せる', () => {
    const cursor = { createdAtMs: 1786000000123, rowId: 9876 };

    expect(
      parseEchoEventArchiveExportCursor(
        formatEchoEventArchiveExportCursor(cursor)
      )
    ).toEqual(cursor);
  });
});
