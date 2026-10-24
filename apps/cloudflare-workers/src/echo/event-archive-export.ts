/**
 * 保存済み Echo event を archive day ごとに取り出す export の入出力定義。
 *
 * Worker runtime の型に依存させず、取り出し用のローカルスクリプトからも
 * 同じ定義を参照できるようにする。
 */

/** 1 ページで返す event 数の既定値。 */
export const ECHO_EVENT_ARCHIVE_EXPORT_DEFAULT_LIMIT = 200;

/** 1 ページで返す event 数の上限。これを超える指定は上限に丸める。 */
export const ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT = 500;

/**
 * 1 ページの NDJSON 本文の目安上限（UTF-16 code unit 数）。
 *
 * Main の model exchange は会話履歴を含む全文を持つため、件数だけでは本文量を抑えられない。
 * Free plan の CPU 時間と DO の memory に収めるため、件数とは別に本文量でもページを区切る。
 */
export const ECHO_EVENT_ARCHIVE_EXPORT_MAX_BODY_CHARS = 2_000_000;

/** export response の metadata を載せる header 名。 */
export const ECHO_EVENT_ARCHIVE_EXPORT_HEADERS = {
  nextCursor: 'x-echo-archive-next-cursor',
  rowsRead: 'x-echo-archive-rows-read',
  eventCount: 'x-echo-archive-event-count',
} as const;

/**
 * archive day 内の読み出し位置。
 * 同じ作成時刻の event が並んでも順序が一意になるよう、SQLite rowid を併用する。
 */
export interface EchoEventArchiveExportCursor {
  createdAtMs: number;
  rowId: number;
}

/** 1 ページ分の export 条件。 */
export interface EchoEventArchiveExportQuery {
  archiveDay: string;
  after: EchoEventArchiveExportCursor | null;
  limit: number;
}

/** export 条件の解釈結果。 */
export type EchoEventArchiveExportQueryResult =
  | { ok: true; query: EchoEventArchiveExportQuery }
  | { ok: false; error: string };

/**
 * NDJSON の 1 行が表す保存済み Echo event。
 * `streams` と `payload` は保存時の JSON をそのまま返すため、型を限定しない。
 */
export interface EchoEventArchiveExportLine {
  id: string;
  createdAtMs: number;
  archiveDay: string;
  sessionId: string | null;
  type: string;
  category: string;
  severity: string;
  streams: unknown;
  summary: string;
  payload: unknown;
}

const ARCHIVE_DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const POSITIVE_INTEGER_PATTERN = /^[1-9]\d*$/;
const CURSOR_PATTERN = /^(\d+):(\d+)$/;

/**
 * query string から 1 ページ分の export 条件を解釈する。
 *
 * @param params request の query string
 * @returns 解釈できた条件、または拒否理由
 */
export function parseEchoEventArchiveExportQuery(
  params: URLSearchParams
): EchoEventArchiveExportQueryResult {
  const archiveDay = params.get('day');
  if (archiveDay === null || !isValidArchiveDay(archiveDay)) {
    return { ok: false, error: 'day must be a valid YYYY-MM-DD archive day' };
  }

  const limit = parseLimit(params.get('limit'));
  if (limit === null) {
    return { ok: false, error: 'limit must be a positive integer' };
  }

  const afterValue = params.get('after');
  const after =
    afterValue === null ? null : parseEchoEventArchiveExportCursor(afterValue);
  if (afterValue !== null && after === null) {
    return { ok: false, error: 'after must be a cursor returned by export' };
  }

  return { ok: true, query: { archiveDay, after, limit } };
}

/**
 * 読み出し位置を response header 用の文字列にする。
 *
 * @param cursor 最後に返した event の位置
 * @returns `<createdAtMs>:<rowId>` 形式の cursor
 */
export function formatEchoEventArchiveExportCursor(
  cursor: EchoEventArchiveExportCursor
): string {
  return `${cursor.createdAtMs}:${cursor.rowId}`;
}

/**
 * `formatEchoEventArchiveExportCursor` が作った文字列を読み出し位置へ戻す。
 *
 * @param value cursor 文字列
 * @returns 読み出し位置。形式が不正なら null
 */
export function parseEchoEventArchiveExportCursor(
  value: string
): EchoEventArchiveExportCursor | null {
  const match = CURSOR_PATTERN.exec(value);
  if (match === null) {
    return null;
  }

  const createdAtMs = Number(match[1]);
  const rowId = Number(match[2]);
  if (!Number.isSafeInteger(createdAtMs) || !Number.isSafeInteger(rowId)) {
    return null;
  }

  return { createdAtMs, rowId };
}

/**
 * 形式だけでなく、暦の上で存在する日付かを確かめる。
 */
function isValidArchiveDay(value: string): boolean {
  if (!ARCHIVE_DAY_PATTERN.test(value)) {
    return false;
  }

  const date = new Date(`${value}T00:00:00.000Z`);
  return (
    !Number.isNaN(date.getTime()) && date.toISOString().slice(0, 10) === value
  );
}

/**
 * 件数指定を解釈し、上限を超える値は上限に丸める。
 *
 * @returns 件数。不正な指定なら null
 */
function parseLimit(value: string | null): number | null {
  if (value === null) {
    return ECHO_EVENT_ARCHIVE_EXPORT_DEFAULT_LIMIT;
  }
  if (!POSITIVE_INTEGER_PATTERN.test(value)) {
    return null;
  }

  return Math.min(Number(value), ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT);
}
