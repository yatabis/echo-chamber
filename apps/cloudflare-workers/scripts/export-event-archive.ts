/**
 * Hosted Echo の DO SQLite に保存された event を、archive day ごとの NDJSON としてローカルへ取り出す。
 *
 * 実行例:
 *   ECHO_ARCHIVE_BASE_URL=https://<production-hostname> \
 *   CF_ACCESS_TOKEN="$(cloudflared access token -app=https://<production-hostname>)" \
 *   pnpm archive:export
 *
 * 取り出した内容には私的な会話が含まれるため、出力先は Git 管理外に置く。
 * Node.js の型除去でそのまま実行するため、実行時に意味を持つ TypeScript 固有構文は使わない。
 */
import {
  appendFile,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  truncate,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import {
  ECHO_EVENT_ARCHIVE_EXPORT_HEADERS,
  ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT,
} from '../src/echo/event-archive-export.ts';

const DAY_MS = 24 * 60 * 60 * 1000;
/** Worker の保持期間と同じ日数。これより古い archive day は日次 sleep 中に削除される。 */
const RETENTION_DAYS = 90;
/**
 * JST 03:00 を archive day の境界にするため、UTC 時刻を 6 時間進めた日付を archive day とみなす。
 * Worker 側の `getEventArchiveDay` と同じ規則。
 */
const ARCHIVE_DAY_OFFSET_MS = 6 * 60 * 60 * 1000;
/**
 * 1 回の実行で読む rows read の既定上限。
 * Free plan の 1 日 5,000,000 rows read は本番 Echo と共有しており、使い切るとその日の
 * DO 操作が失敗する。本番の通常運用の分を残すため、上限の半分未満で止める。
 */
const DEFAULT_MAX_ROWS_READ = 2_000_000;
/** 1 ページで件数のほかに読みうる行数（続きの確認と範囲終端の確認）。 */
const ROWS_READ_OVERHEAD_PER_PAGE = 2;
const MAX_ATTEMPTS = 4;
const ARCHIVE_DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../..'
);

interface ExportConfig {
  baseUrl: URL;
  instances: string[];
  fromDay: string;
  toDay: string;
  outputDir: string;
  maxRowsRead: number;
  authHeaders: Record<string, string>;
}

/** 途中まで取り出した archive day の再開位置。 */
interface DayProgress {
  cursor: string;
  /** cursor までの行を書き終えた時点の partial file の byte 数。 */
  bytes: number;
}

interface FetchedPage {
  body: string;
  eventCount: number;
  rowsRead: number;
  nextCursor: string | null;
}

/** 再試行で回復しうる失敗。 */
class RetryableExportError extends Error {
  override readonly name = 'RetryableExportError';
}

/** 実行中に読んだ rows read の合計。上限に達したら残りの取り出しを止める。 */
class RowsReadBudget {
  private readonly max: number;
  private used = 0;

  /** @param max この実行で読んでよい rows read */
  constructor(max: number) {
    this.max = max;
  }

  /** 次のページを読んでも上限を超えないか判定する。 */
  canRead(limit: number): boolean {
    return this.used + limit + ROWS_READ_OVERHEAD_PER_PAGE <= this.max;
  }

  /** response header で報告された rows read を加える。 */
  add(rowsRead: number): void {
    this.used += rowsRead;
  }

  /** これまでに読んだ rows read。 */
  get total(): number {
    return this.used;
  }
}

/**
 * 環境変数から実行条件を読む。
 */
function readConfig(env: NodeJS.ProcessEnv, now: Date): ExportConfig {
  return {
    baseUrl: new URL(readRequired(env, 'ECHO_ARCHIVE_BASE_URL')),
    instances: (env.ECHO_ARCHIVE_INSTANCES ?? 'rin,marie')
      .split(',')
      .map((instance) => instance.trim())
      .filter((instance) => instance !== ''),
    fromDay: readArchiveDay(
      env.ECHO_ARCHIVE_FROM_DAY,
      new Date(now.getTime() - RETENTION_DAYS * DAY_MS)
    ),
    // 当日分はまだ書き込まれ続けるため、既定では前日までを取り出す。
    toDay: readArchiveDay(
      env.ECHO_ARCHIVE_TO_DAY,
      new Date(now.getTime() - DAY_MS)
    ),
    outputDir:
      env.ECHO_ARCHIVE_OUTPUT_DIR ??
      path.join(REPO_ROOT, '.artifacts/hosted-archive'),
    maxRowsRead: readPositiveInteger(
      env.ECHO_ARCHIVE_MAX_ROWS_READ,
      DEFAULT_MAX_ROWS_READ
    ),
    authHeaders: readAuthHeaders(env),
  };
}

/** 必須の環境変数を読む。 */
function readRequired(env: NodeJS.ProcessEnv, name: string): string {
  const value = env[name];
  if (value === undefined || value === '') {
    throw new Error(`${name} is required`);
  }

  return value;
}

/** 指定された archive day、または既定の時刻が属する archive day を返す。 */
function readArchiveDay(value: string | undefined, fallback: Date): string {
  if (value === undefined) {
    return toArchiveDay(fallback);
  }
  if (!ARCHIVE_DAY_PATTERN.test(value)) {
    throw new Error(`archive day must be YYYY-MM-DD: ${value}`);
  }

  return value;
}

/** 正の整数の設定値を読む。 */
function readPositiveInteger(
  value: string | undefined,
  fallback: number
): number {
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`expected a positive integer: ${value}`);
  }

  return parsed;
}

/**
 * Cloudflare Access を通すための header を選ぶ。
 * `cloudflared access token` の app token、または Access service token を受け付ける。
 */
function readAuthHeaders(env: NodeJS.ProcessEnv): Record<string, string> {
  const appToken = env.CF_ACCESS_TOKEN ?? '';
  if (appToken !== '') {
    return { 'cf-access-token': appToken };
  }

  const clientId = env.CF_ACCESS_CLIENT_ID ?? '';
  const clientSecret = env.CF_ACCESS_CLIENT_SECRET ?? '';
  if (clientId !== '' && clientSecret !== '') {
    return {
      'CF-Access-Client-Id': clientId,
      'CF-Access-Client-Secret': clientSecret,
    };
  }

  throw new Error(
    'Set CF_ACCESS_TOKEN, or CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET'
  );
}

/** 時刻が属する archive day を返す。 */
function toArchiveDay(date: Date): string {
  return new Date(date.getTime() + ARCHIVE_DAY_OFFSET_MS)
    .toISOString()
    .slice(0, 10);
}

/** `fromDay` から `toDay` までの archive day を古い順に列挙する。 */
function listArchiveDays(fromDay: string, toDay: string): string[] {
  const days: string[] = [];
  const end = Date.parse(`${toDay}T00:00:00.000Z`);
  for (
    let time = Date.parse(`${fromDay}T00:00:00.000Z`);
    time <= end;
    time += DAY_MS
  ) {
    days.push(new Date(time).toISOString().slice(0, 10));
  }

  return days;
}

/** ファイルが存在するか判定する。 */
async function exists(filePath: string): Promise<boolean> {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * 1 ページを取得する。一時的な失敗は件数を減らして再試行する。
 * Free plan の CPU 時間を超えた場合も 5xx になるため、件数を減らすと通りやすくなる。
 */
async function fetchPage(
  config: ExportConfig,
  request: { instance: string; day: string; cursor: string | null }
): Promise<FetchedPage> {
  let limit = ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT;
  for (let attempt = 1; ; attempt += 1) {
    const url = new URL(`${request.instance}/archive/events`, config.baseUrl);
    url.searchParams.set('day', request.day);
    url.searchParams.set('limit', String(limit));
    if (request.cursor !== null) {
      url.searchParams.set('after', request.cursor);
    }

    try {
      // eslint-disable-next-line no-await-in-loop -- 失敗したページだけを順に再試行する。
      return await requestPage(url, config.authHeaders);
    } catch (error) {
      if (attempt >= MAX_ATTEMPTS || !(error instanceof RetryableExportError)) {
        throw error;
      }
      limit = Math.max(1, Math.floor(limit / 2));
      console.warn(
        `retry ${request.instance} ${request.day} with limit=${limit}: ${error.message}`
      );
      // eslint-disable-next-line no-await-in-loop -- 再試行の間隔を空ける。
      await sleep(1000 * 2 ** attempt);
    }
  }
}

/**
 * export endpoint を 1 回呼び、本文と metadata を検証して返す。
 */
async function requestPage(
  url: URL,
  authHeaders: Record<string, string>
): Promise<FetchedPage> {
  let response: Response;
  try {
    // Access の login 画面への redirect を成功と取り違えないよう、redirect を追わない。
    response = await fetch(url, { headers: authHeaders, redirect: 'manual' });
  } catch (error) {
    throw new RetryableExportError(
      error instanceof Error ? error.message : String(error)
    );
  }

  if (response.status >= 500) {
    throw new RetryableExportError(`HTTP ${response.status}`);
  }
  const contentType = response.headers.get('content-type') ?? '';
  if (
    response.status !== 200 ||
    !contentType.includes('application/x-ndjson')
  ) {
    throw new Error(
      `Unexpected response HTTP ${response.status} (${contentType}). Check Cloudflare Access credentials.`
    );
  }

  const body = await response.text();
  const eventCount = Number(
    response.headers.get(ECHO_EVENT_ARCHIVE_EXPORT_HEADERS.eventCount)
  );
  assertNdjsonLines(body, eventCount);

  return {
    body,
    eventCount,
    rowsRead: Number(
      response.headers.get(ECHO_EVENT_ARCHIVE_EXPORT_HEADERS.rowsRead)
    ),
    nextCursor: response.headers.get(
      ECHO_EVENT_ARCHIVE_EXPORT_HEADERS.nextCursor
    ),
  };
}

/**
 * 本文の各行が JSON として読め、件数が header と一致することを確かめる。
 * 途中で切れた本文を保存して、完成したファイルとして扱わないようにする。
 */
function assertNdjsonLines(body: string, expectedCount: number): void {
  if (body !== '' && !body.endsWith('\n')) {
    throw new RetryableExportError('NDJSON body is truncated');
  }
  const lines = body === '' ? [] : body.slice(0, -1).split('\n');
  if (lines.length !== expectedCount) {
    throw new RetryableExportError(
      `NDJSON line count ${lines.length} does not match header ${expectedCount}`
    );
  }
  for (const line of lines) {
    JSON.parse(line);
  }
}

/** partial file と progress から再開位置を決める。 */
async function resumeDay(
  partialPath: string,
  progressPath: string
): Promise<DayProgress | null> {
  if (!(await exists(progressPath))) {
    await rm(partialPath, { force: true });
    return null;
  }

  const progress = JSON.parse(
    await readFile(progressPath, 'utf8')
  ) as DayProgress;
  // progress の記録後に追記した行は、再開時に同じ行を重複して書かないよう切り詰める。
  await truncate(partialPath, progress.bytes);
  return progress;
}

/** progress を一時ファイル経由で置き換え、書きかけの progress を残さない。 */
async function writeProgress(
  progressPath: string,
  progress: DayProgress
): Promise<void> {
  const temporaryPath = `${progressPath}.tmp`;
  await writeFile(temporaryPath, JSON.stringify(progress));
  await rename(temporaryPath, progressPath);
}

/**
 * 1 個体の 1 archive day を取り出す。
 *
 * @returns 完了したら true。rows read の上限で中断したら false
 */
async function exportDay(
  config: ExportConfig,
  target: { instance: string; day: string },
  budget: RowsReadBudget
): Promise<boolean> {
  const dayDir = path.join(config.outputDir, target.instance);
  const completePath = path.join(dayDir, `${target.day}.ndjson`);
  if (await exists(completePath)) {
    return true;
  }

  await mkdir(dayDir, { recursive: true });
  const partialPath = `${completePath}.partial`;
  const progressPath = `${completePath}.progress.json`;
  const resumed = await resumeDay(partialPath, progressPath);
  let cursor = resumed?.cursor ?? null;
  let bytes = resumed?.bytes ?? 0;
  let exportedEvents = 0;
  // event の無い archive day でも、完了したことを空のファイルで残す。
  await writeFile(partialPath, '', { flag: 'a' });

  for (;;) {
    if (!budget.canRead(ECHO_EVENT_ARCHIVE_EXPORT_MAX_LIMIT)) {
      return false;
    }
    // eslint-disable-next-line no-await-in-loop -- 続きの位置は前のページの response から決まる。
    const page = await fetchPage(config, { ...target, cursor });
    budget.add(page.rowsRead);
    // eslint-disable-next-line no-await-in-loop -- 追記してから progress を進める順序を保つ。
    await appendFile(partialPath, page.body);
    bytes += Buffer.byteLength(page.body);
    exportedEvents += page.eventCount;
    if (page.nextCursor === null) {
      break;
    }
    cursor = page.nextCursor;
    // eslint-disable-next-line no-await-in-loop -- 追記済みの位置だけを再開位置にする。
    await writeProgress(progressPath, { cursor, bytes });
  }

  await rename(partialPath, completePath);
  await rm(progressPath, { force: true });
  console.log(
    `${target.instance} ${target.day}: ${exportedEvents} events in this run (rows read so far ${budget.total})`
  );
  return true;
}

/**
 * 古い archive day から順に、各個体の event を取り出す。
 * 古い日は毎日削除されていくため、日を外側のループにする。
 */
async function main(): Promise<void> {
  const config = readConfig(process.env, new Date());
  const budget = new RowsReadBudget(config.maxRowsRead);
  console.log(
    `export ${config.instances.join(',')} ${config.fromDay}..${config.toDay} to ${config.outputDir}`
  );

  for (const day of listArchiveDays(config.fromDay, config.toDay)) {
    for (const instance of config.instances) {
      // eslint-disable-next-line no-await-in-loop -- rows read の上限を順に確かめながら進める。
      const completed = await exportDay(config, { instance, day }, budget);
      if (!completed) {
        console.log(
          `Stopped at ${instance} ${day}: rows read ${budget.total} reached the limit. Re-run after 00:00 UTC to resume.`
        );
        return;
      }
    }
  }

  console.log(`done: rows read ${budget.total}`);
}

await main();
