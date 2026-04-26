import { formatDate } from '../utils/datetime';

import type { Usage, UsageRecord } from './types';
import type { ModelUsage } from '../ports/model';

const TOKYO_OFFSET_MS = 9 * 60 * 60 * 1000;
const MINUTES_PER_TOKEN_LIMIT_WINDOW = 20 * 60;
const USAGE_RESET_HOUR = 7;
const MINUTE_MS = 60 * 1000;

/**
 * 今日の日付キーを 'YYYY-MM-DD' 形式で取得
 * 午前7時を日付境界とし、7:00から翌6:59を同一日として扱う
 */
export function getTodayUsageKey(): string {
  return formatDate(new Date(Date.now() - 7 * 60 * 60 * 1000));
}

/**
 * 既存のUsageRecordに新しいUsageを加算する
 * @param usageRecord 既存のUsageRecord
 * @param key Usage加算するキー
 * @param usage 新しいUsage
 * @returns 更新されたUsageRecord
 */
export function addUsage(
  usageRecord: UsageRecord,
  key: string,
  usage: Usage
): UsageRecord {
  if (usageRecord[key]) {
    usageRecord[key].cached_input_tokens += usage.cached_input_tokens;
    usageRecord[key].uncached_input_tokens += usage.uncached_input_tokens;
    usageRecord[key].total_input_tokens += usage.total_input_tokens;
    usageRecord[key].output_tokens += usage.output_tokens;
    usageRecord[key].reasoning_tokens += usage.reasoning_tokens;
    usageRecord[key].total_tokens += usage.total_tokens;
    usageRecord[key].total_cost += usage.total_cost;
  } else {
    usageRecord[key] = usage;
  }

  return usageRecord;
}

/**
 * provider 非依存の ModelUsage を Echo の Usage に変換する
 * @param usage provider 正規化済みの usage
 * @returns EchoのUsage
 */
export function convertUsage(usage: ModelUsage): Usage {
  const cached_input_tokens = usage.cachedInputTokens;
  const uncached_input_tokens = usage.uncachedInputTokens;
  const total_input_tokens = usage.totalInputTokens;
  const output_tokens = usage.outputTokens;
  const reasoning_tokens = usage.reasoningTokens;
  const total_tokens = usage.totalTokens;

  // See https://platform.openai.com/docs/pricing
  const total_cost =
    (cached_input_tokens * 0.125 +
      uncached_input_tokens * 1.25 +
      output_tokens * 10) /
    1_000_000;

  return {
    cached_input_tokens,
    uncached_input_tokens,
    total_input_tokens,
    output_tokens,
    reasoning_tokens,
    total_tokens,
    total_cost,
  };
}

/**
 * Tokyo 時刻で token limit window の開始から何分経過したかを返す。
 * 対象外の時刻（03:01-06:59）は `null` を返す。
 *
 * @param now - 基準時刻
 * @returns token limit window 開始からの経過分。対象外時刻なら `null`
 */
function getElapsedTokenWindowMinutes(now: Date): number | null {
  const tokyoNow = new Date(now.getTime() + TOKYO_OFFSET_MS);
  const hours = tokyoNow.getUTCHours();
  const minutes = tokyoNow.getUTCMinutes();
  const elapsedMinutes =
    hours >= USAGE_RESET_HOUR
      ? (hours - USAGE_RESET_HOUR) * 60 + minutes
      : (hours + (24 - USAGE_RESET_HOUR)) * 60 + minutes;

  if (elapsedMinutes > MINUTES_PER_TOKEN_LIMIT_WINDOW) {
    return null;
  }

  return elapsedMinutes;
}

/**
 * usage 日付が切り替わる次の午前7時（Asia/Tokyo）を返す。
 *
 * @param now - 基準時刻
 * @returns 次回 usage reset 時刻
 */
function getNextUsageResetTime(now: Date): Date {
  const tokyoNow = new Date(now.getTime() + TOKYO_OFFSET_MS);
  const nextReset = new Date(
    Date.UTC(
      tokyoNow.getUTCFullYear(),
      tokyoNow.getUTCMonth(),
      tokyoNow.getUTCDate(),
      USAGE_RESET_HOUR,
      0,
      0,
      0
    ) - TOKYO_OFFSET_MS
  );

  if (tokyoNow.getUTCHours() >= USAGE_RESET_HOUR) {
    nextReset.setUTCDate(nextReset.getUTCDate() + 1);
  }

  return nextReset;
}

/**
 * 秒以下を切り捨てて分単位に正規化する。
 *
 * @param now - 基準時刻
 * @returns 秒以下を切り捨てた時刻
 */
function truncateToMinute(now: Date): Date {
  const truncated = new Date(now);
  truncated.setUTCSeconds(0, 0);
  return truncated;
}

/**
 * 指定時刻における動的なトークン使用制限を計算する。
 * Asia/Tokyoタイムゾーンで午前7時から翌午前3時までの20時間で按分（分単位、少数以下切り捨て）
 *
 * @param tokenLimit - 1日あたりのトークン上限
 * @param bufferFactor - 余裕係数
 * @param now - 評価対象の基準時刻（デフォルト: 現在時刻）
 */
export function calculateDynamicTokenLimit(
  tokenLimit: number,
  bufferFactor = 1.0,
  now: Date = new Date()
): number {
  const elapsedMinutes = getElapsedTokenWindowMinutes(now);
  if (elapsedMinutes == null) {
    return 0;
  }

  // 現在時刻までの按分
  const rawTokenLimit =
    tokenLimit * (elapsedMinutes / MINUTES_PER_TOKEN_LIMIT_WINDOW);

  // バッファ適用（少数部切り捨て）
  const tokenLimitWithBuffer = Math.trunc(rawTokenLimit * bufferFactor);

  // 上限は制限を超えない
  return Math.min(tokenLimitWithBuffer, tokenLimit);
}

/**
 * 現在の totalTokens を再び許容できる最短時刻を計算する。
 * 現 usage 日の中で回復しない場合は、usage がリセットされる次の午前7時（Asia/Tokyo）へフォールバックする。
 *
 * @param totalTokens - 現在の累積トークン数
 * @param tokenLimit - 1日あたりのトークン上限
 * @param bufferFactor - 余裕係数
 * @param now - 基準時刻（デフォルト: 現在時刻）
 * @returns totalTokens を許容できる最短の再開候補時刻
 */
export function findNextTokenLimitRecoveryTime(
  totalTokens: number,
  tokenLimit: number,
  bufferFactor = 1.0,
  now: Date = new Date()
): Date {
  const nextUsageReset = getNextUsageResetTime(now);
  const earliestCandidate = new Date(
    truncateToMinute(now).getTime() + MINUTE_MS
  );

  // 現在時刻の次の分が reset 以降なら、同じ usage 日の中では再開できない。
  if (earliestCandidate >= nextUsageReset) {
    return nextUsageReset;
  }

  const earliestCandidateElapsedMinutes =
    getElapsedTokenWindowMinutes(earliestCandidate);
  // 03:01-06:59 は token limit window 外なので、次の 07:00 reset に任せる。
  if (
    earliestCandidateElapsedMinutes == null ||
    tokenLimit <= 0 ||
    bufferFactor <= 0
  ) {
    return nextUsageReset;
  }

  // 実行許可条件は totalTokens < dynamicLimit なので、同値は回復扱いしない。
  const requiredDynamicLimit = Math.floor(totalTokens) + 1;
  if (requiredDynamicLimit > tokenLimit) {
    return nextUsageReset;
  }

  // dynamicLimit = trunc(tokenLimit * bufferFactor * elapsedMinutes / 1200)
  // となるため、requiredDynamicLimit を満たす最小の elapsedMinutes を直接求める。
  const tokenGrowthPerMinute =
    (tokenLimit * bufferFactor) / MINUTES_PER_TOKEN_LIMIT_WINDOW;
  const requiredElapsedMinutes = Math.ceil(
    requiredDynamicLimit / tokenGrowthPerMinute - Number.EPSILON
  );
  // すでに必要 limit を満たしている場合でも、過去の時刻は返さず次の分以降に丸める。
  const recoveryElapsedMinutes = Math.max(
    requiredElapsedMinutes,
    earliestCandidateElapsedMinutes
  );

  if (recoveryElapsedMinutes > MINUTES_PER_TOKEN_LIMIT_WINDOW) {
    return nextUsageReset;
  }

  const tokenWindowStart = new Date(
    earliestCandidate.getTime() - earliestCandidateElapsedMinutes * MINUTE_MS
  );
  const recoveryTime = new Date(
    tokenWindowStart.getTime() + recoveryElapsedMinutes * MINUTE_MS
  );

  return recoveryTime < nextUsageReset ? recoveryTime : nextUsageReset;
}
