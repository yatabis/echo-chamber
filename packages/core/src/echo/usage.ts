import { formatDate } from '../utils/datetime';

import type { Usage, UsageRecord } from './types';
import type { ResponseUsage } from 'openai/resources/responses/responses';

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
 * OpenAI SDKのResponseUsageをEchoのUsageに変換する
 * @param usage OpenAI SDKのResponseUsage
 * @returns EchoのUsage
 */
export function convertUsage(usage: ResponseUsage): Usage {
  const {
    input_tokens,
    input_tokens_details,
    output_tokens,
    output_tokens_details,
    total_tokens,
  } = usage;
  const { cached_tokens } = input_tokens_details;
  const { reasoning_tokens } = output_tokens_details;

  const cached_input_tokens = cached_tokens;
  const uncached_input_tokens = input_tokens - cached_tokens;

  // See https://platform.openai.com/docs/pricing
  const total_cost =
    (cached_input_tokens * 0.125 +
      uncached_input_tokens * 1.25 +
      output_tokens * 10) /
    1_000_000;

  return {
    cached_input_tokens,
    uncached_input_tokens,
    total_input_tokens: input_tokens,
    output_tokens,
    reasoning_tokens,
    total_tokens,
    total_cost,
  };
}

/**
 * 現在時刻に基づいて動的なトークン使用制限を計算
 * Asia/Tokyoタイムゾーンで午前7時から翌午前3時までの20時間で按分（分単位、少数以下切り捨て）
 */
export function calculateDynamicTokenLimit(
  tokenLimit: number,
  bufferFactor = 1.0
): number {
  // Asia/Tokyoの現在時刻を取得（UTC+9）
  const now = new Date(Date.now() + 9 * 60 * 60 * 1000);
  const hours = now.getUTCHours() + now.getUTCMinutes() / 60;

  // 午前7時からの経過時間
  const elapsed = hours >= 7 ? hours - 7 : hours + 17;

  // 経過時間が20時間を超えた場合（午前3時から午前6時59分）は対象外として0トークンを返す
  if (elapsed > 20) {
    return 0;
  }

  // 現在時刻までの按分
  const rawTokenLimit = tokenLimit * (elapsed / 20);

  // バッファ適用（少数部切り捨て）
  const tokenLimitWithBuffer = Math.trunc(rawTokenLimit * bufferFactor);

  // 上限は制限を超えない
  return Math.min(tokenLimitWithBuffer, tokenLimit);
}
