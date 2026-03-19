import type { Note, Usage, UsageRecord } from '@echo-chamber/core/echo/types';
import { formatDate } from '@echo-chamber/core/utils/datetime';

import type {
  DashboardUsageBreakdownTotals,
  DashboardUsageDays,
  DashboardUsageRatioMetrics,
  DashboardUsageStackedPoint,
} from './types';

const USAGE_DAY_BOUNDARY_OFFSET_HOURS = 7;

/**
 * Dashboard 用の usage 期間キー配列を生成する。
 *
 * Echo は「午前 7 時」を日次の境界として usage を集計するため、
 * `referenceDate` から 7 時間引いた時刻を基準日にしてキーを並べる。
 *
 * @param days 表示対象の日数（7 or 30）
 * @param referenceDate 基準日時（通常は現在時刻）
 * @returns 先頭が最古日・末尾が最新日となる `YYYY-MM-DD` キー配列
 */
function buildUsageDateKeys(
  days: DashboardUsageDays,
  referenceDate: Date
): string[] {
  const shiftedReferenceDate = new Date(
    referenceDate.getTime() - USAGE_DAY_BOUNDARY_OFFSET_HOURS * 60 * 60 * 1000
  );
  const baseDate = new Date(
    shiftedReferenceDate.getFullYear(),
    shiftedReferenceDate.getMonth(),
    shiftedReferenceDate.getDate()
  );

  return Array.from({ length: days }).map((_, index) => {
    const date = new Date(baseDate);
    date.setDate(baseDate.getDate() - (days - 1 - index));
    return formatDate(date);
  });
}

function assertValidUsage(dateKey: string, usage: Usage): void {
  const numericFields: [string, number][] = [
    ['cached_input_tokens', usage.cached_input_tokens],
    ['uncached_input_tokens', usage.uncached_input_tokens],
    ['total_input_tokens', usage.total_input_tokens],
    ['output_tokens', usage.output_tokens],
    ['reasoning_tokens', usage.reasoning_tokens],
    ['total_tokens', usage.total_tokens],
    ['total_cost', usage.total_cost],
  ];

  for (const [fieldName, value] of numericFields) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `Invalid usage value for ${dateKey}: ${fieldName} is not finite`
      );
    }
    if (value < 0) {
      throw new Error(
        `Invalid usage value for ${dateKey}: ${fieldName} is negative`
      );
    }
  }

  if (usage.reasoning_tokens > usage.output_tokens) {
    throw new Error(
      `Invalid usage value for ${dateKey}: reasoning_tokens exceeds output_tokens`
    );
  }

  if (
    usage.cached_input_tokens + usage.uncached_input_tokens !==
    usage.total_input_tokens
  ) {
    throw new Error(
      `Invalid usage value for ${dateKey}: input token fields are inconsistent`
    );
  }

  if (usage.total_input_tokens + usage.output_tokens !== usage.total_tokens) {
    throw new Error(
      `Invalid usage value for ${dateKey}: total_tokens is inconsistent with input/output`
    );
  }
}

/**
 * UsageRecord から Dashboard の積み上げ棒表示用系列を構築する。
 *
 * 出力は `cached input` / `uncached input` / `normal output` / `reasoning output`
 * の 4 区分で返し、欠損日は 0 埋めで補完する。
 *
 * @param usageRecord 日付キーごとの usage 集計
 * @param days 表示対象の日数（7 or 30）
 * @param referenceDate 基準日時（省略時は現在時刻）
 * @returns DashboardUsageStackedPoint の日付昇順配列
 */
export function buildUsageStackedSeries(
  usageRecord: UsageRecord,
  days: DashboardUsageDays,
  referenceDate: Date = new Date()
): DashboardUsageStackedPoint[] {
  return buildUsageDateKeys(days, referenceDate).map((dateKey) => {
    const usage = usageRecord[dateKey];
    if (usage === undefined) {
      return {
        dateKey,
        cachedInputTokens: 0,
        uncachedInputTokens: 0,
        normalOutputTokens: 0,
        reasoningOutputTokens: 0,
        totalInputTokens: 0,
        totalOutputTokens: 0,
        totalTokens: 0,
        totalCost: 0,
      };
    }

    assertValidUsage(dateKey, usage);

    return {
      dateKey,
      cachedInputTokens: usage.cached_input_tokens,
      uncachedInputTokens: usage.uncached_input_tokens,
      normalOutputTokens: usage.output_tokens - usage.reasoning_tokens,
      reasoningOutputTokens: usage.reasoning_tokens,
      totalInputTokens: usage.total_input_tokens,
      totalOutputTokens: usage.output_tokens,
      totalTokens: usage.total_tokens,
      totalCost: usage.total_cost,
    };
  });
}

/**
 * 積み上げ系列を期間合計へ集約する。
 *
 * @param series `buildUsageStackedSeries()` の出力
 * @returns 区分別トークン数と期間合計トークン/コスト
 */
export function sumUsageBreakdown(
  series: DashboardUsageStackedPoint[]
): DashboardUsageBreakdownTotals {
  return series.reduce<DashboardUsageBreakdownTotals>(
    (total, point) => {
      return {
        cachedInputTokens: total.cachedInputTokens + point.cachedInputTokens,
        uncachedInputTokens:
          total.uncachedInputTokens + point.uncachedInputTokens,
        normalOutputTokens: total.normalOutputTokens + point.normalOutputTokens,
        reasoningOutputTokens:
          total.reasoningOutputTokens + point.reasoningOutputTokens,
        totalInputTokens: total.totalInputTokens + point.totalInputTokens,
        totalOutputTokens: total.totalOutputTokens + point.totalOutputTokens,
        totalTokens: total.totalTokens + point.totalTokens,
        totalCost: total.totalCost + point.totalCost,
      };
    },
    {
      cachedInputTokens: 0,
      uncachedInputTokens: 0,
      normalOutputTokens: 0,
      reasoningOutputTokens: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalTokens: 0,
      totalCost: 0,
    }
  );
}

function ratioOrZero(value: number, base: number): number {
  if (base === 0) {
    return 0;
  }
  return value / base;
}

/**
 * Dashboard 表示向けのトークン比率指標を作る。
 *
 * - cache/uncached は `totalInputTokens` を分母にする
 * - input/output は `totalTokens` を分母にする
 *
 * @param totals 期間合計
 * @returns 0.0 - 1.0 の比率群
 */
export function buildUsageRatioMetrics(
  totals: DashboardUsageBreakdownTotals
): DashboardUsageRatioMetrics {
  return {
    cacheRateInInput: ratioOrZero(
      totals.cachedInputTokens,
      totals.totalInputTokens
    ),
    uncachedRateInInput: ratioOrZero(
      totals.uncachedInputTokens,
      totals.totalInputTokens
    ),
    inputRateInTotal: ratioOrZero(totals.totalInputTokens, totals.totalTokens),
    outputRateInTotal: ratioOrZero(
      totals.totalOutputTokens,
      totals.totalTokens
    ),
  };
}

/**
 * ノート一覧をクエリ文字列で部分一致フィルタする。
 *
 * 比較は大文字小文字を区別せず、タイトルまたは本文に一致すれば採用する。
 * クエリが空（trim 後に長さ 0）の場合は入力配列をそのまま返す。
 *
 * @param notes フィルタ対象ノート一覧
 * @param query 検索クエリ
 * @returns フィルタ後のノート一覧
 */
export function filterNotes(notes: Note[], query: string): Note[] {
  const normalizedQuery = query.trim().toLowerCase();

  if (normalizedQuery.length === 0) {
    return notes;
  }

  return notes.filter((note) => {
    return (
      note.title.toLowerCase().includes(normalizedQuery) ||
      note.content.toLowerCase().includes(normalizedQuery)
    );
  });
}
