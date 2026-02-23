import { formatDate } from '../utils/datetime';

import type {
  DashboardUsageDays,
  DashboardUsagePoint,
  DashboardUsageTotals,
} from './types';
import type { Note, UsageRecord } from '../echo/types';

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

/**
 * UsageRecord から Dashboard 表示用の時系列配列を構築する。
 *
 * 指定期間内に usage が存在しない日は `usage: null` で補完するため、
 * フロント側は欠損日を意識せず同一長の配列を描画できる。
 *
 * @param usageRecord 日付キーごとの usage 集計
 * @param days 表示対象の日数（7 or 30）
 * @param referenceDate 基準日時（省略時は現在時刻）
 * @returns DashboardUsagePoint の日付昇順配列
 */
export function buildUsageSeries(
  usageRecord: UsageRecord,
  days: DashboardUsageDays,
  referenceDate: Date = new Date()
): DashboardUsagePoint[] {
  return buildUsageDateKeys(days, referenceDate).map((dateKey) => ({
    dateKey,
    usage: usageRecord[dateKey] ?? null,
  }));
}

/**
 * usage 時系列を合算し、ヘッダー表示用の合計値を返す。
 *
 * `usage: null` の要素はスキップして、存在する日のみ加算する。
 *
 * @param series `buildUsageSeries()` が返す配列
 * @returns 総トークン数と総コスト
 */
export function sumUsage(series: DashboardUsagePoint[]): DashboardUsageTotals {
  return series.reduce<DashboardUsageTotals>(
    (total, point) => {
      if (point.usage === null) {
        return total;
      }

      return {
        totalTokens: total.totalTokens + point.usage.total_tokens,
        totalCost: total.totalCost + point.usage.total_cost,
      };
    },
    {
      totalTokens: 0,
      totalCost: 0,
    }
  );
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
