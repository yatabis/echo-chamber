import { formatDate } from '@echo-chamber/core/utils/datetime';

import type {
  DashboardUsageBreakdownTotals,
  DashboardUsageDays,
  DashboardUsageRatioMetrics,
  DashboardUsageStackedPoint,
  Note,
  TokenUsage,
  Usage,
  UsageModelBreakdown,
  UsageRecord,
} from './types';

const USAGE_DAY_BOUNDARY_OFFSET_HOURS = 7;
const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const TOKENS_PER_MILLION = 1_000_000;

/**
 * 100 万 tokens あたりの model 価格定義。
 */
interface ModelPricing {
  inputPerMillionUsd: number;
  cachedInputPerMillionUsd: number | null;
  cacheWriteInputPerMillionUsd: number | null;
  outputPerMillionUsd: number;
}

/**
 * usage 内訳から計算した推定コスト。
 */
interface UsageCostEstimate {
  costUsd: number | null;
  isPartial: boolean;
}

const OPENAI_MODEL_PRICING: Record<string, ModelPricing> = {
  'gpt-5.6': {
    inputPerMillionUsd: 5,
    cachedInputPerMillionUsd: 0.5,
    cacheWriteInputPerMillionUsd: 6.25,
    outputPerMillionUsd: 30,
  },
  'gpt-5.6-sol': {
    inputPerMillionUsd: 5,
    cachedInputPerMillionUsd: 0.5,
    cacheWriteInputPerMillionUsd: 6.25,
    outputPerMillionUsd: 30,
  },
  'gpt-5.6-terra': {
    inputPerMillionUsd: 2.5,
    cachedInputPerMillionUsd: 0.25,
    cacheWriteInputPerMillionUsd: 3.125,
    outputPerMillionUsd: 15,
  },
  'gpt-5.6-luna': {
    inputPerMillionUsd: 1,
    cachedInputPerMillionUsd: 0.1,
    cacheWriteInputPerMillionUsd: 1.25,
    outputPerMillionUsd: 6,
  },
  'gpt-5.5': {
    inputPerMillionUsd: 5,
    cachedInputPerMillionUsd: 0.5,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 30,
  },
  'gpt-5.4': {
    inputPerMillionUsd: 2.5,
    cachedInputPerMillionUsd: 0.25,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 15,
  },
  'gpt-5.4-mini': {
    inputPerMillionUsd: 0.75,
    cachedInputPerMillionUsd: 0.075,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 4.5,
  },
  'gpt-5.2': {
    inputPerMillionUsd: 1.75,
    cachedInputPerMillionUsd: 0.175,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 14,
  },
  'gpt-5.1': {
    inputPerMillionUsd: 1.25,
    cachedInputPerMillionUsd: 0.125,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 10,
  },
  'gpt-5': {
    inputPerMillionUsd: 1.25,
    cachedInputPerMillionUsd: 0.125,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 10,
  },
  'gpt-5-mini': {
    inputPerMillionUsd: 0.25,
    cachedInputPerMillionUsd: 0.025,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 2,
  },
  'gpt-5-nano': {
    inputPerMillionUsd: 0.05,
    cachedInputPerMillionUsd: 0.005,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 0.4,
  },
  'gpt-4.1': {
    inputPerMillionUsd: 2,
    cachedInputPerMillionUsd: 0.5,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 8,
  },
  'gpt-4.1-mini': {
    inputPerMillionUsd: 0.4,
    cachedInputPerMillionUsd: 0.1,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 1.6,
  },
  'gpt-4.1-nano': {
    inputPerMillionUsd: 0.1,
    cachedInputPerMillionUsd: 0.025,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 0.4,
  },
  'gpt-4o': {
    inputPerMillionUsd: 2.5,
    cachedInputPerMillionUsd: 1.25,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 10,
  },
  'gpt-4o-mini': {
    inputPerMillionUsd: 0.15,
    cachedInputPerMillionUsd: 0.075,
    cacheWriteInputPerMillionUsd: null,
    outputPerMillionUsd: 0.6,
  },
};

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
    referenceDate.getTime() - USAGE_DAY_BOUNDARY_OFFSET_HOURS * HOUR_MS
  );

  return Array.from({ length: days }).map((_, index) => {
    const date = new Date(
      shiftedReferenceDate.getTime() - (days - 1 - index) * DAY_MS
    );
    return formatDate(date);
  });
}

/**
 * token usage の数値整合性を検証する。
 *
 * @param dateKey 検証対象の usage 日付キー
 * @param usage 検証対象の token usage
 * @param path エラー表示用のフィールドパス
 */
function assertValidTokenUsage(
  dateKey: string,
  usage: TokenUsage,
  path = 'usage'
): void {
  const numericFields: [string, number][] = [
    ['cached_input_tokens', usage.cached_input_tokens],
    ['cache_write_input_tokens', usage.cache_write_input_tokens],
    ['uncached_input_tokens', usage.uncached_input_tokens],
    ['total_input_tokens', usage.total_input_tokens],
    ['output_tokens', usage.output_tokens],
    ['reasoning_tokens', usage.reasoning_tokens],
    ['total_tokens', usage.total_tokens],
  ];

  for (const [fieldName, value] of numericFields) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `Invalid usage value for ${dateKey}: ${path}.${fieldName} is not finite`
      );
    }
    if (value < 0) {
      throw new Error(
        `Invalid usage value for ${dateKey}: ${path}.${fieldName} is negative`
      );
    }
  }

  if (usage.reasoning_tokens > usage.output_tokens) {
    throw new Error(
      `Invalid usage value for ${dateKey}: ${path}.reasoning_tokens exceeds output_tokens`
    );
  }

  if (
    usage.cached_input_tokens +
      usage.cache_write_input_tokens +
      usage.uncached_input_tokens !==
    usage.total_input_tokens
  ) {
    throw new Error(
      `Invalid usage value for ${dateKey}: ${path}.input token fields are inconsistent`
    );
  }

  if (usage.total_input_tokens + usage.output_tokens !== usage.total_tokens) {
    throw new Error(
      `Invalid usage value for ${dateKey}: ${path}.total_tokens is inconsistent with input/output`
    );
  }
}

/**
 * 日次 usage と provider/model 別内訳の整合性を検証する。
 *
 * @param dateKey 検証対象の usage 日付キー
 * @param usage 検証対象の日次 usage
 */
function assertValidUsage(dateKey: string, usage: Usage): void {
  assertValidTokenUsage(dateKey, usage);

  if (usage.by_model.length === 0) {
    throw new Error(`Invalid usage value for ${dateKey}: by_model is empty`);
  }

  for (const [index, breakdown] of usage.by_model.entries()) {
    assertValidTokenUsage(dateKey, breakdown, `by_model[${index}]`);
  }

  const breakdownTotal = usage.by_model.reduce<TokenUsage>(
    (total, breakdown) => {
      return {
        cached_input_tokens:
          total.cached_input_tokens + breakdown.cached_input_tokens,
        cache_write_input_tokens:
          total.cache_write_input_tokens + breakdown.cache_write_input_tokens,
        uncached_input_tokens:
          total.uncached_input_tokens + breakdown.uncached_input_tokens,
        total_input_tokens:
          total.total_input_tokens + breakdown.total_input_tokens,
        output_tokens: total.output_tokens + breakdown.output_tokens,
        reasoning_tokens: total.reasoning_tokens + breakdown.reasoning_tokens,
        total_tokens: total.total_tokens + breakdown.total_tokens,
      };
    },
    {
      cached_input_tokens: 0,
      cache_write_input_tokens: 0,
      uncached_input_tokens: 0,
      total_input_tokens: 0,
      output_tokens: 0,
      reasoning_tokens: 0,
      total_tokens: 0,
    }
  );

  if (
    breakdownTotal.cached_input_tokens !== usage.cached_input_tokens ||
    breakdownTotal.cache_write_input_tokens !==
      usage.cache_write_input_tokens ||
    breakdownTotal.uncached_input_tokens !== usage.uncached_input_tokens ||
    breakdownTotal.total_input_tokens !== usage.total_input_tokens ||
    breakdownTotal.output_tokens !== usage.output_tokens ||
    breakdownTotal.reasoning_tokens !== usage.reasoning_tokens ||
    breakdownTotal.total_tokens !== usage.total_tokens
  ) {
    throw new Error(
      `Invalid usage value for ${dateKey}: by_model totals are inconsistent`
    );
  }
}

/**
 * 価格表 lookup 用に model 名を正規化する。
 *
 * @param model usage 内訳に記録された model 名
 * @returns trim と小文字化を適用した model 名
 */
function normalizeModelName(model: string): string {
  return model.trim().toLowerCase();
}

/**
 * provider/model 別 usage 内訳に対応する価格定義を返す。
 *
 * @param breakdown provider/model 別 usage 内訳
 * @returns 対応する価格定義。未対応 provider/model なら `null`
 */
function resolveModelPricing(
  breakdown: UsageModelBreakdown
): ModelPricing | null {
  if (breakdown.provider !== 'openai') {
    return null;
  }

  return OPENAI_MODEL_PRICING[normalizeModelName(breakdown.model)] ?? null;
}

/**
 * provider/model 別 usage 内訳の推定 USD コストを計算する。
 *
 * @param breakdown provider/model 別 usage 内訳
 * @returns 推定 USD コスト。価格定義がない場合は `null`
 */
function estimateBreakdownCostUsd(
  breakdown: UsageModelBreakdown
): number | null {
  const pricing = resolveModelPricing(breakdown);
  if (pricing === null) {
    return null;
  }
  if (
    breakdown.cached_input_tokens > 0 &&
    pricing.cachedInputPerMillionUsd === null
  ) {
    return null;
  }
  if (
    breakdown.cache_write_input_tokens > 0 &&
    pricing.cacheWriteInputPerMillionUsd === null
  ) {
    return null;
  }

  return (
    (breakdown.uncached_input_tokens * pricing.inputPerMillionUsd +
      breakdown.cached_input_tokens *
        (pricing.cachedInputPerMillionUsd ?? pricing.inputPerMillionUsd) +
      breakdown.cache_write_input_tokens *
        (pricing.cacheWriteInputPerMillionUsd ?? pricing.inputPerMillionUsd) +
      breakdown.output_tokens * pricing.outputPerMillionUsd) /
    TOKENS_PER_MILLION
  );
}

/**
 * model/provider 別内訳から、現在の価格表に基づく推定 USD コストを返す。
 *
 * 未対応 provider/model が含まれる場合でも、計算できる内訳は合算し、
 * 不明な内訳が混ざったことを `isPartial` で返す。
 *
 * @param usage 日次 usage
 * @returns 推定 USD コストと部分推定かどうか
 */
export function estimateUsageCostUsd(usage: Usage): UsageCostEstimate {
  let knownCostUsd = 0;
  let hasKnownCost = false;
  let isPartial = false;

  for (const breakdown of usage.by_model) {
    const cost = estimateBreakdownCostUsd(breakdown);
    if (cost === null) {
      isPartial = true;
      continue;
    }

    hasKnownCost = true;
    knownCostUsd += cost;
  }

  return {
    costUsd: hasKnownCost ? knownCostUsd : null,
    isPartial,
  };
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
        cacheWriteInputTokens: 0,
        uncachedInputTokens: 0,
        normalOutputTokens: 0,
        reasoningOutputTokens: 0,
        totalInputTokens: 0,
        totalOutputTokens: 0,
        totalTokens: 0,
        estimatedCostUsd: 0,
        estimatedCostIsPartial: false,
      };
    }

    assertValidUsage(dateKey, usage);
    const costEstimate = estimateUsageCostUsd(usage);

    return {
      dateKey,
      cachedInputTokens: usage.cached_input_tokens,
      cacheWriteInputTokens: usage.cache_write_input_tokens,
      uncachedInputTokens: usage.uncached_input_tokens,
      normalOutputTokens: usage.output_tokens - usage.reasoning_tokens,
      reasoningOutputTokens: usage.reasoning_tokens,
      totalInputTokens: usage.total_input_tokens,
      totalOutputTokens: usage.output_tokens,
      totalTokens: usage.total_tokens,
      estimatedCostUsd: costEstimate.costUsd,
      estimatedCostIsPartial: costEstimate.isPartial,
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
  let estimatedCostUsd = 0;
  let hasKnownCost = false;
  const totals: DashboardUsageBreakdownTotals = {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: 0,
    normalOutputTokens: 0,
    reasoningOutputTokens: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalTokens: 0,
    estimatedCostUsd: null,
    estimatedCostIsPartial: false,
  };

  for (const point of series) {
    totals.cachedInputTokens += point.cachedInputTokens;
    totals.cacheWriteInputTokens += point.cacheWriteInputTokens;
    totals.uncachedInputTokens += point.uncachedInputTokens;
    totals.normalOutputTokens += point.normalOutputTokens;
    totals.reasoningOutputTokens += point.reasoningOutputTokens;
    totals.totalInputTokens += point.totalInputTokens;
    totals.totalOutputTokens += point.totalOutputTokens;
    totals.totalTokens += point.totalTokens;
    totals.estimatedCostIsPartial ||= point.estimatedCostIsPartial;

    if (point.estimatedCostUsd !== null) {
      estimatedCostUsd += point.estimatedCostUsd;
      hasKnownCost ||= point.totalTokens > 0 || point.estimatedCostUsd > 0;
    }
  }

  return {
    ...totals,
    estimatedCostUsd:
      totals.totalTokens === 0 || hasKnownCost ? estimatedCostUsd : null,
  };
}

/**
 * 分母が 0 の場合に 0 を返す安全な比率計算。
 *
 * @param value 分子
 * @param base 分母
 * @returns `value / base`。分母が 0 の場合は 0
 */
function ratioOrZero(value: number, base: number): number {
  if (base === 0) {
    return 0;
  }
  return value / base;
}

/**
 * Dashboard 表示向けのトークン比率指標を作る。
 *
 * - cache read / cache write / uncached は `totalInputTokens` を分母にする
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
    cacheWriteRateInInput: ratioOrZero(
      totals.cacheWriteInputTokens,
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
