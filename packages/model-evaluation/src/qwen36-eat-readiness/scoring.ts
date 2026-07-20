import type {
  EvaluationCategory,
  EvaluationCheck,
  ScoreSummary,
} from './types';

export const TIME_BUDGETS_MS = [5_000, 15_000, 30_000, 60_000, 120_000];

const EVALUATION_CATEGORIES: readonly EvaluationCategory[] = [
  'outcome',
  'protocol',
  'completion',
  'safety',
];

/**
 * ミリ秒の比較窓を結果JSONで安定したキーへ変換する。
 */
export function formatTimeBudgetKey(milliseconds: number): string {
  return `${milliseconds / 1_000}s`;
}

/**
 * 個別条件を総得点、カテゴリ別得点、同一時間内の到達得点へ集約する。
 */
export function summarizeChecks(
  checks: readonly EvaluationCheck[]
): ScoreSummary {
  const byCategory = Object.fromEntries(
    EVALUATION_CATEGORIES.map((category) => [
      category,
      { earned: 0, possible: 0 },
    ])
  ) as ScoreSummary['byCategory'];

  for (const check of checks) {
    byCategory[check.category].possible += check.weight;
    if (check.passed) {
      byCategory[check.category].earned += check.weight;
    }
  }

  const timeComparableChecks = checks.filter((check) => check.timeComparable);
  const timeComparableEarned = Object.fromEntries(
    TIME_BUDGETS_MS.map((budgetMs) => [
      formatTimeBudgetKey(budgetMs),
      timeComparableChecks.reduce((total, check) => {
        if (
          !check.passed ||
          check.firstSatisfiedMs === null ||
          check.firstSatisfiedMs > budgetMs
        ) {
          return total;
        }

        return total + check.weight;
      }, 0),
    ])
  );

  return {
    earned: checks.reduce(
      (total, check) => total + (check.passed ? check.weight : 0),
      0
    ),
    possible: checks.reduce((total, check) => total + check.weight, 0),
    byCategory,
    timeComparableEarned,
    timeComparablePossible: timeComparableChecks.reduce(
      (total, check) => total + check.weight,
      0
    ),
  };
}
