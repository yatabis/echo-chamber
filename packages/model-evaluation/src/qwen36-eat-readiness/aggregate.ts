import { formatTimeBudgetKey, TIME_BUDGETS_MS } from './scoring';

import type { RuntimeScenarioResult, RuntimeWorkflowResult } from './types';

interface AggregateCategoryScore {
  earned: number;
  possible: number;
}

export interface RuntimeAggregate {
  finalScore: AggregateCategoryScore;
  byCategory: Record<string, AggregateCategoryScore>;
  equalTimeAchievement: Record<string, AggregateCategoryScore>;
  finishThinkingCount: number;
  maxTurnsCount: number;
  errorCount: number;
  noToolCallTurnCount: number;
  toolFailureCount: number;
  elapsedMs: number;
  outputTokens: number;
  endToEndOutputTokensPerSecond: number;
}

export interface WorkflowAggregate {
  finalScore: AggregateCategoryScore;
  byCategory: Record<string, AggregateCategoryScore>;
  equalTimeAchievement: Record<string, AggregateCategoryScore>;
  workflowCount: number;
  sessionCount: number;
  finishThinkingSessionCount: number;
  maxTurnsSessionCount: number;
  errorSessionCount: number;
  toolFailureCount: number;
  elapsedMs: number;
  outputTokens: number;
  endToEndOutputTokensPerSecond: number;
}

function safeRate(tokens: number, elapsedMs: number): number {
  return elapsedMs <= 0 ? 0 : tokens / (elapsedMs / 1_000);
}

/**
 * E.C.H.O.シナリオを最終得点、同一時間到達点、運用失敗へ集約する。
 */
export function aggregateRuntimeResults(
  results: readonly RuntimeScenarioResult[]
): RuntimeAggregate {
  const finalScore = results.reduce(
    (total, result) => ({
      earned: total.earned + result.score.earned,
      possible: total.possible + result.score.possible,
    }),
    { earned: 0, possible: 0 }
  );
  const categories = ['outcome', 'protocol', 'completion', 'safety'];
  const byCategory = Object.fromEntries(
    categories.map((category) => [
      category,
      results.reduce(
        (total, result) => {
          const score =
            result.score.byCategory[
              category as keyof typeof result.score.byCategory
            ];
          return {
            earned: total.earned + score.earned,
            possible: total.possible + score.possible,
          };
        },
        { earned: 0, possible: 0 }
      ),
    ])
  );
  const equalTimeAchievement = Object.fromEntries(
    TIME_BUDGETS_MS.map((budgetMs) => {
      const key = formatTimeBudgetKey(budgetMs);
      return [
        key,
        {
          earned: results.reduce(
            (total, result) =>
              total + (result.score.timeComparableEarned[key] ?? 0),
            0
          ),
          possible: results.reduce(
            (total, result) => total + result.score.timeComparablePossible,
            0
          ),
        },
      ];
    })
  );
  const elapsedMs = results.reduce(
    (total, result) => total + result.elapsedMs,
    0
  );
  const outputTokens = results.reduce(
    (total, result) => total + result.usage.outputTokens,
    0
  );

  return {
    finalScore,
    byCategory,
    equalTimeAchievement,
    finishThinkingCount: results.filter(
      (result) => result.terminationReason === 'finish_thinking'
    ).length,
    maxTurnsCount: results.filter(
      (result) => result.terminationReason === 'max_turns'
    ).length,
    errorCount: results.filter((result) => result.terminationReason === 'error')
      .length,
    noToolCallTurnCount: results.reduce(
      (total, result) =>
        total +
        result.events.filter((event) => {
          if (event.type !== 'model.turn.completed') {
            return false;
          }
          const warnings = event.payload?.warnings;
          return Array.isArray(warnings) && warnings.includes('no_tool_calls');
        }).length,
      0
    ),
    toolFailureCount: results.reduce(
      (total, result) =>
        total +
        result.events.filter((event) => event.type === 'tool.failed').length,
      0
    ),
    elapsedMs,
    outputTokens,
    endToEndOutputTokensPerSecond: safeRate(outputTokens, elapsedMs),
  };
}

/** 複数セッションworkflowを成果、時間到達点、運用失敗へ集約する。 */
export function aggregateWorkflowResults(
  results: readonly RuntimeWorkflowResult[]
): WorkflowAggregate {
  const categories = ['outcome', 'protocol', 'completion', 'safety'];
  const sessions = results.flatMap((result) => result.sessions);
  const finalScore = results.reduce(
    (total, result) => ({
      earned: total.earned + result.score.earned,
      possible: total.possible + result.score.possible,
    }),
    { earned: 0, possible: 0 }
  );
  const byCategory = Object.fromEntries(
    categories.map((category) => [
      category,
      results.reduce(
        (total, result) => {
          const score =
            result.score.byCategory[
              category as keyof typeof result.score.byCategory
            ];
          return {
            earned: total.earned + score.earned,
            possible: total.possible + score.possible,
          };
        },
        { earned: 0, possible: 0 }
      ),
    ])
  );
  const equalTimeAchievement = Object.fromEntries(
    TIME_BUDGETS_MS.map((budgetMs) => {
      const key = formatTimeBudgetKey(budgetMs);
      return [
        key,
        {
          earned: results.reduce(
            (total, result) =>
              total + (result.score.timeComparableEarned[key] ?? 0),
            0
          ),
          possible: results.reduce(
            (total, result) => total + result.score.timeComparablePossible,
            0
          ),
        },
      ];
    })
  );
  const elapsedMs = results.reduce(
    (total, result) => total + result.elapsedMs,
    0
  );
  const outputTokens = sessions.reduce(
    (total, session) => total + session.usage.outputTokens,
    0
  );

  return {
    finalScore,
    byCategory,
    equalTimeAchievement,
    workflowCount: results.length,
    sessionCount: sessions.length,
    finishThinkingSessionCount: sessions.filter(
      (session) => session.terminationReason === 'finish_thinking'
    ).length,
    maxTurnsSessionCount: sessions.filter(
      (session) => session.terminationReason === 'max_turns'
    ).length,
    errorSessionCount: sessions.filter(
      (session) => session.terminationReason === 'error'
    ).length,
    toolFailureCount: sessions.reduce(
      (total, session) =>
        total +
        session.events.filter((event) => event.type === 'tool.failed').length,
      0
    ),
    elapsedMs,
    outputTokens,
    endToEndOutputTokensPerSecond: safeRate(outputTokens, elapsedMs),
  };
}
