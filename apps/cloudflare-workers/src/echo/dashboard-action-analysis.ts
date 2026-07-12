import type {
  DashboardActionAnalysisPeriod,
  DashboardActionAnalysisPeriodDays,
  DashboardActionAnalysisResponse,
  DashboardActionAnalysisToolSummary,
} from '@echo-chamber/contracts/dashboard/types';

export const DASHBOARD_ACTION_ANALYSIS_PERIOD_DAYS = [1, 7, 30] as const;

export interface DashboardActionAnalysisEvent {
  archiveDay: string;
  createdAt: string;
  sessionId: string | null;
  type: string;
  severity: 'debug' | 'info' | 'warn' | 'error';
  totalTokens?: number;
  terminationReason?: string;
  toolName?: string;
  finalResultCount?: number;
  warnings: readonly string[];
}

export interface DashboardActionAnalysisEventRange {
  days: number;
  startArchiveDay: string;
  endArchiveDay: string;
  eventCount: number;
  events: readonly DashboardActionAnalysisEvent[];
  metrics?: DashboardActionAnalysisMetrics;
}

export interface DashboardActionAnalysisToolMetrics {
  toolName: string;
  calledCount: number;
  completedCount: number;
  failedCount: number;
}

export interface DashboardActionAnalysisMetrics {
  sessionCount: number;
  completedSessionCount: number;
  failedSessionCount: number;
  warningSessionCount: number;
  maxTurnsSessionCount: number;
  totalTokens: number;
  totalSessionDurationMs: number;
  sessionDurationCount: number;
  totalTurns: number;
  noToolCallTurns: number;
  toolCallCount: number;
  toolCompletedCount: number;
  toolFailedCount: number;
  topTools: readonly DashboardActionAnalysisToolMetrics[];
  memorySearchCompletedCount: number;
  memorySearchFailedCount: number;
  memorySearchZeroResultCount: number;
  memorySearchFinalResultTotal: number;
  storeMemoryCompletedCount: number;
}

interface DashboardActionAnalysisInput {
  archiveDay: string;
  generatedAt: string;
  periods: readonly DashboardActionAnalysisEventRange[];
}

interface MutableToolSummary {
  toolName: string;
  calledCount: number;
  completedCount: number;
  failedCount: number;
}

interface SessionTimes {
  startedAt?: string;
  finishedAt?: string;
}

interface ActionAnalysisAccumulator {
  toolSummaries: Map<string, MutableToolSummary>;
  sessionIds: Set<string>;
  sessionTimes: Map<string, SessionTimes>;
  completedSessionCount: number;
  failedSessionCount: number;
  warningSessionCount: number;
  maxTurnsSessionCount: number;
  totalTokens: number;
  totalTurns: number;
  noToolCallTurns: number;
  toolCallCount: number;
  toolCompletedCount: number;
  toolFailedCount: number;
  memorySearchCompletedCount: number;
  memorySearchFailedCount: number;
  memorySearchZeroResultCount: number;
  memorySearchFinalResultTotal: number;
  storeMemoryCompletedCount: number;
}

/**
 * archive event range 群を dashboard 用 action analysis response へ射影する。
 *
 * @param input archive day と period ごとの raw event range
 * @returns dashboard がそのまま描画できる行動分析 response
 */
export function buildDashboardActionAnalysisResponse(
  input: DashboardActionAnalysisInput
): DashboardActionAnalysisResponse {
  return {
    archiveDay: input.archiveDay,
    generatedAt: input.generatedAt,
    periods: input.periods.map(buildDashboardActionAnalysisPeriod),
  };
}

/**
 * 1 period 分の raw event を行動指標へ集計する。
 */
function buildDashboardActionAnalysisPeriod(
  period: DashboardActionAnalysisEventRange
): DashboardActionAnalysisPeriod {
  if (period.metrics !== undefined) {
    return createDashboardActionAnalysisPeriodFromMetrics(
      period,
      period.metrics
    );
  }

  const accumulator = createActionAnalysisAccumulator();

  for (const event of period.events) {
    consumeActionAnalysisEvent(accumulator, event);
  }

  return createDashboardActionAnalysisPeriod(period, accumulator);
}

/**
 * write 時に作られた集計済み metrics を dashboard contract の period payload に変換する。
 */
function createDashboardActionAnalysisPeriodFromMetrics(
  period: DashboardActionAnalysisEventRange,
  metrics: DashboardActionAnalysisMetrics
): DashboardActionAnalysisPeriod {
  return {
    days: toDashboardActionAnalysisPeriodDays(period.days),
    startArchiveDay: period.startArchiveDay,
    endArchiveDay: period.endArchiveDay,
    eventCount: period.eventCount,
    sessionCount: metrics.sessionCount,
    completedSessionCount: metrics.completedSessionCount,
    failedSessionCount: metrics.failedSessionCount,
    warningSessionCount: metrics.warningSessionCount,
    maxTurnsSessionCount: metrics.maxTurnsSessionCount,
    totalTokens: metrics.totalTokens,
    averageTokensPerCompletedSession: safeAverage(
      metrics.totalTokens,
      metrics.completedSessionCount
    ),
    averageSessionDurationMs: safeAverage(
      metrics.totalSessionDurationMs,
      metrics.sessionDurationCount
    ),
    totalTurns: metrics.totalTurns,
    noToolCallTurns: metrics.noToolCallTurns,
    toolCallCount: metrics.toolCallCount,
    toolCompletedCount: metrics.toolCompletedCount,
    toolFailedCount: metrics.toolFailedCount,
    toolFailureRate: safeAverage(
      metrics.toolFailedCount,
      metrics.toolCompletedCount + metrics.toolFailedCount
    ),
    topTools: buildTopToolSummariesFromMetrics(metrics.topTools),
    memorySearchCompletedCount: metrics.memorySearchCompletedCount,
    memorySearchFailedCount: metrics.memorySearchFailedCount,
    memorySearchZeroResultCount: metrics.memorySearchZeroResultCount,
    memorySearchAverageFinalResultCount: safeAverage(
      metrics.memorySearchFinalResultTotal,
      metrics.memorySearchCompletedCount
    ),
    storeMemoryCompletedCount: metrics.storeMemoryCompletedCount,
  };
}

/**
 * 集計用 accumulator の初期値を作る。
 */
function createActionAnalysisAccumulator(): ActionAnalysisAccumulator {
  return {
    toolSummaries: new Map<string, MutableToolSummary>(),
    sessionIds: new Set<string>(),
    sessionTimes: new Map<string, SessionTimes>(),
    completedSessionCount: 0,
    failedSessionCount: 0,
    warningSessionCount: 0,
    maxTurnsSessionCount: 0,
    totalTokens: 0,
    totalTurns: 0,
    noToolCallTurns: 0,
    toolCallCount: 0,
    toolCompletedCount: 0,
    toolFailedCount: 0,
    memorySearchCompletedCount: 0,
    memorySearchFailedCount: 0,
    memorySearchZeroResultCount: 0,
    memorySearchFinalResultTotal: 0,
    storeMemoryCompletedCount: 0,
  };
}

/**
 * 1 event を対応する小さな handler に渡す。
 */
function consumeActionAnalysisEvent(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  if (event.sessionId !== null) {
    accumulator.sessionIds.add(event.sessionId);
  }

  consumeSessionEvent(accumulator, event);
  consumeModelEvent(accumulator, event);
  consumeToolEvent(accumulator, event);
  consumeMemoryEvent(accumulator, event);
}

/**
 * session 系 event を集計する。
 */
function consumeSessionEvent(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  switch (event.type) {
    case 'session.started':
      recordSessionStarted(accumulator.sessionTimes, event);
      break;
    case 'session.completed':
      recordSessionCompleted(accumulator, event);
      break;
    case 'session.failed':
      recordSessionFinished(accumulator.sessionTimes, event);
      accumulator.failedSessionCount += 1;
      break;
    default:
      break;
  }
}

/**
 * session completed event を集計する。
 */
function recordSessionCompleted(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  recordSessionFinished(accumulator.sessionTimes, event);
  accumulator.completedSessionCount += 1;
  accumulator.totalTokens += event.totalTokens ?? 0;
  if (event.severity === 'warn') {
    accumulator.warningSessionCount += 1;
  }
  if (event.terminationReason === 'max_turns') {
    accumulator.maxTurnsSessionCount += 1;
  }
}

/**
 * model turn event を集計する。
 */
function consumeModelEvent(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  if (event.type !== 'model.turn.completed') {
    return;
  }

  accumulator.totalTurns += 1;
  if (event.warnings.includes('no_tool_calls')) {
    accumulator.noToolCallTurns += 1;
  }
}

/**
 * tool 系 event を集計する。
 */
function consumeToolEvent(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  switch (event.type) {
    case 'tool.called':
      accumulator.toolCallCount += 1;
      getToolSummary(accumulator.toolSummaries, event).calledCount += 1;
      break;
    case 'tool.completed':
      recordToolCompleted(accumulator, event);
      break;
    case 'tool.failed':
      accumulator.toolFailedCount += 1;
      getToolSummary(accumulator.toolSummaries, event).failedCount += 1;
      break;
    default:
      break;
  }
}

/**
 * tool completed event を集計する。
 */
function recordToolCompleted(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  accumulator.toolCompletedCount += 1;
  getToolSummary(accumulator.toolSummaries, event).completedCount += 1;
  if (event.toolName === 'store_memory') {
    accumulator.storeMemoryCompletedCount += 1;
  }
}

/**
 * memory 系 event を集計する。
 */
function consumeMemoryEvent(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  switch (event.type) {
    case 'memory.search.completed':
      recordMemorySearchCompleted(accumulator, event);
      break;
    case 'memory.search.failed':
      accumulator.memorySearchFailedCount += 1;
      break;
    default:
      break;
  }
}

/**
 * memory search completed event を集計する。
 */
function recordMemorySearchCompleted(
  accumulator: ActionAnalysisAccumulator,
  event: DashboardActionAnalysisEvent
): void {
  accumulator.memorySearchCompletedCount += 1;
  const finalResultCount = event.finalResultCount ?? 0;
  accumulator.memorySearchFinalResultTotal += finalResultCount;
  if (finalResultCount === 0) {
    accumulator.memorySearchZeroResultCount += 1;
  }
}

/**
 * accumulator を dashboard contract の period payload に変換する。
 */
function createDashboardActionAnalysisPeriod(
  period: DashboardActionAnalysisEventRange,
  accumulator: ActionAnalysisAccumulator
): DashboardActionAnalysisPeriod {
  return {
    days: toDashboardActionAnalysisPeriodDays(period.days),
    startArchiveDay: period.startArchiveDay,
    endArchiveDay: period.endArchiveDay,
    eventCount: period.eventCount,
    sessionCount: accumulator.sessionIds.size,
    completedSessionCount: accumulator.completedSessionCount,
    failedSessionCount: accumulator.failedSessionCount,
    warningSessionCount: accumulator.warningSessionCount,
    maxTurnsSessionCount: accumulator.maxTurnsSessionCount,
    totalTokens: accumulator.totalTokens,
    averageTokensPerCompletedSession: safeAverage(
      accumulator.totalTokens,
      accumulator.completedSessionCount
    ),
    averageSessionDurationMs: calculateAverageSessionDurationMs(
      accumulator.sessionTimes
    ),
    totalTurns: accumulator.totalTurns,
    noToolCallTurns: accumulator.noToolCallTurns,
    toolCallCount: accumulator.toolCallCount,
    toolCompletedCount: accumulator.toolCompletedCount,
    toolFailedCount: accumulator.toolFailedCount,
    toolFailureRate: safeAverage(
      accumulator.toolFailedCount,
      accumulator.toolCompletedCount + accumulator.toolFailedCount
    ),
    topTools: buildTopToolSummaries(accumulator.toolSummaries),
    memorySearchCompletedCount: accumulator.memorySearchCompletedCount,
    memorySearchFailedCount: accumulator.memorySearchFailedCount,
    memorySearchZeroResultCount: accumulator.memorySearchZeroResultCount,
    memorySearchAverageFinalResultCount: safeAverage(
      accumulator.memorySearchFinalResultTotal,
      accumulator.memorySearchCompletedCount
    ),
    storeMemoryCompletedCount: accumulator.storeMemoryCompletedCount,
  };
}

/**
 * 任意の number を dashboard contract 上の既知 period に丸める。
 */
function toDashboardActionAnalysisPeriodDays(
  days: number
): DashboardActionAnalysisPeriodDays {
  if (days === 1 || days === 7 || days === 30) {
    return days;
  }

  return 30;
}

/**
 * tool event から toolName を取り出し、集計 entry を返す。
 */
function getToolSummary(
  toolSummaries: Map<string, MutableToolSummary>,
  event: DashboardActionAnalysisEvent
): MutableToolSummary {
  const toolName = event.toolName ?? 'tool';
  const existing = toolSummaries.get(toolName);
  if (existing !== undefined) {
    return existing;
  }

  const created: MutableToolSummary = {
    toolName,
    calledCount: 0,
    completedCount: 0,
    failedCount: 0,
  };
  toolSummaries.set(toolName, created);
  return created;
}

/**
 * tool 集計を表示用上位リストへ変換する。
 */
function buildTopToolSummaries(
  toolSummaries: Map<string, MutableToolSummary>
): DashboardActionAnalysisToolSummary[] {
  return buildTopToolSummariesFromMetrics([...toolSummaries.values()]);
}

/**
 * tool 集計 metrics を表示用上位リストへ変換する。
 */
function buildTopToolSummariesFromMetrics(
  toolSummaries: readonly DashboardActionAnalysisToolMetrics[]
): DashboardActionAnalysisToolSummary[] {
  return [...toolSummaries]
    .map((summary) => ({
      ...summary,
      failureRate: safeAverage(
        summary.failedCount,
        summary.completedCount + summary.failedCount
      ),
    }))
    .sort((left, right) => {
      const leftTotal =
        left.calledCount + left.completedCount + left.failedCount;
      const rightTotal =
        right.calledCount + right.completedCount + right.failedCount;
      if (rightTotal !== leftTotal) {
        return rightTotal - leftTotal;
      }
      if (right.failedCount !== left.failedCount) {
        return right.failedCount - left.failedCount;
      }
      return left.toolName.localeCompare(right.toolName);
    })
    .slice(0, 8);
}

/**
 * session 開始時刻を記録する。
 */
function recordSessionStarted(
  sessionTimes: Map<string, SessionTimes>,
  event: DashboardActionAnalysisEvent
): void {
  if (event.sessionId === null) {
    return;
  }

  const times = sessionTimes.get(event.sessionId) ?? {};
  times.startedAt = event.createdAt;
  sessionTimes.set(event.sessionId, times);
}

/**
 * session 終了時刻を記録する。
 */
function recordSessionFinished(
  sessionTimes: Map<string, SessionTimes>,
  event: DashboardActionAnalysisEvent
): void {
  if (event.sessionId === null) {
    return;
  }

  const times = sessionTimes.get(event.sessionId) ?? {};
  times.finishedAt = event.createdAt;
  sessionTimes.set(event.sessionId, times);
}

/**
 * 開始・終了が揃った session duration の平均を計算する。
 */
function calculateAverageSessionDurationMs(
  sessionTimes: Map<string, SessionTimes>
): number {
  let durationTotal = 0;
  let durationCount = 0;

  for (const times of sessionTimes.values()) {
    if (times.startedAt === undefined || times.finishedAt === undefined) {
      continue;
    }
    const startedAt = new Date(times.startedAt).getTime();
    const finishedAt = new Date(times.finishedAt).getTime();
    if (Number.isNaN(startedAt) || Number.isNaN(finishedAt)) {
      continue;
    }

    durationTotal += Math.max(0, finishedAt - startedAt);
    durationCount += 1;
  }

  return safeAverage(durationTotal, durationCount);
}

/**
 * 分母 0 を 0 として扱う平均・比率計算。
 */
function safeAverage(total: number, count: number): number {
  return count === 0 ? 0 : total / count;
}
