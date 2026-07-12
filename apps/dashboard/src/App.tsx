import {
  Link,
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useParams,
} from '@tanstack/react-router';
import { useEffect, useMemo, useState } from 'react';

import {
  parseDashboardActionAnalysisResponse,
  parseDashboardInstancesResponse,
  parseDashboardSessionLogsResponse,
  parseEchoStatus,
} from '@echo-chamber/contracts/dashboard/schemas';
import type {
  DashboardActionAnalysisPeriod,
  DashboardActionAnalysisPeriodDays,
  DashboardActionAnalysisResponse,
  DashboardInstanceSummary,
  DashboardInstancesResponse,
  DashboardRuntimeConfig,
  DashboardSessionLogsResponse,
  DashboardSessionLog,
  DashboardSummaryState,
  DashboardUsageBreakdownTotals,
  DashboardUsageDays,
  DashboardUsageRatioMetrics,
  DashboardUsageStackedPoint,
  EchoMemory,
  EchoStatus,
} from '@echo-chamber/contracts/dashboard/types';
import {
  buildUsageRatioMetrics,
  buildUsageStackedSeries,
  filterNotes,
  sumUsageBreakdown,
} from '@echo-chamber/contracts/dashboard/utils';
import {
  ECHO_INSTANCE_IDS,
  isValidInstanceId,
} from '@echo-chamber/core/types/echo-config';

import type { JSX } from 'react';

const MEMORY_PAGE_SIZE = 20;

interface SignalItem {
  title: string;
  body: string;
  tone: 'critical' | 'warning' | 'neutral' | 'positive';
}

interface CountEntry {
  label: string;
  count: number;
}

interface UsageAnalysis {
  error: string | null;
  ratios7: DashboardUsageRatioMetrics | null;
  ratios30: DashboardUsageRatioMetrics | null;
  series7: DashboardUsageStackedPoint[] | null;
  series30: DashboardUsageStackedPoint[] | null;
  totals7: DashboardUsageBreakdownTotals | null;
  totals30: DashboardUsageBreakdownTotals | null;
}

type DetailTab = 'overview' | 'analysis' | 'activity' | 'notes' | 'memories';

const DETAIL_TABS: {
  id: DetailTab;
  label: string;
}[] = [
  {
    id: 'overview',
    label: 'Overview',
  },
  {
    id: 'analysis',
    label: 'Analysis',
  },
  {
    id: 'activity',
    label: 'Activity',
  },
  {
    id: 'notes',
    label: 'Notes',
  },
  {
    id: 'memories',
    label: 'Memories',
  },
];

/**
 * Dashboard の API から JSON を取得し、契約 parser で検証した値を返す。
 */
async function fetchDashboardJson<T>(
  path: string,
  parse: (value: unknown) => T
): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} failed with ${response.status}`);
  }

  return parse(await response.json());
}

/**
 * API / parser 由来の unknown error を画面表示用の短い文言へ整形する。
 */
function formatLoadError(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }

  return fallback;
}

/**
 * 数値を日本語ロケールで桁区切り表示する。
 */
function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP').format(value);
}

/**
 * 0.0-1.0 の割合をパーセント表示に整形する。
 */
function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * 推定 USD コストを dashboard 表示用に整形する。
 */
function formatEstimatedCost(value: number | null, isPartial = false): string {
  if (value === null) {
    return 'Unknown';
  }

  return isPartial ? `$${value.toFixed(4)}+` : `$${value.toFixed(4)}`;
}

/**
 * provider/model を dashboard 上の短い runtime 表示へ整形する。
 */
function formatMainLlmLabel(runtime: DashboardRuntimeConfig): string {
  return `${runtime.mainLlm.provider}/${runtime.mainLlm.model}`;
}

/**
 * 日次 token limit を dashboard 表示用に整形する。
 */
function formatDailyTokenLimit(runtime: DashboardRuntimeConfig): string {
  return `${formatNumber(runtime.tokenLimits.dailySoftLimit)} / ${formatNumber(runtime.tokenLimits.dailyHardLimit)}`;
}

/**
 * `YYYY-MM-DD` 形式の usage キーから表示用ラベル (`MM-DD`) を作る。
 */
function formatDateLabel(dateKey: string): string {
  return dateKey.slice(5);
}

/**
 * ISO 文字列（または既存のフォーマット済み文字列）を画面表示用に整形する。
 */
function formatDateTime(value: string | null): string {
  if (value === null) {
    return '-';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('ja-JP', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(date);
}

/**
 * activity details を details 内で読める JSON 文字列へ整形する。
 */
function formatActivityDetails(details: Record<string, unknown>): string {
  return JSON.stringify(details, null, 2);
}

/**
 * 最終更新時刻の補助表示を作る。
 */
function formatLastUpdated(value: Date | null): string {
  if (value === null) {
    return 'Not loaded yet';
  }

  return `Last updated: ${formatDateTime(value.toISOString())}`;
}

/**
 * 分単位の差分を読みやすい相対量へ変換する。
 */
function formatRelativeAmount(absMinutes: number): string {
  if (absMinutes < 90) {
    return `${absMinutes} min`;
  }
  if (absMinutes < 60 * 36) {
    return `${Math.round(absMinutes / 60)} hr`;
  }

  return `${Math.round(absMinutes / 60 / 24)} days`;
}

/**
 * ISO 文字列を現在時刻からの相対表示へ整形する。
 */
function formatRelativeDateTime(value: string | null): string {
  if (value === null) {
    return 'Not scheduled';
  }

  const targetTime = new Date(value).getTime();
  if (Number.isNaN(targetTime)) {
    return value;
  }

  const diffMs = targetTime - Date.now();
  const absMinutes = Math.max(1, Math.round(Math.abs(diffMs) / 60_000));
  const amount = formatRelativeAmount(absMinutes);

  return diffMs >= 0 ? `in ${amount}` : `${amount} ago`;
}

/**
 * ミリ秒 duration を dashboard 表示用に短く整形する。
 */
function formatDurationMs(value: number): string {
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  if (value < 60_000) {
    return `${(value / 1000).toFixed(1)} sec`;
  }

  return `${(value / 60_000).toFixed(1)} min`;
}

/**
 * 平均値を小数1桁までの短い数値として表示する。
 */
function formatAverage(value: number): string {
  return value === 0 ? '0' : value.toFixed(1);
}

/**
 * 日時文字列配列のうち最も新しい値を返す。
 */
function findLatestDateTime(values: readonly (string | null)[]): string | null {
  return values.reduce<string | null>((latest, value) => {
    if (value === null) {
      return latest;
    }
    if (latest === null) {
      return value;
    }

    const latestTime = new Date(latest).getTime();
    const valueTime = new Date(value).getTime();
    if (Number.isNaN(valueTime)) {
      return latest;
    }
    if (Number.isNaN(latestTime) || valueTime > latestTime) {
      return value;
    }

    return latest;
  }, null);
}

/**
 * 日時文字列を新しい順に並べるための比較関数。
 */
function compareDateTimeDescending(left: string, right: string): number {
  const leftTime = new Date(left).getTime();
  const rightTime = new Date(right).getTime();

  if (!Number.isNaN(leftTime) && !Number.isNaN(rightTime)) {
    return rightTime - leftTime;
  }
  if (Number.isNaN(leftTime) && !Number.isNaN(rightTime)) {
    return 1;
  }
  if (!Number.isNaN(leftTime) && Number.isNaN(rightTime)) {
    return -1;
  }

  return right.localeCompare(left);
}

/**
 * 文字列配列を出現回数順の上位リストへ集計する。
 */
function buildTopEntries(
  values: readonly string[],
  limit: number
): CountEntry[] {
  const counts = new Map<string, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  return [...counts.entries()]
    .map(([label, count]) => ({ count, label }))
    .sort((left, right) => {
      if (right.count !== left.count) {
        return right.count - left.count;
      }

      return left.label.localeCompare(right.label);
    })
    .slice(0, limit);
}

/**
 * usage record を詳細画面の各セクションで使う解析済みデータへ変換する。
 */
function analyzeUsage(usage: EchoStatus['usage']): UsageAnalysis {
  try {
    const series7 = buildUsageStackedSeries(usage, 7);
    const series30 = buildUsageStackedSeries(usage, 30);
    const totals7 = sumUsageBreakdown(series7);
    const totals30 = sumUsageBreakdown(series30);

    return {
      error: null,
      ratios7: buildUsageRatioMetrics(totals7),
      ratios30: buildUsageRatioMetrics(totals30),
      series7,
      series30,
      totals7,
      totals30,
    };
  } catch (error) {
    console.error('Invalid usage data detected', error);
    return {
      error: formatLoadError(error, 'Invalid usage data detected'),
      ratios7: null,
      ratios30: null,
      series7: null,
      series30: null,
      totals7: null,
      totals30: null,
    };
  }
}

/**
 * usage 系列から最大消費日の point を返す。
 */
function findPeakUsagePoint(
  series: DashboardUsageStackedPoint[]
): DashboardUsageStackedPoint {
  return series.reduce(
    (peak, point) => {
      return point.totalTokens > peak.totalTokens ? point : peak;
    },
    series[0] ?? {
      cachedInputTokens: 0,
      dateKey: '-',
      normalOutputTokens: 0,
      reasoningOutputTokens: 0,
      estimatedCostUsd: 0,
      estimatedCostIsPartial: false,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalTokens: 0,
      uncachedInputTokens: 0,
    }
  );
}

/**
 * 一覧 API 欠損時に表示する Unknown summary を作る。
 */
function createUnknownInstanceSummary(
  id: DashboardInstanceSummary['id']
): DashboardInstanceSummary {
  return {
    id,
    name: id,
    state: 'Unknown',
    nextAlarm: null,
    nextWakeAt: null,
    noteCount: 0,
    memoryCount: 0,
    todayUsageTokens: 0,
    sevenDayUsageTokens: 0,
    thirtyDayUsageTokens: 0,
    runtime: {
      mainLlm: {
        provider: 'unknown',
        model: 'unknown',
      },
      tokenLimits: {
        dailyHardLimit: 0,
        dailySoftLimit: 0,
        hardLimitBufferFactor: 0,
      },
    },
    latestNoteUpdatedAt: null,
    latestMemoryUpdatedAt: null,
  };
}

/**
 * usage 積み上げ棒の区分定義。
 */
const USAGE_SEGMENTS: {
  key: keyof Pick<
    DashboardUsageStackedPoint,
    | 'cachedInputTokens'
    | 'uncachedInputTokens'
    | 'normalOutputTokens'
    | 'reasoningOutputTokens'
  >;
  label: string;
  className: string;
}[] = [
  {
    key: 'cachedInputTokens',
    label: 'Cached input',
    className: 'usage-segment-cached',
  },
  {
    key: 'uncachedInputTokens',
    label: 'Uncached input',
    className: 'usage-segment-uncached',
  },
  {
    key: 'normalOutputTokens',
    label: 'Normal output',
    className: 'usage-segment-normal-output',
  },
  {
    key: 'reasoningOutputTokens',
    label: 'Reasoning output',
    className: 'usage-segment-reasoning-output',
  },
];

/**
 * usage 棒のホバー時に表示する詳細ツールチップ。
 */
function UsageTooltip(props: {
  point: DashboardUsageStackedPoint;
}): JSX.Element {
  const { point } = props;

  return (
    <div className="usage-tooltip" role="tooltip">
      <p className="usage-tooltip-date">{point.dateKey}</p>
      {USAGE_SEGMENTS.map((segment) => {
        return (
          <p key={segment.key} className="usage-tooltip-row">
            <span>{segment.label}</span>
            <strong>{formatNumber(point[segment.key])}</strong>
          </p>
        );
      })}
      <p className="usage-tooltip-row usage-tooltip-total">
        <span>Total tokens</span>
        <strong>{formatNumber(point.totalTokens)}</strong>
      </p>
      <p className="usage-tooltip-row usage-tooltip-total">
        <span>Estimated cost</span>
        <strong>
          {formatEstimatedCost(
            point.estimatedCostUsd,
            point.estimatedCostIsPartial
          )}
        </strong>
      </p>
    </div>
  );
}

/**
 * usage の積み上げ棒グラフと凡例を描画する。
 */
function UsageStackedChart(props: {
  days: DashboardUsageDays;
  series: DashboardUsageStackedPoint[];
  totals: DashboardUsageBreakdownTotals;
}): JSX.Element {
  const { days, series, totals } = props;
  const [activeDateKey, setActiveDateKey] = useState<string | null>(null);

  const maxTokens = Math.max(...series.map((point) => point.totalTokens), 1);

  return (
    <section className="card">
      <div className="section-header">
        <h2>Usage ({days} days)</h2>
        <p>
          total {formatNumber(totals.totalTokens)} tokens / estimated cost{' '}
          {formatEstimatedCost(
            totals.estimatedCostUsd,
            totals.estimatedCostIsPartial
          )}
        </p>
      </div>

      <div className="usage-legend">
        {USAGE_SEGMENTS.map((segment) => {
          return (
            <div key={segment.key} className="usage-legend-item">
              <span className={`usage-legend-dot ${segment.className}`} />
              <span>{segment.label}</span>
            </div>
          );
        })}
      </div>

      <div className="usage-bars">
        {series.map((point) => {
          const stackHeightRatio = point.totalTokens / maxTokens;
          const barHeightPercent =
            point.totalTokens === 0 ? 0 : Math.max(stackHeightRatio * 100, 1);
          const isActive = point.dateKey === activeDateKey;
          const segmentBase = point.totalTokens === 0 ? 1 : point.totalTokens;

          return (
            <div key={point.dateKey} className="usage-bar-item">
              <div
                className="usage-bar-track"
                onMouseEnter={() => {
                  setActiveDateKey(point.dateKey);
                }}
                onMouseLeave={() => {
                  setActiveDateKey(null);
                }}
              >
                {isActive ? <UsageTooltip point={point} /> : null}
                <button
                  type="button"
                  className="usage-bar-button"
                  onFocus={() => {
                    setActiveDateKey(point.dateKey);
                  }}
                  onBlur={() => {
                    setActiveDateKey(null);
                  }}
                  aria-label={`${point.dateKey} total ${formatNumber(point.totalTokens)} tokens`}
                >
                  <div
                    className="usage-bar-stack"
                    style={{ height: `${barHeightPercent}%` }}
                  >
                    {USAGE_SEGMENTS.map((segment) => {
                      const tokenCount = point[segment.key];
                      if (tokenCount === 0) {
                        return null;
                      }
                      return (
                        <div
                          key={segment.key}
                          className={`usage-bar-segment ${segment.className}`}
                          style={{
                            height: `${(tokenCount / segmentBase) * 100}%`,
                          }}
                        />
                      );
                    })}
                  </div>
                </button>
              </div>
              <div className="usage-label">
                {formatDateLabel(point.dateKey)}
              </div>
              <div className="usage-value">
                {formatNumber(point.totalTokens)}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * usage 指標のサマリーカード群を描画する。
 */
function UsageMetricsPanel(props: {
  totals: DashboardUsageBreakdownTotals;
  ratios: DashboardUsageRatioMetrics;
}): JSX.Element {
  const { totals, ratios } = props;

  return (
    <section className="usage-metrics-grid">
      <article className="card usage-metric-card">
        <h3>Total tokens</h3>
        <p className="usage-metric-emphasis">
          {formatNumber(totals.totalTokens)}
        </p>
      </article>

      <article className="card usage-metric-card">
        <h3>Estimated cost</h3>
        <p className="usage-metric-emphasis">
          {formatEstimatedCost(
            totals.estimatedCostUsd,
            totals.estimatedCostIsPartial
          )}
        </p>
        {totals.estimatedCostIsPartial ? <p>Partial estimate</p> : null}
      </article>

      <article className="card usage-metric-card">
        <h3>Cache / Uncached (input)</h3>
        <p>
          Cached: {formatPercent(ratios.cacheRateInInput)} (
          {formatNumber(totals.cachedInputTokens)})
        </p>
        <p>
          Uncached: {formatPercent(ratios.uncachedRateInInput)} (
          {formatNumber(totals.uncachedInputTokens)})
        </p>
      </article>

      <article className="card usage-metric-card">
        <h3>Input / Output (total)</h3>
        <p>
          Input: {formatPercent(ratios.inputRateInTotal)} (
          {formatNumber(totals.totalInputTokens)})
        </p>
        <p>
          Output: {formatPercent(ratios.outputRateInTotal)} (
          {formatNumber(totals.totalOutputTokens)})
        </p>
      </article>
    </section>
  );
}

/**
 * Dashboard 共通レイアウト（ヘッダー + 子ルート）を提供する。
 */
function DashboardShell(): JSX.Element {
  return (
    <div className="dashboard-shell">
      <header className="topbar">
        <Link to="/" className="brand-link">
          E.C.H.O Dashboard
        </Link>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}

/**
 * 一覧画面のタイトル、件数、手動更新ボタンを描画する。
 */
function DashboardListHeader(props: {
  loading: boolean;
  instanceCount: number;
  lastLoadedAt: Date | null;
  onRefresh(): void;
}): JSX.Element {
  return (
    <div className="section-header">
      <div>
        <h1>Instances</h1>
        <p>
          {props.loading ? 'Loading...' : `${props.instanceCount} instances`}
        </p>
      </div>
      <div className="actions">
        <p className="last-updated">{formatLastUpdated(props.lastLoadedAt)}</p>
        <button
          type="button"
          className="secondary"
          disabled={props.loading}
          onClick={() => {
            props.onRefresh();
          }}
        >
          Refresh
        </button>
      </div>
    </div>
  );
}

/**
 * 未来の日時を持つ instance のうち最も近いものを返す。
 */
function findNextFutureInstance(
  instances: readonly DashboardInstanceSummary[],
  getDateTime: (instance: DashboardInstanceSummary) => string | null
): DashboardInstanceSummary | null {
  const now = Date.now();

  return instances.reduce<DashboardInstanceSummary | null>((next, instance) => {
    const value = getDateTime(instance);
    if (value === null) {
      return next;
    }

    const valueTime = new Date(value).getTime();
    if (Number.isNaN(valueTime) || valueTime < now) {
      return next;
    }

    const nextValue = next === null ? null : getDateTime(next);
    const nextTime =
      nextValue === null
        ? Number.POSITIVE_INFINITY
        : new Date(nextValue).getTime();

    return valueTime < nextTime ? instance : next;
  }, null);
}

/**
 * 一覧画面の状態件数と横断メトリクスを集計する。
 */
function buildFleetSummary(instances: DashboardInstanceSummary[]): {
  latestActivityAt: string | null;
  nextInstance: DashboardInstanceSummary | null;
  nextWakeInstance: DashboardInstanceSummary | null;
  stateCounts: Record<DashboardSummaryState, number>;
  totalMemoryCount: number;
  totalNoteCount: number;
  totalTodayUsageTokens: number;
} {
  const stateCounts: Record<DashboardSummaryState, number> = {
    Idling: 0,
    Running: 0,
    Sleeping: 0,
    Unknown: 0,
  };

  for (const instance of instances) {
    stateCounts[instance.state] += 1;
  }

  return {
    latestActivityAt: findLatestDateTime(
      instances.flatMap((instance) => [
        instance.latestNoteUpdatedAt,
        instance.latestMemoryUpdatedAt,
      ])
    ),
    nextInstance: findNextFutureInstance(instances, (instance) => {
      return instance.nextAlarm;
    }),
    nextWakeInstance: findNextFutureInstance(instances, (instance) => {
      return instance.nextWakeAt;
    }),
    stateCounts,
    totalMemoryCount: instances.reduce(
      (sum, item) => sum + item.memoryCount,
      0
    ),
    totalNoteCount: instances.reduce((sum, item) => sum + item.noteCount, 0),
    totalTodayUsageTokens: instances.reduce(
      (sum, item) => sum + item.todayUsageTokens,
      0
    ),
  };
}

/**
 * Fleet 全体の状態を一目で読むための上部サマリー。
 */
function FleetSummary(props: {
  instances: DashboardInstanceSummary[];
}): JSX.Element {
  const summary = useMemo(() => {
    return buildFleetSummary(props.instances);
  }, [props.instances]);

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Fleet Summary</h2>
          <p>Read-only status overview across all instances</p>
        </div>
      </div>
      <div className="summary-grid">
        <div className="summary-metric">
          <span>Total</span>
          <strong>{formatNumber(props.instances.length)}</strong>
        </div>
        <div className="summary-metric">
          <span>Running</span>
          <strong>{formatNumber(summary.stateCounts.Running)}</strong>
        </div>
        <div className="summary-metric">
          <span>Idling</span>
          <strong>{formatNumber(summary.stateCounts.Idling)}</strong>
        </div>
        <div className="summary-metric">
          <span>Sleeping</span>
          <strong>{formatNumber(summary.stateCounts.Sleeping)}</strong>
        </div>
        <div className="summary-metric">
          <span>Unknown</span>
          <strong>{formatNumber(summary.stateCounts.Unknown)}</strong>
        </div>
        <div className="summary-metric">
          <span>Today usage</span>
          <strong>{formatNumber(summary.totalTodayUsageTokens)}</strong>
        </div>
        <div className="summary-metric">
          <span>Knowledge</span>
          <strong>
            {formatNumber(summary.totalNoteCount)} notes /{' '}
            {formatNumber(summary.totalMemoryCount)} memories
          </strong>
        </div>
        <div className="summary-metric">
          <span>Next alarm</span>
          <strong>
            {summary.nextInstance === null
              ? 'None'
              : `${summary.nextInstance.name} (${formatRelativeDateTime(summary.nextInstance.nextAlarm)})`}
          </strong>
        </div>
        <div className="summary-metric">
          <span>Next wake</span>
          <strong>
            {summary.nextWakeInstance === null
              ? 'None'
              : `${summary.nextWakeInstance.name} (${formatRelativeDateTime(summary.nextWakeInstance.nextWakeAt)})`}
          </strong>
        </div>
        <div className="summary-metric">
          <span>Latest activity</span>
          <strong>{formatRelativeDateTime(summary.latestActivityAt)}</strong>
        </div>
      </div>
    </section>
  );
}

/**
 * 一覧画面で優先して見るべき状態シグナルを作る。
 */
function buildFleetSignals(
  instances: DashboardInstanceSummary[]
): SignalItem[] {
  const signals: SignalItem[] = [];
  const unknownInstances = instances.filter((instance) => {
    return instance.state === 'Unknown';
  });
  const noAlarmInstances = instances.filter((instance) => {
    return instance.state !== 'Sleeping' && instance.nextAlarm === null;
  });
  const emptyInventoryInstances = instances.filter((instance) => {
    return instance.noteCount === 0 && instance.memoryCount === 0;
  });
  const topUsageInstance = instances.reduce<DashboardInstanceSummary | null>(
    (top, instance) => {
      if (top === null || instance.todayUsageTokens > top.todayUsageTokens) {
        return instance;
      }
      return top;
    },
    null
  );

  if (unknownInstances.length > 0) {
    signals.push({
      title: 'Unknown instances',
      body: unknownInstances.map((instance) => instance.name).join(', '),
      tone: 'critical',
    });
  }
  if (noAlarmInstances.length > 0) {
    signals.push({
      title: 'No scheduled next alarm',
      body: noAlarmInstances.map((instance) => instance.name).join(', '),
      tone: 'warning',
    });
  }
  if (topUsageInstance !== null && topUsageInstance.todayUsageTokens > 0) {
    signals.push({
      title: 'Highest usage today',
      body: `${topUsageInstance.name}: ${formatNumber(topUsageInstance.todayUsageTokens)} tokens`,
      tone: 'neutral',
    });
  }
  if (emptyInventoryInstances.length > 0) {
    signals.push({
      title: 'Empty knowledge inventory',
      body: emptyInventoryInstances.map((instance) => instance.name).join(', '),
      tone: 'neutral',
    });
  }

  if (signals.length === 0) {
    return [
      {
        title: 'No attention signals',
        body: 'All instances have readable summaries.',
        tone: 'positive',
      },
    ];
  }

  return signals;
}

/**
 * Fleet の注意シグナル一覧を描画する。
 */
function AttentionArea(props: {
  instances: DashboardInstanceSummary[];
}): JSX.Element {
  const signals = useMemo(() => {
    return buildFleetSignals(props.instances);
  }, [props.instances]);

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Attention</h2>
          <p>Signals that are worth checking first</p>
        </div>
      </div>
      <div className="signal-list">
        {signals.map((signal) => {
          return (
            <article
              key={`${signal.title}-${signal.body}`}
              className={`signal signal-${signal.tone}`}
            >
              <h3>{signal.title}</h3>
              <p>{signal.body}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

/**
 * 個別 instance の一覧カードを描画する。
 */
function InstanceStatusCard(props: {
  instance: DashboardInstanceSummary;
}): JSX.Element {
  const { instance } = props;
  const stateClass = `state state-${instance.state.toLowerCase()}`;

  return (
    <Link
      to="/$instanceId"
      params={{ instanceId: instance.id }}
      className="instance-card"
    >
      <div className="instance-head">
        <h2>{instance.name}</h2>
        <span className={stateClass}>{instance.state}</span>
      </div>
      <div className="metric-list">
        <p>
          <span>Next alarm</span>
          <strong>{formatRelativeDateTime(instance.nextAlarm)}</strong>
        </p>
        <p>
          <span>Next wake</span>
          <strong>{formatRelativeDateTime(instance.nextWakeAt)}</strong>
        </p>
        <p>
          <span>Today usage</span>
          <strong>{formatNumber(instance.todayUsageTokens)} tokens</strong>
        </p>
        <p>
          <span>Main LLM</span>
          <strong>{formatMainLlmLabel(instance.runtime)}</strong>
        </p>
        <p>
          <span>Daily limit</span>
          <strong>{formatDailyTokenLimit(instance.runtime)}</strong>
        </p>
        <p>
          <span>Knowledge</span>
          <strong>
            {formatNumber(instance.noteCount)} /{' '}
            {formatNumber(instance.memoryCount)}
          </strong>
        </p>
        <p>
          <span>Latest activity</span>
          <strong>
            {formatRelativeDateTime(
              findLatestDateTime([
                instance.latestNoteUpdatedAt,
                instance.latestMemoryUpdatedAt,
              ])
            )}
          </strong>
        </p>
      </div>
    </Link>
  );
}

/**
 * インスタンス一覧のカードグリッドを描画する。
 */
function InstanceStatusGrid(props: {
  instances: DashboardInstanceSummary[];
}): JSX.Element {
  return (
    <div className="instance-grid">
      {props.instances.map((instance) => {
        return <InstanceStatusCard key={instance.id} instance={instance} />;
      })}
    </div>
  );
}

/**
 * `/dashboard` のインスタンス一覧画面。
 */
function DashboardListPage(): JSX.Element {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [instances, setInstances] = useState<
    DashboardInstancesResponse['instances']
  >([]);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const orderedInstances = useMemo(() => {
    if (loading && instances.length === 0) {
      return [];
    }

    return ECHO_INSTANCE_IDS.map((id) => {
      const instance = instances.find((candidate) => candidate.id === id);
      if (instance !== undefined) {
        return instance;
      }

      return createUnknownInstanceSummary(id);
    });
  }, [instances, loading]);

  useEffect((): (() => void) => {
    let active = true;

    async function load(): Promise<void> {
      setLoading(true);
      setError(null);

      try {
        const payload = await fetchDashboardJson(
          '/instances',
          parseDashboardInstancesResponse
        );
        if (!active) {
          return;
        }

        setInstances(payload.instances);
        setLastLoadedAt(new Date());
      } catch (loadError) {
        if (!active) {
          return;
        }

        setError(formatLoadError(loadError, 'Failed to load instances'));
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void load();

    return (): void => {
      active = false;
    };
  }, [refreshToken]);

  return (
    <section className="stack">
      <DashboardListHeader
        loading={loading}
        instanceCount={orderedInstances.length}
        lastLoadedAt={lastLoadedAt}
        onRefresh={() => {
          setRefreshToken((value) => value + 1);
        }}
      />

      {error !== null ? (
        <div className="error" role="alert">
          {error}
        </div>
      ) : null}

      {orderedInstances.length > 0 ? (
        <>
          <FleetSummary instances={orderedInstances} />
          <AttentionArea instances={orderedInstances} />
          <InstanceStatusGrid instances={orderedInstances} />
        </>
      ) : null}
    </section>
  );
}

/**
 * メモリ一覧セクション。初期表示件数を絞り、Load more で段階表示する。
 */
function MemorySection(props: { memories: EchoMemory[] }): JSX.Element {
  const { memories } = props;
  const [visibleCount, setVisibleCount] = useState(MEMORY_PAGE_SIZE);
  const [memoryQuery, setMemoryQuery] = useState('');

  useEffect(() => {
    setVisibleCount(MEMORY_PAGE_SIZE);
  }, [memories, memoryQuery]);

  const filteredMemories = useMemo(() => {
    const normalizedQuery = memoryQuery.trim().toLowerCase();
    const matchingMemories =
      normalizedQuery.length === 0
        ? memories
        : memories.filter((memory) => {
            return (
              memory.content.toLowerCase().includes(normalizedQuery) ||
              memory.type.toLowerCase().includes(normalizedQuery) ||
              memory.emotion.labels.some((label) => {
                return label.toLowerCase().includes(normalizedQuery);
              })
            );
          });

    return [...matchingMemories].sort((left, right) => {
      const updatedAtCompare = compareDateTimeDescending(
        left.updatedAt,
        right.updatedAt
      );
      if (updatedAtCompare !== 0) {
        return updatedAtCompare;
      }

      return left.content.localeCompare(right.content);
    });
  }, [memories, memoryQuery]);

  const visibleMemories = useMemo(() => {
    return filteredMemories.slice(0, visibleCount);
  }, [filteredMemories, visibleCount]);

  const hasMore = visibleCount < filteredMemories.length;

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Memories</h2>
          <p>
            {formatNumber(filteredMemories.length)} /{' '}
            {formatNumber(memories.length)} items
          </p>
        </div>
        <input
          type="search"
          value={memoryQuery}
          onChange={(event): void => {
            setMemoryQuery(event.currentTarget.value);
          }}
          placeholder="Search memories"
        />
      </div>

      {visibleMemories.length === 0 ? (
        <p className="muted">
          {memoryQuery.trim().length > 0
            ? 'No matching memories.'
            : 'No memories.'}
        </p>
      ) : (
        <div className="item-list">
          {visibleMemories.map((memory, index) => {
            const key = `${memory.createdAt}-${index}`;

            return (
              <article key={key} className="item-card">
                <div className="item-meta">
                  <span className="pill">{memory.type}</span>
                  <span>{formatDateTime(memory.updatedAt)}</span>
                </div>
                {memory.emotion.labels.length > 0 ? (
                  <div className="tag-list">
                    {memory.emotion.labels.map((label) => {
                      return (
                        <span key={label} className="tag">
                          {label}
                        </span>
                      );
                    })}
                  </div>
                ) : null}
                <p className="item-content">{memory.content}</p>
              </article>
            );
          })}
        </div>
      )}

      {hasMore ? (
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setVisibleCount((count) => count + MEMORY_PAGE_SIZE);
          }}
        >
          Load more
        </button>
      ) : null}
    </section>
  );
}

/**
 * 詳細画面の最上部で状態・活動・データ量をまとめて表示する。
 */
function InstanceSnapshot(props: {
  status: EchoStatus;
  usage: UsageAnalysis;
}): JSX.Element {
  const { status, usage } = props;
  const latestActivityAt = findLatestDateTime([
    ...status.notes.map((note) => note.updatedAt),
    ...status.memories.map((memory) => memory.updatedAt),
  ]);

  return (
    <section className="card snapshot-card">
      <div className="section-header">
        <div>
          <h2>Snapshot</h2>
          <p>Current state and recent activity</p>
        </div>
      </div>
      <div className="summary-grid snapshot-grid">
        <div className="summary-metric">
          <span>State</span>
          <strong>{status.state}</strong>
        </div>
        <div className="summary-metric">
          <span>Next alarm</span>
          <strong>{formatRelativeDateTime(status.nextAlarm)}</strong>
        </div>
        <div className="summary-metric">
          <span>Next wake</span>
          <strong>{formatRelativeDateTime(status.nextWakeAt)}</strong>
        </div>
        <div className="summary-metric">
          <span>Main LLM</span>
          <strong>{formatMainLlmLabel(status.runtime)}</strong>
        </div>
        <div className="summary-metric">
          <span>Daily limit</span>
          <strong>{formatDailyTokenLimit(status.runtime)}</strong>
        </div>
        <div className="summary-metric">
          <span>Latest activity</span>
          <strong>{formatRelativeDateTime(latestActivityAt)}</strong>
        </div>
        <div className="summary-metric">
          <span>Notes</span>
          <strong>{formatNumber(status.notes.length)}</strong>
        </div>
        <div className="summary-metric">
          <span>Memories</span>
          <strong>{formatNumber(status.memories.length)}</strong>
        </div>
        <div className="summary-metric">
          <span>7 day usage</span>
          <strong>
            {usage.totals7 === null
              ? '-'
              : `${formatNumber(usage.totals7.totalTokens)} tokens`}
          </strong>
        </div>
      </div>
    </section>
  );
}

/**
 * 詳細画面で確認すべき状態シグナルを作る。
 */
function buildInstanceSignals(
  status: EchoStatus,
  usage: UsageAnalysis
): SignalItem[] {
  const signals: SignalItem[] = [];

  if (status.state !== 'Sleeping' && status.nextAlarm === null) {
    signals.push({
      title: 'Next alarm is not scheduled',
      body: 'This instance is active or idle, but no next alarm is visible.',
      tone: 'warning',
    });
  }
  if (status.notes.length === 0 || status.memories.length === 0) {
    signals.push({
      title: 'Knowledge inventory is sparse',
      body: `${formatNumber(status.notes.length)} notes / ${formatNumber(status.memories.length)} memories`,
      tone: 'neutral',
    });
  }
  if (usage.error !== null) {
    signals.push({
      title: 'Usage data is incomplete',
      body: usage.error,
      tone: 'critical',
    });
  } else if (usage.totals7 !== null && usage.totals7.totalTokens > 0) {
    signals.push({
      title: 'Recent usage observed',
      body: `${formatNumber(usage.totals7.totalTokens)} tokens in the last 7 days`,
      tone: 'neutral',
    });
  }

  if (signals.length === 0) {
    return [
      {
        title: 'No attention signals',
        body: 'The current snapshot has no obvious data gaps.',
        tone: 'positive',
      },
    ];
  }

  return signals;
}

/**
 * 詳細画面の状態シグナルを描画する。
 */
function HealthSignals(props: {
  status: EchoStatus;
  usage: UsageAnalysis;
}): JSX.Element {
  const signals = useMemo(() => {
    return buildInstanceSignals(props.status, props.usage);
  }, [props.status, props.usage]);

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Health Signals</h2>
          <p>Derived read-only signals from the current snapshot</p>
        </div>
      </div>
      <div className="signal-list">
        {signals.map((signal) => {
          return (
            <article
              key={`${signal.title}-${signal.body}`}
              className={`signal signal-${signal.tone}`}
            >
              <h3>{signal.title}</h3>
              <p>{signal.body}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

/**
 * action analysis の期間を選択する。
 */
function selectActionAnalysisPeriod(
  analysis: DashboardActionAnalysisResponse,
  days: DashboardActionAnalysisPeriodDays
): DashboardActionAnalysisPeriod {
  return (
    analysis.periods.find((period) => period.days === days) ??
    analysis.periods[0] ?? {
      averageSessionDurationMs: 0,
      averageTokensPerCompletedSession: 0,
      completedSessionCount: 0,
      days,
      endArchiveDay: analysis.archiveDay,
      eventCount: 0,
      failedSessionCount: 0,
      maxTurnsSessionCount: 0,
      memorySearchAverageFinalResultCount: 0,
      memorySearchCompletedCount: 0,
      memorySearchFailedCount: 0,
      memorySearchZeroResultCount: 0,
      noToolCallTurns: 0,
      sessionCount: 0,
      startArchiveDay: analysis.archiveDay,
      storeMemoryCompletedCount: 0,
      toolCallCount: 0,
      toolCompletedCount: 0,
      toolFailedCount: 0,
      toolFailureRate: 0,
      topTools: [],
      totalTokens: 0,
      totalTurns: 0,
      warningSessionCount: 0,
    }
  );
}

/**
 * action analysis の期間ラベルを作る。
 */
function formatActionPeriodLabel(
  period: DashboardActionAnalysisPeriod
): string {
  if (period.days === 1) {
    return `Archive day ${period.endArchiveDay}`;
  }

  return `${period.startArchiveDay} - ${period.endArchiveDay}`;
}

/**
 * 行動分析の上位 tool 一覧を描画する。
 */
function ToolAnalysisTable(props: {
  tools: DashboardActionAnalysisPeriod['topTools'];
}): JSX.Element {
  if (props.tools.length === 0) {
    return <p className="muted">No tool activity in this period.</p>;
  }

  return (
    <div className="analysis-table" role="table" aria-label="Tool usage">
      <div className="analysis-table-row analysis-table-head" role="row">
        <span role="columnheader">Tool</span>
        <span role="columnheader">Called</span>
        <span role="columnheader">Done</span>
        <span role="columnheader">Failed</span>
        <span role="columnheader">Fail rate</span>
      </div>
      {props.tools.map((tool) => {
        return (
          <div key={tool.toolName} className="analysis-table-row" role="row">
            <strong role="cell">{tool.toolName}</strong>
            <span role="cell">{formatNumber(tool.calledCount)}</span>
            <span role="cell">{formatNumber(tool.completedCount)}</span>
            <span role="cell">{formatNumber(tool.failedCount)}</span>
            <span role="cell">{formatPercent(tool.failureRate)}</span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * action analysis の期間切り替えボタン群を描画する。
 */
function ActionPeriodToggle(props: {
  availableDays: readonly DashboardActionAnalysisPeriodDays[];
  analysisDays: DashboardActionAnalysisPeriodDays;
  setAnalysisDays(days: DashboardActionAnalysisPeriodDays): void;
}): JSX.Element {
  return (
    <div className="usage-toggle">
      {[1, 7, 30].map((days) => {
        const periodDays = days as DashboardActionAnalysisPeriodDays;
        const available = props.availableDays.includes(periodDays);
        const active = available && props.analysisDays === days;

        return (
          <button
            key={days}
            type="button"
            disabled={!available}
            className={active ? 'primary' : 'secondary'}
            onClick={() => {
              props.setAnalysisDays(periodDays);
            }}
          >
            {days}d
          </button>
        );
      })}
    </div>
  );
}

/**
 * action analysis の主要指標を summary grid で描画する。
 */
function ActionSummaryGrid(props: {
  period: DashboardActionAnalysisPeriod;
}): JSX.Element {
  const { period } = props;
  const noToolTurnRate =
    period.totalTurns === 0 ? 0 : period.noToolCallTurns / period.totalTurns;

  return (
    <div className="summary-grid">
      <div className="summary-metric">
        <span>Sessions</span>
        <strong>{formatNumber(period.sessionCount)}</strong>
      </div>
      <div className="summary-metric">
        <span>Completed / failed</span>
        <strong>
          {formatNumber(period.completedSessionCount)} /{' '}
          {formatNumber(period.failedSessionCount)}
        </strong>
      </div>
      <div className="summary-metric">
        <span>Avg duration</span>
        <strong>{formatDurationMs(period.averageSessionDurationMs)}</strong>
      </div>
      <div className="summary-metric">
        <span>Avg tokens/session</span>
        <strong>
          {formatNumber(Math.round(period.averageTokensPerCompletedSession))}
        </strong>
      </div>
      <div className="summary-metric">
        <span>Tool calls</span>
        <strong>{formatNumber(period.toolCallCount)}</strong>
      </div>
      <div className="summary-metric">
        <span>Tool failure rate</span>
        <strong>{formatPercent(period.toolFailureRate)}</strong>
      </div>
      <div className="summary-metric">
        <span>No-tool turns</span>
        <strong>
          {formatNumber(period.noToolCallTurns)} (
          {formatPercent(noToolTurnRate)})
        </strong>
      </div>
      <div className="summary-metric">
        <span>Events analyzed</span>
        <strong>{formatNumber(period.eventCount)}</strong>
      </div>
    </div>
  );
}

/**
 * memory 関連の action analysis 指標を描画する。
 */
function MemoryAnalysisBlock(props: {
  period: DashboardActionAnalysisPeriod;
}): JSX.Element {
  const { period } = props;
  const memoryZeroResultRate =
    period.memorySearchCompletedCount === 0
      ? 0
      : period.memorySearchZeroResultCount / period.memorySearchCompletedCount;

  return (
    <section className="analysis-block">
      <h3>Memory</h3>
      <div className="metric-list">
        <p>
          <span>Search completed</span>
          <strong>{formatNumber(period.memorySearchCompletedCount)}</strong>
        </p>
        <p>
          <span>Search failed</span>
          <strong>{formatNumber(period.memorySearchFailedCount)}</strong>
        </p>
        <p>
          <span>Zero-result rate</span>
          <strong>{formatPercent(memoryZeroResultRate)}</strong>
        </p>
        <p>
          <span>Avg results/search</span>
          <strong>
            {formatAverage(period.memorySearchAverageFinalResultCount)}
          </strong>
        </p>
        <p>
          <span>Store memory</span>
          <strong>{formatNumber(period.storeMemoryCompletedCount)}</strong>
        </p>
      </div>
    </section>
  );
}

/**
 * session 関連の action analysis 指標を描画する。
 */
function SessionAnalysisBlock(props: {
  period: DashboardActionAnalysisPeriod;
}): JSX.Element {
  const { period } = props;

  return (
    <section className="analysis-block">
      <h3>Sessions</h3>
      <div className="metric-list">
        <p>
          <span>Warning sessions</span>
          <strong>{formatNumber(period.warningSessionCount)}</strong>
        </p>
        <p>
          <span>Max-turn sessions</span>
          <strong>{formatNumber(period.maxTurnsSessionCount)}</strong>
        </p>
        <p>
          <span>Total turns</span>
          <strong>{formatNumber(period.totalTurns)}</strong>
        </p>
        <p>
          <span>Total tokens</span>
          <strong>{formatNumber(period.totalTokens)}</strong>
        </p>
      </div>
    </section>
  );
}

/**
 * tool / memory / turn の行動分析を描画する。
 */
function ActionAnalysisSection(props: {
  analysis: DashboardActionAnalysisResponse;
  analysisDays: DashboardActionAnalysisPeriodDays;
  setAnalysisDays(days: DashboardActionAnalysisPeriodDays): void;
}): JSX.Element {
  const availableDays = props.analysis.periods.map((period) => period.days);
  const period = selectActionAnalysisPeriod(props.analysis, props.analysisDays);

  return (
    <section className="card analysis-panel">
      <div className="section-header">
        <div>
          <h2>Action Analysis</h2>
          <p>{formatActionPeriodLabel(period)}</p>
        </div>
        <ActionPeriodToggle
          availableDays={availableDays}
          analysisDays={props.analysisDays}
          setAnalysisDays={(days): void => {
            props.setAnalysisDays(days);
          }}
        />
      </div>

      <ActionSummaryGrid period={period} />

      <div className="analysis-columns">
        <MemoryAnalysisBlock period={period} />
        <SessionAnalysisBlock period={period} />
      </div>

      <section className="analysis-block">
        <div className="section-header">
          <div>
            <h3>Top Tools</h3>
            <p>Most active tools in the selected period</p>
          </div>
        </div>
        <ToolAnalysisTable tools={period.topTools} />
      </section>
    </section>
  );
}

/**
 * 詳細画面のノート検索セクション。
 */
function NotesSection(props: {
  totalNotes: number;
  notes: EchoStatus['notes'];
  noteQuery: string;
  onChangeQuery(value: string): void;
}): JSX.Element {
  const { notes, noteQuery, totalNotes } = props;

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Notes</h2>
          <p>
            {formatNumber(notes.length)} / {formatNumber(totalNotes)} notes
          </p>
        </div>
        <div className="search-actions">
          <input
            type="search"
            value={noteQuery}
            onChange={(event): void => {
              props.onChangeQuery(event.currentTarget.value);
            }}
            placeholder="Search notes"
          />
          {noteQuery.trim().length > 0 ? (
            <button
              type="button"
              className="secondary"
              onClick={() => {
                props.onChangeQuery('');
              }}
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {notes.length === 0 ? (
        <p className="muted">No matching notes.</p>
      ) : (
        <div className="item-list">
          {notes.map((note) => {
            return (
              <article key={note.id} className="item-card">
                <div className="item-meta">
                  <span className="pill">{note.id}</span>
                  <span>{formatDateTime(note.updatedAt)}</span>
                </div>
                <h3>{note.title}</h3>
                <p className="item-content">{note.content}</p>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

/**
 * 現在日の Echo activity log を描画する。
 */
function ActivitySection(props: {
  sessionLogs: DashboardSessionLog[];
  archiveDay: string;
}): JSX.Element {
  const activities = useMemo(() => {
    return props.sessionLogs.flatMap((sessionLog) => sessionLog.activities);
  }, [props.sessionLogs]);
  const summary = useMemo(() => {
    return {
      actions: activities.filter((entry) => entry.kind === 'action').length,
      sessions: props.sessionLogs.length,
      thoughts: activities.filter((entry) => entry.kind === 'thought').length,
      warnings: activities.filter(
        (entry) => entry.tone === 'warning' || entry.tone === 'critical'
      ).length,
    };
  }, [activities, props.sessionLogs]);

  return (
    <section className="card activity-panel">
      <div className="section-header">
        <div>
          <h2>Activity</h2>
          <p>Log day {props.archiveDay}</p>
        </div>
      </div>

      <div className="activity-summary">
        <div className="summary-metric">
          <span>Sessions</span>
          <strong>{formatNumber(summary.sessions)}</strong>
        </div>
        <div className="summary-metric">
          <span>Thoughts</span>
          <strong>{formatNumber(summary.thoughts)}</strong>
        </div>
        <div className="summary-metric">
          <span>Actions</span>
          <strong>{formatNumber(summary.actions)}</strong>
        </div>
        <div className="summary-metric">
          <span>Warnings</span>
          <strong>{formatNumber(summary.warnings)}</strong>
        </div>
      </div>

      {props.sessionLogs.length === 0 ? (
        <p className="muted">No activity for this log day.</p>
      ) : (
        <div className="activity-session-list">
          {props.sessionLogs.map((sessionLog, sessionIndex) => {
            return (
              <details
                key={sessionLog.id}
                className="activity-session"
                open={sessionIndex === 0}
              >
                <summary className="activity-session-summary">
                  <div className="activity-session-heading">
                    <h3>{sessionLog.title}</h3>
                    <div className="activity-meta">
                      {sessionLog.meta.map((item) => {
                        return <span key={item}>{item}</span>;
                      })}
                    </div>
                  </div>
                  <div className="activity-session-stats">
                    <span>{formatNumber(sessionLog.activityCount)}</span>
                    {sessionLog.warningCount === 0 ? null : (
                      <span>{formatNumber(sessionLog.warningCount)} warn</span>
                    )}
                  </div>
                </summary>

                <div className="activity-list activity-session-activities">
                  {sessionLog.activities.map((activity) => {
                    return (
                      <article
                        key={activity.id}
                        className={`activity-item activity-item-${activity.tone} activity-type-${activity.kind}`}
                      >
                        <div className="activity-marker" aria-hidden="true" />
                        <div className="activity-content">
                          <div className="activity-heading">
                            <h3>{activity.title}</h3>
                            <span
                              className={`activity-kind activity-kind-${activity.kind}`}
                            >
                              {activity.kind}
                            </span>
                          </div>
                          <p className="activity-body">{activity.body}</p>
                          <div className="activity-meta">
                            {activity.meta.map((item) => {
                              return <span key={item}>{item}</span>;
                            })}
                          </div>
                          {activity.details === null ? null : (
                            <details className="activity-details">
                              <summary>Details</summary>
                              <pre>
                                {formatActivityDetails(activity.details)}
                              </pre>
                            </details>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </details>
            );
          })}
        </div>
      )}
    </section>
  );
}

/**
 * 選択中の usage 期間に対応する解析結果を取り出す。
 */
function selectUsagePeriod(
  usage: UsageAnalysis,
  usageDays: DashboardUsageDays
): {
  ratios: DashboardUsageRatioMetrics | null;
  series: DashboardUsageStackedPoint[] | null;
  totals: DashboardUsageBreakdownTotals | null;
} {
  if (usageDays === 7) {
    return {
      ratios: usage.ratios7,
      series: usage.series7,
      totals: usage.totals7,
    };
  }

  return {
    ratios: usage.ratios30,
    series: usage.series30,
    totals: usage.totals30,
  };
}

/**
 * usage chart の前に置く期間内サマリーを描画する。
 */
function UsagePeriodSummary(props: {
  peakPoint: DashboardUsageStackedPoint;
  totals: DashboardUsageBreakdownTotals;
  usageDays: DashboardUsageDays;
}): JSX.Element {
  const dailyAverage = Math.round(props.totals.totalTokens / props.usageDays);
  const reasoningOutputRate =
    props.totals.totalOutputTokens === 0
      ? 0
      : props.totals.reasoningOutputTokens / props.totals.totalOutputTokens;

  return (
    <section className="summary-grid">
      <article className="card usage-metric-card">
        <h3>Daily average</h3>
        <p className="usage-metric-emphasis">{formatNumber(dailyAverage)}</p>
      </article>
      <article className="card usage-metric-card">
        <h3>Peak day</h3>
        <p className="usage-metric-emphasis">{props.peakPoint.dateKey}</p>
        <p>{formatNumber(props.peakPoint.totalTokens)} tokens</p>
      </article>
      <article className="card usage-metric-card">
        <h3>Reasoning output</h3>
        <p className="usage-metric-emphasis">
          {formatPercent(reasoningOutputRate)}
        </p>
      </article>
    </section>
  );
}

/**
 * 詳細画面の usage 期間切り替え + 棒グラフセクション。
 */
function UsageSection(props: {
  usage: UsageAnalysis;
  usageDays: DashboardUsageDays;
  setUsageDays(days: DashboardUsageDays): void;
}): JSX.Element {
  const { usage, usageDays } = props;
  const { ratios, series, totals } = selectUsagePeriod(usage, usageDays);

  return (
    <section className="stack">
      <div className="usage-toggle">
        {[7, 30].map((days) => {
          const active = usageDays === days;

          return (
            <button
              key={days}
              type="button"
              className={active ? 'primary' : 'secondary'}
              onClick={() => {
                props.setUsageDays(days as DashboardUsageDays);
              }}
            >
              {days} days
            </button>
          );
        })}
      </div>

      {usage.error !== null ? (
        <div className="error" role="alert">
          {usage.error}
        </div>
      ) : null}

      {series !== null && totals !== null && ratios !== null ? (
        <>
          <UsagePeriodSummary
            peakPoint={findPeakUsagePoint(series)}
            totals={totals}
            usageDays={usageDays}
          />
          <UsageStackedChart days={usageDays} series={series} totals={totals} />
          <UsageMetricsPanel totals={totals} ratios={ratios} />
        </>
      ) : null}
    </section>
  );
}

/**
 * 詳細画面の note / memory 保有傾向を表示する。
 */
function KnowledgeInventory(props: { status: EchoStatus }): JSX.Element {
  const { status } = props;
  const memoryTypes = useMemo(() => {
    return buildTopEntries(
      status.memories.map((memory) => memory.type),
      6
    );
  }, [status.memories]);
  const emotionLabels = useMemo(() => {
    return buildTopEntries(
      status.memories.flatMap((memory) => memory.emotion.labels),
      8
    );
  }, [status.memories]);
  const embeddingModels = useMemo(() => {
    return buildTopEntries(
      status.memories.map((memory) => memory.embedding_model),
      4
    );
  }, [status.memories]);
  const latestNoteUpdatedAt = findLatestDateTime(
    status.notes.map((note) => note.updatedAt)
  );
  const latestMemoryUpdatedAt = findLatestDateTime(
    status.memories.map((memory) => memory.updatedAt)
  );

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Knowledge Inventory</h2>
          <p>Stored notes and memories by recency and shape</p>
        </div>
      </div>
      <div className="summary-grid">
        <div className="summary-metric">
          <span>Latest note</span>
          <strong>{formatRelativeDateTime(latestNoteUpdatedAt)}</strong>
        </div>
        <div className="summary-metric">
          <span>Latest memory</span>
          <strong>{formatRelativeDateTime(latestMemoryUpdatedAt)}</strong>
        </div>
        <div className="summary-metric">
          <span>Memory types</span>
          <strong>
            {memoryTypes.length === 0
              ? '-'
              : memoryTypes
                  .map((entry) => `${entry.label} ${entry.count}`)
                  .join(' / ')}
          </strong>
        </div>
        <div className="summary-metric">
          <span>Embedding models</span>
          <strong>
            {embeddingModels.length === 0
              ? '-'
              : embeddingModels
                  .map((entry) => `${entry.label} ${entry.count}`)
                  .join(' / ')}
          </strong>
        </div>
      </div>
      <div className="tag-list">
        {emotionLabels.length === 0 ? (
          <span className="muted">No emotion labels.</span>
        ) : (
          emotionLabels.map((entry) => {
            return (
              <span key={entry.label} className="tag">
                {entry.label} ({entry.count})
              </span>
            );
          })
        )}
      </div>
    </section>
  );
}

/**
 * 次回起動へ引き継ぐ runtime context を表示する。
 */
function RuntimeContextPanel(props: {
  context: EchoStatus['context'];
}): JSX.Element {
  const { context } = props;

  return (
    <section className="card">
      <div className="section-header">
        <div>
          <h2>Runtime Context</h2>
          <p>Persisted context for the next thinking session</p>
        </div>
      </div>

      {context === null ? (
        <p className="muted">No persisted context.</p>
      ) : (
        <>
          <div className="summary-grid">
            <div className="summary-metric">
              <span>Updated</span>
              <strong>{formatRelativeDateTime(context.updatedAt)}</strong>
            </div>
            <div className="summary-metric">
              <span>Created</span>
              <strong>{formatRelativeDateTime(context.createdAt)}</strong>
            </div>
            <div className="summary-metric">
              <span>Valence</span>
              <strong>{formatNumber(context.emotion.valence)}</strong>
            </div>
            <div className="summary-metric">
              <span>Arousal</span>
              <strong>{formatNumber(context.emotion.arousal)}</strong>
            </div>
          </div>
          {context.emotion.labels.length === 0 ? null : (
            <div className="tag-list">
              {context.emotion.labels.map((label) => {
                return (
                  <span key={label} className="tag">
                    {label}
                  </span>
                );
              })}
            </div>
          )}
          <p className="context-content">{context.content}</p>
        </>
      )}
    </section>
  );
}

/**
 * 詳細画面のタイトル、戻るリンク、手動更新ボタンを描画する。
 */
function DashboardDetailHeader(props: {
  instanceId: string;
  instanceName: string;
  loading: boolean;
  lastLoadedAt: Date | null;
  onRefresh(): void;
}): JSX.Element {
  return (
    <div className="section-header">
      <div>
        <h1>Instance: {props.instanceId}</h1>
        <p>{props.instanceName}</p>
      </div>
      <div className="actions">
        <p className="last-updated">{formatLastUpdated(props.lastLoadedAt)}</p>
        <Link to="/" className="secondary">
          Back
        </Link>
        <button
          type="button"
          className="primary"
          disabled={props.loading}
          onClick={() => {
            props.onRefresh();
          }}
        >
          Refresh
        </button>
      </div>
    </div>
  );
}

/**
 * 詳細タブに表示する件数バッジを返す。
 */
function getDetailTabCount(
  tab: DetailTab,
  counts: { activityCount: number; memoryCount: number; noteCount: number }
): number | null {
  if (tab === 'activity') {
    return counts.activityCount;
  }
  if (tab === 'notes') {
    return counts.noteCount;
  }
  if (tab === 'memories') {
    return counts.memoryCount;
  }

  return null;
}

/**
 * 詳細画面のタブナビゲーションを描画する。
 */
function DetailTabs(props: {
  activeTab: DetailTab;
  activityCount: number;
  memoryCount: number;
  noteCount: number;
  onChange(tab: DetailTab): void;
}): JSX.Element {
  return (
    <div className="detail-tabs" role="tablist" aria-label="Detail sections">
      {DETAIL_TABS.map((tab) => {
        const active = props.activeTab === tab.id;
        const count = getDetailTabCount(tab.id, {
          activityCount: props.activityCount,
          memoryCount: props.memoryCount,
          noteCount: props.noteCount,
        });

        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active}
            className={active ? 'detail-tab detail-tab-active' : 'detail-tab'}
            onClick={() => {
              props.onChange(tab.id);
            }}
          >
            {tab.label}
            {count === null ? null : (
              <span className="detail-tab-count">{formatNumber(count)}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/**
 * 選択中の詳細タブの本文を描画する。
 */
function DetailTabPanel(props: {
  activeTab: DetailTab;
  actionAnalysis: DashboardActionAnalysisResponse;
  analysisDays: DashboardActionAnalysisPeriodDays;
  archiveDay: string;
  noteQuery: string;
  notes: EchoStatus['notes'];
  sessionLogs: DashboardSessionLog[];
  setAnalysisDays(days: DashboardActionAnalysisPeriodDays): void;
  setNoteQuery(value: string): void;
  setUsageDays(days: DashboardUsageDays): void;
  status: EchoStatus;
  usageAnalysis: UsageAnalysis;
  usageDays: DashboardUsageDays;
}): JSX.Element {
  switch (props.activeTab) {
    case 'overview':
      return (
        <div className="stack" role="tabpanel">
          <InstanceSnapshot status={props.status} usage={props.usageAnalysis} />
          <HealthSignals status={props.status} usage={props.usageAnalysis} />
          <UsageSection
            usage={props.usageAnalysis}
            usageDays={props.usageDays}
            setUsageDays={(days): void => {
              props.setUsageDays(days);
            }}
          />
          <KnowledgeInventory status={props.status} />
          <RuntimeContextPanel context={props.status.context} />
        </div>
      );
    case 'analysis':
      return (
        <div className="stack" role="tabpanel">
          <ActionAnalysisSection
            analysis={props.actionAnalysis}
            analysisDays={props.analysisDays}
            setAnalysisDays={(days): void => {
              props.setAnalysisDays(days);
            }}
          />
        </div>
      );
    case 'activity':
      return (
        <div className="stack" role="tabpanel">
          <ActivitySection
            archiveDay={props.archiveDay}
            sessionLogs={props.sessionLogs}
          />
        </div>
      );
    case 'notes':
      return (
        <div className="stack" role="tabpanel">
          <NotesSection
            totalNotes={props.status.notes.length}
            notes={props.notes}
            noteQuery={props.noteQuery}
            onChangeQuery={(value): void => {
              props.setNoteQuery(value);
            }}
          />
        </div>
      );
    case 'memories':
      return (
        <div className="stack" role="tabpanel">
          <MemorySection memories={props.status.memories} />
        </div>
      );
  }
}

/**
 * 詳細画面の読み込み完了後の情報セクション群を描画する。
 */
function DashboardDetailContent(props: {
  actionAnalysisResponse: DashboardActionAnalysisResponse;
  analysisDays: DashboardActionAnalysisPeriodDays;
  noteQuery: string;
  sessionLogsResponse: DashboardSessionLogsResponse;
  setAnalysisDays(days: DashboardActionAnalysisPeriodDays): void;
  setNoteQuery(value: string): void;
  setUsageDays(days: DashboardUsageDays): void;
  status: EchoStatus;
  usageDays: DashboardUsageDays;
}): JSX.Element {
  const [activeTab, setActiveTab] = useState<DetailTab>('overview');
  const notes = useMemo(() => {
    return filterNotes(props.status.notes, props.noteQuery);
  }, [props.status.notes, props.noteQuery]);
  const usageAnalysis = useMemo(() => {
    return analyzeUsage(props.status.usage);
  }, [props.status.usage]);
  const sessionLogs = props.sessionLogsResponse.sessionLogs;
  const activityCount = sessionLogs.reduce((total, sessionLog) => {
    return total + sessionLog.activityCount;
  }, 0);

  return (
    <section className="stack">
      <DetailTabs
        activeTab={activeTab}
        activityCount={activityCount}
        noteCount={props.status.notes.length}
        memoryCount={props.status.memories.length}
        onChange={(tab): void => {
          setActiveTab(tab);
        }}
      />
      <DetailTabPanel
        activeTab={activeTab}
        actionAnalysis={props.actionAnalysisResponse}
        analysisDays={props.analysisDays}
        archiveDay={props.sessionLogsResponse.archiveDay}
        sessionLogs={sessionLogs}
        status={props.status}
        usageAnalysis={usageAnalysis}
        usageDays={props.usageDays}
        setUsageDays={(days): void => {
          props.setUsageDays(days);
        }}
        setAnalysisDays={(days): void => {
          props.setAnalysisDays(days);
        }}
        notes={notes}
        noteQuery={props.noteQuery}
        setNoteQuery={(value): void => {
          props.setNoteQuery(value);
        }}
      />
    </section>
  );
}

/**
 * `/dashboard/:instanceId` の詳細画面。
 */
function DashboardDetailPage(): JSX.Element {
  const { instanceId } = useParams({ from: '/$instanceId' });
  const [status, setStatus] = useState<EchoStatus | null>(null);
  const [sessionLogsResponse, setSessionLogsResponse] =
    useState<DashboardSessionLogsResponse | null>(null);
  const [actionAnalysisResponse, setActionAnalysisResponse] =
    useState<DashboardActionAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [noteQuery, setNoteQuery] = useState('');
  const [usageDays, setUsageDaysState] = useState<DashboardUsageDays>(7);
  const [analysisDays, setAnalysisDaysState] =
    useState<DashboardActionAnalysisPeriodDays>(7);
  const [refreshToken, setRefreshToken] = useState(0);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);

  useEffect((): (() => void) => {
    let active = true;

    async function load(): Promise<void> {
      setLoading(true);
      setError(null);

      if (!isValidInstanceId(instanceId)) {
        setStatus(null);
        setSessionLogsResponse(null);
        setActionAnalysisResponse(null);
        setError(`Invalid instance ID: ${instanceId}`);
        setLoading(false);
        return;
      }

      try {
        const [statusPayload, sessionLogsPayload, actionAnalysisPayload] =
          await Promise.all([
            fetchDashboardJson(`/${instanceId}`, parseEchoStatus),
            fetchDashboardJson(
              `/${instanceId}/session-logs`,
              parseDashboardSessionLogsResponse
            ),
            fetchDashboardJson(
              `/${instanceId}/action-analysis`,
              parseDashboardActionAnalysisResponse
            ),
          ]);

        if (!active) {
          return;
        }

        setStatus(statusPayload);
        setSessionLogsResponse(sessionLogsPayload);
        setActionAnalysisResponse(actionAnalysisPayload);
        setLastLoadedAt(new Date());
      } catch (loadError) {
        if (!active) {
          return;
        }

        setStatus(null);
        setSessionLogsResponse(null);
        setActionAnalysisResponse(null);
        setError(formatLoadError(loadError, 'Failed to load instance'));
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void load();

    return (): void => {
      active = false;
    };
  }, [instanceId, refreshToken]);

  return (
    <section className="stack">
      <DashboardDetailHeader
        instanceId={instanceId}
        instanceName={status?.name ?? '-'}
        loading={loading}
        lastLoadedAt={lastLoadedAt}
        onRefresh={() => {
          setRefreshToken((value) => value + 1);
        }}
      />

      {loading ? <p>Loading...</p> : null}
      {error !== null ? (
        <div className="error" role="alert">
          {error}
        </div>
      ) : null}

      {status !== null &&
      sessionLogsResponse !== null &&
      actionAnalysisResponse !== null ? (
        <DashboardDetailContent
          status={status}
          sessionLogsResponse={sessionLogsResponse}
          actionAnalysisResponse={actionAnalysisResponse}
          noteQuery={noteQuery}
          setNoteQuery={(value): void => {
            setNoteQuery(value);
          }}
          usageDays={usageDays}
          setUsageDays={(days): void => {
            setUsageDaysState(days);
          }}
          analysisDays={analysisDays}
          setAnalysisDays={(days): void => {
            setAnalysisDaysState(days);
          }}
        />
      ) : null}
    </section>
  );
}

const rootRoute = createRootRoute({
  component: DashboardShell,
});

const listRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: DashboardListPage,
});

const detailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/$instanceId',
  component: DashboardDetailPage,
});

const routeTree = rootRoute.addChildren([listRoute, detailRoute]);

export const router = createRouter({
  routeTree,
  basepath: '/dashboard',
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
