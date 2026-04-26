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
  parseDashboardInstancesResponse,
  parseEchoStatus,
} from '@echo-chamber/contracts/dashboard/schemas';
import type {
  DashboardInstanceSummary,
  DashboardInstancesResponse,
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

type DetailTab = 'overview' | 'notes' | 'memories';

const DETAIL_TABS: {
  id: DetailTab;
  label: string;
}[] = [
  {
    id: 'overview',
    label: 'Overview',
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
  }).format(date);
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
      totalCost: 0,
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
    noteCount: 0,
    memoryCount: 0,
    todayUsageTokens: 0,
    sevenDayUsageTokens: 0,
    thirtyDayUsageTokens: 0,
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
        <span>Cost</span>
        <strong>${point.totalCost.toFixed(4)}</strong>
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
          total {formatNumber(totals.totalTokens)} tokens / cost $
          {totals.totalCost.toFixed(4)}
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
        <h3>Total cost</h3>
        <p className="usage-metric-emphasis">${totals.totalCost.toFixed(4)}</p>
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
 * 一覧画面の状態件数と横断メトリクスを集計する。
 */
function buildFleetSummary(instances: DashboardInstanceSummary[]): {
  latestActivityAt: string | null;
  nextInstance: DashboardInstanceSummary | null;
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
  let nextInstance: DashboardInstanceSummary | null = null;

  for (const instance of instances) {
    stateCounts[instance.state] += 1;

    if (instance.nextAlarm !== null) {
      const nextTime = new Date(instance.nextAlarm).getTime();
      const currentNextTime =
        nextInstance?.nextAlarm === undefined || nextInstance.nextAlarm === null
          ? Number.POSITIVE_INFINITY
          : new Date(nextInstance.nextAlarm).getTime();
      if (
        !Number.isNaN(nextTime) &&
        nextTime >= Date.now() &&
        nextTime < currentNextTime
      ) {
        nextInstance = instance;
      }
    }
  }

  return {
    latestActivityAt: findLatestDateTime(
      instances.flatMap((instance) => [
        instance.latestNoteUpdatedAt,
        instance.latestMemoryUpdatedAt,
      ])
    ),
    nextInstance,
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
          <span>Today usage</span>
          <strong>{formatNumber(instance.todayUsageTokens)} tokens</strong>
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
    if (normalizedQuery.length === 0) {
      return memories;
    }

    return memories.filter((memory) => {
      return (
        memory.content.toLowerCase().includes(normalizedQuery) ||
        memory.type.toLowerCase().includes(normalizedQuery) ||
        memory.emotion.labels.some((label) => {
          return label.toLowerCase().includes(normalizedQuery);
        })
      );
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
  counts: { memoryCount: number; noteCount: number }
): number | null {
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
  memoryCount: number;
  noteCount: number;
  onChange(tab: DetailTab): void;
}): JSX.Element {
  return (
    <div className="detail-tabs" role="tablist" aria-label="Detail sections">
      {DETAIL_TABS.map((tab) => {
        const active = props.activeTab === tab.id;
        const count = getDetailTabCount(tab.id, {
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
  noteQuery: string;
  notes: EchoStatus['notes'];
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
  noteQuery: string;
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

  return (
    <section className="stack">
      <DetailTabs
        activeTab={activeTab}
        noteCount={props.status.notes.length}
        memoryCount={props.status.memories.length}
        onChange={(tab): void => {
          setActiveTab(tab);
        }}
      />
      <DetailTabPanel
        activeTab={activeTab}
        status={props.status}
        usageAnalysis={usageAnalysis}
        usageDays={props.usageDays}
        setUsageDays={(days): void => {
          props.setUsageDays(days);
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [noteQuery, setNoteQuery] = useState('');
  const [usageDays, setUsageDaysState] = useState<DashboardUsageDays>(7);
  const [refreshToken, setRefreshToken] = useState(0);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);

  useEffect((): (() => void) => {
    let active = true;

    async function load(): Promise<void> {
      setLoading(true);
      setError(null);

      if (!isValidInstanceId(instanceId)) {
        setStatus(null);
        setError(`Invalid instance ID: ${instanceId}`);
        setLoading(false);
        return;
      }

      try {
        const payload = await fetchDashboardJson(
          `/${instanceId}`,
          parseEchoStatus
        );

        if (!active) {
          return;
        }

        setStatus(payload);
        setLastLoadedAt(new Date());
      } catch (loadError) {
        if (!active) {
          return;
        }

        setStatus(null);
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

      {status !== null ? (
        <DashboardDetailContent
          status={status}
          noteQuery={noteQuery}
          setNoteQuery={(value): void => {
            setNoteQuery(value);
          }}
          usageDays={usageDays}
          setUsageDays={(days): void => {
            setUsageDaysState(days);
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
