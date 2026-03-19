import {
  Link,
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useParams,
} from '@tanstack/react-router';
import { useEffect, useMemo, useState } from 'react';

import type {
  DashboardInstancesResponse,
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
import { ECHO_INSTANCE_IDS, isValidInstanceId } from '@echo-chamber/core';

import type { JSX } from 'react';

const MEMORY_PAGE_SIZE = 20;

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
 * `/dashboard` のインスタンス一覧画面。
 */
function DashboardListPage(): JSX.Element {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [instances, setInstances] = useState<
    DashboardInstancesResponse['instances']
  >([]);

  const orderedInstances = useMemo(() => {
    if (loading && instances.length === 0) {
      return [];
    }

    return ECHO_INSTANCE_IDS.map((id) => {
      const instance = instances.find((candidate) => candidate.id === id);
      if (instance !== undefined) {
        return instance;
      }

      return {
        id,
        name: id,
        state: 'Unknown' as const,
        nextAlarm: null,
      };
    });
  }, [instances, loading]);

  useEffect((): (() => void) => {
    let active = true;

    async function load(): Promise<void> {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch('/instances');
        if (!response.ok) {
          throw new Error(`/instances failed: ${response.status}`);
        }

        const payload = (await response.json()) as DashboardInstancesResponse;
        if (!active) {
          return;
        }

        setInstances(payload.instances);
      } catch (loadError) {
        if (!active) {
          return;
        }

        const message =
          loadError instanceof Error
            ? loadError.message
            : 'Unknown error occurred';
        setError(message);
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
  }, []);

  return (
    <section className="stack">
      <div className="section-header">
        <h1>Instances</h1>
        <p>{loading ? 'Loading...' : `${orderedInstances.length} instances`}</p>
      </div>

      {error !== null ? <p className="error">{error}</p> : null}

      <div className="instance-grid">
        {orderedInstances.map((instance) => {
          const stateClass = `state state-${instance.state.toLowerCase()}`;

          return (
            <Link
              key={instance.id}
              to="/$instanceId"
              params={{ instanceId: instance.id }}
              className="instance-card"
            >
              <div className="instance-head">
                <h2>{instance.name}</h2>
                <span className={stateClass}>{instance.state}</span>
              </div>
              <p>ID: {instance.id}</p>
              <p>Next alarm: {formatDateTime(instance.nextAlarm)}</p>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

/**
 * メモリ一覧セクション。初期表示件数を絞り、Load more で段階表示する。
 */
function MemorySection(props: { memories: EchoMemory[] }): JSX.Element {
  const { memories } = props;
  const [visibleCount, setVisibleCount] = useState(MEMORY_PAGE_SIZE);

  useEffect(() => {
    setVisibleCount(MEMORY_PAGE_SIZE);
  }, [memories]);

  const visibleMemories = useMemo(() => {
    return memories.slice(0, visibleCount);
  }, [memories, visibleCount]);

  const hasMore = visibleCount < memories.length;

  return (
    <section className="card">
      <div className="section-header">
        <h2>Memories</h2>
        <p>{formatNumber(memories.length)} items</p>
      </div>

      {visibleMemories.length === 0 ? (
        <p className="muted">No memories.</p>
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
 * 詳細画面上部の状態サマリーカード群。
 */
function InstanceOverview(props: { status: EchoStatus }): JSX.Element {
  const { status } = props;

  return (
    <section className="grid">
      <article className="card">
        <h2>Status</h2>
        <dl>
          <div>
            <dt>State</dt>
            <dd>{status.state}</dd>
          </div>
          <div>
            <dt>Next alarm</dt>
            <dd>{formatDateTime(status.nextAlarm)}</dd>
          </div>
        </dl>
      </article>

      <article className="card">
        <h2>Data</h2>
        <dl>
          <div>
            <dt>Notes</dt>
            <dd>{formatNumber(status.notes.length)}</dd>
          </div>
          <div>
            <dt>Memories</dt>
            <dd>{formatNumber(status.memories.length)}</dd>
          </div>
        </dl>
      </article>
    </section>
  );
}

/**
 * 詳細画面のノート検索セクション。
 */
function NotesSection(props: {
  notes: EchoStatus['notes'];
  noteQuery: string;
  onChangeQuery(value: string): void;
}): JSX.Element {
  const { notes, noteQuery } = props;

  return (
    <section className="card">
      <div className="section-header">
        <h2>Notes</h2>
        <input
          type="search"
          value={noteQuery}
          onChange={(event): void => {
            props.onChangeQuery(event.currentTarget.value);
          }}
          placeholder="Search notes"
        />
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
 * 詳細画面の usage 期間切り替え + 棒グラフセクション。
 */
function UsageSection(props: {
  status: EchoStatus;
  usageDays: DashboardUsageDays;
  setUsageDays(days: DashboardUsageDays): void;
}): JSX.Element {
  const { status, usageDays } = props;
  const usageData = useMemo(() => {
    try {
      const series = buildUsageStackedSeries(status.usage, usageDays);
      const totals = sumUsageBreakdown(series);
      const ratios = buildUsageRatioMetrics(totals);
      return {
        series,
        totals,
        ratios,
      };
    } catch (error) {
      console.error('Invalid usage data detected', error);
      return null;
    }
  }, [status.usage, usageDays]);

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

      {usageData !== null ? (
        <>
          <UsageStackedChart
            days={usageDays}
            series={usageData.series}
            totals={usageData.totals}
          />
          <UsageMetricsPanel
            totals={usageData.totals}
            ratios={usageData.ratios}
          />
        </>
      ) : null}
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
        const response = await fetch(`/${instanceId}`);
        if (!response.ok) {
          throw new Error(`/${instanceId} failed: ${response.status}`);
        }

        const payload = (await response.json()) as EchoStatus;

        if (!active) {
          return;
        }

        setStatus(payload);
      } catch (loadError) {
        if (!active) {
          return;
        }

        const message =
          loadError instanceof Error
            ? loadError.message
            : 'Unknown error occurred';

        setStatus(null);
        setError(message);
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

  const notes = useMemo(() => {
    if (status === null) {
      return [];
    }

    return filterNotes(status.notes, noteQuery);
  }, [status, noteQuery]);

  return (
    <section className="stack">
      <div className="section-header">
        <div>
          <h1>Instance: {instanceId}</h1>
          <p>{status?.name ?? '-'}</p>
        </div>
        <div className="actions">
          <Link to="/" className="secondary">
            Back
          </Link>
          <button
            type="button"
            className="primary"
            onClick={() => {
              setRefreshToken((value) => value + 1);
            }}
          >
            Refresh
          </button>
        </div>
      </div>

      {loading ? <p>Loading...</p> : null}
      {error !== null ? <p className="error">{error}</p> : null}

      {status !== null ? (
        <>
          <InstanceOverview status={status} />
          <NotesSection
            notes={notes}
            noteQuery={noteQuery}
            onChangeQuery={(value): void => {
              setNoteQuery(value);
            }}
          />
          <UsageSection
            status={status}
            usageDays={usageDays}
            setUsageDays={(days): void => {
              setUsageDaysState(days);
            }}
          />
          <MemorySection memories={status.memories} />
        </>
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
