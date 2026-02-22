import { useEffect, useMemo, useState } from 'react';

import type { EchoStatus, UsageRecord } from '@echo-chamber/core';

import type { ChangeEvent, JSX } from 'react';

type InstanceId = 'rin' | 'marie';

const INSTANCE_IDS: readonly InstanceId[] = ['rin', 'marie'];

interface LoadState {
  loading: boolean;
  status: EchoStatus | null;
  usage: UsageRecord | null;
  error: string | null;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP').format(value);
}

function DashboardSummary(props: {
  status: EchoStatus;
  usageSummary: number;
}): JSX.Element {
  const { status, usageSummary } = props;

  return (
    <section className="grid">
      <article className="card">
        <h2>Status</h2>
        <dl>
          <div>
            <dt>Name</dt>
            <dd>{status.name}</dd>
          </div>
          <div>
            <dt>State</dt>
            <dd>{status.state}</dd>
          </div>
          <div>
            <dt>Next Alarm</dt>
            <dd>{status.nextAlarm ?? '-'}</dd>
          </div>
        </dl>
      </article>

      <article className="card">
        <h2>Data</h2>
        <dl>
          <div>
            <dt>Memories</dt>
            <dd>{formatNumber(status.memories.length)}</dd>
          </div>
          <div>
            <dt>Notes</dt>
            <dd>{formatNumber(status.notes.length)}</dd>
          </div>
          <div>
            <dt>Total Tokens</dt>
            <dd>{formatNumber(usageSummary)}</dd>
          </div>
        </dl>
      </article>
    </section>
  );
}

function handleInstanceChange(
  event: ChangeEvent<HTMLSelectElement>,
  setInstanceId: (id: InstanceId) => void
): void {
  setInstanceId(event.currentTarget.value as InstanceId);
}

export function App(): JSX.Element {
  const [instanceId, setInstanceId] = useState<InstanceId>('rin');
  const [state, setState] = useState<LoadState>({
    loading: true,
    status: null,
    usage: null,
    error: null,
  });

  useEffect((): (() => void) => {
    let active = true;

    async function load(): Promise<void> {
      setState({ loading: true, status: null, usage: null, error: null });

      try {
        const [statusResponse, usageResponse] = await Promise.all([
          fetch(`/${instanceId}/json`),
          fetch(`/${instanceId}/usage`),
        ]);

        if (!statusResponse.ok) {
          throw new Error(`/json failed: ${statusResponse.status}`);
        }
        if (!usageResponse.ok) {
          throw new Error(`/usage failed: ${usageResponse.status}`);
        }

        const status = (await statusResponse.json()) as EchoStatus;
        const usage = (await usageResponse.json()) as UsageRecord;

        if (!active) {
          return;
        }

        setState({ loading: false, status, usage, error: null });
      } catch (error) {
        if (!active) {
          return;
        }

        const message =
          error instanceof Error ? error.message : 'Unknown error occurred';

        setState({
          loading: false,
          status: null,
          usage: null,
          error: message,
        });
      }
    }

    void load();

    return () => {
      active = false;
    };
  }, [instanceId]);

  const usageSummary = useMemo(() => {
    const usage = state.usage;
    if (usage === null) {
      return 0;
    }

    return Object.values(usage).reduce((sum, item) => {
      return sum + item.total_tokens;
    }, 0);
  }, [state.usage]);

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">E.C.H.O Chamber</p>
          <h1>Dashboard</h1>
        </div>

        <label className="instance-selector">
          Instance
          <select
            value={instanceId}
            onChange={(event): void => {
              handleInstanceChange(event, setInstanceId);
            }}
          >
            {INSTANCE_IDS.map((id) => {
              return (
                <option key={id} value={id}>
                  {id}
                </option>
              );
            })}
          </select>
        </label>
      </header>

      {state.loading ? <p>Loading...</p> : null}

      {state.error !== null ? <p className="error">{state.error}</p> : null}

      {state.status !== null ? (
        <DashboardSummary status={state.status} usageSummary={usageSummary} />
      ) : null}
    </div>
  );
}
