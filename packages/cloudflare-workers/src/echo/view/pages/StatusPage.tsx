import { formatDate } from '@echo-chamber/core';
import type { Note, UsageRecord } from '@echo-chamber/core';

import { Card } from '../components/Card';
import { Layout } from '../components/Layout';
import { NoteList } from '../components/NoteList';
import { UsageChart } from '../components/UsageChart';

import type { FC } from 'hono/jsx';

type EchoState = 'Idling' | 'Running' | 'Sleeping';

export interface StatusPageProps {
  id: string;
  name: string;
  state: EchoState;
  nextAlarm: string | null;
  notes: Note[];
  noteQuery: string;
  usage: UsageRecord;
}

export const StatusPage: FC<StatusPageProps> = async ({
  id,
  name,
  state,
  nextAlarm,
  notes,
  noteQuery,
  usage,
}) => {
  const today = new Date(Date.now() + 9 * 60 * 60 * 1000);
  const usageData = Array.from({ length: 7 }).map((_, i) => {
    const date = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() - 6 + i
    );
    return {
      date,
      usage: usage[formatDate(date)],
    };
  });
  const totalTokens = usageData.reduce(
    (sum, { usage }) => sum + (usage ? usage.total_tokens : 0),
    0
  );

  return (
    <Layout title={`Echo | ${name}`}>
      <header className="header" style={{ marginBottom: '12px' }}>
        <div className="row" style={{ width: '100%' }}>
          <div>
            <div className="pill mono">{id}</div>
            <h1 style={{ margin: '6px 0 0 0', fontSize: '18px' }}>{name}</h1>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="pill">State: {state}</div>
            <div className="small muted" style={{ marginTop: '6px' }}>
              next run: {nextAlarm ?? '—'}
            </div>
          </div>
        </div>
      </header>

      <main>
        <div className="grid">
          <Card
            title="Usage (7days)"
            right={<div className="pill mono">total {totalTokens} tokens</div>}
          >
            <UsageChart data={usageData} />
          </Card>
        </div>

        <div style={{ marginTop: '12px' }}>
          <Card title="Notes">
            <NoteList id={id} notes={notes} noteQuery={noteQuery} />
          </Card>
        </div>
      </main>
    </Layout>
  );
};
