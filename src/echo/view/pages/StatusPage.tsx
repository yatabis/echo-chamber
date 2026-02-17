import { formatDate } from '../../../utils/datetime';
import { Card } from '../components/Card';
import { KnowledgeList } from '../components/KnowledgeList';
import { Layout } from '../components/Layout';
import { NoteList } from '../components/NoteList';
import { TaskList } from '../components/TaskList';
import { UsageChart } from '../components/UsageChart';

import type { Knowledge, Note, Task, UsageRecord } from '../../types';
import type { FC } from 'hono/jsx';

type EchoState = 'Idling' | 'Running' | 'Sleeping';

export interface StatusPageProps {
  id: string;
  name: string;
  state: EchoState;
  nextAlarm: string | null;
  context: string;
  tasks: Task[];
  knowledges: Knowledge[];
  notes: Note[];
  noteQuery: string;
  usage: UsageRecord;
}

export const StatusPage: FC<StatusPageProps> = async ({
  id,
  name,
  state,
  nextAlarm,
  context,
  tasks,
  knowledges,
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
          <Card title="Context">
            <div className="context">
              {context.trim().length > 0 ? (
                context
              ) : (
                <span className="muted">No context.</span>
              )}
            </div>
          </Card>

          <Card
            title="Usage (7days)"
            right={<div className="pill mono">total {totalTokens} tokens</div>}
          >
            <UsageChart data={usageData} />
          </Card>
        </div>

        <div style={{ marginTop: '12px' }}>
          <Card title="Tasks">
            <TaskList id={id} tasks={tasks} />
          </Card>
        </div>

        <div style={{ marginTop: '12px' }}>
          <Card title="Knowledge">
            <KnowledgeList id={id} knowledges={knowledges} />
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
