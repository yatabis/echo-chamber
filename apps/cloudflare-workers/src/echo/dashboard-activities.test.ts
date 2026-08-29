import { describe, expect, it } from 'vitest';

import type { DashboardEchoEvent } from '@echo-chamber/contracts/dashboard/types';

import { buildDashboardSessionLogsResponse } from './dashboard-activities';

function createEvent(
  overrides: Partial<DashboardEchoEvent> & Pick<DashboardEchoEvent, 'type'>
): DashboardEchoEvent {
  return {
    id: `event-${overrides.type}`,
    archiveDay: '2026-06-01',
    category: 'system',
    createdAt: '2026-06-01T12:00:00.000Z',
    payload: null,
    sessionId: 'session-1',
    severity: 'info',
    streams: ['system', 'analysis'],
    summary: overrides.type,
    ...overrides,
  };
}

describe('buildDashboardSessionLogsResponse', () => {
  it('tool called / completed を dashboard 上の 1 activity に畳む', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'tool-completed',
          type: 'tool.completed',
          category: 'tool',
          createdAt: '2026-06-01T12:00:02.000Z',
          streams: ['system', 'analysis'],
          payload: {
            callId: 'call-1',
            durationMs: 30,
            entityId: 'note-1',
            entityType: 'note',
            operation: 'note.create',
            outputLength: 120,
            success: true,
            toolName: 'create_note',
            turnIndex: 1,
          },
        }),
        createEvent({
          id: 'tool-called',
          type: 'tool.called',
          category: 'tool',
          createdAt: '2026-06-01T12:00:01.000Z',
          streams: ['thought', 'analysis'],
          payload: {
            callId: 'call-1',
            input: '{"title":"Daily note"}',
            toolName: 'create_note',
            turnIndex: 1,
          },
        }),
      ],
    });

    expect(response).toMatchObject({
      archiveDay: '2026-06-01',
      sessionLogs: [
        {
          id: 'session:session-1',
          activities: [
            {
              id: 'tool-completed',
              title: 'Used create_note',
              kind: 'action',
              tone: 'positive',
              details: {
                callId: 'call-1',
                input: {
                  title: 'Daily note',
                },
                operation: 'note.create',
              },
            },
          ],
        },
      ],
    });
  });

  it('No tool calls returned を activity として返す', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'turn-completed',
          type: 'model.turn.completed',
          category: 'model',
          severity: 'warn',
          streams: ['analysis'],
          payload: {
            durationMs: 200,
            outputItemCount: 1,
            toolCallCount: 0,
            turnIndex: 2,
            warnings: ['no_tool_calls'],
          },
        }),
      ],
    });

    expect(response.sessionLogs).toHaveLength(1);
    expect(response.sessionLogs[0]?.activities[0]).toMatchObject({
      id: 'turn-completed',
      kind: 'decision',
      title: 'No tool calls returned',
      tone: 'warning',
    });
  });

  it('dashboard に不要な低レベル event は返さない', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'alarm-scheduled',
          type: 'system.schedule.alarm_scheduled',
          category: 'system',
          severity: 'debug',
          streams: ['system', 'analysis'],
        }),
        createEvent({
          id: 'next-wake-invalid',
          type: 'system.schedule.next_wake_at_invalidated',
          category: 'system',
          severity: 'warn',
          streams: ['system', 'analysis'],
        }),
      ],
    });

    expect(response.sessionLogs).toEqual([]);
  });

  it('session 中の明示的な issue event は activity として返す', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'model-warning',
          type: 'model.provider.warning',
          category: 'model',
          payload: {
            code: 'missing_response_token',
            model: 'gpt-5.5',
            provider: 'openai.responses',
            turnIndex: 1,
          },
          severity: 'warn',
          streams: ['system', 'analysis'],
          summary: 'response token was missing',
        }),
        createEvent({
          id: 'memory-failed',
          type: 'memory.search.failed',
          category: 'memory',
          payload: {
            error: 'vector store unavailable',
            query: 'startup context',
            source: 'startup_context',
          },
          severity: 'warn',
          streams: ['system', 'analysis'],
          summary: 'failed to load related memories',
        }),
      ],
    });

    expect(
      response.sessionLogs[0]?.activities.map((activity) => ({
        id: activity.id,
        kind: activity.kind,
        title: activity.title,
        tone: activity.tone,
      }))
    ).toEqual([
      {
        id: 'model-warning',
        kind: 'issue',
        title: 'Model warning',
        tone: 'warning',
      },
      {
        id: 'memory-failed',
        kind: 'issue',
        title: 'Memory search failed',
        tone: 'warning',
      },
    ]);
  });

  it('Cognitive commit / failure を operator-facing activity にする', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'cognitive-commit',
          type: 'cognitive.phase.committed',
          category: 'memory',
          payload: {
            boundaryId: 'activation-1:2:post_main',
            committedVersion: 1,
            memoryUpdates: 1,
            phase: 'post_main',
          },
        }),
        createEvent({
          id: 'cognitive-failure',
          type: 'cognitive.phase.failed',
          category: 'memory',
          createdAt: '2026-06-01T12:00:01.000Z',
          payload: {
            boundaryId: 'activation-2:1:pre_main',
            commitError: 'sqlite disk full',
            memory: {
              attempts: 2,
              error: 'temporary failure',
              reason: 'retry_exhausted',
              status: 'failed',
            },
            phase: 'pre_main',
          },
          severity: 'error',
        }),
      ],
    });

    expect(
      response.sessionLogs[0]?.activities.map((activity) => ({
        id: activity.id,
        kind: activity.kind,
        title: activity.title,
        tone: activity.tone,
      }))
    ).toEqual([
      {
        id: 'cognitive-commit',
        kind: 'knowledge',
        title: 'Cognitive phase committed',
        tone: 'positive',
      },
      {
        id: 'cognitive-failure',
        kind: 'issue',
        title: 'Cognitive phase failed',
        tone: 'critical',
      },
    ]);
    expect(response.sessionLogs[0]?.activities[1]?.details).toMatchObject({
      commitError: 'sqlite disk full',
    });
  });

  it('Cognitive model の本文を Main と同じ activity に保ち、module を識別する', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'memory-output',
          type: 'model.output.emitted',
          category: 'model',
          payload: {
            cognitiveModule: 'memory',
            content: '{"query":"前回の設計判断"}',
            model: 'gpt-5.6-luna',
            provider: 'openai.responses',
            turnIndex: 1,
          },
          streams: ['thought', 'analysis'],
        }),
        createEvent({
          id: 'main-output',
          type: 'model.output.emitted',
          category: 'model',
          createdAt: '2026-06-01T12:00:01.000Z',
          payload: {
            content: '次の行動を検討する。',
            model: 'gpt-5.6-sol',
            provider: 'openai.responses',
            turnIndex: 1,
          },
          streams: ['thought', 'analysis'],
        }),
      ],
    });

    expect(response.sessionLogs[0]?.activities).toMatchObject([
      {
        body: '{"query":"前回の設計判断"}',
        details: { cognitiveModule: 'memory' },
        title: 'Memory Module',
      },
      {
        body: '次の行動を検討する。',
        title: 'Echo',
      },
    ]);
  });

  it('sessionId のない event は session log の材料にしない', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'orphan-warning',
          type: 'system.echo_state.changed',
          category: 'system',
          payload: {
            nextState: 'Sleeping',
            previousState: 'Idling',
            reason: 'daily_sleep',
          },
          sessionId: null,
          severity: 'warn',
          streams: ['system', 'analysis'],
        }),
      ],
    });

    expect(response.sessionLogs).toEqual([]);
  });

  it('時刻表示は JST に固定する', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'session-started',
          type: 'session.started',
          category: 'session',
          createdAt: '2026-06-01T15:00:00.000Z',
          streams: ['thought', 'system', 'analysis'],
        }),
      ],
    });

    expect(response.sessionLogs[0]?.meta[0]).toBe('2026/06/02 00:00:00');
    expect(response.sessionLogs[0]?.activities[0]?.meta[0]).toBe(
      '2026/06/02 00:00:00'
    );
  });

  it('session log は新しい順、log 内 activity は時系列順に並べる', () => {
    const response = buildDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      events: [
        createEvent({
          id: 'old-output',
          type: 'model.output.emitted',
          category: 'model',
          createdAt: '2026-06-01T12:05:00.000Z',
          payload: {
            content: 'Old session output.',
          },
          sessionId: 'session-old',
          streams: ['thought', 'analysis'],
        }),
        createEvent({
          id: 'new-start',
          type: 'session.started',
          category: 'session',
          createdAt: '2026-06-01T13:00:00.000Z',
          sessionId: 'session-new',
          streams: ['thought', 'system', 'analysis'],
        }),
        createEvent({
          id: 'old-start',
          type: 'session.started',
          category: 'session',
          createdAt: '2026-06-01T12:00:00.000Z',
          sessionId: 'session-old',
          streams: ['thought', 'system', 'analysis'],
        }),
      ],
    });

    expect(response.sessionLogs.map((sessionLog) => sessionLog.id)).toEqual([
      'session:session-new',
      'session:session-old',
    ]);
    expect(
      response.sessionLogs[1]?.activities.map((activity) => activity.id)
    ).toEqual(['old-start', 'old-output']);
  });
});
