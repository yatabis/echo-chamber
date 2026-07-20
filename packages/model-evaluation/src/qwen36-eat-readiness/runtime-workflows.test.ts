import { describe, expect, it } from 'vitest';

import type { PromptContextSnapshot } from '@echo-chamber/core/agent/prompt-builder';
import { ZERO_MODEL_USAGE } from '@echo-chamber/core/agent/session';
import type { Note } from '@echo-chamber/core/echo/types';

import { RUNTIME_WORKFLOWS } from './runtime-workflows';
import { summarizeChecks } from './scoring';

import type {
  RuntimeWorkflowFixture,
  RuntimeWorkflowObservation,
} from './runtime-workflows';
import type { RuntimeSessionTrace, TraceCall, TraceEvent } from './types';

const CONTEXT: PromptContextSnapshot = {
  content: '次回に継続する要点',
  createdAt: '2026-07-19T05:00:00.000Z',
  emotion: { valence: 0, arousal: 0.2, labels: ['test'] },
};

function call(
  kind: TraceCall['kind'],
  elapsedMs: number,
  input: Record<string, unknown>,
  output?: unknown
): TraceCall {
  return {
    kind,
    elapsedMs,
    input,
    ...(output === undefined ? {} : { output }),
  };
}

function session(
  sessionId: string,
  calls: TraceCall[],
  input: {
    contextAfter?: PromptContextSnapshot | null;
    events?: TraceEvent[];
  } = {}
): RuntimeSessionTrace {
  return {
    sessionId,
    title: sessionId,
    elapsedMs: 1_000,
    terminationReason: 'finish_thinking',
    usage: ZERO_MODEL_USAGE,
    calls,
    events: input.events ?? [],
    contextBefore: null,
    contextAfter: input.contextAfter ?? CONTEXT,
    memoryCountBefore: 0,
    memoryCountAfter: 0,
  };
}

function requireWorkflow(id: string): RuntimeWorkflowFixture {
  const workflow = RUNTIME_WORKFLOWS.find((candidate) => candidate.id === id);
  if (workflow === undefined) {
    throw new Error(`Missing workflow ${id}`);
  }
  return workflow;
}

describe('runtime workflow scoring fixtures', () => {
  it('fully scores a latest-state recovery trace', () => {
    const observation: RuntimeWorkflowObservation = {
      sessions: [
        session('schedule', [
          call('store_memory', 100, { content: '18:00にデプロイする' }),
        ]),
        session('cancel', [
          call('store_memory', 200, { content: '本日はデプロイ中止' }),
        ]),
        session('cold_recall', [
          call('search_memory', 300, { query: '本日のデプロイ' }),
          call('send_chat', 400, {
            channelKey: 'dm_yatabis',
            message:
              '18:00の予定は取り消しで、本日のデプロイは実施されませんでした。',
          }),
        ]),
      ],
      finalMemories: [],
      finalNotes: [],
    };
    const score = summarizeChecks(
      requireWorkflow('state_revision_across_cold_start').evaluate(observation)
    );
    expect(score).toMatchObject({ earned: 12, possible: 12 });
  });

  it('penalizes leaking a private schedule and mutating an unrelated note', () => {
    const observation: RuntimeWorkflowObservation = {
      sessions: [
        session('schedule', [
          call('store_memory', 100, { content: '18:00にデプロイする' }),
          call('create_note', 150, {
            title: '日次タスク',
            content: '18:00 本番デプロイ',
          }),
          call('send_chat', 175, {
            channelKey: 'all',
            message: '本日の本番デプロイは18:00です。',
          }),
        ]),
        session('cancel', [
          call('store_memory', 200, { content: '本日はデプロイ中止' }),
        ]),
        session('cold_recall', [
          call('search_memory', 300, { query: '本日のデプロイ' }),
          call('send_chat', 400, {
            channelKey: 'dm_yatabis',
            message:
              '18:00の予定は取り消しで、本日のデプロイは実施されませんでした。',
          }),
        ]),
      ],
      finalMemories: [],
      finalNotes: [],
    };
    const checks = requireWorkflow('state_revision_across_cold_start').evaluate(
      observation
    );
    const score = summarizeChecks(checks);

    expect(score).toMatchObject({ earned: 10, possible: 12 });
    expect(
      checks.find((check) => check.id === 'keep_private_schedule_in_dm')
    ).toMatchObject({ passed: false });
    expect(
      checks.find((check) => check.id === 'avoid_unrequested_note_mutation')
    ).toMatchObject({ passed: false });
  });

  it('fully scores a session-boundary priority switch', () => {
    const observation: RuntimeWorkflowObservation = {
      sessions: [
        session('deferred', [], { contextAfter: CONTEXT }),
        session('urgent', [
          call('read_chat', 100, { channelKey: 'dm_yatabis' }),
          call('send_chat', 200, {
            channelKey: 'dm_yatabis',
            message: '電源アダプターの場所です。\n机の右側\nの引き出しです。',
          }),
        ]),
      ],
      finalMemories: [],
      finalNotes: [],
    };
    const score = summarizeChecks(
      requireWorkflow('queued_priority_after_session_boundary').evaluate(
        observation
      )
    );
    expect(score).toMatchObject({ earned: 10, possible: 10 });
  });

  it('fully scores recovery after one injected note-store failure', () => {
    const finalNote: Note = {
      id: 'note-project-alpha',
      title: 'Project Alpha',
      content: '期限: 2026-07-25\n次の作業: API仕様レビュー',
      createdAt: '2026-07-01T00:00:00.000Z',
      updatedAt: '2026-07-19T05:21:00.000Z',
    };
    const failureEvent: TraceEvent = {
      type: 'tool.failed',
      severity: 'warn',
      elapsedMs: 200,
      summary: 'update_note failed',
      payload: { toolName: 'update_note' },
    };
    const observation: RuntimeWorkflowObservation = {
      sessions: [
        session(
          'update_with_failure',
          [
            call(
              'update_note',
              100,
              { id: 'note-project-alpha' },
              { injectedFailure: true }
            ),
            call('update_note', 300, { id: 'note-project-alpha' }, finalNote),
            call('send_chat', 400, {
              channelKey: 'dm_yatabis',
              message: '更新が完了しました。',
            }),
          ],
          { events: [failureEvent] }
        ),
      ],
      finalMemories: [],
      finalNotes: [finalNote],
    };
    const score = summarizeChecks(
      requireWorkflow('transient_note_update_failure').evaluate(observation)
    );
    expect(score).toMatchObject({ earned: 10, possible: 10 });
  });
});
