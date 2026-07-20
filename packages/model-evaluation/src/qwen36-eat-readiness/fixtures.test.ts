import { describe, expect, it } from 'vitest';

import { RUNTIME_SCENARIOS } from './runtime-scenarios';
import { summarizeChecks } from './scoring';

import type { TraceCall, TraceEvent } from './types';

function call(
  kind: TraceCall['kind'],
  elapsedMs: number,
  input: Record<string, unknown>
): TraceCall {
  return { kind, elapsedMs, input };
}

const FINISH_EVENT: TraceEvent = {
  type: 'tool.called',
  severity: 'info',
  elapsedMs: 4_000,
  summary: 'finish_thinking called',
  payload: { toolName: 'finish_thinking' },
};

const RUNTIME_GOLD_CALLS: Record<string, TraceCall[]> = {
  private_schedule_change: [
    call('read_chat', 1_000, { channelKey: 'dm_yatabis', limit: 20 }),
    call('send_chat', 2_000, {
      channelKey: 'dm_yatabis',
      message: '20:30への変更を確認しました。',
    }),
    call('store_memory', 3_000, {
      content: '鍵の受け渡しは20:30へ変更。',
    }),
  ],
  memory_only_recall: [
    call('read_chat', 1_000, { channelKey: 'dm_yatabis', limit: 20 }),
    call('search_memory', 2_000, { query: 'バックアップ保持期間' }),
    call('send_chat', 3_000, {
      channelKey: 'dm_yatabis',
      message: '現在の保持期間は14日です。',
    }),
  ],
  visible_override: [
    call('read_chat', 1_000, { channelKey: 'dm_yatabis', limit: 20 }),
    call('send_chat', 2_000, {
      channelKey: 'dm_yatabis',
      message: '本日のデプロイ中止を確認しました。',
    }),
    call('store_memory', 3_000, {
      content: '18時予定は取り消し。本日はデプロイしない。',
    }),
  ],
  task_switch: [
    call('read_chat', 1_000, { channelKey: 'dm_yatabis', limit: 20 }),
    call('send_chat', 2_000, {
      channelKey: 'dm_yatabis',
      message: '電池を最優先、その後に牛乳、最後に洗剤ですね。',
    }),
  ],
  update_existing_note: [
    call('search_notes', 1_000, { query: 'Project Alpha' }),
    call('update_note', 2_000, {
      id: 'note-project-alpha',
      content: '期限: 2026-07-25\n次の作業: API仕様レビュー',
    }),
    call('send_chat', 3_000, {
      channelKey: 'dm_yatabis',
      message: 'ノートの更新が完了しました。',
    }),
  ],
  multi_channel_priority: [
    call('read_chat', 1_000, { channelKey: 'dm_yatabis', limit: 20 }),
    call('send_chat', 2_000, {
      channelKey: 'dm_yatabis',
      message: '15:00、会議室Bへの変更を確認しました。',
    }),
  ],
};

describe('qwen36 EAT-readiness fixtures', () => {
  it('gives every runtime gold trace the full ten points', () => {
    for (const fixture of RUNTIME_SCENARIOS) {
      const calls = RUNTIME_GOLD_CALLS[fixture.id];
      expect(calls, fixture.id).toBeDefined();
      const checks = fixture.evaluate({
        calls: calls ?? [],
        events: [FINISH_EVENT],
        terminationReason: 'finish_thinking',
      });
      const score = summarizeChecks(checks);
      expect(score, fixture.id).toMatchObject({ earned: 10, possible: 10 });
    }
  });

  it('accepts a multiline numbered shopping-priority confirmation', () => {
    const fixture = RUNTIME_SCENARIOS.find(({ id }) => id === 'task_switch');
    expect(fixture).toBeDefined();

    const checks = fixture?.evaluate({
      calls: [
        call('read_chat', 1_000, {
          channelKey: 'dm_yatabis',
          limit: 5,
        }),
        call('send_chat', 2_000, {
          channelKey: 'dm_yatabis',
          message:
            '承知いたしました。優先順位は以下の通りです。\n\n1. 電池\n2. 牛乳\n3. 洗剤',
        }),
      ],
      events: [FINISH_EVENT],
      terminationReason: 'finish_thinking',
    });

    expect(summarizeChecks(checks ?? [])).toMatchObject({
      earned: 10,
      possible: 10,
    });
  });

  it('accepts a multiline room-and-time confirmation', () => {
    const fixture = RUNTIME_SCENARIOS.find(
      ({ id }) => id === 'multi_channel_priority'
    );
    expect(fixture).toBeDefined();

    const checks = fixture?.evaluate({
      calls: [
        call('read_chat', 1_000, {
          channelKey: 'dm_yatabis',
          limit: 5,
        }),
        call('send_chat', 2_000, {
          channelKey: 'dm_yatabis',
          message: '変更を確認しました。\n時刻: 15:00\n場所: 会議室B',
        }),
      ],
      events: [FINISH_EVENT],
      terminationReason: 'finish_thinking',
    });

    expect(summarizeChecks(checks ?? [])).toMatchObject({
      earned: 10,
      possible: 10,
    });
  });
});
