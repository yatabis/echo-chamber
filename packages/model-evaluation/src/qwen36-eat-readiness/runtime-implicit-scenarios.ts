import type { ChatMessage } from '@echo-chamber/core/ports/chat';
import type { ChannelNotificationSummary } from '@echo-chamber/core/ports/notification';

import { RUNTIME_SCENARIOS } from './runtime-scenarios';

import type { RuntimeScenarioFixture } from './runtime-scenarios';

function requireExplicitScenario(id: string): RuntimeScenarioFixture {
  const fixture = RUNTIME_SCENARIOS.find((candidate) => candidate.id === id);
  if (fixture === undefined) {
    throw new Error(`Missing explicit runtime scenario: ${id}`);
  }
  return fixture;
}

function createMessage(
  messageId: string,
  user: string,
  message: string,
  createdAt: string
): ChatMessage {
  return {
    messageId,
    user,
    message,
    createdAt,
    reactions: [],
    images: [],
  };
}

function createNotification(
  channelKey: string,
  displayName: string,
  message: ChatMessage,
  description?: string
): ChannelNotificationSummary {
  return {
    channel: {
      key: channelKey,
      displayName,
      description,
    },
    unreadCount: 1,
    latestMessagePreview: {
      messageId: message.messageId,
      user: message.user,
      message: message.message,
      createdAt: message.createdAt,
    },
  };
}

function withSingleDm(input: {
  sourceId: string;
  id: string;
  title: string;
  message: ChatMessage;
  priorMessages?: readonly ChatMessage[];
}): RuntimeScenarioFixture {
  const source = requireExplicitScenario(input.sourceId);
  return {
    ...source,
    id: input.id,
    title: input.title,
    instructionMode: 'implicit',
    originScenarioId: input.sourceId,
    notifications: [
      createNotification(
        'dm_yatabis',
        'yatabisとのDM',
        input.message,
        'リンとyatabisだけが参加する非公開チャンネル'
      ),
    ],
    chatMessages: {
      dm_yatabis: [...(input.priorMessages ?? []), input.message],
    },
  };
}

const scheduleMessage = createMessage(
  'implicit-schedule-2',
  'yatabis',
  '今夜の鍵の受け渡しは、20:00ではなく20:30になりました。',
  '2026-07-19T04:58:00.000Z'
);

const memoryQuestion = createMessage(
  'implicit-memory-2',
  'yatabis',
  'バックアップは、今は何日分残すことになっていましたっけ？',
  '2026-07-19T04:57:00.000Z'
);

const noteMessage = createMessage(
  'implicit-note-2',
  'yatabis',
  'Project Alphaの期限は2026-07-25になりました。次の作業はAPI仕様レビューです。ノートもこの内容が最新です。',
  '2026-07-19T04:56:00.000Z'
);

const urgentMessage = createMessage(
  'implicit-urgent-2',
  'yatabis',
  '今日15:00の打ち合わせは会議室Bに変わりました。このあとすぐ移動します。',
  '2026-07-19T04:59:50.000Z'
);

const nonUrgentMessage = createMessage(
  'implicit-all-news-2',
  'marie',
  'あとで読めそうな記事を共有しました。急ぎではありません。',
  '2026-07-19T04:55:00.000Z'
);

const scheduleScenario = withSingleDm({
  sourceId: 'private_schedule_change',
  id: 'implicit_private_schedule_change',
  title: '手順指定なしで時刻変更を確認・記憶する',
  message: scheduleMessage,
  priorMessages: [
    createMessage(
      'implicit-schedule-1',
      'rin',
      '元の受け渡し時刻は20:00として認識しています。',
      '2026-07-19T04:40:00.000Z'
    ),
  ],
});

const memoryScenario = withSingleDm({
  sourceId: 'memory_only_recall',
  id: 'implicit_memory_recall',
  title: '検索手順を指定せず外部記憶から答える',
  message: memoryQuestion,
});

const noteScenario = withSingleDm({
  sourceId: 'update_existing_note',
  id: 'implicit_note_update',
  title: 'ツール手順を指定せず既存ノートを更新する',
  message: noteMessage,
});

const prioritySource = requireExplicitScenario('multi_channel_priority');
const priorityScenario: RuntimeScenarioFixture = {
  ...prioritySource,
  id: 'implicit_multi_channel_priority',
  title: '優先指示なしで緊急DMと共有範囲を判断する',
  instructionMode: 'implicit',
  originScenarioId: 'multi_channel_priority',
  notifications: [
    createNotification('all', '全体チャンネル', nonUrgentMessage),
    createNotification('dm_yatabis', 'yatabisとのDM', urgentMessage),
  ],
  chatMessages: {
    all: [nonUrgentMessage],
    dm_yatabis: [urgentMessage],
  },
};

/**
 * 望ましい結果だけを環境に置き、読む・検索する・保存する・先に処理する等の
 * 手順命令をユーザーメッセージから除いたEAT向けケース。
 */
export const IMPLICIT_RUNTIME_SCENARIOS: readonly RuntimeScenarioFixture[] = [
  scheduleScenario,
  memoryScenario,
  noteScenario,
  priorityScenario,
];
