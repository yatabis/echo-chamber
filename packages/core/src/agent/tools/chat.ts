import { z } from 'zod';

import { createToolResultSchema, defineToolSpecification } from './shared';

const chatMessageReactionSchema = z.object({
  emoji: z.string().nullable(),
  me: z.boolean(),
});

const chatMessageSchema = z.object({
  messageId: z.string(),
  user: z.string(),
  message: z.string(),
  created_at: z.string(),
  reactions: z.array(chatMessageReactionSchema).optional(),
});

const notificationPreviewSchema = z.object({
  messageId: z.string(),
  user: z.string(),
  message: z.string(),
  created_at: z.string(),
});

export const checkNotificationsToolSpec = defineToolSpecification({
  name: 'check_notifications',
  description:
    'チャットチャンネルの新しい通知を確認する。未読メッセージ数と最新メッセージのプレビューを返す。通知が見つかった場合は、内容を確認し、必要に応じて対応することを推奨する。',
  parameters: {},
  outputSchema: createToolResultSchema({
    notifications: z.object({
      channel: z.literal('chat'),
      unreadCount: z.union([z.number(), z.literal('99+')]),
      latestMessagePreview: notificationPreviewSchema.nullable(),
    }),
  }),
});

export const readChatMessagesToolSpec = defineToolSpecification({
  name: 'read_chat_messages',
  description:
    'チャットチャンネルからチャットメッセージを読み取る。最新のメッセージをタイムスタンプの昇順で返す。会話の文脈を理解するために、十分な数のメッセージを取得するのが良い。取得したメッセージ数では状況を完全に把握できない場合は、より大きな制限値でこのツールを再度呼び出すことができる。',
  parameters: {
    limit: z.int().min(1).max(100).describe('取得するメッセージ数'),
  },
  outputSchema: createToolResultSchema({
    messages: z.array(chatMessageSchema),
  }),
});

export const sendChatMessageToolSpec = defineToolSpecification({
  name: 'send_chat_message',
  description:
    'チャットチャンネルにメッセージを送信する。あなたの考えは、それを伝える行動を起こさなければ伝わらない。チャットにメッセージを送ることはその方法の一つである。',
  parameters: {
    message: z
      .string()
      .min(1)
      .max(2000)
      .describe('送信するメッセージ内容。最大2000文字。'),
  },
  outputSchema: createToolResultSchema({}),
});

export const addReactionToChatMessageToolSpec = defineToolSpecification({
  name: 'add_reaction_to_chat_message',
  description:
    '特定のチャットメッセージにリアクションを追加する。リアクションは有効な絵文字文字列である必要がある。メッセージにリアクションすると、そこまでのメッセージは既読としてマークされる。メッセージに返信する必要性を感じないが、読んだことを示したい場合は、リアクションを付けることができる。返信もリアクションもしなければ、他者はあなたがそのメッセージを読んだかどうかすら分からない。',
  parameters: {
    messageId: z.string().describe('リアクションを付けるメッセージのID'),
    reaction: z.string().describe('追加するリアクション（絵文字文字列）'),
  },
  outputSchema: createToolResultSchema({}),
});
