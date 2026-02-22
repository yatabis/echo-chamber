import { z } from 'zod';

import { formatDatetimeForAgent, getErrorMessage } from '@echo-chamber/core';
import type * as DiscordApi from '@echo-chamber/core/discord';

import { Tool } from '.';

async function getDiscordApi(): Promise<typeof DiscordApi> {
  return import('@echo-chamber/core/discord');
}

export const checkNotificationsFunction = new Tool(
  'check_notifications',
  'チャットチャンネルの新しい通知を確認する。未読メッセージ数と最新メッセージのプレビューを返す。通知が見つかった場合は、内容を確認し、必要に応じて対応することを推奨する。',
  {},
  async (_, ctx) => {
    try {
      const { chatChannelId, discordBotToken } = ctx.instanceConfig;
      const { getNotificationDetails } = await getDiscordApi();

      const notificationDetails = await getNotificationDetails(
        discordBotToken,
        chatChannelId
      );

      return {
        success: true,
        notifications: {
          channel: 'chat',
          unreadCount:
            notificationDetails.unreadCount > 99
              ? '99+'
              : notificationDetails.unreadCount,
          latestMessagePreview: notificationDetails.latestMessagePreview,
        },
      };
    } catch (error) {
      await ctx.logger.error(
        `Error checking notifications: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to fetch notifications',
      };
    }
  }
);

export const readChatMessagesFunction = new Tool(
  'read_chat_messages',
  'チャットチャンネルからチャットメッセージを読み取る。最新のメッセージをタイムスタンプの昇順で返す。会話の文脈を理解するために、十分な数のメッセージを取得するのが良い。取得したメッセージ数では状況を完全に把握できない場合は、より大きな制限値でこのツールを再度呼び出すことができる。',
  {
    limit: z.int().min(1).max(100).describe('取得するメッセージ数'),
  },
  async ({ limit }, ctx) => {
    try {
      const { chatChannelId, discordBotToken } = ctx.instanceConfig;
      const { getChannelMessages } = await getDiscordApi();

      const messages = await getChannelMessages(
        discordBotToken,
        chatChannelId,
        {
          limit,
        }
      );

      return {
        success: true,
        // 投稿日時の昇順
        messages: messages.reverse().map((message) => ({
          messageId: message.id,
          user: message.author.username,
          message: message.content,
          created_at: formatDatetimeForAgent(new Date(message.timestamp)),
          reactions: message.reactions?.map((reaction) => ({
            emoji: reaction.emoji.name,
            me: reaction.me,
          })),
        })),
      };
    } catch (error) {
      await ctx.logger.error(
        `Error reading chat messages: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to read messages',
      };
    }
  }
);

export const sendChatMessageFunction = new Tool(
  'send_chat_message',
  'チャットチャンネルにメッセージを送信する。あなたの考えは、それを伝える行動を起こさなければ伝わらない。チャットにメッセージを送ることはその方法の一つである。',
  {
    message: z
      .string()
      .min(1)
      .max(2000)
      .describe('送信するメッセージ内容。最大2000文字。'),
  },
  async ({ message }, ctx) => {
    try {
      const { chatChannelId, discordBotToken } = ctx.instanceConfig;
      const { sendChannelMessage } = await getDiscordApi();

      await sendChannelMessage(discordBotToken, chatChannelId, {
        content: message,
      });

      return {
        success: true,
      };
    } catch (error) {
      await ctx.logger.error(
        `Error sending chat message: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to send message',
      };
    }
  }
);

export const addReactionToChatMessageFunction = new Tool(
  'add_reaction_to_chat_message',
  '特定のチャットメッセージにリアクションを追加する。リアクションは有効な絵文字文字列である必要がある。メッセージにリアクションすると、そこまでのメッセージは既読としてマークされる。メッセージに返信する必要性を感じないが、読んだことを示したい場合は、リアクションを付けることができる。返信もリアクションもしなければ、他者はあなたがそのメッセージを読んだかどうかすら分からない。',
  {
    messageId: z.string().describe('リアクションを付けるメッセージのID'),
    reaction: z.string().describe('追加するリアクション（絵文字文字列）'),
  },
  async ({ messageId, reaction }, ctx) => {
    try {
      const { chatChannelId, discordBotToken } = ctx.instanceConfig;
      const { addReactionToMessage } = await getDiscordApi();

      await addReactionToMessage(
        discordBotToken,
        chatChannelId,
        messageId,
        reaction
      );

      return {
        success: true,
      };
    } catch (error) {
      await ctx.logger.error(
        `Error adding reaction to chat message: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to add reaction',
      };
    }
  }
);
