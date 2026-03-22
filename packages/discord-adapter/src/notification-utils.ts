import type { ChannelNotificationSummary } from '@echo-chamber/core/ports/notification';
import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';

import { getChannelMessages, getCurrentUser } from './api';

import type { APIMessage } from 'discord-api-types/v10';

/**
 * Bot が未読として扱うメッセージ数を取得する。
 *
 * @param token Discord bot token
 * @param channelId Discord channel ID
 * @returns 未読メッセージ数
 */
export async function getUnreadMessageCount(
  token: string,
  channelId: string
): Promise<number> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });
  return getUnreadCount(messages, user.id);
}

/**
 * 未読件数と最新メッセージのプレビューを取得する。
 *
 * @param token Discord bot token
 * @param channelId Discord channel ID
 * @returns 未読件数と最新プレビュー
 */
export async function getNotificationDetails(
  token: string,
  channelId: string
): Promise<Omit<ChannelNotificationSummary, 'channel'>> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });
  const latestMessage = messages[0];

  return {
    unreadCount: getUnreadCount(messages, user.id),
    latestMessagePreview: latestMessage
      ? {
          messageId: latestMessage.id,
          user: latestMessage.author.username,
          message: latestMessage.content,
          createdAt: formatDatetimeForAgent(new Date(latestMessage.timestamp)),
        }
      : null,
  };
}

/**
 * Discord メッセージ列から未読件数を計算する。
 *
 * @param messages Discord API が返すメッセージ列
 * @param currentUserId 現在の bot ユーザー ID
 * @returns 先頭から最初の既読境界までの件数。境界がなければ全件数
 */
function getUnreadCount(messages: APIMessage[], currentUserId: string): number {
  const unreadCount = messages.findIndex((message) => {
    if (message.author.id === currentUserId) {
      return true;
    }

    return message.reactions?.some((reaction) => reaction.me) ?? false;
  });

  if (unreadCount === -1) {
    return messages.length;
  }

  return unreadCount;
}
