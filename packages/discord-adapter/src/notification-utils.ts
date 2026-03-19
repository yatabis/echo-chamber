import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';

import { getChannelMessages, getCurrentUser } from './api';

export interface DiscordNotificationDetails {
  unreadCount: number;
  latestMessagePreview: {
    messageId: string;
    user: string;
    message: string;
    createdAt: string;
  } | null;
}

/**
 * Bot が未読として扱うメッセージ数を取得する。
 */
export async function getUnreadMessageCount(
  token: string,
  channelId: string
): Promise<number> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });
  const unreadCount = messages.findIndex((message) => {
    if (message.author.id === user.id) {
      return true;
    }

    const reactions = message.reactions;
    if (!reactions) {
      return false;
    }

    if (reactions.some((reaction) => reaction.me)) {
      return true;
    }

    return false;
  });

  if (unreadCount === -1) {
    return messages.length;
  }

  return unreadCount;
}

/**
 * 未読件数と最新メッセージのプレビューを取得する。
 */
export async function getNotificationDetails(
  token: string,
  channelId: string
): Promise<DiscordNotificationDetails> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });

  const unreadCount = messages.findIndex((message) => {
    if (message.author.id === user.id) {
      return true;
    }

    const reactions = message.reactions;
    if (!reactions) {
      return false;
    }

    if (reactions.some((reaction) => reaction.me)) {
      return true;
    }

    return false;
  });

  const finalUnreadCount = unreadCount === -1 ? messages.length : unreadCount;
  const latestMessage = messages[0];

  return {
    unreadCount: finalUnreadCount,
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
