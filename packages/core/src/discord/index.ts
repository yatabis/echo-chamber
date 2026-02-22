import { formatDatetimeForAgent } from '../utils/datetime';

import { getChannelMessages, getCurrentUser } from './api';

export {
  addReactionToMessage,
  getChannelMessages,
  sendChannelMessage,
  getCurrentUser,
} from './api';

/**
 * ボットが未読のメッセージ数を取得（最大100件）
 * @param token Discord Bot Token
 * @param channelId チャンネルID
 * @returns 未読メッセージ数
 */
export async function getUnreadMessageCount(
  token: string,
  channelId: string
): Promise<number> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });
  const unreadCount = messages.findIndex((msg) => {
    // 自分が送信したメッセージは既読
    if (msg.author.id === user.id) {
      return true;
    }
    // リアクションがついていないメッセージは未読
    const reactions = msg.reactions;
    if (!reactions) {
      return false;
    }
    // 自分がリアクションをつけているメッセージは既読
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
 * 未読メッセージ数と最新メッセージプレビューを取得
 * @param token Discord Bot Token
 * @param channelId チャンネルID
 * @returns 未読メッセージ数と最新メッセージプレビュー
 */
export async function getNotificationDetails(
  token: string,
  channelId: string
): Promise<{
  unreadCount: number;
  latestMessagePreview: {
    messageId: string;
    user: string;
    message: string;
    created_at: string;
  } | null;
}> {
  const limit = 100;
  const user = await getCurrentUser(token);
  const messages = await getChannelMessages(token, channelId, { limit });

  // 未読メッセージ数を計算
  const unreadCount = messages.findIndex((msg) => {
    // 自分が送信したメッセージは既読
    if (msg.author.id === user.id) {
      return true;
    }
    // リアクションがついていないメッセージは未読
    const reactions = msg.reactions;
    if (!reactions) {
      return false;
    }
    // 自分がリアクションをつけているメッセージは既読
    if (reactions.some((reaction) => reaction.me)) {
      return true;
    }
    return false;
  });

  const finalUnreadCount = unreadCount === -1 ? messages.length : unreadCount;

  // 最新メッセージプレビューを生成
  const latestMessage = messages[0];
  const latestMessagePreview = latestMessage
    ? {
        messageId: latestMessage.id,
        user: latestMessage.author.username,
        message: latestMessage.content,
        created_at: formatDatetimeForAgent(new Date(latestMessage.timestamp)),
      }
    : null;

  return {
    unreadCount: finalUnreadCount,
    latestMessagePreview,
  };
}
