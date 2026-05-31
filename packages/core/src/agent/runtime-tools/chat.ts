import {
  addReactionToChatMessageToolSpec,
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
} from '../tools/chat';

import { createToolErrorResult, Tool } from './tool';

export const checkNotificationsTool = new Tool(
  checkNotificationsToolSpec,
  async (_, ctx) => {
    try {
      const notificationDetails =
        await ctx.notifications.getNotificationSummary();

      return {
        success: true,
        notifications: notificationDetails.map((summary) => ({
          channelKey: summary.channel.key,
          channelName: summary.channel.displayName,
          channelDescription: summary.channel.description ?? null,
          unreadCount: summary.unreadCount > 99 ? '99+' : summary.unreadCount,
          latestMessagePreview:
            summary.latestMessagePreview === null
              ? null
              : {
                  messageId: summary.latestMessagePreview.messageId,
                  user: summary.latestMessagePreview.user,
                  message: summary.latestMessagePreview.message,
                  created_at: summary.latestMessagePreview.createdAt,
                },
        })),
      };
    } catch (error) {
      return createToolErrorResult('Failed to fetch notifications', error);
    }
  }
);

export const readChatMessagesTool = new Tool(
  readChatMessagesToolSpec,
  async ({ channelKey, limit }, ctx) => {
    try {
      const messages = await ctx.chat.readMessages(channelKey, limit);

      return {
        success: true,
        channelKey,
        messages: messages.map((message) => ({
          messageId: message.messageId,
          user: message.user,
          message: message.message,
          created_at: message.createdAt,
          reactions: message.reactions,
        })),
      };
    } catch (error) {
      return createToolErrorResult('Failed to read messages', error);
    }
  }
);

export const sendChatMessageTool = new Tool(
  sendChatMessageToolSpec,
  async ({ channelKey, message }, ctx) => {
    try {
      await ctx.chat.sendMessage(channelKey, message);

      return {
        success: true,
      };
    } catch (error) {
      return createToolErrorResult('Failed to send message', error);
    }
  }
);

export const addReactionToChatMessageTool = new Tool(
  addReactionToChatMessageToolSpec,
  async ({ channelKey, messageId, reaction }, ctx) => {
    try {
      await ctx.chat.addReaction(channelKey, messageId, reaction);

      return {
        success: true,
      };
    } catch (error) {
      return createToolErrorResult('Failed to add reaction', error);
    }
  }
);
