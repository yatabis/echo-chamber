import { getErrorMessage } from '../../utils/error';
import {
  addReactionToChatMessageToolSpec,
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
} from '../tools/chat';

import { Tool } from './tool';

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

export const sendChatMessageTool = new Tool(
  sendChatMessageToolSpec,
  async ({ channelKey, message }, ctx) => {
    try {
      await ctx.chat.sendMessage(channelKey, message);

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

export const addReactionToChatMessageTool = new Tool(
  addReactionToChatMessageToolSpec,
  async ({ channelKey, messageId, reaction }, ctx) => {
    try {
      await ctx.chat.addReaction(channelKey, messageId, reaction);

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
