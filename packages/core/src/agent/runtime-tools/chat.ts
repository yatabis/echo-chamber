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
        notifications: {
          channel: 'chat',
          unreadCount:
            notificationDetails.unreadCount > 99
              ? '99+'
              : notificationDetails.unreadCount,
          latestMessagePreview:
            notificationDetails.latestMessagePreview === null
              ? null
              : {
                  messageId: notificationDetails.latestMessagePreview.messageId,
                  user: notificationDetails.latestMessagePreview.user,
                  message: notificationDetails.latestMessagePreview.message,
                  created_at:
                    notificationDetails.latestMessagePreview.createdAt,
                },
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

export const readChatMessagesTool = new Tool(
  readChatMessagesToolSpec,
  async ({ limit }, ctx) => {
    try {
      const messages = await ctx.chat.readMessages(limit);

      return {
        success: true,
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
  async ({ message }, ctx) => {
    try {
      await ctx.chat.sendMessage(message);

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
  async ({ messageId, reaction }, ctx) => {
    try {
      await ctx.chat.addReaction(messageId, reaction);

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
