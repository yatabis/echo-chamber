import { getErrorMessage } from '@echo-chamber/core';
import {
  addReactionToChatMessageToolSpec,
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
} from '@echo-chamber/core/agent/tools/chat';

import { Tool } from '.';

export const checkNotificationsFunction = new Tool(
  checkNotificationsToolSpec.name,
  checkNotificationsToolSpec.description,
  checkNotificationsToolSpec.parameters,
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

export const readChatMessagesFunction = new Tool(
  readChatMessagesToolSpec.name,
  readChatMessagesToolSpec.description,
  readChatMessagesToolSpec.parameters,
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

export const sendChatMessageFunction = new Tool(
  sendChatMessageToolSpec.name,
  sendChatMessageToolSpec.description,
  sendChatMessageToolSpec.parameters,
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

export const addReactionToChatMessageFunction = new Tool(
  addReactionToChatMessageToolSpec.name,
  addReactionToChatMessageToolSpec.description,
  addReactionToChatMessageToolSpec.parameters,
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
