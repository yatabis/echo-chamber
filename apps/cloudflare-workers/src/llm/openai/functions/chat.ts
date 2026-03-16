import { formatDatetimeForAgent, getErrorMessage } from '@echo-chamber/core';
import {
  addReactionToChatMessageToolSpec,
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
} from '@echo-chamber/core/agent/tools/chat';
import type * as DiscordApi from '@echo-chamber/core/discord';

import { Tool } from '.';

async function getDiscordApi(): Promise<typeof DiscordApi> {
  return import('@echo-chamber/core/discord');
}

export const checkNotificationsFunction = new Tool(
  checkNotificationsToolSpec.name,
  checkNotificationsToolSpec.description,
  checkNotificationsToolSpec.parameters,
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
  readChatMessagesToolSpec.name,
  readChatMessagesToolSpec.description,
  readChatMessagesToolSpec.parameters,
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
  sendChatMessageToolSpec.name,
  sendChatMessageToolSpec.description,
  sendChatMessageToolSpec.parameters,
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
  addReactionToChatMessageToolSpec.name,
  addReactionToChatMessageToolSpec.description,
  addReactionToChatMessageToolSpec.parameters,
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
