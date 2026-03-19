import type { ChatMessage, ChatPort } from '@echo-chamber/core/ports/chat';
import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';

import {
  addReactionToMessage,
  getChannelMessages,
  sendChannelMessage,
} from './api';

export interface DiscordChatPortOptions {
  token: string;
  channelId: string;
}

/**
 * Discord channel を `ChatPort` として扱う adapter。
 */
export function createDiscordChatPort(
  options: DiscordChatPortOptions
): ChatPort {
  return {
    async readMessages(limit: number): Promise<ChatMessage[]> {
      const messages = await getChannelMessages(
        options.token,
        options.channelId,
        { limit }
      );

      return messages.reverse().map((message) => ({
        messageId: message.id,
        user: message.author.username,
        message: message.content,
        createdAt: formatDatetimeForAgent(new Date(message.timestamp)),
        reactions:
          message.reactions?.map((reaction) => ({
            emoji: reaction.emoji.name,
            me: reaction.me,
          })) ?? [],
      }));
    },

    async sendMessage(message: string): Promise<void> {
      await sendChannelMessage(options.token, options.channelId, {
        content: message,
      });
    },

    async addReaction(messageId: string, reaction: string): Promise<void> {
      await addReactionToMessage(
        options.token,
        options.channelId,
        messageId,
        reaction
      );
    },
  };
}
