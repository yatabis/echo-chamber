import type {
  ChatChannel,
  ChatMessage,
  ChatPort,
} from '@echo-chamber/core/ports/chat';
import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';

import {
  addReactionToMessage,
  getChannelMessages,
  sendChannelMessage,
} from './api';

export interface DiscordChatChannel extends ChatChannel {
  discordChannelId: string;
}

export interface DiscordChatPortOptions {
  token: string;
  channels: readonly DiscordChatChannel[];
}

function getChannelOrThrow(
  channels: readonly DiscordChatChannel[],
  channelKey: string
): DiscordChatChannel {
  const channel = channels.find((candidate) => candidate.key === channelKey);
  if (channel === undefined) {
    throw new Error(`Unknown chat channel key: ${channelKey}`);
  }

  return channel;
}

/**
 * Discord channel を `ChatPort` として扱う adapter。
 *
 * @param options Discord token と対象 channel ID
 * @returns `ChatPort` 実装
 */
export function createDiscordChatPort(
  options: DiscordChatPortOptions
): ChatPort {
  return {
    async readMessages(
      channelKey: string,
      limit: number
    ): Promise<ChatMessage[]> {
      const channel = getChannelOrThrow(options.channels, channelKey);
      const messages = await getChannelMessages(
        options.token,
        channel.discordChannelId,
        {
          limit,
        }
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

    async sendMessage(channelKey: string, message: string): Promise<void> {
      const channel = getChannelOrThrow(options.channels, channelKey);
      await sendChannelMessage(options.token, channel.discordChannelId, {
        content: message,
      });
    },

    async addReaction(
      channelKey: string,
      messageId: string,
      reaction: string
    ): Promise<void> {
      const channel = getChannelOrThrow(options.channels, channelKey);
      await addReactionToMessage(
        options.token,
        channel.discordChannelId,
        messageId,
        reaction
      );
    },
  };
}
