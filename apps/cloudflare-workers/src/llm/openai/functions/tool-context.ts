import { formatDatetimeForAgent } from '@echo-chamber/core';
import type { EchoInstanceConfig, Note } from '@echo-chamber/core';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type * as DiscordApi from '@echo-chamber/core/discord';
import type { ChatMessage, ChatPort } from '@echo-chamber/core/ports/chat';
import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import type { NotePort } from '@echo-chamber/core/ports/note';
import type {
  NotificationPort,
  NotificationSummary,
} from '@echo-chamber/core/ports/notification';

import type { MemorySystem } from '../../../echo/memory-system';
import type { NoteSystem } from '../../../echo/note-system';
import type { Logger } from '../../../utils/logger';

async function getDiscordApi(): Promise<typeof DiscordApi> {
  return import('@echo-chamber/core/discord');
}

function createChatPort(instanceConfig: EchoInstanceConfig): ChatPort {
  return {
    async readMessages(limit): Promise<ChatMessage[]> {
      const { getChannelMessages } = await getDiscordApi();
      const messages = await getChannelMessages(
        instanceConfig.discordBotToken,
        instanceConfig.chatChannelId,
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

    async sendMessage(message): Promise<void> {
      const { sendChannelMessage } = await getDiscordApi();
      await sendChannelMessage(
        instanceConfig.discordBotToken,
        instanceConfig.chatChannelId,
        { content: message }
      );
    },

    async addReaction(messageId, reaction): Promise<void> {
      const { addReactionToMessage } = await getDiscordApi();
      await addReactionToMessage(
        instanceConfig.discordBotToken,
        instanceConfig.chatChannelId,
        messageId,
        reaction
      );
    },
  };
}

function createNotificationPort(
  instanceConfig: EchoInstanceConfig
): NotificationPort {
  return {
    async getNotificationSummary(): Promise<NotificationSummary> {
      const { getNotificationDetails } = await getDiscordApi();
      const notificationDetails = await getNotificationDetails(
        instanceConfig.discordBotToken,
        instanceConfig.chatChannelId
      );

      return {
        unreadCount: notificationDetails.unreadCount,
        latestMessagePreview:
          notificationDetails.latestMessagePreview === null
            ? null
            : {
                messageId: notificationDetails.latestMessagePreview.messageId,
                user: notificationDetails.latestMessagePreview.user,
                message: notificationDetails.latestMessagePreview.message,
                createdAt: notificationDetails.latestMessagePreview.created_at,
              },
      };
    },
  };
}

function createMemoryPort(
  memorySystem: MemorySystem
): ToolExecutionContext['memory'] {
  return {
    async store(content, emotion, type): Promise<void> {
      await memorySystem.storeMemory(content, emotion, type);
    },

    async search(query, type): Promise<MemorySearchResult[]> {
      return await memorySystem.searchMemory(query, type);
    },
  };
}

function createNotePort(noteSystem: NoteSystem): NotePort {
  return {
    async list(): Promise<Note[]> {
      return await noteSystem.listNotes();
    },

    async get(id): Promise<Note | null> {
      return await noteSystem.getNote(id);
    },

    async search(query): Promise<Note[]> {
      return await noteSystem.searchNotes(query);
    },

    async create(input): Promise<Note> {
      return await noteSystem.createNote(input);
    },

    async update(id, patch): Promise<Note | null> {
      return await noteSystem.updateNote(id, patch);
    },

    async delete(id): Promise<boolean> {
      return await noteSystem.deleteNote(id);
    },
  };
}

function createLoggerPort(logger: Logger): LoggerPort {
  return {
    async log(level, message, context): Promise<void> {
      switch (level) {
        case 'debug':
          await logger.debug(message, context);
          break;
        case 'info':
          await logger.info(message, context);
          break;
        case 'warn':
          await logger.warn(message, context);
          break;
        case 'error':
          await logger.error(message);
          break;
      }
    },

    async debug(message, context): Promise<void> {
      await logger.debug(message, context);
    },

    async info(message, context): Promise<void> {
      await logger.info(message, context);
    },

    async warn(message, context): Promise<void> {
      await logger.warn(message, context);
    },

    async error(message, error): Promise<void> {
      await logger.error(message, error);
    },
  };
}

export function createToolExecutionContext(options: {
  instanceConfig: EchoInstanceConfig;
  memorySystem: MemorySystem;
  noteSystem: NoteSystem;
  logger: Logger;
}): ToolExecutionContext {
  return {
    chat: createChatPort(options.instanceConfig),
    notifications: createNotificationPort(options.instanceConfig),
    memory: createMemoryPort(options.memorySystem),
    notes: createNotePort(options.noteSystem),
    logger: createLoggerPort(options.logger),
  };
}
