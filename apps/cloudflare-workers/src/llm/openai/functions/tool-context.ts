import type { EchoInstanceConfig, Note } from '@echo-chamber/core';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import type { NotePort } from '@echo-chamber/core/ports/note';

import {
  createDiscordChatPort,
  createDiscordNotificationPort,
} from '../../../discord/client';

import type { MemorySystem } from '../../../runtime/memory-system';
import type { NoteSystem } from '../../../runtime/note-system';
import type { Logger } from '../../../utils/logger';

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
    chat: createDiscordChatPort({
      token: options.instanceConfig.discordBotToken,
      channelId: options.instanceConfig.chatChannelId,
    }),
    notifications: createDiscordNotificationPort({
      token: options.instanceConfig.discordBotToken,
      channelId: options.instanceConfig.chatChannelId,
    }),
    memory: createMemoryPort(options.memorySystem),
    notes: createNotePort(options.noteSystem),
    logger: createLoggerPort(options.logger),
  };
}
