import type { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import type { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { Note } from '@echo-chamber/core/echo/types';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import type { NotePort } from '@echo-chamber/core/ports/note';
import { createDiscordChatPort } from '@echo-chamber/discord-adapter/chat-port';
import { createDiscordNotificationPort } from '@echo-chamber/discord-adapter/notification-port';

import { createZennPort } from '../zenn/create-zenn-port';

import type { EchoChatRuntimeBindings } from '../config/echo-runtime-bindings';

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

export function createToolExecutionContext(options: {
  chatBindings: EchoChatRuntimeBindings;
  memorySystem: MemorySystem;
  noteSystem: NoteSystem;
}): ToolExecutionContext {
  return {
    chat: createDiscordChatPort({
      token: options.chatBindings.discordBotToken,
      channels: options.chatBindings.chatChannels,
    }),
    notifications: createDiscordNotificationPort({
      token: options.chatBindings.discordBotToken,
      channels: options.chatBindings.chatChannels,
    }),
    memory: createMemoryPort(options.memorySystem),
    notes: createNotePort(options.noteSystem),
    zenn: createZennPort(),
  };
}
