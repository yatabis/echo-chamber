import type { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import type { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { Emotion, Note } from '@echo-chamber/core/echo/types';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import type { NotePort } from '@echo-chamber/core/ports/note';
import { createDiscordChatPort } from '@echo-chamber/discord-adapter/chat-port';
import { createDiscordNotificationPort } from '@echo-chamber/discord-adapter/notification-port';

import { createCloudflareWebPageReader } from '../web/cloudflare-web-page-reader';
import { createZennPort } from '../zenn/create-zenn-port';

import type { EchoChatRuntimeBindings } from '../config/echo-runtime-bindings';

/** MemorySystem を Main runtime tool が使う最小 port へ変換する。 */
function createMemoryPort(
  memorySystem: MemorySystem,
  getCurrentEmotion: () => Promise<Emotion>
): ToolExecutionContext['memory'] {
  return {
    async store(content, type): Promise<void> {
      const emotion = await getCurrentEmotion();
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
  getCurrentEmotion(): Promise<Emotion>;
  beforeExternalRequest?(): void;
}): ToolExecutionContext {
  const beforeDiscordRequest =
    options.beforeExternalRequest === undefined
      ? undefined
      : (): void => options.beforeExternalRequest?.();
  const chatOptions = {
    token: options.chatBindings.discordBotToken,
    channels: options.chatBindings.chatChannels,
    ...(beforeDiscordRequest === undefined
      ? {}
      : { beforeRequest: beforeDiscordRequest }),
  };

  return {
    chat: createDiscordChatPort(chatOptions),
    notifications: createDiscordNotificationPort(chatOptions),
    memory: createMemoryPort(
      options.memorySystem,
      async (): Promise<Emotion> => await options.getCurrentEmotion()
    ),
    notes: createNotePort(options.noteSystem),
    webPageReader:
      options.beforeExternalRequest === undefined
        ? createCloudflareWebPageReader()
        : createCloudflareWebPageReader({
            fetcher: async (input, init) => {
              options.beforeExternalRequest?.();
              return await fetch(input, init);
            },
          }),
    zenn:
      options.beforeExternalRequest === undefined
        ? createZennPort()
        : createZennPort(async (input, init) => {
            options.beforeExternalRequest?.();
            return await fetch(input, init);
          }),
  };
}
