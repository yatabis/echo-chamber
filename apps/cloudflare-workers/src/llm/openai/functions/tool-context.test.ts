import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import type { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import type { Emotion, Note } from '@echo-chamber/core/echo/types';
import { createDiscordChatPort } from '@echo-chamber/discord-adapter/chat-port';
import { createDiscordNotificationPort } from '@echo-chamber/discord-adapter/notification-port';

import { createToolExecutionContext } from './tool-context';

import type { Logger } from '../../../utils/logger';

vi.mock('@echo-chamber/discord-adapter/chat-port', () => ({
  createDiscordChatPort: vi.fn(),
}));

vi.mock('@echo-chamber/discord-adapter/notification-port', () => ({
  createDiscordNotificationPort: vi.fn(),
}));

const mockChatPort = {
  readMessages: vi.fn(),
  sendMessage: vi.fn(),
  addReaction: vi.fn(),
};

const mockNotificationPort = {
  getNotificationSummary: vi.fn(),
};

function createMemorySystemMock(): {
  memorySystem: MemorySystem;
  storeMemory: ReturnType<typeof vi.fn>;
} {
  const storeMemory = vi.fn(async () => Promise.resolve());
  const searchMemory = vi.fn(async () => Promise.resolve([] as []));

  return {
    memorySystem: {
      storeMemory,
      searchMemory,
    } as unknown as MemorySystem,
    storeMemory,
  };
}

function createNoteSystemMock(): NoteSystem {
  const note: Note = {
    id: 'note-1',
    title: 'title',
    content: 'content',
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
  };

  return {
    listNotes: vi.fn(async () => Promise.resolve([note])),
    getNote: vi.fn(async () => Promise.resolve(note)),
    searchNotes: vi.fn(async () => Promise.resolve([note])),
    createNote: vi.fn(async () => Promise.resolve(note)),
    updateNote: vi.fn(async () => Promise.resolve(note)),
    deleteNote: vi.fn(async () => Promise.resolve(true)),
  } as unknown as NoteSystem;
}

function createLoggerMock(): Logger {
  return {
    debug: vi.fn(async () => Promise.resolve()),
    info: vi.fn(async () => Promise.resolve()),
    warn: vi.fn(async () => Promise.resolve()),
    error: vi.fn(async () => Promise.resolve()),
  } as unknown as Logger;
}

describe('createToolExecutionContext', () => {
  beforeEach(() => {
    vi.mocked(createDiscordChatPort).mockReturnValue(mockChatPort);
    vi.mocked(createDiscordNotificationPort).mockReturnValue(
      mockNotificationPort
    );
  });

  it('chat 用 runtime bindings だけで Discord ports を構築する', async () => {
    const { memorySystem, storeMemory } = createMemorySystemMock();
    const noteSystem = createNoteSystemMock();
    const logger = createLoggerMock();
    const emotion: Emotion = {
      valence: 0.3,
      arousal: 0.4,
      labels: ['curious'],
    };

    const context = createToolExecutionContext({
      chatBindings: {
        discordBotToken: 'discord-token',
        chatChannelId: 'chat-channel',
      },
      memorySystem,
      noteSystem,
      logger,
    });

    expect(createDiscordChatPort).toHaveBeenCalledWith({
      token: 'discord-token',
      channelId: 'chat-channel',
    });
    expect(createDiscordNotificationPort).toHaveBeenCalledWith({
      token: 'discord-token',
      channelId: 'chat-channel',
    });
    expect(context.chat).toBe(mockChatPort);
    expect(context.notifications).toBe(mockNotificationPort);

    await context.memory.store('memory', emotion, 'episode');
    expect(storeMemory).toHaveBeenCalledWith('memory', emotion, 'episode');
  });
});
