import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import type { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import type { Emotion, Note } from '@echo-chamber/core/echo/types';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import { createDiscordChatPort } from '@echo-chamber/discord-adapter/chat-port';
import { createDiscordNotificationPort } from '@echo-chamber/discord-adapter/notification-port';

import { createCloudflareWebPageReader } from '../web/cloudflare-web-page-reader';
import { createZennPort } from '../zenn/create-zenn-port';

import { createToolExecutionContext } from './tool-context';

vi.mock('@echo-chamber/discord-adapter/chat-port', () => ({
  createDiscordChatPort: vi.fn(),
}));

vi.mock('@echo-chamber/discord-adapter/notification-port', () => ({
  createDiscordNotificationPort: vi.fn(),
}));

vi.mock('../zenn/create-zenn-port', () => ({
  createZennPort: vi.fn(),
}));

vi.mock('../web/cloudflare-web-page-reader', () => ({
  createCloudflareWebPageReader: vi.fn(),
}));

const mockChatPort = {
  readMessages: vi.fn(),
  sendMessage: vi.fn(),
  addReaction: vi.fn(),
};

const mockNotificationPort = {
  getNotificationSummary: vi.fn(),
};

const mockZennPort = {
  listTrendingArticles: vi.fn(),
  getArticleBySlug: vi.fn(),
};

const mockWebPageReader = {
  readPage: vi.fn(),
};

function createMemorySystemMock(): {
  memorySystem: MemorySystem;
  storeMemory: ReturnType<typeof vi.fn>;
  searchMemory: ReturnType<typeof vi.fn>;
} {
  const storeMemory = vi.fn(async () => Promise.resolve());
  const searchMemory = vi.fn(async () =>
    Promise.resolve([] as MemorySearchResult[])
  );

  return {
    memorySystem: {
      storeMemory,
      searchMemory,
    } as unknown as MemorySystem,
    storeMemory,
    searchMemory,
  };
}

function getRequired<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new Error(`Expected ${label}`);
  }
  return value;
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

describe('createToolExecutionContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(createDiscordChatPort).mockReturnValue(mockChatPort);
    vi.mocked(createDiscordNotificationPort).mockReturnValue(
      mockNotificationPort
    );
    vi.mocked(createZennPort).mockReturnValue(mockZennPort);
    vi.mocked(createCloudflareWebPageReader).mockReturnValue(mockWebPageReader);
  });

  it('runtime bindings と systems から各 tool port を構築する', async () => {
    const { memorySystem, storeMemory, searchMemory } =
      createMemorySystemMock();
    const noteSystem = createNoteSystemMock();
    const emotion: Emotion = {
      valence: 0.3,
      arousal: 0.4,
      labels: ['curious'],
    };
    const getCurrentEmotion = vi.fn().mockResolvedValue(emotion);

    const context = createToolExecutionContext({
      chatBindings: {
        discordBotToken: 'discord-token',
        chatChannels: [
          {
            key: 'main',
            displayName: 'メイン',
            description: '主な会話用チャンネル',
            discordChannelId: 'chat-channel-main',
          },
          {
            key: 'sub',
            displayName: 'サブ',
            discordChannelId: 'chat-channel-sub',
          },
        ],
      },
      memorySystem,
      noteSystem,
      getCurrentEmotion,
    });

    expect(createDiscordChatPort).toHaveBeenCalledWith({
      token: 'discord-token',
      channels: [
        {
          key: 'main',
          displayName: 'メイン',
          description: '主な会話用チャンネル',
          discordChannelId: 'chat-channel-main',
        },
        {
          key: 'sub',
          displayName: 'サブ',
          discordChannelId: 'chat-channel-sub',
        },
      ],
    });
    expect(createDiscordNotificationPort).toHaveBeenCalledWith({
      token: 'discord-token',
      channels: [
        {
          key: 'main',
          displayName: 'メイン',
          description: '主な会話用チャンネル',
          discordChannelId: 'chat-channel-main',
        },
        {
          key: 'sub',
          displayName: 'サブ',
          discordChannelId: 'chat-channel-sub',
        },
      ],
    });
    expect(context.chat).toBe(mockChatPort);
    expect(context.notifications).toBe(mockNotificationPort);
    expect(createZennPort).toHaveBeenCalledWith();
    expect(context.zenn).toBe(mockZennPort);
    expect(createCloudflareWebPageReader).toHaveBeenCalledWith();
    expect(context.webPageReader).toBe(mockWebPageReader);

    await expect(context.notes.list()).resolves.toHaveLength(1);
    await context.memory.store('memory', 'episode');
    await context.memory.search('query', 'semantic');
    expect(getCurrentEmotion).toHaveBeenCalledTimes(1);
    expect(storeMemory).toHaveBeenCalledWith('memory', emotion, 'episode');
    expect(searchMemory).toHaveBeenCalledWith('query', 'semantic');
  });

  it('外部 request gate を Discord / Web / Zenn の実送信境界へ渡す', async () => {
    const beforeExternalRequest = vi.fn();
    const { memorySystem } = createMemorySystemMock();
    const noteSystem = createNoteSystemMock();

    createToolExecutionContext({
      chatBindings: {
        discordBotToken: 'discord-token',
        chatChannels: [
          {
            key: 'main',
            displayName: 'メイン',
            discordChannelId: 'chat-channel-main',
          },
        ],
      },
      memorySystem,
      noteSystem,
      getCurrentEmotion: vi.fn().mockResolvedValue({
        valence: 0,
        arousal: 0,
        labels: [],
      }),
      beforeExternalRequest,
    });

    const chatCalls = vi.mocked(createDiscordChatPort).mock.calls;
    const notificationCalls = vi.mocked(createDiscordNotificationPort).mock
      .calls;
    const webCalls = vi.mocked(createCloudflareWebPageReader).mock.calls;
    const zennCalls = vi.mocked(createZennPort).mock.calls;
    const chatOptions = getRequired(
      chatCalls[chatCalls.length - 1]?.[0],
      'guarded chat options'
    );
    const notificationOptions = getRequired(
      notificationCalls[notificationCalls.length - 1]?.[0],
      'guarded notification options'
    );
    const webOptions = getRequired(
      webCalls[webCalls.length - 1]?.[0],
      'guarded Web options'
    );
    const zennFetcher = getRequired(
      zennCalls[zennCalls.length - 1]?.[0],
      'guarded Zenn fetcher'
    );
    const beforeChatRequest = getRequired(
      chatOptions.beforeRequest,
      'chat request guard'
    );
    const beforeNotificationRequest = getRequired(
      notificationOptions.beforeRequest,
      'notification request guard'
    );
    const webFetcher = getRequired(webOptions.fetcher, 'Web request guard');
    expect(chatOptions.token).toBe('discord-token');
    expect(typeof beforeChatRequest).toBe('function');
    expect(typeof beforeNotificationRequest).toBe('function');
    expect(typeof webFetcher).toBe('function');
    expect(typeof zennFetcher).toBe('function');

    await beforeChatRequest();
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{}'));
    await webFetcher('https://example.com', {});
    await zennFetcher('https://zenn.dev/api/articles');

    expect(beforeExternalRequest.mock.calls).toEqual([[], [], []]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    fetchMock.mockRestore();
  });
});
