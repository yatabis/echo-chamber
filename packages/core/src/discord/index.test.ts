import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createDiscordCurrentUserResponse,
  createDiscordMessagesResponse,
} from '../../test/helpers/discord';

import {
  getChannelMessages,
  getCurrentUser,
  getNotificationDetails,
  getUnreadMessageCount,
} from './index';

const TOKEN = 'TEST_DISCORD_BOT_TOKEN';
const CHANNEL_ID = 'test-channel-id';
const BOT_USER_ID = 'bot-user-123';

vi.mock('./api', () => ({
  getChannelMessages: vi.fn(),
  getCurrentUser: vi.fn(),
}));

describe('getUnreadMessageCount', () => {
  it('すべてのメッセージが未読の場合、取得した数を返す', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(2);
  });

  it('自分が送信したメッセージがある場合、そこまでの未読数を返す', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'OK',
        user: BOT_USER_ID,
        userId: BOT_USER_ID,
        timestamp: new Date().toISOString(),
      },
      {
        message: 'Test',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(2);
  });

  it('自分がリアクションをつけたメッセージがある場合、そこまでの未読数を返す', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-2',
        timestamp: new Date().toISOString(),
        reactions: [{ emoji: '👍', me: true }],
      },
      {
        message: 'Test',
        user: 'user-3',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(1);
  });

  it('リアクションが自分のものではない場合、既読として扱わない', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-2',
        timestamp: new Date().toISOString(),
        reactions: [{ emoji: '👍', me: false }],
      },
      {
        message: 'Test',
        user: 'user-3',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(3);
  });

  it('リアクションが複数ある場合、1つでも自分のものがあれば既読として扱う', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-2',
        timestamp: new Date().toISOString(),
        reactions: [
          { emoji: '👍', me: true },
          { emoji: '👎', me: false },
        ],
      },
      {
        message: 'Test',
        user: 'user-3',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(1);
  });

  it('最初のメッセージが自分のメッセージである場合、未読数は0', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: BOT_USER_ID,
        userId: BOT_USER_ID,
        timestamp: new Date().toISOString(),
      },
      {
        message: 'World',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(0);
  });

  it('最初のメッセージに自分がリアクションをつけている場合、未読数は0', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: 'Hello',
        user: 'user-1',
        timestamp: new Date().toISOString(),
        reactions: [{ emoji: '👍', me: true }],
      },
      {
        message: 'World',
        user: 'user-1',
        timestamp: new Date().toISOString(),
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);
    const result = await getUnreadMessageCount(TOKEN, CHANNEL_ID);
    expect(result).toBe(0);
  });
});

describe('getNotificationDetails', () => {
  // テストデータのタイムスタンプ: 2025-01-23T04:56:07.089Z (UTC)
  // 基準時刻を2025-01-25の深夜に固定して「2日前」を期待
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-25T23:59:59.999Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('基本ケース：未読メッセージありで未読数と最新プレビューを返す', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: '最新メッセージ',
        user: 'user-1',
        timestamp: '2025-01-23T04:56:07.089Z',
      },
      {
        message: '2番目のメッセージ',
        user: 'user-2',
        timestamp: '2025-01-23T04:55:00.000Z',
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);

    const result = await getNotificationDetails(TOKEN, CHANNEL_ID);

    expect(result).toEqual({
      unreadCount: 2,
      latestMessagePreview: {
        messageId: 'message-1',
        user: 'user-1',
        message: '最新メッセージ',
        created_at: '2日前 (2025年01月23日 13:56:07)',
      },
    });
  });

  it('未読0件だが最新メッセージあり：最新メッセージをプレビュー表示', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: '最新メッセージ（既読）',
        user: BOT_USER_ID,
        userId: BOT_USER_ID,
        timestamp: '2025-01-23T04:56:07.089Z',
      },
      {
        message: '過去のメッセージ',
        user: 'user-1',
        timestamp: '2025-01-23T04:55:00.000Z',
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);

    const result = await getNotificationDetails(TOKEN, CHANNEL_ID);

    expect(result).toEqual({
      unreadCount: 0,
      latestMessagePreview: {
        messageId: 'message-1',
        user: BOT_USER_ID,
        message: '最新メッセージ（既読）',
        created_at: '2日前 (2025年01月23日 13:56:07)',
      },
    });
  });

  it('空のチャンネル：未読数0でプレビューはnull', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const mockMessages = createDiscordMessagesResponse([]);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);

    const result = await getNotificationDetails(TOKEN, CHANNEL_ID);

    expect(result).toEqual({
      unreadCount: 0,
      latestMessagePreview: null,
    });
  });

  it('リアクション付きメッセージ：既読として扱われ未読数は正しい', async () => {
    const mockUser = createDiscordCurrentUserResponse(BOT_USER_ID);
    vi.mocked(getCurrentUser).mockResolvedValue(mockUser);
    const messages = [
      {
        message: '最新メッセージ',
        user: 'user-1',
        timestamp: '2025-01-23T04:56:07.089Z',
      },
      {
        message: '既読メッセージ',
        user: 'user-2',
        timestamp: '2025-01-23T04:55:00.000Z',
        reactions: [{ emoji: '👍', me: true }],
      },
      {
        message: '過去のメッセージ',
        user: 'user-3',
        timestamp: '2025-01-23T04:54:00.000Z',
      },
    ];
    const mockMessages = createDiscordMessagesResponse(messages);
    vi.mocked(getChannelMessages).mockResolvedValue(mockMessages);

    const result = await getNotificationDetails(TOKEN, CHANNEL_ID);

    expect(result).toEqual({
      unreadCount: 1,
      latestMessagePreview: {
        messageId: 'message-1',
        user: 'user-1',
        message: '最新メッセージ',
        created_at: '2日前 (2025年01月23日 13:56:07)',
      },
    });
  });
});
