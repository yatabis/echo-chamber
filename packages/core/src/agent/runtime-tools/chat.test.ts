import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  addReactionToChatMessageTool,
  checkNotificationsTool,
  readChatMessagesTool,
  sendChatMessageTool,
} from './chat';
import { mockToolContext } from './mock-tool-context';

const CHAT_API_ERROR = 'Chat API error';

const mockedGetNotificationSummary = vi.mocked(
  // eslint-disable-next-line @typescript-eslint/unbound-method
  mockToolContext.notifications.getNotificationSummary
);
// eslint-disable-next-line @typescript-eslint/unbound-method
const mockedReadMessages = vi.mocked(mockToolContext.chat.readMessages);
// eslint-disable-next-line @typescript-eslint/unbound-method
const mockedSendMessage = vi.mocked(mockToolContext.chat.sendMessage);
// eslint-disable-next-line @typescript-eslint/unbound-method
const mockedAddReaction = vi.mocked(mockToolContext.chat.addReaction);

beforeEach(() => {
  vi.resetAllMocks();
  mockedGetNotificationSummary.mockResolvedValue([]);
  mockedReadMessages.mockResolvedValue([]);
  mockedSendMessage.mockResolvedValue(undefined);
  mockedAddReaction.mockResolvedValue(undefined);
});

describe('checkNotificationsTool', () => {
  it('name', () => {
    expect(checkNotificationsTool.name).toBe('check_notifications');
  });

  it('0件（最新メッセージをプレビュー表示）', async () => {
    mockedGetNotificationSummary.mockResolvedValue([
      {
        channel: {
          key: 'main',
          displayName: 'メイン',
          description: '主な会話用チャンネル',
        },
        unreadCount: 0,
        latestMessagePreview: {
          messageId: 'message-latest',
          user: 'user1',
          message: 'Latest message (already read)',
          createdAt: '2025/01/23 13:56:07',
        },
      },
    ]);

    const result = await checkNotificationsTool.handler({}, mockToolContext);

    expect(result).toEqual({
      success: true,
      notifications: [
        {
          channelKey: 'main',
          channelName: 'メイン',
          channelDescription: '主な会話用チャンネル',
          unreadCount: 0,
          latestMessagePreview: {
            messageId: 'message-latest',
            user: 'user1',
            message: 'Latest message (already read)',
            created_at: '2025/01/23 13:56:07',
          },
        },
      ],
    });
  });

  it('100件以上は99+を返す', async () => {
    mockedGetNotificationSummary.mockResolvedValue([
      {
        channel: {
          key: 'main',
          displayName: 'メイン',
          description: '主な会話用チャンネル',
        },
        unreadCount: 100,
        latestMessagePreview: {
          messageId: 'message-100',
          user: 'testuser',
          message: 'Test message',
          createdAt: '2025/01/23 13:56:07',
        },
      },
    ]);

    const result = await checkNotificationsTool.handler({}, mockToolContext);

    expect(result).toEqual({
      success: true,
      notifications: [
        {
          channelKey: 'main',
          channelName: 'メイン',
          channelDescription: '主な会話用チャンネル',
          unreadCount: '99+',
          latestMessagePreview: {
            messageId: 'message-100',
            user: 'testuser',
            message: 'Test message',
            created_at: '2025/01/23 13:56:07',
          },
        },
      ],
    });
  });

  it('通知がない場合は空配列を返す', async () => {
    const result = await checkNotificationsTool.handler({}, mockToolContext);

    expect(result).toEqual({
      success: true,
      notifications: [],
    });
  });

  it('NotificationPortエラー時は失敗を返す', async () => {
    mockedGetNotificationSummary.mockRejectedValue(new Error(CHAT_API_ERROR));

    const result = await checkNotificationsTool.handler({}, mockToolContext);

    expect(result).toEqual({
      success: false,
      error: 'Failed to fetch notifications',
    });
  });
});

describe('readChatMessagesTool', () => {
  it('name', () => {
    expect(readChatMessagesTool.name).toBe('read_chat_messages');
  });

  it('メッセージ一覧をそのまま返す', async () => {
    mockedReadMessages.mockResolvedValue([
      {
        messageId: 'message-1',
        user: 'user1',
        message: 'Hello',
        createdAt: '2日前 (2025年01月23日 13:56:07)',
        reactions: [],
      },
    ]);

    const result = await readChatMessagesTool.handler(
      { channelKey: 'main', limit: 1 },
      mockToolContext
    );

    expect(mockedReadMessages).toHaveBeenCalledWith('main', 1);
    expect(result).toEqual({
      success: true,
      channelKey: 'main',
      messages: [
        {
          messageId: 'message-1',
          user: 'user1',
          message: 'Hello',
          created_at: '2日前 (2025年01月23日 13:56:07)',
          reactions: [],
        },
      ],
    });
  });

  it('リアクション付きメッセージを返す', async () => {
    mockedReadMessages.mockResolvedValue([
      {
        messageId: 'message-1',
        user: 'user1',
        message: 'Hello with reactions',
        createdAt: '2日前 (2025年01月23日 13:56:07)',
        reactions: [
          { emoji: '👍', me: false },
          { emoji: '😄', me: true },
        ],
      },
    ]);

    const result = await readChatMessagesTool.handler(
      { channelKey: 'main', limit: 1 },
      mockToolContext
    );

    expect(result).toEqual({
      success: true,
      channelKey: 'main',
      messages: [
        {
          messageId: 'message-1',
          user: 'user1',
          message: 'Hello with reactions',
          created_at: '2日前 (2025年01月23日 13:56:07)',
          reactions: [
            { emoji: '👍', me: false },
            { emoji: '😄', me: true },
          ],
        },
      ],
    });
  });

  it('ChatPortエラー時は失敗を返す', async () => {
    mockedReadMessages.mockRejectedValue(new Error(CHAT_API_ERROR));

    const result = await readChatMessagesTool.handler(
      { channelKey: 'main', limit: 1 },
      mockToolContext
    );

    expect(result).toEqual({
      success: false,
      error: 'Failed to read messages',
    });
  });
});

describe('sendChatMessageTool', () => {
  it('name', () => {
    expect(sendChatMessageTool.name).toBe('send_chat_message');
  });

  it('ChatPort.sendMessageを呼ぶ', async () => {
    const result = await sendChatMessageTool.handler(
      { channelKey: 'main', message: 'Hello' },
      mockToolContext
    );

    expect(mockedSendMessage).toHaveBeenCalledWith('main', 'Hello');
    expect(result).toEqual({ success: true });
  });

  it('ChatPortエラー時は失敗を返す', async () => {
    mockedSendMessage.mockRejectedValue(new Error(CHAT_API_ERROR));

    const result = await sendChatMessageTool.handler(
      { channelKey: 'main', message: 'Hello' },
      mockToolContext
    );

    expect(result).toEqual({
      success: false,
      error: 'Failed to send message',
    });
  });
});

describe('addReactionToChatMessageTool', () => {
  it('name', () => {
    expect(addReactionToChatMessageTool.name).toBe(
      'add_reaction_to_chat_message'
    );
  });

  it('ChatPort.addReactionを呼ぶ', async () => {
    const result = await addReactionToChatMessageTool.handler(
      {
        channelKey: 'main',
        messageId: '123456789',
        reaction: '👍',
      },
      mockToolContext
    );

    expect(mockedAddReaction).toHaveBeenCalledWith('main', '123456789', '👍');
    expect(result).toEqual({ success: true });
  });

  it('ChatPortエラー時は失敗を返す', async () => {
    mockedAddReaction.mockRejectedValue(new Error(CHAT_API_ERROR));

    const result = await addReactionToChatMessageTool.handler(
      {
        channelKey: 'main',
        messageId: '123456789',
        reaction: '👍',
      },
      mockToolContext
    );

    expect(result).toEqual({
      success: false,
      error: 'Failed to add reaction',
    });
  });
});
