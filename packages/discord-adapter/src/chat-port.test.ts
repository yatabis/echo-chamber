import { afterEach, describe, expect, it, vi } from 'vitest';

import { formatDatetimeForAgent } from '@echo-chamber/core/utils/datetime';

import { createDiscordMessagesResponse } from '../test/helpers/discord';

import {
  addReactionToMessage,
  getChannelMessages,
  sendChannelMessage,
} from './api';
import { createDiscordChatPort } from './chat-port';

const TOKEN = 'TEST_DISCORD_BOT_TOKEN';
const CHANNEL_ID = 'test-channel-id';

vi.mock('./api', () => ({
  addReactionToMessage: vi.fn(),
  getChannelMessages: vi.fn(),
  sendChannelMessage: vi.fn(),
}));

describe('createDiscordChatPort', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('Discord のメッセージ一覧を ChatPort の形式に変換する', async () => {
    vi.mocked(getChannelMessages).mockResolvedValue(
      createDiscordMessagesResponse([
        {
          message: 'latest',
          user: 'alice',
          timestamp: '2025-01-26T00:00:00.000Z',
          reactions: [{ emoji: '👍', me: true }],
        },
        {
          message: 'older',
          user: 'bob',
          timestamp: '2025-01-25T23:59:00.000Z',
        },
      ])
    );

    const port = createDiscordChatPort({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });

    const result = await port.readMessages(10);

    expect(getChannelMessages).toHaveBeenCalledWith(TOKEN, CHANNEL_ID, {
      limit: 10,
    });
    expect(result).toEqual([
      {
        messageId: 'message-2',
        user: 'bob',
        message: 'older',
        createdAt: formatDatetimeForAgent(new Date('2025-01-25T23:59:00.000Z')),
        reactions: [],
      },
      {
        messageId: 'message-1',
        user: 'alice',
        message: 'latest',
        createdAt: formatDatetimeForAgent(new Date('2025-01-26T00:00:00.000Z')),
        reactions: [{ emoji: '👍', me: true }],
      },
    ]);
  });

  it('sendMessage で Discord API に送信する', async () => {
    const port = createDiscordChatPort({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });

    await port.sendMessage('hello');

    expect(sendChannelMessage).toHaveBeenCalledWith(TOKEN, CHANNEL_ID, {
      content: 'hello',
    });
  });

  it('addReaction で Discord API にリアクションを追加する', async () => {
    const port = createDiscordChatPort({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });

    await port.addReaction('message-1', '👍');

    expect(addReactionToMessage).toHaveBeenCalledWith(
      TOKEN,
      CHANNEL_ID,
      'message-1',
      '👍'
    );
  });
});
