import { afterEach, describe, expect, it, vi } from 'vitest';

import { sendChannelMessage } from './api';
import { DiscordThoughtLog } from './discord-thought-log';

const TOKEN = 'TEST_DISCORD_BOT_TOKEN';
const CHANNEL_ID = 'thinking-channel-id';

vi.mock('./api', () => ({
  sendChannelMessage: vi.fn(),
}));

describe('DiscordThoughtLog', () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  it('空文字は送信しない', async () => {
    const thoughtLog = new DiscordThoughtLog({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });

    await thoughtLog.send('');

    expect(sendChannelMessage).not.toHaveBeenCalled();
  });

  it('2000文字を超える内容は切り詰めて送信する', async () => {
    const thoughtLog = new DiscordThoughtLog({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });
    const originalContent = 'a'.repeat(2100);
    const expectedContent = `${originalContent.substring(
      0,
      2000 - 15
    )}...(truncated)`;

    await thoughtLog.send(originalContent);

    expect(sendChannelMessage).toHaveBeenCalledTimes(1);
    expect(sendChannelMessage).toHaveBeenCalledWith(TOKEN, CHANNEL_ID, {
      content: expectedContent,
    });
    expect(expectedContent.length).toBeLessThanOrEqual(2000);
  });

  it('Discord 送信失敗は握りつぶして console に出す', async () => {
    const thoughtLog = new DiscordThoughtLog({
      token: TOKEN,
      channelId: CHANNEL_ID,
    });
    vi.mocked(sendChannelMessage).mockRejectedValueOnce(new Error('boom'));
    const consoleErrorSpy = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);

    await expect(thoughtLog.send('hello')).resolves.toBeUndefined();

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Failed to send thinking to Discord:',
      expect.any(Error)
    );
  });
});
