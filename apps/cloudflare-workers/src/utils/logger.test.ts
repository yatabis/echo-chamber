import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { LoggerPort } from '@echo-chamber/core/ports/logger';

import type { LoggerOptions } from './logger';

async function createTestLogger(options: LoggerOptions): Promise<LoggerPort> {
  const module = await vi.importActual<{
    Logger: new (config: LoggerOptions) => LoggerPort;
  }>('./logger');
  return new module.Logger(options);
}

describe('Logger', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('level に応じた console に JSON 文字列を出力する', async () => {
    const debugSpy = vi
      .spyOn(console, 'debug')
      .mockImplementation(() => undefined);
    const logger = await createTestLogger({
      discordNotifyLevel: 'error',
    });

    await logger.debug('debug message', { scope: 'test' });

    expect(debugSpy).toHaveBeenCalledWith(
      JSON.stringify({
        timestamp: '2026-03-21T00:00:00.000Z',
        level: 'debug',
        message: 'debug message',
        context: { scope: 'test' },
      })
    );
  });

  it('error は Error を context に正規化して console.error に出す', async () => {
    const errorSpy = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const logger = await createTestLogger({
      discordNotifyLevel: 'error',
    });
    const error = new Error('boom');
    error.stack = 'stack-trace';

    await logger.error('failed to do something', error);

    expect(errorSpy).toHaveBeenCalledWith(
      JSON.stringify({
        timestamp: '2026-03-21T00:00:00.000Z',
        level: 'error',
        message: 'failed to do something',
        context: {
          error: {
            name: 'Error',
            message: 'boom',
            stack: 'stack-trace',
          },
        },
      })
    );
  });

  it('しきい値以上のログだけ Discord に送る', async () => {
    vi.spyOn(console, 'info').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const sendDiscordMessage = vi.fn<
      (token: string, channelId: string, content: string) => Promise<void>
    >(async (): Promise<void> => {
      await Promise.resolve();
    });
    const logger = await createTestLogger({
      discordNotifyLevel: 'warn',
      discord: {
        token: 'token',
        channelId: 'channel',
      },
      sendDiscordMessage,
    });

    await logger.info('not forwarded');
    await logger.warn('forwarded');

    expect(sendDiscordMessage).toHaveBeenCalledTimes(1);
    expect(sendDiscordMessage).toHaveBeenCalledWith(
      'token',
      'channel',
      '⚠️ **[WARN]** forwarded'
    );
  });

  it('context がある場合は Discord に json code block を付ける', async () => {
    vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const sendDiscordMessage = vi.fn<
      (token: string, channelId: string, content: string) => Promise<void>
    >(async (): Promise<void> => {
      await Promise.resolve();
    });
    const logger = await createTestLogger({
      discordNotifyLevel: 'info',
      discord: {
        token: 'token',
        channelId: 'channel',
      },
      sendDiscordMessage,
    });

    await logger.info('hello', { scope: 'test', count: 1 });

    expect(sendDiscordMessage).toHaveBeenCalledWith(
      'token',
      'channel',
      'ℹ️ **[INFO]** hello\n```json\n{\n  "scope": "test",\n  "count": 1\n}\n```'
    );
  });

  it('Discord 通知が 2000 文字を超える場合は切り詰める', async () => {
    vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const sendDiscordMessage = vi.fn<
      (token: string, channelId: string, content: string) => Promise<void>
    >(async (): Promise<void> => {
      await Promise.resolve();
    });
    const logger = await createTestLogger({
      discordNotifyLevel: 'info',
      discord: {
        token: 'token',
        channelId: 'channel',
      },
      sendDiscordMessage,
    });

    await logger.info('a'.repeat(2100));

    expect(sendDiscordMessage).toHaveBeenCalledTimes(1);
    const sentMessage = sendDiscordMessage.mock.calls[0]?.[2];
    expect(sentMessage).toBeDefined();
    if (typeof sentMessage !== 'string') {
      throw new Error('Expected Discord message to be a string');
    }

    expect(sentMessage).toHaveLength(1994);
    expect(sentMessage).toContain('...(truncated)');
  });

  it('Discord 送信失敗は握りつぶして console.error に出す', async () => {
    const consoleWarnSpy = vi
      .spyOn(console, 'warn')
      .mockImplementation(() => undefined);
    const consoleErrorSpy = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const sendDiscordMessage = vi.fn<
      (token: string, channelId: string, content: string) => Promise<void>
    >(async (): Promise<void> => {
      await Promise.reject(new Error('discord failed'));
    });
    const logger = await createTestLogger({
      discordNotifyLevel: 'warn',
      discord: {
        token: 'token',
        channelId: 'channel',
      },
      sendDiscordMessage,
    });

    await expect(logger.warn('warn message')).resolves.toBeUndefined();

    expect(consoleWarnSpy).toHaveBeenCalledWith(
      JSON.stringify({
        timestamp: '2026-03-21T00:00:00.000Z',
        level: 'warn',
        message: 'warn message',
      })
    );
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Failed to send log to Discord:',
      expect.any(Error)
    );
  });
});
