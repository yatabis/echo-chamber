import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  EchoEvent,
  EchoEventSeverity,
} from '@echo-chamber/core/ports/echo-event';

import {
  CompositeEchoEventPort,
  ConsoleEchoEventPort,
  createCloudflareEchoEventPort,
  type DiscordEchoEventConfig,
  DiscordEchoEventPort,
  shouldNotifyDiscord,
} from './echo-event';

const { mockSendChannelMessage } = vi.hoisted(() => ({
  mockSendChannelMessage: vi.fn(),
}));

vi.mock('@echo-chamber/discord-adapter/api', () => ({
  sendChannelMessage: mockSendChannelMessage,
}));

describe('ConsoleEchoEventPort', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it.each([
    ['debug', 'debug'],
    ['info', 'info'],
    ['warn', 'warn'],
    ['error', 'error'],
  ] satisfies [EchoEventSeverity, keyof Console][])(
    'severity=%s の Echo event を context 付き JSON として console.%s に出す',
    async (severity, consoleMethod) => {
      const logSpy = vi
        .spyOn(console, consoleMethod)
        .mockImplementation(() => undefined);
      const events = new ConsoleEchoEventPort({
        source: 'test-source',
        getInstanceId: (): string => 'rin',
        getSessionId: (): string => 'session-1',
      });

      await events.emit({
        type: 'tool.completed',
        category: 'tool',
        severity,
        streams: ['system', 'analysis'],
        summary: 'tool completed',
        payload: {
          toolName: 'search_memory',
        },
      });

      expect(logSpy).toHaveBeenCalledWith(
        JSON.stringify({
          timestamp: '2026-03-21T00:00:00.000Z',
          kind: 'echo_event',
          source: 'test-source',
          instanceId: 'rin',
          sessionId: 'session-1',
          type: 'tool.completed',
          category: 'tool',
          severity,
          streams: ['system', 'analysis'],
          summary: 'tool completed',
          payload: {
            toolName: 'search_memory',
          },
        })
      );
    }
  );

  it('severity と異なる console method には出さない', async () => {
    const debugSpy = vi
      .spyOn(console, 'debug')
      .mockImplementation(() => undefined);
    const infoSpy = vi
      .spyOn(console, 'info')
      .mockImplementation(() => undefined);
    const warnSpy = vi
      .spyOn(console, 'warn')
      .mockImplementation(() => undefined);
    const errorSpy = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const events = new ConsoleEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string => 'rin',
      getSessionId: (): string => 'session-1',
    });

    await events.emit({
      type: 'tool.completed',
      category: 'tool',
      severity: 'warn',
      streams: ['system', 'analysis'],
      summary: 'tool completed',
      payload: {
        toolName: 'search_memory',
      },
    });

    expect(debugSpy).not.toHaveBeenCalled();
    expect(infoSpy).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy).not.toHaveBeenCalled();
  });
});

describe('DiscordEchoEventPort', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
    vi.clearAllMocks();
    mockSendChannelMessage.mockResolvedValue({});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('session category の event は info でも Discord に送る', async () => {
    const beforeRequest = vi.fn();
    const events = new DiscordEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string => 'rin',
      getSessionId: (): string => 'session-1',
      beforeRequest,
      getDiscordConfig: (): DiscordEchoEventConfig => ({
        token: 'discord-token',
        channelId: 'thinking-channel',
      }),
    });

    await events.emit({
      type: 'session.started',
      category: 'session',
      severity: 'info',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session started',
    });

    expect(mockSendChannelMessage).toHaveBeenCalledWith(
      'discord-token',
      'thinking-channel',
      {
        content:
          '**[INFO] session.started**\nthinking session started\nsource: test-source\ninstance: rin\nsession: session-1',
      },
      expect.any(Function)
    );
    const forwardedBeforeRequest: unknown =
      mockSendChannelMessage.mock.calls[0]?.[3];
    if (typeof forwardedBeforeRequest !== 'function') {
      throw new Error('Expected Discord request admission hook');
    }
    await (forwardedBeforeRequest as () => void | Promise<void>)();
    expect(beforeRequest).toHaveBeenCalledTimes(1);
  });

  it('system stream を持つ warn event は Discord に送る', async () => {
    const events = new DiscordEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string => 'rin',
      getSessionId: (): string => 'session-1',
      getDiscordConfig: (): DiscordEchoEventConfig => ({
        token: 'discord-token',
        channelId: 'thinking-channel',
      }),
    });

    await events.emit({
      type: 'tool.failed',
      category: 'tool',
      severity: 'warn',
      streams: ['thought', 'system', 'analysis'],
      summary: 'read_chat_messages failed',
      payload: {
        toolName: 'read_chat_messages',
        error: 'Failed to read messages',
      },
    });

    expect(mockSendChannelMessage).toHaveBeenCalledWith(
      'discord-token',
      'thinking-channel',
      {
        content:
          '**[WARN] tool.failed**\nread_chat_messages failed\nsource: test-source\ninstance: rin\nsession: session-1\n\n```json\n{\n  "toolName": "read_chat_messages",\n  "error": "Failed to read messages"\n}\n```',
      }
    );
  });

  it('analysis 専用の warn event は Discord に送らない', async () => {
    const events = new DiscordEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string => 'rin',
      getSessionId: (): string => 'session-1',
      getDiscordConfig: (): DiscordEchoEventConfig => ({
        token: 'discord-token',
        channelId: 'thinking-channel',
      }),
    });

    await events.emit({
      type: 'model.turn.completed',
      category: 'model',
      severity: 'warn',
      streams: ['analysis'],
      summary: 'model turn completed',
      payload: {
        warnings: ['no_tool_calls'],
      },
    });

    expect(mockSendChannelMessage).not.toHaveBeenCalled();
  });

  it('Discord config が未解決なら通知対象 event でも送らない', async () => {
    const events = new DiscordEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string | null => null,
      getSessionId: (): string | null => null,
      getDiscordConfig: (): null => null,
    });

    await events.emit({
      type: 'session.started',
      category: 'session',
      severity: 'info',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session started',
    });

    expect(mockSendChannelMessage).not.toHaveBeenCalled();
  });
});

describe('CompositeEchoEventPort', () => {
  it('登録された port に順番に event を流す', async () => {
    const first = {
      emit: vi.fn(async () => Promise.resolve()),
    };
    const second = {
      emit: vi.fn(async () => Promise.resolve()),
    };
    const event: EchoEvent = {
      type: 'session.completed',
      category: 'session',
      severity: 'info',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session completed',
    };
    const events = new CompositeEchoEventPort([first, second]);

    await events.emit(event);

    expect(first.emit).toHaveBeenCalledWith(event);
    expect(second.emit).toHaveBeenCalledWith(event);
  });

  it('途中の port が失敗しても残りの port へ event を流す', async () => {
    const firstError = new Error('first failed');
    const first = {
      emit: vi.fn(async (): Promise<void> => {
        await Promise.resolve();
        throw firstError;
      }),
    };
    const second = {
      emit: vi.fn(async () => Promise.resolve()),
    };
    const event: EchoEvent = {
      type: 'session.completed',
      category: 'session',
      severity: 'info',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session completed',
    };
    const events = new CompositeEchoEventPort([first, second]);

    await expect(events.emit(event)).rejects.toThrow(firstError);

    expect(first.emit).toHaveBeenCalledWith(event);
    expect(second.emit).toHaveBeenCalledWith(event);
  });
});

describe('createCloudflareEchoEventPort', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-21T00:00:00.000Z'));
    vi.clearAllMocks();
    mockSendChannelMessage.mockResolvedValue({});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('console / archive / Discord へ event を配送する', async () => {
    const eventArchive = {
      recordEvent: vi.fn(async () => Promise.resolve()),
    };
    const logSpy = vi
      .spyOn(console, 'info')
      .mockImplementation(() => undefined);
    const events = createCloudflareEchoEventPort({
      source: 'test-source',
      getInstanceId: (): string => 'rin',
      getSessionId: (): string => 'session-1',
      eventArchive,
      getDiscordConfig: (): DiscordEchoEventConfig => ({
        token: 'discord-token',
        channelId: 'thinking-channel',
      }),
    });
    const event: EchoEvent = {
      type: 'session.started',
      category: 'session',
      severity: 'info',
      streams: ['thought', 'system', 'analysis'],
      summary: 'thinking session started',
    };

    await events.emit(event);

    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(eventArchive.recordEvent).toHaveBeenCalledWith(event, {
      sessionId: 'session-1',
    });
    expect(mockSendChannelMessage).toHaveBeenCalledTimes(1);
  });
});

describe('shouldNotifyDiscord', () => {
  it('session category は severity に関係なく通知する', () => {
    expect(
      shouldNotifyDiscord({
        type: 'session.completed',
        category: 'session',
        severity: 'debug',
        streams: ['thought', 'system', 'analysis'],
        summary: 'done',
      })
    ).toBe(true);
  });

  it('error は stream に関係なく通知する', () => {
    expect(
      shouldNotifyDiscord({
        type: 'model.exchange.recorded',
        category: 'model',
        severity: 'error',
        streams: ['analysis'],
        summary: 'failed',
      })
    ).toBe(true);
  });

  it('warn は system stream を持つ場合だけ通知する', () => {
    expect(
      shouldNotifyDiscord({
        type: 'system.schedule.next_wake_at_invalidated',
        category: 'system',
        severity: 'warn',
        streams: ['system', 'analysis'],
        summary: 'invalidated',
      })
    ).toBe(true);
    expect(
      shouldNotifyDiscord({
        type: 'model.turn.completed',
        category: 'model',
        severity: 'warn',
        streams: ['analysis'],
        summary: 'no tool calls',
      })
    ).toBe(false);
  });
});
