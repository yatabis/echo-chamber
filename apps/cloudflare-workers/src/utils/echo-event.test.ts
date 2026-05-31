import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EchoEventSeverity } from '@echo-chamber/core/ports/echo-event';

import { ConsoleEchoEventPort } from './echo-event';

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
