import { describe, expect, it } from 'vitest';

import type { DashboardEchoEvent } from '@echo-chamber/contracts/dashboard/types';

import { buildDashboardActionAnalysisResponse } from './dashboard-action-analysis';

function createEvent(
  overrides: Partial<DashboardEchoEvent> & Pick<DashboardEchoEvent, 'type'>
): DashboardEchoEvent {
  return {
    id: `event-${overrides.type}`,
    archiveDay: '2026-06-01',
    category: 'system',
    createdAt: '2026-06-01T12:00:00.000Z',
    payload: null,
    sessionId: 'session-1',
    severity: 'info',
    streams: ['system', 'analysis'],
    summary: overrides.type,
    ...overrides,
  };
}

describe('buildDashboardActionAnalysisResponse', () => {
  it('tool / memory / session 行動指標を集計する', () => {
    const response = buildDashboardActionAnalysisResponse({
      archiveDay: '2026-06-01',
      generatedAt: '2026-06-01T12:10:00.000Z',
      periods: [
        {
          days: 7,
          startArchiveDay: '2026-05-26',
          endArchiveDay: '2026-06-01',
          events: [
            createEvent({
              id: 'session-started',
              type: 'session.started',
              category: 'session',
              createdAt: '2026-06-01T12:00:00.000Z',
              streams: ['thought', 'system', 'analysis'],
            }),
            createEvent({
              id: 'model-turn-completed',
              type: 'model.turn.completed',
              category: 'model',
              createdAt: '2026-06-01T12:00:01.000Z',
              payload: {
                toolCallCount: 0,
                turnIndex: 1,
                warnings: ['no_tool_calls'],
              },
              streams: ['analysis'],
            }),
            createEvent({
              id: 'tool-called-search',
              type: 'tool.called',
              category: 'tool',
              createdAt: '2026-06-01T12:00:02.000Z',
              payload: {
                callId: 'call-search',
                toolName: 'search_memory',
              },
              streams: ['thought', 'analysis'],
            }),
            createEvent({
              id: 'tool-completed-search',
              type: 'tool.completed',
              category: 'tool',
              createdAt: '2026-06-01T12:00:03.000Z',
              payload: {
                callId: 'call-search',
                success: true,
                toolName: 'search_memory',
              },
              streams: ['system', 'analysis'],
            }),
            createEvent({
              id: 'memory-search-completed',
              type: 'memory.search.completed',
              category: 'memory',
              createdAt: '2026-06-01T12:00:04.000Z',
              payload: {
                finalResultCount: 0,
                query: 'latest context',
              },
            }),
            createEvent({
              id: 'tool-completed-store',
              type: 'tool.completed',
              category: 'tool',
              createdAt: '2026-06-01T12:00:05.000Z',
              payload: {
                callId: 'call-store',
                success: true,
                toolName: 'store_memory',
              },
              streams: ['system', 'analysis'],
            }),
            createEvent({
              id: 'tool-failed-chat',
              type: 'tool.failed',
              category: 'tool',
              createdAt: '2026-06-01T12:00:06.000Z',
              payload: {
                callId: 'call-chat',
                success: false,
                toolName: 'read_chat_messages',
              },
              streams: ['thought', 'system', 'analysis'],
            }),
            createEvent({
              id: 'session-completed',
              type: 'session.completed',
              category: 'session',
              createdAt: '2026-06-01T12:00:10.000Z',
              payload: {
                terminationReason: 'max_turns',
                totalTokens: 900,
              },
              severity: 'warn',
              streams: ['thought', 'system', 'analysis'],
            }),
          ],
        },
      ],
    });

    expect(response.periods[0]).toMatchObject({
      days: 7,
      sessionCount: 1,
      completedSessionCount: 1,
      warningSessionCount: 1,
      maxTurnsSessionCount: 1,
      totalTokens: 900,
      averageTokensPerCompletedSession: 900,
      averageSessionDurationMs: 10_000,
      totalTurns: 1,
      noToolCallTurns: 1,
      toolCallCount: 1,
      toolCompletedCount: 2,
      toolFailedCount: 1,
      toolFailureRate: 1 / 3,
      memorySearchCompletedCount: 1,
      memorySearchZeroResultCount: 1,
      memorySearchAverageFinalResultCount: 0,
      storeMemoryCompletedCount: 1,
    });
    expect(response.periods[0]?.topTools).toEqual([
      {
        toolName: 'search_memory',
        calledCount: 1,
        completedCount: 1,
        failedCount: 0,
        failureRate: 0,
      },
      {
        toolName: 'read_chat_messages',
        calledCount: 0,
        completedCount: 0,
        failedCount: 1,
        failureRate: 1,
      },
      {
        toolName: 'store_memory',
        calledCount: 0,
        completedCount: 1,
        failedCount: 0,
        failureRate: 0,
      },
    ]);
  });
});
