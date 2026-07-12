import { describe, expect, it } from 'vitest';

import { buildDashboardActionAnalysisResponse } from './dashboard-action-analysis';

import type { DashboardActionAnalysisEvent } from './dashboard-action-analysis';

function createEvent(
  overrides: Partial<DashboardActionAnalysisEvent> &
    Pick<DashboardActionAnalysisEvent, 'type'>
): DashboardActionAnalysisEvent {
  return {
    archiveDay: '2026-06-01',
    createdAt: '2026-06-01T12:00:00.000Z',
    sessionId: 'session-1',
    severity: 'info',
    warnings: [],
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
          eventCount: 8,
          events: [
            createEvent({
              type: 'session.started',
              createdAt: '2026-06-01T12:00:00.000Z',
            }),
            createEvent({
              type: 'model.turn.completed',
              createdAt: '2026-06-01T12:00:01.000Z',
              warnings: ['no_tool_calls'],
            }),
            createEvent({
              type: 'tool.called',
              createdAt: '2026-06-01T12:00:02.000Z',
              toolName: 'search_memory',
            }),
            createEvent({
              type: 'tool.completed',
              createdAt: '2026-06-01T12:00:03.000Z',
              toolName: 'search_memory',
            }),
            createEvent({
              type: 'memory.search.completed',
              createdAt: '2026-06-01T12:00:04.000Z',
              finalResultCount: 0,
            }),
            createEvent({
              type: 'tool.completed',
              createdAt: '2026-06-01T12:00:05.000Z',
              toolName: 'store_memory',
            }),
            createEvent({
              type: 'tool.failed',
              createdAt: '2026-06-01T12:00:06.000Z',
              toolName: 'read_chat_messages',
            }),
            createEvent({
              type: 'session.completed',
              createdAt: '2026-06-01T12:00:10.000Z',
              terminationReason: 'max_turns',
              totalTokens: 900,
              severity: 'warn',
            }),
          ],
        },
      ],
    });

    expect(response.periods[0]).toMatchObject({
      days: 7,
      eventCount: 8,
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

  it('集計済み metrics から period payload を作る', () => {
    const response = buildDashboardActionAnalysisResponse({
      archiveDay: '2026-06-01',
      generatedAt: '2026-06-01T12:10:00.000Z',
      periods: [
        {
          days: 30,
          startArchiveDay: '2026-05-03',
          endArchiveDay: '2026-06-01',
          eventCount: 12,
          events: [],
          metrics: {
            sessionCount: 2,
            completedSessionCount: 2,
            failedSessionCount: 0,
            warningSessionCount: 1,
            maxTurnsSessionCount: 1,
            totalTokens: 1200,
            totalSessionDurationMs: 15_000,
            sessionDurationCount: 2,
            totalTurns: 3,
            noToolCallTurns: 1,
            toolCallCount: 3,
            toolCompletedCount: 2,
            toolFailedCount: 1,
            topTools: [
              {
                toolName: 'search_memory',
                calledCount: 2,
                completedCount: 1,
                failedCount: 1,
              },
              {
                toolName: 'store_memory',
                calledCount: 1,
                completedCount: 1,
                failedCount: 0,
              },
            ],
            memorySearchCompletedCount: 2,
            memorySearchFailedCount: 1,
            memorySearchZeroResultCount: 1,
            memorySearchFinalResultTotal: 3,
            storeMemoryCompletedCount: 1,
          },
        },
      ],
    });

    expect(response.periods[0]).toMatchObject({
      days: 30,
      eventCount: 12,
      sessionCount: 2,
      completedSessionCount: 2,
      averageTokensPerCompletedSession: 600,
      averageSessionDurationMs: 7500,
      toolFailureRate: 1 / 3,
      memorySearchAverageFinalResultCount: 1.5,
      storeMemoryCompletedCount: 1,
    });
    expect(response.periods[0]?.topTools).toEqual([
      {
        toolName: 'search_memory',
        calledCount: 2,
        completedCount: 1,
        failedCount: 1,
        failureRate: 0.5,
      },
      {
        toolName: 'store_memory',
        calledCount: 1,
        completedCount: 1,
        failedCount: 0,
        failureRate: 0,
      },
    ]);
  });
});
