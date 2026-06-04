import { describe, expect, it } from 'vitest';

import {
  dashboardInstancesResponseSchema,
  echoStatusSchema,
  parseDashboardInstancesResponse,
  parseDashboardSessionLogsResponse,
  parseEchoStatus,
  dashboardSessionLogsResponseSchema,
} from './schemas';

describe('dashboard contract schemas', () => {
  it('parses /:instanceId/session-logs payload', () => {
    const payload = parseDashboardSessionLogsResponse({
      archiveDay: '2026-06-01',
      sessionLogs: [
        {
          id: 'session:session-1',
          activities: [
            {
              id: 'activity-1',
              body: 'A thinking session started.',
              createdAt: '2026-06-01T12:00:00.000Z',
              details: null,
              kind: 'session',
              meta: ['2026/06/01 21:00:00'],
              tone: 'positive',
              title: 'Started thinking',
            },
          ],
          activityCount: 1,
          latestActivityAt: '2026-06-01T12:00:00.000Z',
          meta: ['2026/06/01 21:00:00', '1 activity'],
          sessionId: 'session-1',
          startedAt: '2026-06-01T12:00:00.000Z',
          title: 'Session session-1',
          warningCount: 0,
        },
      ],
    });

    expect(payload.archiveDay).toBe('2026-06-01');
    expect(payload.sessionLogs[0]?.activities[0]?.kind).toBe('session');
  });

  it('rejects invalid /:instanceId/session-logs payload', () => {
    expect(() => {
      dashboardSessionLogsResponseSchema.parse({
        archiveDay: '2026-06-01',
        sessionLogs: [
          {
            id: 'session:session-1',
            activities: [
              {
                id: 'activity-1',
                body: 'A thinking session started.',
                createdAt: '2026-06-01T12:00:00.000Z',
                details: null,
                kind: 'raw_event',
                meta: [],
                tone: 'positive',
                title: 'Started thinking',
              },
            ],
            activityCount: 1,
            latestActivityAt: '2026-06-01T12:00:00.000Z',
            meta: [],
            sessionId: 'session-1',
            startedAt: '2026-06-01T12:00:00.000Z',
            title: 'Session session-1',
            warningCount: 0,
          },
        ],
      });
    }).toThrow();
  });

  it('rejects /:instanceId/session-logs activity with duplicated sessionId', () => {
    expect(() => {
      dashboardSessionLogsResponseSchema.parse({
        archiveDay: '2026-06-01',
        sessionLogs: [
          {
            id: 'session:session-1',
            activities: [
              {
                id: 'activity-1',
                body: 'A thinking session started.',
                createdAt: '2026-06-01T12:00:00.000Z',
                details: null,
                kind: 'session',
                meta: [],
                sessionId: 'session-1',
                tone: 'positive',
                title: 'Started thinking',
              },
            ],
            activityCount: 1,
            latestActivityAt: '2026-06-01T12:00:00.000Z',
            meta: [],
            sessionId: 'session-1',
            startedAt: '2026-06-01T12:00:00.000Z',
            title: 'Session session-1',
            warningCount: 0,
          },
        ],
      });
    }).toThrow();
  });

  it('parses /instances payload', () => {
    const payload = parseDashboardInstancesResponse({
      instances: [
        {
          id: 'rin',
          name: 'リン',
          state: 'Idling',
          nextAlarm: '2026-03-19T12:00:00.000Z',
          nextWakeAt: '2026-03-19T13:00:00.000Z',
          noteCount: 4,
          memoryCount: 8,
          todayUsageTokens: 1200,
          sevenDayUsageTokens: 5400,
          thirtyDayUsageTokens: 22000,
          runtime: {
            mainLlm: {
              provider: 'openai',
              model: 'gpt-5.5',
            },
            tokenLimits: {
              dailyHardLimit: 500_000,
              dailySoftLimit: 300_000,
              hardLimitBufferFactor: 1.5,
            },
          },
          latestNoteUpdatedAt: '2026-03-19T11:00:00.000Z',
          latestMemoryUpdatedAt: '2026-03-19T10:00:00.000Z',
        },
        {
          id: 'marie',
          name: 'marie',
          state: 'Unknown',
          nextAlarm: null,
          nextWakeAt: null,
          noteCount: 0,
          memoryCount: 0,
          todayUsageTokens: 0,
          sevenDayUsageTokens: 0,
          thirtyDayUsageTokens: 0,
          runtime: {
            mainLlm: {
              provider: 'unknown',
              model: 'unknown',
            },
            tokenLimits: {
              dailyHardLimit: 0,
              dailySoftLimit: 0,
              hardLimitBufferFactor: 0,
            },
          },
          latestNoteUpdatedAt: null,
          latestMemoryUpdatedAt: null,
        },
      ],
    });

    expect(payload.instances).toHaveLength(2);
    expect(payload.instances[1]?.state).toBe('Unknown');
  });

  it('rejects invalid /instances payload', () => {
    expect(() => {
      dashboardInstancesResponseSchema.parse({
        instances: [
          {
            id: 'rin',
            name: 'リン',
            state: 'Broken',
            nextAlarm: null,
            nextWakeAt: null,
            noteCount: 0,
            memoryCount: 0,
            todayUsageTokens: 0,
            sevenDayUsageTokens: 0,
            thirtyDayUsageTokens: 0,
            runtime: {
              mainLlm: {
                provider: 'openai',
                model: 'gpt-5.5',
              },
              tokenLimits: {
                dailyHardLimit: 500_000,
                dailySoftLimit: 300_000,
                hardLimitBufferFactor: 1.5,
              },
            },
            latestNoteUpdatedAt: null,
            latestMemoryUpdatedAt: null,
          },
        ],
      });
    }).toThrow();
  });

  it('parses /:instanceId payload', () => {
    const payload = parseEchoStatus({
      id: 'rin',
      name: 'リン',
      state: 'Idling',
      nextAlarm: null,
      nextWakeAt: '2026-03-19T13:00:00.000Z',
      context: {
        content: 'Continue from the latest dashboard work.',
        emotion: {
          valence: 0.3,
          arousal: 0.4,
          labels: ['focused'],
        },
        createdAt: '2026-03-19T11:00:00.000Z',
        updatedAt: '2026-03-19T12:00:00.000Z',
      },
      runtime: {
        mainLlm: {
          provider: 'openai',
          model: 'gpt-5.5',
        },
        tokenLimits: {
          dailyHardLimit: 500_000,
          dailySoftLimit: 300_000,
          hardLimitBufferFactor: 1.5,
        },
      },
      memories: [
        {
          content: 'remember this',
          type: 'semantic',
          emotion: {
            valence: 0.4,
            arousal: 0.2,
            labels: ['focus'],
          },
          embedding_model: 'text-embedding-3-small',
          createdAt: '2026-03-19T12:00:00.000Z',
          updatedAt: '2026-03-19T12:00:00.000Z',
        },
      ],
      notes: [
        {
          id: 'note-1',
          title: 'Title',
          content: 'Body',
          createdAt: '2026-03-19T12:00:00.000Z',
          updatedAt: '2026-03-19T12:00:00.000Z',
        },
      ],
      usage: {
        '2026-03-19': {
          cached_input_tokens: 10,
          uncached_input_tokens: 20,
          total_input_tokens: 30,
          output_tokens: 5,
          reasoning_tokens: 1,
          total_tokens: 35,
          by_model: [
            {
              provider: 'openai',
              model: 'gpt-5',
              cached_input_tokens: 10,
              uncached_input_tokens: 20,
              total_input_tokens: 30,
              output_tokens: 5,
              reasoning_tokens: 1,
              total_tokens: 35,
            },
          ],
        },
      },
    });

    expect(payload.memories[0]?.type).toBe('semantic');
    expect(payload.notes[0]?.id).toBe('note-1');
    expect(payload.context?.content).toBe(
      'Continue from the latest dashboard work.'
    );
  });

  it('rejects invalid /:instanceId payload', () => {
    expect(() => {
      echoStatusSchema.parse({
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: null,
        nextWakeAt: null,
        context: null,
        runtime: {
          mainLlm: {
            provider: 'openai',
            model: 'gpt-5.5',
          },
          tokenLimits: {
            dailyHardLimit: 500_000,
            dailySoftLimit: 300_000,
            hardLimitBufferFactor: 1.5,
          },
        },
        memories: [],
        notes: [],
        usage: {
          '2026-03-19': {
            cached_input_tokens: 10,
            uncached_input_tokens: 20,
            total_input_tokens: 30,
            output_tokens: 5,
            reasoning_tokens: 1,
            total_tokens: '35',
            by_model: [
              {
                provider: 'openai',
                model: 'gpt-5',
                cached_input_tokens: 10,
                uncached_input_tokens: 20,
                total_input_tokens: 30,
                output_tokens: 5,
                reasoning_tokens: 1,
                total_tokens: '35',
              },
            ],
          },
        },
      });
    }).toThrow();
  });
});
