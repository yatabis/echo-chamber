import { describe, expect, it } from 'vitest';

import {
  dashboardInstancesResponseSchema,
  echoStatusSchema,
  parseDashboardInstancesResponse,
  parseEchoStatus,
} from './schemas';

describe('dashboard contract schemas', () => {
  it('parses /instances payload', () => {
    const payload = parseDashboardInstancesResponse({
      instances: [
        {
          id: 'rin',
          name: 'リン',
          state: 'Idling',
          nextAlarm: '2026-03-19T12:00:00.000Z',
          noteCount: 4,
          memoryCount: 8,
          todayUsageTokens: 1200,
          sevenDayUsageTokens: 5400,
          thirtyDayUsageTokens: 22000,
          latestNoteUpdatedAt: '2026-03-19T11:00:00.000Z',
          latestMemoryUpdatedAt: '2026-03-19T10:00:00.000Z',
        },
        {
          id: 'marie',
          name: 'marie',
          state: 'Unknown',
          nextAlarm: null,
          noteCount: 0,
          memoryCount: 0,
          todayUsageTokens: 0,
          sevenDayUsageTokens: 0,
          thirtyDayUsageTokens: 0,
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
            noteCount: 0,
            memoryCount: 0,
            todayUsageTokens: 0,
            sevenDayUsageTokens: 0,
            thirtyDayUsageTokens: 0,
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
          total_cost: 0.001,
        },
      },
    });

    expect(payload.memories[0]?.type).toBe('semantic');
    expect(payload.notes[0]?.id).toBe('note-1');
  });

  it('rejects invalid /:instanceId payload', () => {
    expect(() => {
      echoStatusSchema.parse({
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: null,
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
            total_cost: 0.001,
          },
        },
      });
    }).toThrow();
  });
});
