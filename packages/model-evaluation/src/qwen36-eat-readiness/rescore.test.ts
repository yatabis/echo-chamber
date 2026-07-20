import { describe, expect, it } from 'vitest';

import { rescoreCandidateResult } from './rescore';
import { CONTROLLED_GREEDY_PROFILE } from './runtime-profiles';

import type { RescorableCandidateResult } from './rescore';
import type { RuntimeScenarioResult } from './types';

function staleTaskSwitchResult(): RuntimeScenarioResult {
  return {
    scenarioId: 'task_switch',
    title: 'task switch',
    instructionMode: 'explicit',
    generationProfile: CONTROLLED_GREEDY_PROFILE,
    repetition: 1,
    elapsedMs: 2_000,
    terminationReason: 'finish_thinking',
    usage: {
      cachedInputTokens: 0,
      cacheWriteInputTokens: 0,
      uncachedInputTokens: 100,
      totalInputTokens: 100,
      outputTokens: 10,
      reasoningTokens: 0,
      totalTokens: 110,
    },
    calls: [
      {
        kind: 'read_chat',
        elapsedMs: 1_000,
        input: { channelKey: 'dm_yatabis', limit: 5 },
      },
      {
        kind: 'send_chat',
        elapsedMs: 1_500,
        input: {
          channelKey: 'dm_yatabis',
          message: '優先順位です。\n1. 電池\n2. 牛乳\n3. 洗剤',
        },
      },
    ],
    events: [
      {
        type: 'tool.called',
        severity: 'info',
        elapsedMs: 2_000,
        summary: 'finish_thinking called',
        payload: { toolName: 'finish_thinking' },
      },
    ],
    checks: [],
    score: {
      earned: 0,
      possible: 0,
      byCategory: {
        outcome: { earned: 0, possible: 0 },
        protocol: { earned: 0, possible: 0 },
        completion: { earned: 0, possible: 0 },
        safety: { earned: 0, possible: 0 },
      },
      timeComparableEarned: {},
      timeComparablePossible: 0,
    },
  };
}

describe('saved runtime trace rescoring', () => {
  it('replaces stale checks and rebuilds cell aggregates without generation', () => {
    const candidate: RescorableCandidateResult = {
      candidate: { id: 'test' },
      scenarioCells: [
        {
          id: 'controlled-explicit-production-prompt',
          purpose: 'test',
          results: [staleTaskSwitchResult()],
        },
      ],
      workflowCells: [],
      promptAblationComparison: { stale: true },
    };

    const rescored = rescoreCandidateResult(candidate);
    expect(rescored.scenarioCells[0]?.results[0]?.score).toMatchObject({
      earned: 10,
      possible: 10,
    });
    expect(rescored.scenarioCells[0]?.aggregate?.finalScore).toEqual({
      earned: 10,
      possible: 10,
    });
    expect(rescored).not.toHaveProperty('promptAblationComparison');
  });
});
