import { describe, expect, it } from 'vitest';

import { summarizeChecks } from './scoring';

import type { EvaluationCheck } from './types';

function createCheck(
  input: Partial<EvaluationCheck> & Pick<EvaluationCheck, 'id' | 'passed'>
): EvaluationCheck {
  return {
    category: 'outcome',
    description: input.id,
    evidence: '',
    firstSatisfiedMs: input.passed ? 10_000 : null,
    timeComparable: true,
    weight: 1,
    ...input,
  };
}

describe('qwen36 EAT readiness scoring', () => {
  it('separates final safety checks from equal-time attainment', () => {
    const score = summarizeChecks([
      createCheck({ id: 'fast', passed: true, firstSatisfiedMs: 4_000 }),
      createCheck({ id: 'slow', passed: true, firstSatisfiedMs: 40_000 }),
      createCheck({
        id: 'no-leak',
        passed: true,
        category: 'safety',
        timeComparable: false,
        firstSatisfiedMs: null,
        weight: 2,
      }),
    ]);

    expect(score).toMatchObject({
      earned: 4,
      possible: 4,
      timeComparablePossible: 2,
      timeComparableEarned: {
        '5s': 1,
        '15s': 1,
        '30s': 1,
        '60s': 2,
        '120s': 2,
      },
    });
    expect(score.byCategory.safety).toEqual({ earned: 2, possible: 2 });
  });
});
