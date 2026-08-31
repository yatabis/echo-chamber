import { describe, expect, it } from 'vitest';

import {
  ExternalRequestBudget,
  ExternalRequestBudgetError,
} from './external-request-budget';

describe('ExternalRequestBudget', () => {
  it('同一 invocation の外部 request を1つの上限で集計する', () => {
    const budget = new ExternalRequestBudget(4);

    budget.reserve();
    budget.reserve(2);

    expect(budget.snapshot()).toEqual({
      limit: 4,
      used: 3,
      remaining: 1,
    });
  });

  it('上限を超える request は消費前に非一時エラーで拒否する', () => {
    const budget = new ExternalRequestBudget(2);
    budget.reserve(2);

    let caught: unknown;
    try {
      budget.reserve();
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ExternalRequestBudgetError);
    if (!(caught instanceof ExternalRequestBudgetError)) {
      throw new Error('Expected ExternalRequestBudgetError');
    }
    expect(caught).toMatchObject({
      name: 'ExternalRequestBudgetError',
      code: 'external_request_budget_exceeded',
      limit: 2,
      used: 2,
      requested: 1,
    });
    expect(budget.snapshot().used).toBe(2);
  });

  it('不正な上限と消費数を拒否する', () => {
    expect(() => new ExternalRequestBudget(0)).toThrow(
      'External request budget limit must be a positive integer'
    );

    const budget = new ExternalRequestBudget(1);
    expect(() => {
      budget.reserve(0);
    }).toThrow('External request count must be a positive integer');
  });
});
