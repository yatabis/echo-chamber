import { describe, expect, it } from 'vitest';

import { mockToolContext } from '../../../../test/mocks/tool';

import { finishThinkingFunction } from './finish';

describe('finishThinkingFunction', () => {
  it('name', () => {
    expect(finishThinkingFunction.name).toBe('finish_thinking');
  });

  it('description', () => {
    expect(finishThinkingFunction.description).toBeDefined();
    expect(finishThinkingFunction.description).toContain('終了');
  });

  describe('parameters', () => {
    const { parameters } = finishThinkingFunction;
    expect(parameters).toBeDefined();

    it('reason', () => {
      expect(parameters).toHaveProperty('reason');
      expect(parameters.reason.def.type).toBe('string');
      expect(parameters.reason.description).toBeDefined();
    });

    it('next_wake_at (optional)', () => {
      expect(parameters).toHaveProperty('next_wake_at');
      expect(parameters.next_wake_at.unwrap().def.type).toBe('string');
      expect(parameters.next_wake_at.description).toBeDefined();
    });
  });

  describe('handler', () => {
    it('returns success with next_wake_at', () => {
      const result = finishThinkingFunction.handler(
        {
          reason: '十分な情報を得たので思考を終了する',
          next_wake_at: '2025-01-01T12:00:00Z',
        },
        mockToolContext
      );
      expect(result).toEqual({ success: true });
    });

    it('returns success without next_wake_at', () => {
      const result = finishThinkingFunction.handler(
        { reason: '十分な情報を得たので思考を終了する' },
        mockToolContext
      );
      expect(result).toEqual({ success: true });
    });
  });
});
