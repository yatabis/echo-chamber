import { describe, expect, it } from 'vitest';

import { finishThinkingTool } from './finish';
import { mockToolContext } from './mock-tool-context';

describe('finishThinkingTool', () => {
  it('name', () => {
    expect(finishThinkingTool.name).toBe('finish_thinking');
  });

  it('description', () => {
    expect(finishThinkingTool.description).toBeDefined();
    expect(finishThinkingTool.description).toContain('終了');
  });

  describe('parameters', () => {
    const { parameters } = finishThinkingTool;
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
      const result = finishThinkingTool.handler(
        {
          reason: '十分な情報を得たので思考を終了する',
          next_wake_at: '2025-01-01T12:00:00Z',
        },
        mockToolContext
      );
      expect(result).toEqual({ success: true });
    });

    it('returns success without next_wake_at', () => {
      const result = finishThinkingTool.handler(
        { reason: '十分な情報を得たので思考を終了する' },
        mockToolContext
      );
      expect(result).toEqual({ success: true });
    });
  });
});
