import { describe, expect, it } from 'vitest';

import { mockToolContext } from './mock-tool-context';
import { thinkDeeplyTool } from './think';

describe('thinkDeeplyTool', () => {
  it('name', () => {
    expect(thinkDeeplyTool.name).toBe('think_deeply');
  });

  it('description', () => {
    expect(thinkDeeplyTool.description).toBeDefined();
  });

  describe('parameters', () => {
    const { parameters } = thinkDeeplyTool;
    expect(parameters).toBeDefined();

    it('thought', () => {
      expect(parameters).toHaveProperty('thought');
      expect(parameters.thought.def.type).toBe('string');
      expect(parameters.thought.description).toBeDefined();
    });
  });

  describe('handler', () => {
    it('thought', () => {
      const result = thinkDeeplyTool.handler(
        { thought: 'What is the meaning of life?' },
        mockToolContext
      );
      const expected = {
        success: true,
      };
      expect(result).toEqual(expected);
    });
  });
});
