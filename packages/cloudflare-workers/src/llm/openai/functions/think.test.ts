import { describe, expect, it } from 'vitest';

import { mockToolContext } from '../../../../test/mocks/tool';

import { thinkDeeplyFunction } from './think';

describe('thinkDeeplyFunction', () => {
  it('name', () => {
    expect(thinkDeeplyFunction.name).toBe('think_deeply');
  });

  it('description', () => {
    expect(thinkDeeplyFunction.description).toBeDefined();
  });

  describe('parameters', () => {
    const { parameters } = thinkDeeplyFunction;
    expect(parameters).toBeDefined();

    it('thought', () => {
      expect(parameters).toHaveProperty('thought');
      expect(parameters.thought.def.type).toBe('string');
      expect(parameters.thought.description).toBeDefined();
    });
  });

  describe('handler', () => {
    it('thought', () => {
      const result = thinkDeeplyFunction.handler(
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
