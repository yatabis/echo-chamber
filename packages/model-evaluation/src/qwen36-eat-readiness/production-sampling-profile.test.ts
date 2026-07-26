import { describe, expect, it } from 'vitest';

import { parseProductionSamplingOverride } from './production-sampling-profile';

const VALID_OVERRIDE = {
  description:
    'Agents-A1 official sampling values in the E.C.H.O. non-thinking evaluator.',
  temperature: 0.85,
  topP: 0.95,
  topK: 20,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 1.1,
};

describe('production sampling profile override', () => {
  it('replaces sampling values without changing evaluator safety limits', () => {
    expect(parseProductionSamplingOverride(VALID_OVERRIDE)).toEqual({
      id: 'production-sampling',
      ...VALID_OVERRIDE,
      maxTokensPerTurn: 1_024,
      enableThinking: false,
    });
  });

  it.each([
    ['temperature', -0.1],
    ['topP', 1.1],
    ['topK', 0],
    ['topK', 1.5],
    ['minP', -0.1],
    ['repetitionPenalty', 0],
    ['presencePenalty', 2.1],
  ])('rejects an invalid %s', (key, value) => {
    expect(() =>
      parseProductionSamplingOverride({
        ...VALID_OVERRIDE,
        [key]: value,
      })
    ).toThrow(`Production sampling override has invalid ${key}`);
  });

  it('rejects missing required values', () => {
    const { topP: _topP, ...missingTopP } = VALID_OVERRIDE;

    expect(() => parseProductionSamplingOverride(missingTopP)).toThrow(
      'Production sampling override has invalid topP'
    );
  });

  it('rejects unknown keys that could hide a misspelling', () => {
    expect(() =>
      parseProductionSamplingOverride({
        ...VALID_OVERRIDE,
        top_p: 0.95,
      })
    ).toThrow('Production sampling override has unknown key: top_p');
  });
});
