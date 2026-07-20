import { describe, expect, it } from 'vitest';

import { parseEvaluationTargets } from './evaluation-targets';

const VALID_TARGET = {
  id: 'qwen36-35b-a3b',
  displayName: 'Qwen3.6-35B-A3B',
  modelPath: '/models/qwen36-35b-a3b',
  servedModelName: 'qwen36-35b-a3b-eval',
};

describe('evaluation targets', () => {
  it('accepts one or more self-contained local models', () => {
    expect(
      parseEvaluationTargets([
        VALID_TARGET,
        {
          id: 'qwen36-27b',
          displayName: 'Qwen3.6-27B',
          modelPath: '/models/qwen36-27b',
          servedModelName: 'qwen36-27b-eval',
        },
      ])
    ).toHaveLength(2);
  });

  it('rejects an empty list', () => {
    expect(() => parseEvaluationTargets([])).toThrow(
      'Evaluation targets must be a non-empty array'
    );
  });

  it.each(['id', 'displayName', 'modelPath', 'servedModelName'])(
    'rejects an invalid %s',
    (key) => {
      expect(() =>
        parseEvaluationTargets([{ ...VALID_TARGET, [key]: ' ' }])
      ).toThrow(`Evaluation target 0 has invalid ${key}`);
    }
  );

  it('rejects duplicate ids', () => {
    expect(() =>
      parseEvaluationTargets([
        VALID_TARGET,
        { ...VALID_TARGET, displayName: 'Duplicate ID' },
      ])
    ).toThrow('Evaluation targets contain duplicate ids');
  });
});
