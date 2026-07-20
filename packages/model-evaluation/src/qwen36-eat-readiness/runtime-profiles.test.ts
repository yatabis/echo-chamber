import { describe, expect, it } from 'vitest';

import {
  CONTROLLED_GREEDY_PROFILE,
  PRODUCTION_SAMPLING_PROFILE,
} from './runtime-profiles';

describe('qwen36 runtime generation profiles', () => {
  it('keeps the controlled comparison free of repetition penalties', () => {
    expect(CONTROLLED_GREEDY_PROFILE).toMatchObject({
      temperature: 0,
      topP: 1,
      topK: 1,
      minP: 0,
      repetitionPenalty: 1,
      presencePenalty: 0,
    });
  });

  it('copies every production sampling parameter except the documented output cap', () => {
    expect(PRODUCTION_SAMPLING_PROFILE).toMatchObject({
      temperature: 0.7,
      topP: 0.8,
      topK: 20,
      minP: 0,
      repetitionPenalty: 1,
      presencePenalty: 1.5,
      maxTokensPerTurn: 1_024,
    });
  });
});
