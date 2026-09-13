import { describe, expect, it } from 'vitest';

import {
  createEmotionCognitiveModuleOutputFormat,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
} from '@echo-chamber/core/agent/cognitive-module-schema';
import { ZERO_MODEL_USAGE } from '@echo-chamber/core/agent/session';

import { prepareStructuredOutput } from './structured-output';

describe('Native structured output', () => {
  it.each([
    [
      createMemoryRecallCognitiveModuleOutputFormat(),
      { query: '週末の予定' },
      { query: '' },
    ],
    [
      createMemoryStoreCognitiveModuleOutputFormat(),
      { content: '散歩した', type: 'episode' },
      { content: '散歩した', type: 'unknown' },
    ],
    [
      createEmotionCognitiveModuleOutputFormat(),
      { valence: 0.2, arousal: 0.1, labels: ['平静'] },
      { valence: 1.1, arousal: 0.1, labels: [] },
    ],
  ])(
    'accepts valid Cognitive output and rejects invalid output for %j',
    (format, valid, invalid) => {
      const contract = prepareStructuredOutput(format);
      const validate = (value: unknown): void =>
        contract?.validate(
          [
            {
              type: 'message',
              role: 'assistant',
              content: JSON.stringify(value),
            },
          ],
          ZERO_MODEL_USAGE,
          'token'
        );
      expect(() => {
        validate(valid);
      }).not.toThrow();
      expect(() => {
        validate(invalid);
      }).toThrow('schema_mismatch');
      expect(() => {
        validate({ ...valid, unexpected: true });
      }).toThrow('schema_mismatch');
    }
  );

  it.each([
    { type: 'string', unknownRule: true },
    { type: 'string', format: 'unknown-format' },
    { $ref: 'https://example.com/external-schema' },
    { $async: true, type: 'string' },
    { $schema: 'https://json-schema.org/draft/2020-12/schema', type: 'string' },
    { type: 'string', minLength: -1 },
    { type: 'number', maximum: Infinity },
  ])('rejects unsupported or invalid schema %j before generation', (schema) => {
    expect(() =>
      prepareStructuredOutput({
        type: 'json_schema',
        strict: true,
        name: 'unsupported',
        schema,
      })
    ).toThrow();
  });

  it('captures the schema before caller mutation and does not coerce generated values', () => {
    const schema = { type: 'integer', minimum: 7 };
    const contract = prepareStructuredOutput({
      type: 'json_schema',
      strict: true,
      name: 'number',
      schema,
    });
    schema.minimum = 1;
    expect(() =>
      contract?.validate(
        [{ type: 'message', role: 'assistant', content: '3' }],
        ZERO_MODEL_USAGE,
        'token'
      )
    ).toThrow('schema_mismatch');
    expect(() =>
      contract?.validate(
        [{ type: 'message', role: 'assistant', content: '"7"' }],
        ZERO_MODEL_USAGE,
        'token'
      )
    ).toThrow('schema_mismatch');
    expect(() =>
      contract?.validate(
        [{ type: 'tool_call', callId: 'call', toolName: 'answer', input: '7' }],
        ZERO_MODEL_USAGE,
        'token'
      )
    ).toThrow('unexpected_output');
  });
});
