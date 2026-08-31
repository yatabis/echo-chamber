import { describe, expect, it } from 'vitest';

import {
  MAX_MEMORY_CONTENT_LENGTH,
  MAX_MEMORY_QUERY_LENGTH,
} from '../echo/schemas';

import {
  CognitiveModuleSchemaValidationError,
  createEmotionCognitiveModuleOutputFormat,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  parseEmotionCognitiveModuleOutput,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
  type EmotionCognitiveModuleOutput,
  type MemoryRecallCognitiveModuleOutput,
  type MemoryStoreCognitiveModuleOutput,
} from './cognitive-module-schema';

function createRecallOutput(): MemoryRecallCognitiveModuleOutput {
  return { query: 'current implementation task and relevant decisions' };
}

function createStoreOutput(): MemoryStoreCognitiveModuleOutput {
  return {
    content: 'Memory and Emotion use phase-specific interfaces.',
    type: 'semantic',
  };
}

function createEmotionOutput(): EmotionCognitiveModuleOutput {
  return {
    valence: 0.1,
    arousal: 0.2,
    labels: ['calm', 'directive'],
  };
}

function captureError(callback: () => void): unknown {
  try {
    callback();
  } catch (error) {
    return error;
  }
  throw new Error('Expected callback to throw');
}

describe('cognitive module output schemas', () => {
  it('Memory recallではsearch_memoryのqueryだけを受理する', () => {
    expect(
      parseMemoryRecallCognitiveModuleOutput(createRecallOutput())
    ).toEqual(createRecallOutput());
    expect(() =>
      parseMemoryRecallCognitiveModuleOutput({
        ...createRecallOutput(),
        type: 'semantic',
      })
    ).toThrow(CognitiveModuleSchemaValidationError);
  });

  it('Memory storeではstore_memoryのcontentとtypeだけを受理する', () => {
    expect(parseMemoryStoreCognitiveModuleOutput(createStoreOutput())).toEqual(
      createStoreOutput()
    );
    expect(() =>
      parseMemoryStoreCognitiveModuleOutput({
        ...createStoreOutput(),
        emotion: createEmotionOutput(),
      })
    ).toThrow(CognitiveModuleSchemaValidationError);
  });

  it('Memoryのphase別schemaを相互に取り違えない', () => {
    expect(() =>
      parseMemoryRecallCognitiveModuleOutput(createStoreOutput())
    ).toThrow(CognitiveModuleSchemaValidationError);
    expect(() =>
      parseMemoryStoreCognitiveModuleOutput(createRecallOutput())
    ).toThrow(CognitiveModuleSchemaValidationError);
  });

  it('Memory queryと保存本文の上限超過をboundedな診断で拒否する', () => {
    const oversizedQuery = 'q'.repeat(MAX_MEMORY_QUERY_LENGTH + 1);
    const queryError = captureError(() =>
      parseMemoryRecallCognitiveModuleOutput({ query: oversizedQuery })
    );
    expect(queryError).toMatchObject({
      code: 'strict_schema',
      issues: [{ path: 'query', code: 'too_big' }],
    });
    expect(String(queryError)).not.toContain(oversizedQuery);

    const oversizedContent = 'x'.repeat(MAX_MEMORY_CONTENT_LENGTH + 1);
    const contentError = captureError(() =>
      parseMemoryStoreCognitiveModuleOutput({
        ...createStoreOutput(),
        content: oversizedContent,
      })
    );
    expect(contentError).toMatchObject({
      code: 'strict_schema',
      issues: [{ path: 'content', code: 'too_big' }],
    });
    expect(String(contentError)).not.toContain(oversizedContent);
  });

  it('Emotionの感情状態を受理する', () => {
    expect(parseEmotionCognitiveModuleOutput(createEmotionOutput())).toEqual(
      createEmotionOutput()
    );
  });

  it('Emotionのlabelを各12文字・最大5件に制限する', () => {
    expect(
      parseEmotionCognitiveModuleOutput({
        ...createEmotionOutput(),
        labels: ['123456789012', 'two', 'three', 'four', 'five'],
      })
    ).toBeDefined();

    expect(() =>
      parseEmotionCognitiveModuleOutput({
        ...createEmotionOutput(),
        labels: ['1234567890123'],
      })
    ).toThrow(CognitiveModuleSchemaValidationError);
    expect(() =>
      parseEmotionCognitiveModuleOutput({
        ...createEmotionOutput(),
        labels: ['one', 'two', 'three', 'four', 'five', 'six'],
      })
    ).toThrow(CognitiveModuleSchemaValidationError);
  });

  it('E.C.H.O. Chamberのphase別strict output contractを公開する', () => {
    expect(createMemoryRecallCognitiveModuleOutputFormat()).toMatchObject({
      type: 'json_schema',
      name: 'cognitive_memory_recall',
      strict: true,
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['query'],
      },
    });
    expect(createMemoryStoreCognitiveModuleOutputFormat()).toMatchObject({
      type: 'json_schema',
      name: 'cognitive_memory_store',
      strict: true,
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['content', 'type'],
      },
    });
    expect(createEmotionCognitiveModuleOutputFormat()).toMatchObject({
      type: 'json_schema',
      name: 'cognitive_emotion_update',
      strict: true,
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['valence', 'arousal', 'labels'],
      },
    });
  });
});
