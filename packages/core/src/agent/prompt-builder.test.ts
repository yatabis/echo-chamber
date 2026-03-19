import { describe, expect, it } from 'vitest';

import {
  buildAgentPromptMessages,
  buildRuntimeContextPrompt,
  buildToolCatalogPrompt,
} from './prompt-builder';
import { canonicalToolSpecifications } from './tools/catalog';

describe('buildToolCatalogPrompt', () => {
  it('canonical tool definitions からツール一覧を生成する', () => {
    const result = buildToolCatalogPrompt();

    expect(result).toContain('<available_tools>');
    expect(result).toContain('</available_tools>');

    for (const tool of canonicalToolSpecifications) {
      expect(result).toContain(`- ${tool.name}: ${tool.description}`);
    }

    expect(result).toContain('limit (required): 取得するメッセージ数');
    expect(result).toContain('arguments: none');
  });
});

describe('buildRuntimeContextPrompt', () => {
  it('latest memory がない場合の runtime context block を生成する', () => {
    const result = buildRuntimeContextPrompt(
      new Date('2025-01-25T15:00:00.000Z'),
      null
    );

    expect(result).toContain('<runtime_context>');
    expect(result).toContain('No persisted context loaded.');
    expect(result).toContain('Current datetime: 2025年01月26日 00:00:00');
    expect(result).toContain('</runtime_context>');
  });

  it('latest memory がある場合の runtime context block を生成する', () => {
    const result = buildRuntimeContextPrompt(
      new Date('2025-01-25T15:00:00.000Z'),
      {
        content: 'Had a meaningful conversation.',
        createdAt: '2日前 (2025年01月23日 13:56:07)',
        emotion: {
          valence: 0.7,
          arousal: 0.4,
          labels: ['joy', 'interest'],
        },
      }
    );

    expect(result).toContain('Latest memory:');
    expect(result).toContain('"content": "Had a meaningful conversation."');
    expect(result).toContain('"created_at": "2日前 (2025年01月23日 13:56:07)"');
    expect(result).toContain('"valence": 0.7');
    expect(result).toContain('"labels": [');
  });
});

describe('buildAgentPromptMessages', () => {
  it('system prompt と generated blocks から developer messages を組み立てる', () => {
    const result = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      latestMemory: null,
    });

    expect(result).toEqual([
      expect.objectContaining({
        role: 'developer',
      }),
      expect.objectContaining({
        role: 'developer',
      }),
    ]);
    expect(result[0]?.content).toContain('<persona>Test persona</persona>');
    expect(result[0]?.content).toContain('<available_tools>');
    expect(result[1]?.content).toContain('<runtime_context>');
  });
});
