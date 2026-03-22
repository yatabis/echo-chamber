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
  it('persisted context がない場合の runtime context block を生成する', () => {
    const result = buildRuntimeContextPrompt(
      new Date('2025-01-25T15:00:00.000Z'),
      null
    );

    expect(result).toContain('<runtime_context>');
    expect(result).toContain('No persisted context loaded.');
    expect(result).toContain('Current datetime: 2025年01月26日 00:00:00');
    expect(result).toContain('</runtime_context>');
  });

  it('latest context がある場合の runtime context block を生成する', () => {
    const result = buildRuntimeContextPrompt(
      new Date('2025-01-25T15:00:00.000Z'),
      {
        content: 'Finished replying to the urgent thread and queued the rest.',
        createdAt: '2025-01-25T15:00:00.000Z',
        emotion: {
          valence: 0.4,
          arousal: 0.2,
          labels: ['calm', 'satisfied'],
        },
      }
    );

    expect(result).toContain('Latest context:');
    expect(result).toContain(
      '"content": "Finished replying to the urgent thread and queued the rest."'
    );
    expect(result).toContain('"created_at": "2025-01-25T15:00:00.000Z"');
    expect(result).toContain('"labels": [');
  });
});

describe('buildAgentPromptMessages', () => {
  it('system prompt と generated blocks から developer messages を組み立てる', () => {
    const result = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: new Date('2025-01-25T15:00:00.000Z'),
      latestContext: null,
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
