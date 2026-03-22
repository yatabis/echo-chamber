import { describe, expect, it } from 'vitest';

import {
  buildAgentPromptMessages,
  buildRuntimeContextPrompt,
  buildToolCatalogPrompt,
} from './prompt-builder';
import { canonicalToolSpecifications } from './tools/catalog';

const testCurrentDatetime = new Date('2025-01-25T15:00:00.000Z');
const testLatestContext = {
  content: 'Finished replying to the urgent thread and queued the rest.',
  createdAt: '2025-01-25T15:00:00.000Z',
  emotion: {
    valence: 0.4,
    arousal: 0.2,
    labels: ['calm', 'satisfied'],
  },
};
const testRelatedMemories = [
  {
    content: 'Remembered an earlier similar conversation.',
    type: 'episode' as const,
    createdAt: '2日前 (2025年01月24日 10:00:00)',
    emotion: {
      valence: 0.3,
      arousal: 0.1,
      labels: ['calm'],
    },
  },
];

describe('buildToolCatalogPrompt', () => {
  it('canonical tool definitions からツール一覧を生成する', () => {
    const result = buildToolCatalogPrompt();

    expect(result).toContain('<available_tools>');
    expect(result).toContain('</available_tools>');

    for (const tool of canonicalToolSpecifications) {
      expect(result).toContain(`- ${tool.name}: ${tool.description}`);
    }

    expect(result).toContain(
      'channelKey (required): 読み取り対象の channelKey。check_notifications の結果に含まれる channelKey を使う。'
    );
    expect(result).toContain('limit (required): 取得するメッセージ数');
    expect(result).toContain('arguments: none');
  });
});

describe('buildRuntimeContextPrompt', () => {
  it('runtime context block の開始タグと終了タグを含む', () => {
    const result = buildRuntimeContextPrompt(testCurrentDatetime, null);

    expect(result).toContain('<runtime_context>');
    expect(result).toContain('</runtime_context>');
  });

  it('persisted context がない場合はプレースホルダを表示する', () => {
    const result = buildRuntimeContextPrompt(testCurrentDatetime, null);

    expect(result).toContain('No persisted context loaded.');
  });

  it('現在時刻を含める', () => {
    const result = buildRuntimeContextPrompt(testCurrentDatetime, null);

    expect(result).toContain('Current datetime: 2025年01月26日 00:00:00');
  });

  it('latest context がある場合は context 本文を含める', () => {
    const result = buildRuntimeContextPrompt(
      testCurrentDatetime,
      testLatestContext
    );

    expect(result).toContain('Latest context:');
    expect(result).toContain(
      '"content": "Finished replying to the urgent thread and queued the rest."'
    );
  });

  it('latest context がある場合は context の created_at と emotion を含める', () => {
    const result = buildRuntimeContextPrompt(
      testCurrentDatetime,
      testLatestContext
    );

    expect(result).toContain('"created_at": "2025-01-25T15:00:00.000Z"');
    expect(result).toContain('"labels": [');
  });

  it('関連メモリがある場合は related memories block を含める', () => {
    const result = buildRuntimeContextPrompt(
      testCurrentDatetime,
      testLatestContext,
      testRelatedMemories
    );

    expect(result).toContain('Related memories:');
    expect(result).toContain(
      '"content": "Remembered an earlier similar conversation."'
    );
    expect(result).toContain('"type": "episode"');
    expect(result).not.toContain('"similarity"');
  });
});

describe('buildAgentPromptMessages', () => {
  it('system prompt と generated blocks から developer messages を組み立てる', () => {
    const result = buildAgentPromptMessages({
      systemPrompt: '<persona>Test persona</persona>',
      currentDatetime: testCurrentDatetime,
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
