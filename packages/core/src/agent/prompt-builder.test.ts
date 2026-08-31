import { describe, expect, it } from 'vitest';

import {
  buildAgentPromptMessages,
  buildEmotionCognitiveModuleSystemPrompt,
  buildMemoryCognitiveModuleSystemPrompt,
  buildRuntimeContextPrompt,
  buildToolCatalogPrompt,
} from './prompt-builder';
import { canonicalRuntimeTools } from './runtime-tools/catalog';

import type { ModelToolContract } from '../ports/model';

const testCurrentDatetime = new Date('2025-01-25T15:00:00.000Z');
const canonicalToolContracts = canonicalRuntimeTools.map(
  (tool) => tool.contract
);

function createToolContract(name: string): ModelToolContract {
  return {
    name,
    description: `${name} の説明`,
    inputSchema: {
      type: 'object',
      properties: {},
      additionalProperties: false,
    },
    strict: true,
  };
}

describe('buildToolCatalogPrompt', () => {
  it('bind対象のtool contractsからツール一覧を生成する', () => {
    const result = buildToolCatalogPrompt(canonicalToolContracts);

    expect(result).toContain('<available_tools>');
    expect(result).toContain('</available_tools>');

    for (const tool of canonicalToolContracts) {
      expect(result).toContain(`- ${tool.name}: ${tool.description}`);
    }

    expect(result).toContain(
      'channelKey (必須): 読み取り対象の channelKey。check_notifications の結果に含まれる channelKey を使う。'
    );
    expect(result).toContain('limit (必須): 取得するメッセージ数');
    expect(result).toContain('引数: なし');
  });

  it('渡されていないtoolをstatic catalogから補完しない', () => {
    const result = buildToolCatalogPrompt([
      createToolContract('only_bound_tool'),
    ]);

    expect(result).toContain('- only_bound_tool: only_bound_tool の説明');
    expect(result).not.toContain('finish_thinking');
    expect(result).not.toContain('read_web_page');
  });
});

describe('buildRuntimeContextPrompt', () => {
  it('共有runtime contextに現在日時を含める', () => {
    const result = buildRuntimeContextPrompt(testCurrentDatetime);

    expect(result).toBe(
      '<runtime_context>\n現在日時: 2025年01月26日 00:00:00\n</runtime_context>'
    );
  });
});

describe('buildMemoryCognitiveModuleSystemPrompt', () => {
  it.each([
    [
      'pre_main',
      'あなたはE.C.H.O. Chamberで動作する「リン」の記憶モジュールです。記憶の想起と記銘を担います。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。共有コンテキストから、「リン」が次の思考で必要とする可能性のある記憶を想起してください。その記憶を検索するためのクエリを1つ返してください。',
    ],
    [
      'post_main',
      'あなたはE.C.H.O. Chamberで動作する「リン」の記憶モジュールです。記憶の想起と記銘を担います。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。完了した思考セッションの共有コンテキストから、「リン」が記憶しておく内容を選んでください。記憶の本文と種類を返してください。',
    ],
  ] as const)('%s phaseの責務を示す', (phase, expected) => {
    expect(buildMemoryCognitiveModuleSystemPrompt('リン', phase)).toBe(
      expected
    );
  });
});

describe('buildEmotionCognitiveModuleSystemPrompt', () => {
  it('現在の感情状態を更新する責務を示す', () => {
    expect(buildEmotionCognitiveModuleSystemPrompt('リン')).toBe(
      'あなたはE.C.H.O. Chamberで動作する「リン」の感情モジュールです。「リン」の感情状態を管理します。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。共有コンテキストに基づいて現在の感情状態を更新してください。感情価（valence）、覚醒度（arousal）、ラベル（labels）を返してください。'
    );
  });
});

describe('buildAgentPromptMessages', () => {
  it('Main専用promptと共有runtime contextを別のdeveloper messageとして返す', () => {
    const result = buildAgentPromptMessages({
      systemPrompt: '<persona>テスト用ペルソナ</persona>',
      currentDatetime: testCurrentDatetime,
      toolContracts: [createToolContract('only_bound_tool')],
    });

    expect(result.mainSystemPrompt.role).toBe('developer');
    expect(result.sharedRuntimeContext).toEqual({
      role: 'developer',
      content:
        '<runtime_context>\n現在日時: 2025年01月26日 00:00:00\n</runtime_context>',
    });
    expect(result.mainSystemPrompt.content).toContain(
      '<persona>テスト用ペルソナ</persona>'
    );
    expect(result.mainSystemPrompt.content).toContain('<available_tools>');
    expect(result.mainSystemPrompt.content).toContain('only_bound_tool');
    expect(result.mainSystemPrompt.content).toContain(
      'システムが追加する search_memory と update_emotion のツール往復'
    );
  });
});
