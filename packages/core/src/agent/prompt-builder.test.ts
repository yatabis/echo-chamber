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

const testPersonaProfile =
  '<persona_definition>\nテスト用の人物像\n</persona_definition>';

describe('buildMemoryCognitiveModuleSystemPrompt', () => {
  it.each([
    [
      'pre_main',
      [
        'あなたはE.C.H.O. Chamberで動作する「リン」の記憶モジュールです。記憶の想起と記銘を担います。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。',
        '次は「リン」の人物像です。リンが何に気づき、何を大切にし、誰とどのような関係にあるかを、想起と記銘の手がかりにしてください。\n<persona_profile>\n<persona_definition>\nテスト用の人物像\n</persona_definition>\n</persona_profile>',
        '共有コンテキストから、「リン」が次の思考で必要とする可能性のある記憶を想起してください。その記憶を検索するためのクエリを1つ返してください。',
      ].join('\n\n'),
    ],
    [
      'post_main',
      [
        'あなたはE.C.H.O. Chamberで動作する「リン」の記憶モジュールです。記憶の想起と記銘を担います。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。',
        '次は「リン」の人物像です。リンが何に気づき、何を大切にし、誰とどのような関係にあるかを、想起と記銘の手がかりにしてください。\n<persona_profile>\n<persona_definition>\nテスト用の人物像\n</persona_definition>\n</persona_profile>',
        '完了した思考セッションの共有コンテキストから、「リン」が記憶しておく内容を選んでください。記憶の本文と種類を返してください。',
      ].join('\n\n'),
    ],
  ] as const)(
    '%s phaseの責務と、判断の手がかりになる個体の人物像を示す',
    (phase, expected) => {
      expect(
        buildMemoryCognitiveModuleSystemPrompt(
          'リン',
          phase,
          testPersonaProfile
        )
      ).toBe(expected);
    }
  );
});

describe('buildEmotionCognitiveModuleSystemPrompt', () => {
  it('現在の感情状態を更新する責務と、判断の手がかりになる個体の人物像を示す', () => {
    expect(
      buildEmotionCognitiveModuleSystemPrompt('リン', testPersonaProfile)
    ).toBe(
      [
        'あなたはE.C.H.O. Chamberで動作する「リン」の感情モジュールです。「リン」の感情状態を管理します。共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。',
        '次は「リン」の人物像です。リンの気質や感情の動き方、大切にしているもの、他者との関係を手がかりに、共有コンテキストの出来事がリンの感情をどう動かすかを考えてください。\n<persona_profile>\n<persona_definition>\nテスト用の人物像\n</persona_definition>\n</persona_profile>',
        '共有コンテキストに基づいて現在の感情状態を更新してください。感情価（valence）、覚醒度（arousal）、ラベル（labels）を返してください。',
      ].join('\n\n')
    );
  });
});

describe('buildAgentPromptMessages', () => {
  it('Main専用指示をsystem roleにし、共有runtime contextと分離する', () => {
    const result = buildAgentPromptMessages({
      systemPrompt: '<persona>テスト用ペルソナ</persona>',
      currentDatetime: testCurrentDatetime,
      toolContracts: [createToolContract('only_bound_tool')],
    });

    expect(result.mainSystemPrompt.role).toBe('system');
    expect(result.sharedRuntimeContext).toEqual({
      role: 'system',
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
