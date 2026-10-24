import { formatJapaneseDatetime } from '../utils/datetime';

import type { CognitiveModulePhase } from './cognitive-module-orchestrator';
import type {
  ModelInputItem,
  ModelMessageContent,
  ModelMessageContentPart,
  ModelToolContract,
} from '../ports/model';

/**
 * Agent の起動時の指示と共有contextを組み立てるための入力。
 * static prompt、本時刻、利用可能ツール一覧をまとめて受け取る。
 */
export interface BuildAgentPromptInput {
  systemPrompt: string;
  currentDatetime: Date;
  toolContracts: readonly ModelToolContract[];
}

/**
 * prompt builder が返す専用指示またはruntime contextのメッセージ。
 * E.C.H.O. Chamber の model protocol に渡す中間表現として使う。
 */
export interface AgentPromptMessage {
  role: 'system';
  content: string;
}

/** Main専用指示と各modelが共有するruntime context。 */
export interface AgentPromptMessages {
  mainSystemPrompt: AgentPromptMessage;
  sharedRuntimeContext: AgentPromptMessage;
}

/**
 * JSON Schema から、prompt に埋め込める引数説明の箇条書きを生成する。
 * required/optional を明示して、LLM が tool 呼び出し時の前提を読み取りやすくする。
 */
function buildToolParameterDescriptions(inputSchema: unknown): string[] {
  const schema = getRecord(inputSchema);
  const properties = getRecord(schema?.properties) ?? {};
  const required = new Set(
    Array.isArray(schema?.required)
      ? schema.required.filter(
          (property): property is string => typeof property === 'string'
        )
      : []
  );

  return Object.entries(properties).map(([name, property]) => {
    const requiredLabel = required.has(name) ? '必須' : '任意';
    const propertySchema = getRecord(property);
    const description =
      typeof propertySchema?.description === 'string'
        ? propertySchema.description
        : '説明なし。';

    return `  - ${name} (${requiredLabel}): ${description}`;
  });
}

/**
 * unknown 値を配列ではないobjectとして扱える場合だけ返す。
 */
function getRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

/**
 * bind済みtool contractsをもとに `<available_tools>` ブロックを生成する。
 * prompt、provider contract、実行可能toolの広告対象を同じ集合へ揃える。
 */
export function buildToolCatalogPrompt(
  toolContracts: readonly ModelToolContract[]
): string {
  const lines = toolContracts.flatMap((tool) => {
    const parameterLines = buildToolParameterDescriptions(tool.inputSchema);

    return [
      `- ${tool.name}: ${tool.description}`,
      ...(parameterLines.length > 0 ? parameterLines : ['  - 引数: なし']),
    ];
  });

  return [
    '<available_tools>',
    '利用可能なツールは次のとおりです。',
    ...lines,
    '</available_tools>',
  ].join('\n');
}

/**
 * 起動時の runtime context を表す `<runtime_context>` ブロックを生成する。
 * Domain continuity は Cognitive Module のsystem-owned tool exchangeから受け取り、
 * ここでは時刻だけを渡す。
 */
export function buildRuntimeContextPrompt(currentDatetime: Date): string {
  const currentDatetimeText = formatJapaneseDatetime(currentDatetime);

  return [
    '<runtime_context>',
    `現在日時: ${currentDatetimeText}`,
    '</runtime_context>',
  ].join('\n');
}

const COGNITIVE_SHARED_CONTEXT_INSTRUCTION =
  '共有コンテキストにあるthinkはMainの自然言語出力を表します。その他のツール利用も含め、いずれもMainの履歴であり、あなた自身の過去の出力ではありません。';

/**
 * Cognitive Moduleの判断材料として、個体の人物像を区切って示す。
 * 人物像は個体を理解する手がかりとして渡し、moduleに個体として振る舞わせる指示にはしない。
 */
function buildCognitivePersonaSection(
  intro: string,
  personaProfile: string
): string {
  return `${intro}\n<persona_profile>\n${personaProfile}\n</persona_profile>`;
}

/**
 * Memory Cognitive Moduleが現在phaseで担う役割をsystem promptにする。
 *
 * 人物像は phase ごとの指示より前に置き、想起と記銘で同じ前置きを共有する。
 *
 * @param instanceName 個体名
 * @param phase 想起（pre_main）か記銘（post_main）か
 * @param personaProfile 個体を理解するための人物像
 */
export function buildMemoryCognitiveModuleSystemPrompt(
  instanceName: string,
  phase: CognitiveModulePhase,
  personaProfile: string
): string {
  const identity = `あなたはE.C.H.O. Chamberで動作する「${instanceName}」の記憶モジュールです。記憶の想起と記銘を担います。${COGNITIVE_SHARED_CONTEXT_INSTRUCTION}`;
  const persona = buildCognitivePersonaSection(
    `次は「${instanceName}」の人物像です。${instanceName}が何に気づき、何を大切にし、誰とどのような関係にあるかを、想起と記銘の手がかりにしてください。`,
    personaProfile
  );
  const task =
    phase === 'pre_main'
      ? `共有コンテキストから、「${instanceName}」が次の思考で必要とする可能性のある記憶を想起してください。その記憶を検索するためのクエリを1つ返してください。`
      : `完了した思考セッションの共有コンテキストから、「${instanceName}」が記憶しておく内容を選んでください。記憶の本文と種類を返してください。`;
  return [identity, persona, task].join('\n\n');
}

/**
 * Emotion Cognitive Moduleが現在の感情状態を更新するsystem promptを作る。
 *
 * @param instanceName 個体名
 * @param personaProfile 個体を理解するための人物像
 */
export function buildEmotionCognitiveModuleSystemPrompt(
  instanceName: string,
  personaProfile: string
): string {
  const identity = `あなたはE.C.H.O. Chamberで動作する「${instanceName}」の感情モジュールです。「${instanceName}」の感情状態を管理します。${COGNITIVE_SHARED_CONTEXT_INSTRUCTION}`;
  const persona = buildCognitivePersonaSection(
    `次は「${instanceName}」の人物像です。${instanceName}の気質や感情の動き方、大切にしているもの、他者との関係を手がかりに、共有コンテキストの出来事が${instanceName}の感情をどう動かすかを考えてください。`,
    personaProfile
  );
  const task =
    '共有コンテキストに基づいて現在の感情状態を更新してください。感情価（valence）、覚醒度（arousal）、ラベル（labels）を返してください。';
  return [identity, persona, task].join('\n\n');
}

/**
 * Main専用system promptと共有runtime contextを組み立てる。
 */
export function buildAgentPromptMessages(
  input: BuildAgentPromptInput
): AgentPromptMessages {
  const toolCatalog = buildToolCatalogPrompt(input.toolContracts);
  const runtimeContext = buildRuntimeContextPrompt(input.currentDatetime);

  return {
    mainSystemPrompt: {
      role: 'system',
      content: [
        input.systemPrompt,
        toolCatalog,
        '各メインモデルターンの前にシステムが追加する search_memory と update_emotion のツール往復によって、思考の継続に必要な記憶と感情状態が渡されます。',
      ].join('\n\n'),
    },
    sharedRuntimeContext: {
      role: 'system',
      content: runtimeContext,
    },
  };
}

/**
 * Coreの専用指示と共有system contextを、先頭のsystemメッセージに組み立てる。
 * 共有contextでは前sessionの状態が日時に先行するため、systemだけを取り出す。
 * その他のメッセージ・tool履歴はrole、内容、順序を変えず接続する。
 */
export function buildModelInputWithSystemPrompt(
  systemPrompt: string,
  sharedContext: readonly ModelInputItem[]
): ModelInputItem[] {
  const instructions: ModelMessageContent[] = [systemPrompt];
  const observations: ModelInputItem[] = [];
  for (const item of sharedContext) {
    if ('role' in item && item.role === 'system') {
      instructions.push(item.content);
    } else {
      observations.push(item);
    }
  }
  const content = instructions.every(
    (instruction) => typeof instruction === 'string'
  )
    ? instructions.join('\n\n')
    : instructions.flatMap<ModelMessageContentPart>((instruction, index) => [
        ...(index === 0 ? [] : [{ type: 'text' as const, text: '\n\n' }]),
        ...(typeof instruction === 'string'
          ? [{ type: 'text' as const, text: instruction }]
          : instruction),
      ]);
  return [{ role: 'system', content }, ...observations];
}
