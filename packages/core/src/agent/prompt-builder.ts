import { formatJapaneseDatetime } from '../utils/datetime';

import type { CognitiveModulePhase } from './cognitive-module-orchestrator';
import type { ModelToolContract } from '../ports/model';

/**
 * Agent の初期 developer prompt を組み立てるための入力。
 * static prompt、本時刻、利用可能ツール一覧をまとめて受け取る。
 */
export interface BuildAgentPromptInput {
  systemPrompt: string;
  currentDatetime: Date;
  toolContracts: readonly ModelToolContract[];
}

/**
 * prompt builder が返す developer message。
 * E.C.H.O. Chamber の model protocol に渡す中間表現として使う。
 */
export interface AgentPromptMessage {
  role: 'developer';
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
 * Memory Cognitive Moduleが現在phaseで担う役割をsystem promptにする。
 */
export function buildMemoryCognitiveModuleSystemPrompt(
  instanceName: string,
  phase: CognitiveModulePhase
): string {
  const identity = `あなたはE.C.H.O. Chamberで動作する「${instanceName}」の記憶モジュールです。記憶の想起と記銘を担います。${COGNITIVE_SHARED_CONTEXT_INSTRUCTION}`;
  return phase === 'pre_main'
    ? `${identity}共有コンテキストから、「${instanceName}」が次の思考で必要とする可能性のある記憶を想起してください。その記憶を検索するためのクエリを1つ返してください。`
    : `${identity}完了した思考セッションの共有コンテキストから、「${instanceName}」が記憶しておく内容を選んでください。記憶の本文と種類を返してください。`;
}

/**
 * Emotion Cognitive Moduleが現在の感情状態を更新するsystem promptを作る。
 */
export function buildEmotionCognitiveModuleSystemPrompt(
  instanceName: string
): string {
  return `あなたはE.C.H.O. Chamberで動作する「${instanceName}」の感情モジュールです。「${instanceName}」の感情状態を管理します。${COGNITIVE_SHARED_CONTEXT_INSTRUCTION}共有コンテキストに基づいて現在の感情状態を更新してください。感情価（valence）、覚醒度（arousal）、ラベル（labels）を返してください。`;
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
      role: 'developer',
      content: [
        input.systemPrompt,
        toolCatalog,
        '各メインモデルターンの前にシステムが追加する search_memory と update_emotion のツール往復によって、思考の継続に必要な記憶と感情状態が渡されます。',
      ].join('\n\n'),
    },
    sharedRuntimeContext: {
      role: 'developer',
      content: runtimeContext,
    },
  };
}
