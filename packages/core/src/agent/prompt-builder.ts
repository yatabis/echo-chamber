import { formatJapaneseDatetime } from '../utils/datetime';

import type { Emotion, MemoryType } from '../echo/types';
import type { ModelToolContract } from '../ports/model';

/**
 * 起動時に prompt へ注入する最新 context の要約。
 * 直前セッションの要点だけを表し、長期記憶とは別枠で扱う。
 */
export interface PromptContextSnapshot {
  content: string;
  createdAt: string;
  emotion: Emotion;
}

/**
 * 起動時の context から引いた関連メモリの最小表現。
 * memory record 全体ではなく、再開判断に必要な要点だけを prompt に渡す。
 */
export interface PromptRelatedMemorySnapshot {
  content: string;
  type: MemoryType;
  createdAt: string;
  emotion: Emotion;
}

/**
 * Agent の初期 developer prompt を組み立てるための入力。
 * static prompt、本時刻、直近 context、利用可能ツール一覧をまとめて受け取る。
 */
export interface BuildAgentPromptInput {
  systemPrompt: string;
  currentDatetime: Date;
  latestContext: PromptContextSnapshot | null;
  relatedMemories?: readonly PromptRelatedMemorySnapshot[];
  toolContracts: readonly ModelToolContract[];
}

/**
 * prompt builder が返す developer message。
 * 現在は OpenAI Responses API の入力に変換する前段の中間表現として使う。
 */
export interface AgentPromptMessage {
  role: 'developer';
  content: string;
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
    const requiredLabel = required.has(name) ? 'required' : 'optional';
    const propertySchema = getRecord(property);
    const description =
      typeof propertySchema?.description === 'string'
        ? propertySchema.description
        : 'No description provided.';

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
      ...(parameterLines.length > 0 ? parameterLines : ['  - arguments: none']),
    ];
  });

  return [
    '<available_tools>',
    'You have access to the following tools:',
    ...lines,
    '</available_tools>',
  ].join('\n');
}

/**
 * latest context を prompt に埋め込む JSON ブロックへ整形する。
 *
 * @param latestContext 起動時に再注入する最新 context
 * @returns `Latest context:` 見出し付きの整形済みブロック
 */
function formatLatestContextBlock(
  latestContext: PromptContextSnapshot
): string {
  return `Latest context:\n${JSON.stringify(
    {
      content: latestContext.content,
      created_at: latestContext.createdAt,
      emotion: {
        valence: latestContext.emotion.valence,
        arousal: latestContext.emotion.arousal,
        labels: latestContext.emotion.labels,
      },
    },
    null,
    2
  )}`;
}

/**
 * 最新 context から検索した関連メモリを prompt 用ブロックへ整形する。
 */
function formatRelatedMemoriesBlock(
  relatedMemories: readonly PromptRelatedMemorySnapshot[]
): string {
  return `Related memories:\n${JSON.stringify(
    relatedMemories.map((memory) => ({
      content: memory.content,
      type: memory.type,
      created_at: memory.createdAt,
      emotion: {
        valence: memory.emotion.valence,
        arousal: memory.emotion.arousal,
        labels: memory.emotion.labels,
      },
    })),
    null,
    2
  )}`;
}

/**
 * 起動時の runtime context を表す `<runtime_context>` ブロックを生成する。
 * 直近 context と現在時刻をひとまとめにし、
 * 思考再開時の足掛かりとして prompt に差し込む。
 */
export function buildRuntimeContextPrompt(
  currentDatetime: Date,
  latestContext: PromptContextSnapshot | null,
  relatedMemories: readonly PromptRelatedMemorySnapshot[] = []
): string {
  const currentDatetimeText = formatJapaneseDatetime(currentDatetime);
  const persistedContextBlock =
    latestContext === null
      ? 'No persisted context loaded.'
      : [
          formatLatestContextBlock(latestContext),
          formatRelatedMemoriesBlock(relatedMemories),
        ].join('\n');

  return [
    '<runtime_context>',
    persistedContextBlock,
    `Current datetime: ${currentDatetimeText}`,
    '</runtime_context>',
  ].join('\n');
}

/**
 * Agent 起動時に渡す developer messages を組み立てる。
 * 1通目に static prompt と generated tool catalog、
 * 2通目に runtime context block を載せる構成にしている。
 */
export function buildAgentPromptMessages(
  input: BuildAgentPromptInput
): AgentPromptMessage[] {
  const toolCatalog = buildToolCatalogPrompt(input.toolContracts);
  const runtimeContext = buildRuntimeContextPrompt(
    input.currentDatetime,
    input.latestContext,
    input.relatedMemories ?? []
  );

  return [
    {
      role: 'developer',
      content: `${input.systemPrompt}\n\n${toolCatalog}`,
    },
    {
      role: 'developer',
      content: runtimeContext,
    },
  ];
}
