import type {
  CognitiveModuleCommittedState,
  CognitiveModuleOutcome,
  CognitiveModulePhaseResult,
} from './cognitive-module-orchestrator';
import type {
  MemoryCognitiveModuleOutput,
  MemoryRecallCognitiveModuleOutput,
} from './cognitive-module-schema';
import type { ModelInputItem } from '../ports/model';

const UPDATE_EMOTION_TOOL_NAME = 'update_emotion';

/** Mainの一会話内で一意かつprovider制限に収まる擬似tool call IDを作る。 */
function createCognitiveHandoffCallId(
  sequence: number,
  toolName: 'search_memory' | typeof UPDATE_EMOTION_TOOL_NAME
): string {
  return `cognitive:${sequence}:${toolName}`;
}

/** Main handoff に使える成功値だけを取り出す。 */
function getReadyValue<T>(outcome: CognitiveModuleOutcome<T>): T {
  if (outcome.status === 'failed') {
    throw new Error('Failed cognitive outcome cannot become a Main handoff');
  }
  return outcome.value;
}

/** Memory成功値がpre_main用queryであることを保証する。 */
function getRecallOutput(
  outcome: CognitiveModuleOutcome<MemoryCognitiveModuleOutput>
): MemoryRecallCognitiveModuleOutput {
  const value = getReadyValue(outcome);
  if (!('query' in value)) {
    throw new Error('Memory pre_main output must contain query');
  }
  return value;
}

/** 両moduleの確定結果をsystem-owned tool exchangeとしてMainへ渡す。 */
export function formatCognitiveModuleHandoff(
  result: CognitiveModulePhaseResult,
  committed: CognitiveModuleCommittedState
): readonly ModelInputItem[] {
  if (result.phase !== 'pre_main') {
    return [];
  }

  const memory = getRecallOutput(result.memory);
  const emotion = getReadyValue(result.emotion);
  const memoryCallId = createCognitiveHandoffCallId(
    result.sequence,
    'search_memory'
  );
  const emotionCallId = createCognitiveHandoffCallId(
    result.sequence,
    UPDATE_EMOTION_TOOL_NAME
  );

  return [
    {
      type: 'tool_call',
      origin: 'runtime',
      callId: memoryCallId,
      toolName: 'search_memory',
      input: JSON.stringify(memory),
    },
    {
      type: 'tool_result',
      callId: memoryCallId,
      output: JSON.stringify({
        success: true,
        results: committed.recalledMemories,
      }),
    },
    {
      type: 'tool_call',
      origin: 'runtime',
      callId: emotionCallId,
      toolName: UPDATE_EMOTION_TOOL_NAME,
      input: JSON.stringify(emotion),
    },
    {
      type: 'tool_result',
      callId: emotionCallId,
      output: JSON.stringify({ success: true }),
    },
  ];
}

/** 前sessionで確定したMemoryとEmotionをmodule専用の初期状態として復元する。 */
export function formatInitialCognitiveModuleContext(
  committed: CognitiveModuleCommittedState
): readonly ModelInputItem[] {
  if (committed.previousSessionMemory === null && committed.emotion === null) {
    return [];
  }

  return [
    {
      role: 'user',
      content: [
        '前回の思考セッション終了時の状態です。',
        JSON.stringify({
          memory: committed.previousSessionMemory,
          emotion: committed.emotion,
        }),
      ].join('\n'),
    },
  ];
}
