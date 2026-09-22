import type {
  CognitiveModuleCommittedState,
  CognitiveModuleDomainCommitInput,
  CognitiveModuleDomainPort,
  CognitiveModulePhaseResult,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import type { AgentSessionTool } from '@echo-chamber/core/agent/session';
import { finishThinkingInputSchema } from '@echo-chamber/core/agent/tools/thinking';

/** Test-only domain: model outputs are real in the opt-in test; storage/search are in memory. */
export class CognitiveFixtureDomain implements CognitiveModuleDomainPort {
  readonly commits: CognitiveModuleDomainCommitInput[] = [];
  readonly failures: CognitiveModulePhaseResult[] = [];
  state: CognitiveModuleCommittedState = {
    version: 0,
    emotion: null,
    previousSessionMemory: null,
    recalledMemories: [],
  };

  /** Reuse only the prior committed session result in a new activation. */
  async beginActivation(): Promise<CognitiveModuleCommittedState> {
    return await Promise.resolve({
      ...structuredClone(this.state),
      recalledMemories: [],
    });
  }

  /** Test storage has no external phase log. */
  async startPhase(): Promise<void> {
    await Promise.resolve();
  }

  /** One synchronous replacement models a successful atomic domain commit. */
  async commitPhase(
    input: CognitiveModuleDomainCommitInput
  ): Promise<CognitiveModuleCommittedState> {
    if (input.phase.committed.version !== this.state.version)
      throw new Error('fixture version conflict');
    const memory = input.memory.value;
    const emotion = input.emotion.value;
    const previousSessionMemory =
      'content' in memory
        ? { ...memory, emotion, createdAt: '2026-09-21T00:00:00.000Z' }
        : this.state.previousSessionMemory;
    // The fixture's bounded search corpus contains the previous session item.
    const recalledMemories =
      'query' in memory && this.state.previousSessionMemory !== null
        ? [this.state.previousSessionMemory]
        : [];
    this.state = {
      version: this.state.version + 1,
      emotion,
      previousSessionMemory,
      recalledMemories,
    };
    this.commits.push(structuredClone(input));
    return await Promise.resolve(structuredClone(this.state));
  }

  /** Keep failed phases separate from committed domain data. */
  async failPhase(result: CognitiveModulePhaseResult): Promise<void> {
    this.failures.push(result);
    await Promise.resolve();
  }
}

/** Bounded Main task for testing two turns; it is not a production persona prompt. */
export const COGNITIVE_FIXTURE_PROMPT =
  'これはローカルの接続テストです。最初の応答では inspect_note を1回呼んでください。' +
  '結果が来たら、その確認コードを reason に含めて finish_thinking を呼んで終了してください。' +
  'それ以外の操作は必要ありません。MemoryとEmotionは実行環境が各turnの前後に処理します。';

/** Local tools with no network or external effects, using the ordinary finish contract. */
export function createCognitiveFixtureTools(): {
  tools: AgentSessionTool[];
  calls: string[];
  finishedReasons: string[];
} {
  const calls: string[] = [];
  const finishedReasons: string[] = [];
  const tools: AgentSessionTool[] = [
    'check_notifications',
    'inspect_note',
    'finish_thinking',
  ].map((name) => ({
    name,
    contract: {
      name,
      description:
        name === 'inspect_note' ? '接続テストの確認コードを読む。' : name,
      inputSchema:
        name === 'finish_thinking'
          ? {
              type: 'object',
              properties: { reason: { type: 'string' } },
              required: ['reason'],
              additionalProperties: false,
            }
          : { type: 'object', properties: {}, additionalProperties: false },
    },
    execute: async (raw): Promise<string> => {
      calls.push(name);
      if (name === 'finish_thinking') {
        const input = finishThinkingInputSchema.parse(JSON.parse(raw));
        finishedReasons.push(input.reason);
      }
      return await Promise.resolve(
        JSON.stringify(
          name === 'inspect_note'
            ? {
                success: true,
                code: 'KITE-47',
                content: '明日は図書館へ行く。',
              }
            : { success: true }
        )
      );
    },
  }));
  return { tools, calls, finishedReasons };
}

/** A second session reviews the committed result; it introduces no new lookup task. */
export const COGNITIVE_REVIEW_PROMPT =
  '前回の確認作業は完了しています。この起動では、実行環境から渡される記憶と感情を参照し、' +
  'その内容を reason に簡潔に含めて finish_thinking で終了してください。新しい確認作業はありません。';
