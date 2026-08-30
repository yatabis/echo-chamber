import { expect, test } from 'vitest';

import type {
  CognitiveModuleCommittedState,
  CognitiveModuleDomainPort,
  CognitiveModuleName,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import { getEchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';
import type { EchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { ModelRequest } from '@echo-chamber/core/ports/model';

import { createCognitiveModuleOrchestrator } from '../../../apps/cloudflare-workers/src/echo/cognitive-modules';

const LIVE_EVALUATION_ENABLED = process.env.ECHO_COGNITIVE_HOSTED_EVAL === '1';
const liveTest = LIVE_EVALUATION_ENABLED ? test : test.skip;

type CognitiveState = CognitiveModuleCommittedState;

interface ModelRequestObservation {
  module: CognitiveModuleName;
  maxOutputTokens: number | undefined;
  previousResponseToken: string | undefined;
  responseFormatType: string | undefined;
}

/** Live smoke 用に、production と同じ phase state 遷移を再現する。 */
function createEvaluationDomain(): CognitiveModuleDomainPort {
  let state: CognitiveState = {
    version: 0,
    emotion: null,
    previousSessionMemory: null,
    recalledMemories: [],
  };

  return {
    beginActivation: async (): Promise<CognitiveState> =>
      await Promise.resolve(state),
    startPhase: async (): Promise<void> => {
      await Promise.resolve();
    },
    commitPhase: async (input): Promise<CognitiveState> => {
      state = {
        ...state,
        version: state.version + 1,
        emotion: input.emotion.value,
      };
      return await Promise.resolve(state);
    },
    failPhase: async (): Promise<void> => {
      await Promise.resolve();
    },
  };
}

function getRequiredApiKey(): string {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (apiKey === undefined || apiKey === '') {
    throw new Error('OPENAI_API_KEY is required for hosted cognitive eval');
  }
  return apiKey;
}

liveTest(
  'Hosted Cognitive Module は専用system promptと共有contextからphase別schema出力を返す',
  async () => {
    const startedAt = Date.now();
    const modelRequests: ModelRequestObservation[] = [];
    const events: EchoEvent[] = [];
    const orchestrator = createCognitiveModuleOrchestrator({
      env: { OPENAI_API_KEY: getRequiredApiKey() },
      instance: {
        ...getEchoInstanceDefinition('rin'),
        name: 'リン（合成評価）',
      },
      domain: createEvaluationDomain(),
      createActivationId: () => 'synthetic-hosted-eval-activation',
      events: {
        emit: async (event): Promise<void> => {
          events.push(event);
          await Promise.resolve();
        },
      },
      beforeModelRequest: (
        module: CognitiveModuleName,
        request: ModelRequest
      ) => {
        modelRequests.push({
          module,
          maxOutputTokens: request.maxOutputTokens,
          previousResponseToken: request.previousResponseToken,
          responseFormatType: request.responseFormat?.type,
        });
      },
    });
    const activation = orchestrator.beginActivation();

    const preMainHandoff = await activation.beforeMain([
      {
        role: 'user',
        content:
          '[合成評価データ] 実際のユーザー判断: 本番投入前にレビューとテストを必須とする。',
      },
    ]);
    expect(preMainHandoff).toHaveLength(4);
    expect(preMainHandoff).toMatchObject([
      { type: 'tool_call', toolName: 'search_memory' },
      { type: 'tool_result' },
      { type: 'tool_call', toolName: 'update_emotion' },
      { type: 'tool_result', output: '{"success":true}' },
    ]);

    await activation.onMainTurnBoundary({
      turnIndex: 1,
      responseOutput: [
        {
          type: 'tool_call',
          callId: 'finish-eval',
          toolName: 'finish_thinking',
          input: '{"reason":"合成評価完了"}',
        },
      ],
      toolCalls: [
        {
          type: 'tool_call',
          callId: 'finish-eval',
          toolName: 'finish_thinking',
          input: '{"reason":"合成評価完了"}',
        },
      ],
      resolvedInput: [
        {
          type: 'tool_result',
          callId: 'finish-eval',
          output: '{"success":true}',
        },
      ],
      terminationReason: 'finish_thinking',
    });
    const result = activation.getResultSnapshot();

    expect(result.phases).toHaveLength(2);
    expect(result.phases.map((phase) => phase.phase)).toEqual([
      'pre_main',
      'post_main',
    ]);
    expect(
      result.phases.every((phase) => phase.memory.status === 'ready')
    ).toBe(true);
    expect(
      result.phases.every((phase) => phase.emotion.status === 'ready')
    ).toBe(true);
    expect(result.usage.totalTokens).toBeGreaterThan(0);
    expect(modelRequests).toHaveLength(4);
    expect(
      modelRequests.every(
        ({ maxOutputTokens, responseFormatType }) =>
          maxOutputTokens === 2048 && responseFormatType === 'json_schema'
      )
    ).toBe(true);
    for (const module of ['memory', 'emotion'] as const) {
      const moduleRequests = modelRequests.filter(
        (request) => request.module === module
      );
      expect(moduleRequests).toHaveLength(2);
      expect(
        moduleRequests.every(
          ({ previousResponseToken }) => previousResponseToken === undefined
        )
      ).toBe(true);
    }

    const exchangeEvents = events.filter(
      (event) => event.type === 'model.exchange.recorded'
    );
    expect(exchangeEvents).toHaveLength(4);
    expect(
      exchangeEvents.every(
        (event) =>
          (event.payload?.cognitiveModule === 'memory' ||
            event.payload?.cognitiveModule === 'emotion') &&
          Object.prototype.hasOwnProperty.call(event.payload, 'request') &&
          Object.prototype.hasOwnProperty.call(event.payload, 'response') &&
          !Object.prototype.hasOwnProperty.call(event.payload, 'redacted')
      )
    ).toBe(true);
    expect(JSON.stringify(events)).toContain(
      '実際のユーザー判断: 本番投入前にレビューとテストを必須とする。'
    );

    process.stdout.write(
      `cognitive-hosted-eval ${JSON.stringify({
        activationId: result.activationId,
        durationMs: Date.now() - startedAt,
        modelCalls: modelRequests.length,
        phases: result.phases.length,
        totalInputTokens: result.usage.totalInputTokens,
        totalOutputTokens: result.usage.outputTokens,
        totalTokens: result.usage.totalTokens,
      })}\n`
    );
  },
  120_000
);
