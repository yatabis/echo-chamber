import { describe, expect, it, vi } from 'vitest';

import {
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
  type MemoryCognitiveModuleOutput,
} from './cognitive-module-schema';
import { ModelCognitiveModuleRunner } from './model-cognitive-module';

import type {
  CognitiveModuleOutputValidationError,
  CognitiveModulePhase,
  CognitiveModulePhaseInput,
} from './cognitive-module-orchestrator';
import type { ModelInputItem, ModelPort, ModelUsage } from '../ports/model';

function createUsage(totalTokens: number): ModelUsage {
  return {
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: totalTokens,
    totalInputTokens: totalTokens,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens,
  };
}

function createPhaseInput(
  phase: CognitiveModulePhase = 'pre_main',
  sequence = 1
): CognitiveModulePhaseInput {
  return {
    activationId: 'activation-1',
    boundaryId: `activation-1:${sequence}:${phase}`,
    sequence,
    phase,
    committed: {
      version: sequence - 1,
      emotion: null,
      previousSessionMemory: null,
      recalledMemories: [],
    },
  };
}

function createMemoryRunner(
  model: ModelPort
): ModelCognitiveModuleRunner<MemoryCognitiveModuleOutput> {
  const recallFormat = createMemoryRecallCognitiveModuleOutputFormat();
  const storeFormat = createMemoryStoreCognitiveModuleOutputFormat();
  return new ModelCognitiveModuleRunner<MemoryCognitiveModuleOutput>({
    model,
    resolveSystemPrompt: ({ phase }) =>
      phase === 'pre_main'
        ? 'あなたは記憶モジュールです。次のメインターンに役立つ記憶を想起してください。'
        : 'あなたは記憶モジュールです。完了した思考セッションから記憶を記銘してください。',
    resolveOutputContract: ({ phase }) =>
      phase === 'pre_main'
        ? {
            format: recallFormat,
            parse: parseMemoryRecallCognitiveModuleOutput,
          }
        : {
            format: storeFormat,
            parse: parseMemoryStoreCognitiveModuleOutput,
          },
  });
}

describe('ModelCognitiveModuleRunner', () => {
  it('専用system promptだけをdeveloper roleにして共有contextを観測として渡す', async () => {
    const usage = createUsage(12);
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: JSON.stringify({ query: 'relevant memory' }),
        },
      ],
      usage,
    });
    const runner = createMemoryRunner({ generate });
    const phaseInput = createPhaseInput();
    const sharedContext: ModelInputItem[] = [
      { role: 'developer', content: '現在日時: 2026年08月24日' },
      { role: 'user', content: '現在の会話' },
      {
        type: 'tool_result',
        callId: 'check_notifications',
        output: '{"success":true}',
      },
    ];

    await expect(runner.run(phaseInput, { sharedContext })).resolves.toEqual({
      value: { query: 'relevant memory' },
      usage,
    });
    expect(generate).toHaveBeenCalledWith({
      input: [
        {
          role: 'developer',
          content:
            'あなたは記憶モジュールです。次のメインターンに役立つ記憶を想起してください。',
        },
        { role: 'user', content: '現在日時: 2026年08月24日' },
        { role: 'user', content: '現在の会話' },
        {
          type: 'tool_result',
          callId: 'check_notifications',
          output: '{"success":true}',
        },
      ],
      tools: [],
      turnIndex: 1,
      responseFormat: createMemoryRecallCognitiveModuleOutputFormat(),
      maxOutputTokens: 2048,
    });
  });

  it('post_mainでは専用system promptとstore schemaを選ぶ', async () => {
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: JSON.stringify({
            content: 'remember the completed session',
            type: 'episode',
          }),
        },
      ],
      usage: createUsage(4),
    });
    const runner = createMemoryRunner({ generate });
    const phaseInput = createPhaseInput('post_main', 2);
    const sharedContext: ModelInputItem[] = [
      { role: 'developer', content: '現在日時: 2026年08月24日' },
      {
        type: 'tool_call',
        callId: 'finish-1',
        toolName: 'finish_thinking',
        input: '{"reason":"done"}',
      },
      {
        type: 'tool_result',
        callId: 'finish-1',
        output: '{"success":true}',
      },
    ];

    await runner.run(phaseInput, { sharedContext });

    expect(generate).toHaveBeenCalledWith({
      input: [
        {
          role: 'developer',
          content:
            'あなたは記憶モジュールです。完了した思考セッションから記憶を記銘してください。',
        },
        { role: 'user', content: '現在日時: 2026年08月24日' },
        {
          type: 'tool_call',
          callId: 'finish-1',
          toolName: 'finish_thinking',
          input: '{"reason":"done"}',
        },
        {
          type: 'tool_result',
          callId: 'finish-1',
          output: '{"success":true}',
        },
      ],
      tools: [],
      turnIndex: 2,
      responseFormat: createMemoryStoreCognitiveModuleOutputFormat(),
      maxOutputTokens: 2048,
    });
  });

  it('空のassistant出力を成功扱いにしない', async () => {
    const usage = createUsage(2);
    const generate = vi.fn<ModelPort['generate']>().mockResolvedValue({
      output: [{ type: 'message', role: 'assistant', content: '   ' }],
      usage,
      responseToken: 'empty-response',
    });
    const runner = createMemoryRunner({ generate });

    await expect(
      runner.run(createPhaseInput(), { sharedContext: [] })
    ).rejects.toMatchObject({
      name: 'CognitiveModuleOutputValidationError',
      message:
        'Cognitive module returned no assistant content at activation-1:1:pre_main',
      usage,
    } satisfies Partial<CognitiveModuleOutputValidationError>);
  });

  it('不正JSON、refusal、phase schema不一致を課金済みfailureにする', async () => {
    const usage = createUsage(7);
    const generate = vi
      .fn<ModelPort['generate']>()
      .mockResolvedValueOnce({
        output: [{ type: 'message', role: 'assistant', content: '{not-json' }],
        usage,
        responseToken: 'invalid-json-response',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: '<refusal>cannot comply</refusal>',
          },
        ],
        usage,
        responseToken: 'refusal-response',
      })
      .mockResolvedValueOnce({
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: JSON.stringify({
              content: 'wrong phase output',
              type: 'semantic',
            }),
          },
        ],
        usage,
        responseToken: 'schema-response',
      });
    const runner = createMemoryRunner({ generate });

    await expect(
      runner.run(createPhaseInput(), { sharedContext: [] })
    ).rejects.toMatchObject({ code: 'invalid_json', usage });
    await expect(
      runner.run(createPhaseInput(), { sharedContext: [] })
    ).rejects.toMatchObject({ code: 'refusal', usage });
    await expect(
      runner.run(createPhaseInput(), { sharedContext: [] })
    ).rejects.toMatchObject({
      code: 'schema_mismatch',
      diagnostic: {
        code: 'strict_schema',
        issues: [
          { path: 'query', code: 'invalid_type' },
          { path: '', code: 'unrecognized_keys' },
        ],
      },
      usage,
    });
  });
});
