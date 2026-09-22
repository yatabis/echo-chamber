import {
  formatCognitiveModuleHandoff,
  formatInitialCognitiveModuleContext,
} from './cognitive-module-handoff';
import {
  CognitiveModuleOutputValidationError,
  ParallelCognitiveModuleOrchestrator,
  type CognitiveModuleOrchestrator,
  type ParallelCognitiveModuleOrchestratorOptions,
  type CognitiveModulePhaseInput,
  type CognitiveModuleRunContext,
  type CognitiveModuleRunner,
  type CognitiveModuleRunResult,
} from './cognitive-module-orchestrator';
import {
  CognitiveModuleSchemaValidationError,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  createEmotionCognitiveModuleOutputFormat,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
  parseEmotionCognitiveModuleOutput,
  type MemoryCognitiveModuleOutput,
} from './cognitive-module-schema';
import {
  buildModelInputWithSystemPrompt,
  buildMemoryCognitiveModuleSystemPrompt,
  buildEmotionCognitiveModuleSystemPrompt,
} from './prompt-builder';

import type {
  ModelInputItem,
  ModelPort,
  ModelRequest,
  ModelStructuredOutputFormat,
} from '../ports/model';

const DEFAULT_MAX_OUTPUT_TOKENS = 2048;

/** Provider実装と保存先を注入する、共通のCognitive構築入力。 */
export interface ModelCognitiveModuleOrchestratorOptions extends Pick<
  ParallelCognitiveModuleOrchestratorOptions,
  'domain' | 'retryPolicy' | 'createActivationId' | 'createRequestSignal'
> {
  instanceName: string;
  memoryModel: ModelPort;
  emotionModel: ModelPort;
  maxOutputTokens?: number;
}

/** Coreのprompt・schema・handoffを使い、同じ処理を各providerへ接続する。 */
export function createModelCognitiveModuleOrchestrator(
  options: ModelCognitiveModuleOrchestratorOptions
): CognitiveModuleOrchestrator {
  const recallFormat = createMemoryRecallCognitiveModuleOutputFormat();
  const storeFormat = createMemoryStoreCognitiveModuleOutputFormat();
  const emotionFormat = createEmotionCognitiveModuleOutputFormat();
  const outputLimit =
    options.maxOutputTokens === undefined
      ? {}
      : { maxOutputTokens: options.maxOutputTokens };
  return new ParallelCognitiveModuleOrchestrator({
    domain: options.domain,
    retryPolicy: options.retryPolicy,
    createActivationId: options.createActivationId,
    ...(options.createRequestSignal === undefined
      ? {}
      : { createRequestSignal: options.createRequestSignal }),
    memory: new ModelCognitiveModuleRunner<MemoryCognitiveModuleOutput>({
      model: options.memoryModel,
      ...outputLimit,
      resolveSystemPrompt: ({ phase }) =>
        buildMemoryCognitiveModuleSystemPrompt(options.instanceName, phase),
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
    }),
    emotion: new ModelCognitiveModuleRunner({
      model: options.emotionModel,
      ...outputLimit,
      resolveSystemPrompt: () =>
        buildEmotionCognitiveModuleSystemPrompt(options.instanceName),
      resolveOutputContract: () => ({
        format: emotionFormat,
        parse: parseEmotionCognitiveModuleOutput,
      }),
    }),
    formatInitialContext: formatInitialCognitiveModuleContext,
    formatHandoff: formatCognitiveModuleHandoff,
  });
}

/** 1 phaseで使用するstructured output contract。 */
export interface ModelCognitiveModuleOutputContract<TOutput> {
  format: ModelStructuredOutputFormat;
  parse(value: unknown): TOutput;
}

/** Provider-neutralなmodule runnerの構築入力。 */
export interface ModelCognitiveModuleRunnerOptions<TOutput> {
  model: ModelPort;
  resolveSystemPrompt(input: CognitiveModulePhaseInput): string;
  resolveOutputContract(
    input: CognitiveModulePhaseInput
  ): ModelCognitiveModuleOutputContract<TOutput>;
  maxOutputTokens?: number;
}

/**
 * 専用system promptと共有contextから構造化された結果を返す。
 */
export class ModelCognitiveModuleRunner<
  TOutput,
> implements CognitiveModuleRunner<TOutput> {
  private readonly maxOutputTokens: number;

  /** @param options module model、phase別system prompt、output contract */
  constructor(
    private readonly options: ModelCognitiveModuleRunnerOptions<TOutput>
  ) {
    this.maxOutputTokens = options.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
    if (!Number.isInteger(this.maxOutputTokens) || this.maxOutputTokens < 1) {
      throw new Error('Cognitive module maxOutputTokens must be positive');
    }
  }

  /** 1 phaseを生成し、検証済みresultとusageを返す。 */
  async run(
    input: CognitiveModulePhaseInput,
    context: CognitiveModuleRunContext
  ): Promise<CognitiveModuleRunResult<TOutput>> {
    const systemPrompt = this.options.resolveSystemPrompt(input).trim();
    if (systemPrompt === '') {
      throw new Error('Cognitive module system prompt must not be empty');
    }
    const outputContract = this.options.resolveOutputContract(input);
    const response = await this.options.model.generate(
      this.createModelRequest(input, context, outputContract, systemPrompt)
    );
    const content = response.output
      .flatMap((item) => (item.type === 'message' ? [item.content.trim()] : []))
      .filter((item) => item !== '')
      .join('\n\n');

    if (content === '') {
      throw new CognitiveModuleOutputValidationError(
        `Cognitive module returned no assistant content at ${input.boundaryId}`,
        response.usage,
        { code: 'empty_output' }
      );
    }
    if (content.startsWith('<refusal>')) {
      throw new CognitiveModuleOutputValidationError(
        `Cognitive module refused structured output at ${input.boundaryId}`,
        response.usage,
        { code: 'refusal' }
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(content) as unknown;
    } catch {
      throw new CognitiveModuleOutputValidationError(
        `Cognitive module returned invalid JSON at ${input.boundaryId}`,
        response.usage,
        { code: 'invalid_json' }
      );
    }

    let value: TOutput;
    try {
      value = outputContract.parse(parsed);
    } catch (error) {
      throw new CognitiveModuleOutputValidationError(
        `Cognitive module output failed schema validation at ${input.boundaryId}`,
        response.usage,
        {
          code: 'schema_mismatch',
          ...(error instanceof CognitiveModuleSchemaValidationError
            ? {
                diagnostic: { code: error.code, issues: error.issues },
              }
            : {}),
        }
      );
    }

    return {
      value,
      usage: response.usage,
    };
  }

  /** Phaseと共有contextからprovider-neutral requestを作る。 */
  private createModelRequest(
    input: CognitiveModulePhaseInput,
    context: CognitiveModuleRunContext,
    outputContract: ModelCognitiveModuleOutputContract<TOutput>,
    systemPrompt: string
  ): ModelRequest {
    const request: ModelRequest = {
      input: this.createModelInput(context, systemPrompt),
      tools: [],
      turnIndex: input.sequence,
      responseFormat: outputContract.format,
      maxOutputTokens: this.maxOutputTokens,
    };
    if (context.signal !== undefined) {
      request.signal = context.signal;
    }
    return request;
  }

  /** 専用system promptを先頭へ1件だけ置き、共有contextを接続する。 */
  private createModelInput(
    context: CognitiveModuleRunContext,
    systemPrompt: string
  ): ModelInputItem[] {
    return buildModelInputWithSystemPrompt(systemPrompt, context.sharedContext);
  }
}
