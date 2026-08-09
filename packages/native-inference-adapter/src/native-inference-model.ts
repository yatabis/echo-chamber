import { randomBytes } from 'node:crypto';

import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import type {
  ModelInputItem,
  ModelOutputItem,
  ModelPort,
  ModelRequest,
  ModelResponse,
  ModelToolContract,
  ModelUsage,
} from '@echo-chamber/core/ports/model';

import { NativeTokenListenerCompletionError } from './native-inference-client';
import {
  toModelOutputItem,
  toNativeWireInput,
  toNativeWireTool,
} from './protocol';

import type {
  NativeInferenceClient,
  NativeTokenListener,
} from './native-inference-client';
import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
  NativeOpenStateCommand,
  NativeSamplingConfig,
  NativeSnapshotPublishedEvent,
  NativeStatePersistence,
  NativeStateTransition,
} from './protocol';

const PROVIDER_NAME = 'echo.native_inference';

type NativeRequestFlow = NativeStateTransition;

/** E.C.H.O.'s current Qwen non-thinking production profile. */
export const ECHO_NATIVE_PRODUCTION_SAMPLING = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 1.5,
} as const satisfies Omit<NativeSamplingConfig, 'seed'>;

/** Supplies one reproducible, request-local MLX categorical seed. */
export type NativeSamplingSeedSource = () => number;

/** Construction options for one stateful E.C.H.O. instance. */
export interface NativeInferenceModelOptions {
  client: NativeInferenceClient;
  instanceId: string;
  maxTokens?: number;
  sampling?: Omit<NativeSamplingConfig, 'seed'>;
  seedSource?: NativeSamplingSeedSource;
  events?: EchoEventPort;
  onToken?: NativeTokenListener;
}

/** Observable adapter-owned lifecycle state for one E.C.H.O. instance. */
export interface NativeInferenceModelState {
  instanceId: string;
  stateOpened: boolean;
  persistence?: NativeStatePersistence;
  hasState: boolean;
  snapshotDirty: boolean;
  currentPath?: string;
  responseToken?: string;
  pendingToolCallIds?: string[];
  stateSequenceLength?: number;
  activeRequestId?: string;
}

/** Explicit storage authority used when one Native state lane opens. */
export type NativeStateOpenOptions =
  | { persistence: 'durable'; snapshotRoot: string }
  | { persistence: 'ephemeral' };

/**
 * Error returned after a length-limited generation was closed with EOS and
 * committed successfully.
 *
 * The opaque response token permits a caller that explicitly accepts the
 * truncated semantic result to continue the same live thinking session.
 */
export class NativeInferenceIncompleteGenerationError extends Error {
  /**
   * @param responseToken Opaque process-local continuation capability
   */
  constructor(public readonly responseToken: string) {
    super('native inference committed a length-limited generation');
    this.name = 'NativeInferenceIncompleteGenerationError';
  }

  /** Native terminal reason retained for provider-neutral diagnostics. */
  readonly finishReason = 'length' as const;
}

/**
 * ModelPort for one E.C.H.O. existence backed by a shared native owner.
 *
 * The native process owns the model-native KV/GDN state. The adapter tracks
 * only whether a live response token was issued in this process: token
 * presence requests continuation, while absence selects initial or
 * new-session from whether current state exists. Token contents are opaque and
 * intentionally neither decoded nor compared. Continuation input must still
 * answer the exact pending tool calls emitted by the preceding completion.
 */
export class NativeInferenceModel implements ModelPort {
  private readonly maxTokens: number | undefined;
  private readonly sampling: Omit<NativeSamplingConfig, 'seed'>;
  private readonly seedSource: () => number;
  private readonly events: EchoEventPort | undefined;
  private readonly onToken: NativeTokenListener | undefined;
  private readonly client: NativeInferenceClient;
  private readonly instanceId: string;
  private resolvedMaxTokens: number | undefined;
  private stateOpened = false;
  private persistence: NativeStatePersistence | undefined;
  private hasState = false;
  private snapshotDirty = false;
  private currentPath: string | undefined;
  private stateSequenceLength: number | undefined;
  private responseToken: string | undefined;
  private pendingToolCallIds: string[] = [];
  private toolFingerprint: string | undefined;
  private engineId: number | undefined;
  private activeRequestId: string | undefined;
  private stopping = false;
  private requestSequence = 0;

  /**
   * @param options Shared native client, stable instance identity, and
   * production generation controls
   */
  constructor(options: NativeInferenceModelOptions) {
    this.client = options.client;
    this.instanceId = requireNonEmpty(options.instanceId, 'instanceId');
    this.maxTokens = options.maxTokens;
    if (
      this.maxTokens !== undefined &&
      (!Number.isSafeInteger(this.maxTokens) || this.maxTokens < 1)
    ) {
      throw new Error('maxTokens must be a positive safe integer');
    }
    this.sampling = options.sampling ?? ECHO_NATIVE_PRODUCTION_SAMPLING;
    this.seedSource = options.seedSource ?? createSamplingSeed;
    this.events = options.events;
    this.onToken = options.onToken;
  }

  /**
   * Registers this state lane and optionally restores durable
   * current.safetensors.
   */
  async openState(
    options: NativeStateOpenOptions
  ): Promise<NativeInferenceModelState> {
    if (this.stateOpened) {
      throw new Error('native instance state is already open');
    }
    const ready = await this.client.ready();
    const resolvedMaxTokens =
      this.maxTokens ?? ready.max_new_tokens_per_request;
    if (resolvedMaxTokens > ready.max_new_tokens_per_request) {
      throw new Error(
        `maxTokens ${resolvedMaxTokens} exceeds resident owner limit ${ready.max_new_tokens_per_request}`
      );
    }
    const requestId = this.beginLifecycleRequest();
    try {
      const command: NativeOpenStateCommand =
        options.persistence === 'durable'
          ? {
              type: 'open_state',
              request_id: requestId,
              instance_id: this.instanceId,
              persistence: 'durable',
              snapshot_root: requireNonEmpty(
                options.snapshotRoot,
                'snapshotRoot'
              ),
            }
          : {
              type: 'open_state',
              request_id: requestId,
              instance_id: this.instanceId,
              persistence: 'ephemeral',
            };
      const event = await this.client.openState(command);
      if (event.instance_id !== this.instanceId) {
        throw new Error(
          `native opened instance ${event.instance_id} does not match ${this.instanceId}`
        );
      }
      if (event.persistence !== options.persistence) {
        throw new Error(
          `native opened ${event.persistence} state, expected ${options.persistence}`
        );
      }
      this.stateOpened = true;
      this.resolvedMaxTokens = resolvedMaxTokens;
      this.persistence = event.persistence;
      this.hasState = event.restored;
      this.snapshotDirty = false;
      this.currentPath = event.current_path;
      // A restored state intentionally does not recreate a live continuation
      // capability. The next request therefore starts a new thinking session.
      this.responseToken = undefined;
      this.pendingToolCallIds = [];
      this.toolFingerprint = undefined;
      this.stateSequenceLength = undefined;
    } finally {
      this.activeRequestId = undefined;
    }
    return this.state();
  }

  /** Executes one native generation and accepts state only on completion. */
  async generate(request: ModelRequest): Promise<ModelResponse> {
    if (!this.stateOpened) {
      throw new Error('native instance state must be opened before generation');
    }
    if (this.stopping) {
      throw new Error(`native instance ${this.instanceId} is stopping`);
    }
    if (this.activeRequestId !== undefined) {
      throw new Error(
        `native instance ${this.instanceId} already has an active generation`
      );
    }
    const flow = this.classifyRequestFlow(request.previousResponseToken);
    return await this.executeRequest(request, flow);
  }

  private async executeRequest(
    request: ModelRequest,
    flow: NativeRequestFlow
  ): Promise<ModelResponse> {
    if (!this.stateOpened) {
      throw new Error('native instance state must be opened before generation');
    }
    if (this.activeRequestId !== undefined) {
      throw new Error(
        `native instance ${this.instanceId} already has an active generation`
      );
    }
    const requestId = this.nextRequestId();
    this.activeRequestId = requestId;
    try {
      const prepared = this.prepareCommand(
        request,
        requestId,
        flow,
        this.requireResolvedMaxTokens()
      );
      try {
        const event = await this.client.generate(
          prepared.command,
          this.onToken
        );
        return await this.acceptCompleted(
          request,
          prepared.toolFingerprint,
          event,
          flow
        );
      } catch (error) {
        if (!(error instanceof NativeTokenListenerCompletionError)) {
          throw error;
        }
        try {
          await this.acceptCompleted(
            request,
            prepared.toolFingerprint,
            error.completedEvent,
            flow
          );
        } catch (completionError) {
          throw new AggregateError(
            [completionError, error.listenerError],
            'native generation committed after a token listener error, but completion handling also failed'
          );
        }
        throw error.listenerError;
      }
    } finally {
      this.activeRequestId = undefined;
    }
  }

  /** Cancels this instance's active request, if one exists. */
  async cancelActive(): Promise<boolean> {
    if (this.activeRequestId === undefined) {
      return false;
    }
    await this.client.cancel(this.activeRequestId);
    return true;
  }

  /** Permanently rejects generation requests admitted after owner shutdown begins. */
  stopAcceptingGeneration(): void {
    this.stopping = true;
  }

  /** Atomically replaces the opened instance's current.safetensors. */
  async snapshot(): Promise<NativeSnapshotPublishedEvent> {
    if (!this.stateOpened || !this.hasState) {
      throw new Error('native instance has no current state to snapshot');
    }
    if (this.persistence !== 'durable') {
      throw new Error('native ephemeral state cannot be snapshotted');
    }
    const requestId = this.beginLifecycleRequest();
    try {
      const event = await this.client.snapshot({
        type: 'snapshot',
        request_id: requestId,
        instance_id: this.instanceId,
      });
      if (
        event.instance_id !== this.instanceId ||
        (this.currentPath !== undefined && event.path !== this.currentPath)
      ) {
        throw new Error(
          'native snapshot acknowledgement differs from the opened state owner'
        );
      }
      this.currentPath = event.path;
      this.snapshotDirty = false;
      return event;
    } finally {
      this.activeRequestId = undefined;
    }
  }

  /** Whether a committed in-memory state has not yet been published. */
  needsSnapshot(): boolean {
    return this.snapshotDirty;
  }

  /** Returns a copy of the adapter-owned lifecycle state. */
  state(): NativeInferenceModelState {
    return {
      instanceId: this.instanceId,
      stateOpened: this.stateOpened,
      ...(this.persistence === undefined
        ? {}
        : { persistence: this.persistence }),
      hasState: this.hasState,
      snapshotDirty: this.snapshotDirty,
      ...(this.currentPath === undefined
        ? {}
        : { currentPath: this.currentPath }),
      ...(this.responseToken === undefined
        ? {}
        : { responseToken: this.responseToken }),
      ...(this.pendingToolCallIds.length === 0
        ? {}
        : { pendingToolCallIds: [...this.pendingToolCallIds] }),
      ...(this.stateSequenceLength === undefined
        ? {}
        : { stateSequenceLength: this.stateSequenceLength }),
      ...(this.activeRequestId === undefined
        ? {}
        : { activeRequestId: this.activeRequestId }),
    };
  }

  private prepareCommand(
    request: ModelRequest,
    requestId: string,
    flow: NativeRequestFlow,
    maxTokens: number
  ): { command: NativeGenerateCommand; toolFingerprint: string } {
    if (flow === 'continuation') {
      validateContinuationInput(request.input, this.pendingToolCallIds);
    }
    const wireTools = request.tools.map(toNativeWireTool);
    const toolFingerprint = fingerprintTools(request.tools, wireTools);
    if (
      flow === 'continuation' &&
      this.toolFingerprint !== undefined &&
      toolFingerprint !== this.toolFingerprint
    ) {
      throw new Error(
        'native tool catalog cannot change inside one live E.C.H.O. thinking session'
      );
    }
    const seed = this.seedSource();
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new Error('seedSource must return a non-negative safe integer');
    }
    return {
      command: {
        type: 'generate',
        request_id: requestId,
        instance_id: this.instanceId,
        state_transition: flow,
        stream_tokens: this.onToken !== undefined,
        input: request.input.map(toNativeWireInput),
        tools: flow === 'continuation' ? [] : wireTools,
        max_new_tokens: maxTokens,
        sampling: { ...this.sampling, seed },
      },
      toolFingerprint,
    };
  }

  private requireResolvedMaxTokens(): number {
    if (this.resolvedMaxTokens === undefined) {
      throw new Error('native generation limit was not resolved at state open');
    }
    return this.resolvedMaxTokens;
  }

  private async acceptCompleted(
    request: ModelRequest,
    toolFingerprint: string,
    event: NativeCompletedEvent,
    flow: NativeRequestFlow
  ): Promise<ModelResponse> {
    this.validateCompleted(event);
    const output = event.output.map(toModelOutputItem);
    const usage = toModelUsage(event);
    const responseToken = createResponseToken(event);

    this.hasState = true;
    this.snapshotDirty = this.persistence === 'durable';
    this.stateSequenceLength = event.response.state_sequence_length;
    this.responseToken = responseToken;
    this.pendingToolCallIds = output
      .filter((item) => item.type === 'tool_call')
      .map((item) => item.callId);
    this.toolFingerprint = toolFingerprint;
    this.engineId = event.response.engine_id;

    await this.emitExchange({ request, event, output, usage, flow });
    if (event.tool_parse_warning !== undefined) {
      await this.emitToolParseWarning(request, event.tool_parse_warning);
    }
    await this.emitOutput(request, output);

    if (event.response.finish_reason === 'length') {
      throw new NativeInferenceIncompleteGenerationError(responseToken);
    }
    return { output, usage, responseToken };
  }

  private validateCompleted(event: NativeCompletedEvent): void {
    if (event.response.instance_id !== this.instanceId) {
      throw new Error(
        `native response instance ${event.response.instance_id} does not match ${this.instanceId}`
      );
    }
    if (
      this.engineId !== undefined &&
      event.response.engine_id !== this.engineId
    ) {
      throw new Error('native response came from a different resident owner');
    }
  }

  private classifyRequestFlow(token: string | undefined): NativeRequestFlow {
    if (token !== undefined) {
      if (!this.hasState || this.responseToken === undefined) {
        throw new Error(
          'native continuation requires a response token issued by this live process'
        );
      }
      return 'continuation';
    }
    return this.hasState ? 'new_session' : 'initial';
  }

  private nextRequestId(): string {
    this.requestSequence += 1;
    const safeInstance = this.instanceId
      .replaceAll(/[^A-Za-z0-9_.-]/g, '_')
      .slice(0, 80);
    return `${safeInstance}:${this.requestSequence}`;
  }

  private beginLifecycleRequest(): string {
    if (this.activeRequestId !== undefined) {
      throw new Error(
        `native instance ${this.instanceId} already has an active request`
      );
    }
    const requestId = this.nextRequestId();
    this.activeRequestId = requestId;
    return requestId;
  }

  private async emitExchange(input: {
    request: ModelRequest;
    event: NativeCompletedEvent;
    output: readonly ModelOutputItem[];
    usage: ModelUsage;
    flow: NativeRequestFlow;
  }): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'model.exchange.recorded',
      severity: 'debug',
      summary: `model exchange recorded: native engine ${input.event.response.engine_id}`,
      payload: {
        provider: PROVIDER_NAME,
        instanceId: this.instanceId,
        turnIndex: input.request.turnIndex,
        stateTransition: input.flow,
        stateSequenceLength: input.event.response.state_sequence_length,
        inputItemCount: input.request.input.length,
        outputItemCount: input.output.length,
        finishReason: input.event.response.finish_reason,
        usage: input.usage,
        metrics: input.event.response.metrics,
      },
    });
  }

  private async emitToolParseWarning(
    request: ModelRequest,
    warning: string
  ): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'model.provider.warning',
      severity: 'warn',
      summary: 'native Qwen tool output remained unparsed',
      payload: {
        provider: PROVIDER_NAME,
        instanceId: this.instanceId,
        turnIndex: request.turnIndex,
        code: 'tool_parse_warning',
        warning,
      },
    });
  }

  private async emitOutput(
    request: ModelRequest,
    output: readonly ModelOutputItem[]
  ): Promise<void> {
    const content = output
      .filter(
        (item): item is Extract<ModelOutputItem, { type: 'message' }> =>
          item.type === 'message'
      )
      .map((item) => `*thinking: ${item.content}*`)
      .join('\n\n')
      .trim();
    if (content === '') {
      return;
    }
    await emitEchoEvent(this.events, {
      type: 'model.output.emitted',
      severity: 'info',
      summary: 'model output emitted',
      payload: {
        provider: PROVIDER_NAME,
        instanceId: this.instanceId,
        turnIndex: request.turnIndex,
        content,
      },
    });
  }
}

function validateContinuationInput(
  input: readonly ModelInputItem[],
  pendingToolCallIds: readonly string[]
): void {
  if (pendingToolCallIds.length === 0) {
    if (input.length === 0) {
      return;
    }
    throw new Error(
      'native continuation requires a pending tool call from the preceding completion'
    );
  }
  const resultCallIds = input.map((item) => {
    if (!('type' in item) || item.type !== 'tool_result') {
      throw new Error(
        'native continuation accepts only results for the pending tool calls'
      );
    }
    return item.callId;
  });
  if (
    resultCallIds.length !== pendingToolCallIds.length ||
    resultCallIds.some((callId, index) => callId !== pendingToolCallIds[index])
  ) {
    throw new Error(
      `native continuation tool results do not match pending calls: expected ${JSON.stringify(pendingToolCallIds)}, observed ${JSON.stringify(resultCallIds)}`
    );
  }
}

function toModelUsage(event: NativeCompletedEvent): ModelUsage {
  const metrics = event.response.metrics;
  const totalInputTokens =
    metrics.cached_prefix_tokens + metrics.input_tokens_processed;
  const outputTokens = metrics.generated_tokens;
  return {
    cachedInputTokens: metrics.cached_prefix_tokens,
    cacheWriteInputTokens: 0,
    uncachedInputTokens: metrics.input_tokens_processed,
    totalInputTokens,
    outputTokens,
    reasoningTokens: 0,
    totalTokens: totalInputTokens + outputTokens,
  };
}

function createResponseToken(event: NativeCompletedEvent): string {
  return `echo-native-v3:${Buffer.from(
    JSON.stringify({
      instanceId: event.response.instance_id,
      engineId: event.response.engine_id,
      requestId: event.request_id,
    })
  ).toString('base64url')}`;
}

function fingerprintTools(
  source: readonly ModelToolContract[],
  wire: readonly unknown[]
): string {
  try {
    return JSON.stringify(wire);
  } catch (error) {
    throw new Error(
      `native tool catalog is not JSON-serializable (${source.length} tools): ${String(error)}`
    );
  }
}

function createSamplingSeed(): number {
  return randomBytes(6).readUIntBE(0, 6);
}

function requireNonEmpty(value: string, label: string): string {
  if (value.length === 0) {
    throw new Error(`${label} must not be empty`);
  }
  return value;
}
