import type {
  ModelInputItem,
  ModelOutputItem,
  ModelToolContract,
} from '@echo-chamber/core/ports/model';

/** Exact native wire contract required by this adapter. */
export const NATIVE_INFERENCE_PROTOCOL_VERSION = 10;

/** Sampling controls admitted by the specialized native engine. */
export interface NativeSamplingConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
  seed: number;
}

/** Relation between one request and the instance's single current state. */
export type NativeStateTransition = 'initial' | 'continuation' | 'new_session';

/** Whether one Native state lane is checkpointed or process-local. */
export type NativeStatePersistence = 'durable' | 'ephemeral';

interface NativeWireTextPart {
  type: 'text';
  text: string;
}

interface NativeWireImagePart {
  type: 'image';
  image_url: string;
  detail?: string;
}

type NativeWireContentPart = NativeWireTextPart | NativeWireImagePart;

type NativeWireInputItem =
  | {
      role: 'system' | 'developer' | 'user' | 'assistant';
      content: string | NativeWireContentPart[];
    }
  | {
      type: 'tool_call';
      call_id: string;
      tool_name: string;
      input: string;
    }
  | {
      type: 'tool_result';
      call_id: string;
      output: string;
    };

interface NativeWireToolContract {
  name: string;
  description: string;
  input_schema: unknown;
  output_schema?: unknown;
  strict: boolean;
}

/** One request sent to the native NDJSON owner. */
export interface NativeGenerateCommand {
  type: 'generate';
  request_id: string;
  instance_id: string;
  state_transition: NativeStateTransition;
  /** Whether this request pays for provisional per-token diagnostic events. */
  stream_tokens: boolean;
  input: NativeWireInputItem[];
  tools: NativeWireToolContract[];
  max_new_tokens: number;
  sampling: NativeSamplingConfig;
}

/** Registers one independently mutable Native state lane. */
export type NativeOpenStateCommand =
  | {
      type: 'open_state';
      request_id: string;
      instance_id: string;
      persistence: 'durable';
      snapshot_root: string;
    }
  | {
      type: 'open_state';
      request_id: string;
      instance_id: string;
      persistence: 'ephemeral';
      snapshot_root?: never;
    };

/** Atomically replaces the opened instance's current state payload. */
export interface NativeSnapshotCommand {
  type: 'snapshot';
  request_id: string;
  instance_id: string;
}

/** Commands owned by the typed TypeScript process/lifecycle adapter. */
export type NativeWireCommand =
  | NativeGenerateCommand
  | NativeOpenStateCommand
  | NativeSnapshotCommand
  | { type: 'cancel'; request_id: string }
  | { type: 'shutdown' };

/** Request-level native performance metrics. */
export interface NativeRuntimeMetrics {
  queue_wait_nanos: number;
  cached_prefix_tokens: number;
  input_tokens_processed: number;
  generated_tokens: number;
  maximum_decode_batch_size: number;
  decode_batch_membership_changes: number;
  model_step_count: number;
  input_model_execution_count: number;
  input_execution_nanos: number;
  input_graph_construction_nanos: number;
  input_materialization_nanos: number;
  /** First sampled-token latency, or null when no token was generated. */
  first_generated_token_nanos: number | null;
  decode_execution_nanos: number;
  decode_graph_construction_nanos: number;
  decode_schedule_nanos: number;
  decode_token_wait_nanos: number;
  decode_finalization_nanos: number;
  model_execution_nanos: number;
  request_nanos: number;
  committed_state_logical_nbytes: number;
  /** Post-commit allocator counters, or null when MLX could not observe them. */
  metal_memory: NativeMetalMemoryStats | null;
}

/** Process-wide MLX Metal allocator observations at one native boundary. */
export interface NativeMetalMemoryStats {
  active_nbytes: number;
  cache_nbytes: number;
  peak_nbytes: number;
}

/** Requires allocator counters for a diagnostic that cannot run without them. */
export function requireNativeMetalMemory(
  metrics: NativeRuntimeMetrics,
  label: string
): NativeMetalMemoryStats {
  if (metrics.metal_memory === null) {
    throw new Error(`${label} has no post-commit Metal memory observation`);
  }
  return metrics.metal_memory;
}

/** State-advancing native response included in a completed event. */
export interface NativeInferenceResponse {
  engine_id: number;
  instance_id: string;
  model: Record<string, unknown>;
  state_sequence_length: number;
  generated_tokens: number[];
  finish_reason: 'length' | 'stop_token';
  metrics: NativeRuntimeMetrics;
}

type NativeWireOutputItem =
  | {
      type: 'message';
      role: 'assistant';
      content: string;
    }
  | {
      type: 'tool_call';
      call_id: string;
      tool_name: string;
      input: string;
    };

/** Terminal successful generation event. */
export interface NativeCompletedEvent {
  event: 'completed';
  request_id: string;
  response: NativeInferenceResponse;
  text: string;
  output: NativeWireOutputItem[];
  tool_parse_warning?: string;
}

/** Successful acquisition of one instance's durable state directory. */
export interface NativeStateOpenedEvent {
  event: 'state_opened';
  request_id: string;
  instance_id: string;
  persistence: NativeStatePersistence;
  restored: boolean;
  current_path?: string;
}

/** Successful atomic replacement of current.safetensors. */
export interface NativeSnapshotPublishedEvent {
  event: 'snapshot_published';
  request_id: string;
  instance_id: string;
  path: string;
  physical_nbytes: number;
}

/** Native owner event surface consumed by the adapter. */
export type NativeWireEvent =
  | {
      event: 'ready';
      protocol_version: number;
      engine: Record<string, unknown>;
      eos_token_id: number;
      chat_template_sha256: string;
      max_new_tokens_per_request: number;
      max_outstanding_requests: number;
      max_active_batch_size: number;
      max_late_join_batch_size: number;
    }
  | {
      event: 'queued';
      request_id: string;
      outstanding_requests: number;
    }
  | {
      event: 'started';
      request_id: string;
      prompt_tokens: number;
    }
  | {
      event: 'token';
      request_id: string;
      index: number;
      token_id: number;
      text?: string;
      terminal: boolean;
    }
  | NativeCompletedEvent
  | {
      event: 'cancel_acknowledged';
      request_id: string;
      accepted: boolean;
    }
  | {
      event: 'cancelled';
      request_id: string;
    }
  | NativeStateOpenedEvent
  | NativeSnapshotPublishedEvent
  | {
      event: 'failed';
      request_id?: string;
      phase: string;
      error: string;
    }
  | { event: 'shutdown' };

const EVENT_NAMES = new Set<NativeWireEvent['event']>([
  'ready',
  'queued',
  'started',
  'token',
  'completed',
  'cancel_acknowledged',
  'cancelled',
  'state_opened',
  'snapshot_published',
  'failed',
  'shutdown',
]);

/** Parses one trusted-local, but still fail-closed, native event line. */
export function parseNativeWireEvent(line: string): NativeWireEvent {
  const parsed: unknown = JSON.parse(line);
  if (!isRecord(parsed) || typeof parsed.event !== 'string') {
    throw new Error('native inference event must be a JSON object with event');
  }
  if (!EVENT_NAMES.has(parsed.event as NativeWireEvent['event'])) {
    throw new Error(`unsupported native inference event: ${parsed.event}`);
  }
  const event = parsed as { event: NativeWireEvent['event'] } & Record<
    string,
    unknown
  >;
  validateEventEnvelope(event);
  return event as NativeWireEvent;
}

/** Converts a provider-neutral request item to the native snake-case wire. */
export function toNativeWireInput(item: ModelInputItem): NativeWireInputItem {
  if ('role' in item) {
    return {
      role: item.role,
      content:
        typeof item.content === 'string'
          ? item.content
          : item.content.map((part) =>
              part.type === 'text'
                ? { type: 'text', text: part.text }
                : {
                    type: 'image',
                    image_url: part.imageUrl,
                    ...(part.detail === undefined
                      ? {}
                      : { detail: part.detail }),
                  }
            ),
    };
  }
  if (item.type === 'tool_call') {
    return {
      type: 'tool_call',
      call_id: item.callId,
      tool_name: item.toolName,
      input: item.input,
    };
  }
  return {
    type: 'tool_result',
    call_id: item.callId,
    output: item.output,
  };
}

/** Converts one provider-neutral tool contract to the native wire. */
export function toNativeWireTool(
  tool: ModelToolContract
): NativeWireToolContract {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
    ...(tool.outputSchema === undefined
      ? {}
      : { output_schema: tool.outputSchema }),
    strict: tool.strict ?? false,
  };
}

/** Converts parsed native output to the core model-port shape. */
export function toModelOutputItem(item: NativeWireOutputItem): ModelOutputItem {
  if (item.type === 'message') {
    return item;
  }
  return {
    type: 'tool_call',
    callId: item.call_id,
    toolName: item.tool_name,
    input: item.input,
  };
}

function validateEventEnvelope(
  event: { event: NativeWireEvent['event'] } & Record<string, unknown>
): void {
  if (event.event === 'ready') {
    validateReadyEvent(event);
    return;
  }
  if (event.event === 'shutdown') {
    return;
  }
  if (event.event === 'failed') {
    validateFailedEvent(event);
    return;
  }
  requireString(event, 'request_id');
  validateRequestEvent(event);
}

function validateRequestEvent(event: Record<string, unknown>): void {
  switch (event.event) {
    case 'queued':
      requireNonnegativeSafeInteger(event, 'outstanding_requests');
      return;
    case 'started':
      requireNonnegativeSafeInteger(event, 'prompt_tokens');
      return;
    case 'token':
      validateTokenEvent(event);
      return;
    case 'completed':
      validateCompletedEvent(event);
      return;
    case 'cancel_acknowledged':
      requireBoolean(event, 'accepted');
      return;
    case 'cancelled':
      return;
    case 'state_opened':
      validateStateOpenedEvent(event);
      return;
    case 'snapshot_published':
      validateSnapshotPublishedEvent(event);
      return;
    default:
      throw new Error(
        `unsupported native inference request event: ${String(event.event)}`
      );
  }
}

function validateReadyEvent(event: Record<string, unknown>): void {
  requireNonnegativeSafeInteger(event, 'protocol_version');
  requireNonnegativeSafeInteger(event, 'eos_token_id');
  requireString(event, 'chat_template_sha256');
  if (!isRecord(event.engine)) {
    throw new Error('native ready event field engine must be an object');
  }
  requirePositiveSafeInteger(event, 'max_new_tokens_per_request');
  const maxOutstandingRequests = requirePositiveSafeInteger(
    event,
    'max_outstanding_requests'
  );
  const maxActiveBatchSize = requirePositiveSafeInteger(
    event,
    'max_active_batch_size'
  );
  const maxLateJoinBatchSize = requirePositiveSafeInteger(
    event,
    'max_late_join_batch_size'
  );
  if (maxActiveBatchSize > maxOutstandingRequests) {
    throw new Error(
      'native ready event max_active_batch_size exceeds max_outstanding_requests'
    );
  }
  if (maxLateJoinBatchSize > maxActiveBatchSize) {
    throw new Error(
      'native ready event max_late_join_batch_size exceeds max_active_batch_size'
    );
  }
}

function validateFailedEvent(event: Record<string, unknown>): void {
  if (event.request_id !== undefined) {
    requireString(event, 'request_id');
  }
  requireString(event, 'phase');
  requireString(event, 'error');
}

function validateTokenEvent(event: Record<string, unknown>): void {
  requireNonnegativeSafeInteger(event, 'index');
  requireNonnegativeSafeInteger(event, 'token_id');
  if (event.text !== undefined) {
    requireString(event, 'text');
  }
  requireBoolean(event, 'terminal');
}

function validateCompletedEvent(event: Record<string, unknown>): void {
  if (
    !isRecord(event.response) ||
    !isRecord(event.response.metrics) ||
    !Array.isArray(event.output)
  ) {
    throw new Error('native completed event is missing response/output');
  }
  requireNonnegativeSafeInteger(event.response, 'engine_id');
  requireString(event.response, 'instance_id');
  if (!isRecord(event.response.model)) {
    throw new Error(
      'native completed event field response.model must be an object'
    );
  }
  requireNonnegativeSafeInteger(event.response, 'state_sequence_length');
  requireNonnegativeSafeIntegerArray(event.response, 'generated_tokens');
  const finishReason = requireString(event.response, 'finish_reason');
  if (finishReason !== 'length' && finishReason !== 'stop_token') {
    throw new Error(
      'native completed event field response.finish_reason must be length or stop_token'
    );
  }
  validateRuntimeMetrics(event.response.metrics);
  requireString(event, 'text');
  event.output.forEach(validateOutputItem);
  if (event.tool_parse_warning !== undefined) {
    requireString(event, 'tool_parse_warning');
  }
}

function validateRuntimeMetrics(metrics: Record<string, unknown>): void {
  for (const key of NONNEGATIVE_RUNTIME_METRIC_FIELDS) {
    requireNonnegativeSafeInteger(metrics, key);
  }
  requirePositiveSafeInteger(metrics, 'maximum_decode_batch_size');
  requireNonnegativeSafeInteger(metrics, 'decode_batch_membership_changes');
  const firstGeneratedTokenNanos = metrics.first_generated_token_nanos;
  if (firstGeneratedTokenNanos !== null) {
    requireNonnegativeSafeInteger(metrics, 'first_generated_token_nanos');
  }
  const metalMemory = metrics.metal_memory;
  if (metalMemory !== null && !isRecord(metalMemory)) {
    throw new Error(
      'native completed event field metrics.metal_memory must be an object or null'
    );
  }
  if (metalMemory !== null) {
    requireNonnegativeSafeInteger(metalMemory, 'active_nbytes');
    requireNonnegativeSafeInteger(metalMemory, 'cache_nbytes');
    requireNonnegativeSafeInteger(metalMemory, 'peak_nbytes');
  }
}

const NONNEGATIVE_RUNTIME_METRIC_FIELDS = [
  'queue_wait_nanos',
  'cached_prefix_tokens',
  'input_tokens_processed',
  'generated_tokens',
  'model_step_count',
  'input_model_execution_count',
  'input_execution_nanos',
  'input_graph_construction_nanos',
  'input_materialization_nanos',
  'decode_execution_nanos',
  'decode_graph_construction_nanos',
  'decode_schedule_nanos',
  'decode_token_wait_nanos',
  'decode_finalization_nanos',
  'model_execution_nanos',
  'request_nanos',
  'committed_state_logical_nbytes',
] as const satisfies readonly (keyof NativeRuntimeMetrics)[];

function validateOutputItem(item: unknown, index: number): void {
  if (!isRecord(item) || typeof item.type !== 'string') {
    throw new Error(
      `native completed event field output[${index}] must be an output object`
    );
  }
  if (item.type === 'message') {
    if (item.role !== 'assistant' || typeof item.content !== 'string') {
      throw new Error(
        `native completed event field output[${index}] must be an assistant message`
      );
    }
    return;
  }
  if (
    item.type === 'tool_call' &&
    typeof item.call_id === 'string' &&
    typeof item.tool_name === 'string' &&
    typeof item.input === 'string'
  ) {
    return;
  }
  throw new Error(
    `native completed event field output[${index}] must be a complete tool call or assistant message`
  );
}

function validateStateOpenedEvent(event: Record<string, unknown>): void {
  requireString(event, 'instance_id');
  const persistence = requireString(event, 'persistence');
  if (persistence !== 'durable' && persistence !== 'ephemeral') {
    throw new Error(
      'native state_opened event field persistence must be durable or ephemeral'
    );
  }
  if (typeof event.restored !== 'boolean') {
    throw new Error('native state_opened event field restored must be boolean');
  }
  if (persistence === 'durable') {
    requireString(event, 'current_path');
  } else if (event.restored || event.current_path !== undefined) {
    throw new Error(
      'native ephemeral state_opened event cannot restore or publish current_path'
    );
  }
}

function validateSnapshotPublishedEvent(event: Record<string, unknown>): void {
  requireString(event, 'instance_id');
  requireString(event, 'path');
  requireNonnegativeSafeInteger(event, 'physical_nbytes');
}

function requireString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string') {
    throw new Error(`native inference event field ${key} must be a string`);
  }
  return value;
}

function requireBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (typeof value !== 'boolean') {
    throw new Error(`native inference event field ${key} must be a boolean`);
  }
  return value;
}

function requireNonnegativeSafeIntegerArray(
  record: Record<string, unknown>,
  key: string
): void {
  const value = record[key];
  if (
    !Array.isArray(value) ||
    value.some(
      (item) =>
        typeof item !== 'number' || !Number.isSafeInteger(item) || item < 0
    )
  ) {
    throw new Error(
      `native inference event field ${key} must be an array of nonnegative safe integers`
    );
  }
}

function requireNonnegativeSafeInteger(
  record: Record<string, unknown>,
  key: string
): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(
      `native inference event field ${key} must be a nonnegative safe integer`
    );
  }
  return value;
}

function requirePositiveSafeInteger(
  record: Record<string, unknown>,
  key: string
): number {
  const value = requireNonnegativeSafeInteger(record, key);
  if (value === 0) {
    throw new Error(
      `native inference event field ${key} must be a positive safe integer`
    );
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
