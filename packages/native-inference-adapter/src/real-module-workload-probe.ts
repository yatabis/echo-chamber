import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';

import { NativeInferenceClient } from './native-inference-client';

import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
  NativeStateOpenedEvent,
} from './protocol';

const [binaryPath, modelDirectory] = process.argv.slice(2);
if (binaryPath === undefined || modelDirectory === undefined) {
  throw new Error(
    'usage: pnpm probe:module-workload <echo-inference-binary> <model-directory>'
  );
}

const SAMPLING = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
} as const;
const ECHO_IDS = ['echo-a', 'echo-b', 'echo-c'] as const;
const AUXILIARY_MODULES = ['memory', 'emotion'] as const;
type EchoId = (typeof ECHO_IDS)[number];
type AuxiliaryModule = (typeof AUXILIARY_MODULES)[number];
type NativeToolCallOutput = Extract<
  NativeCompletedEvent['output'][number],
  { type: 'tool_call' }
>;

interface StateLane {
  echoId: EchoId;
  module: 'main' | AuxiliaryModule;
  instanceId: string;
}

const mainLanes = ECHO_IDS.map((echoId) => lane(echoId, 'main'));
const auxiliaryLanes = ECHO_IDS.flatMap((echoId) =>
  AUXILIARY_MODULES.map((module) => lane(echoId, module))
);
const stateRoot = await mkdtemp(join(tmpdir(), 'echo-module-workload-'));
const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const client = NativeInferenceClient.spawn({
  binaryPath,
  modelDirectory,
  maxOutstandingRequests: 8,
  environment:
    nativeLibraryPath === undefined
      ? process.env
      : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath },
});

try {
  const ready = await client.ready();
  const openedMain = await Promise.all(
    mainLanes.map(
      async (stateLane, index) =>
        await client.openState({
          type: 'open_state',
          request_id: `module-probe:open-main:${index + 1}`,
          instance_id: stateLane.instanceId,
          persistence: 'durable',
          snapshot_root: join(stateRoot, stateLane.instanceId),
        })
    )
  );
  const openedAuxiliary = await Promise.all(
    auxiliaryLanes.map(
      async (stateLane, index) =>
        await client.openState({
          type: 'open_state',
          request_id: `module-probe:open-aux:${index + 1}`,
          instance_id: stateLane.instanceId,
          persistence: 'ephemeral',
        })
    )
  );

  const auxiliaryInitial = await timedCohort(
    auxiliaryLanes.map((stateLane, index) =>
      generation({
        requestId: `module-probe:aux-initial:${index + 1}`,
        stateLane,
        stateTransition: 'initial',
        input: [
          {
            role: 'system',
            content:
              '要求されたモジュール更新は利用可能なツールで実行し、summaryを改変しないでください。説明文は出力しないでください。ツールの実行結果には次の観測が含まれます。',
          },
          {
            role: 'user',
            content: toolInstruction(stateLane, initialSummary(stateLane)),
          },
        ],
        tools: toolContract(stateLane),
        maxNewTokens: 64,
        seed: 101_000 + index,
      })
    )
  );
  const auxiliaryBaselines = sequenceLengths(auxiliaryInitial.events);
  const auxiliaryPendingCalls = pendingCalls(auxiliaryInitial.events);

  const mainInitial = await timedCohort(
    mainLanes.map((stateLane, index) =>
      generation({
        requestId: `module-probe:main-initial:${index + 1}`,
        stateLane,
        stateTransition: 'initial',
        input: [
          {
            role: 'system',
            content:
              '要求された思考ステップは利用可能なツールで実行し、summaryを改変しないでください。説明文は出力しないでください。ツールの実行結果には次の観測が含まれます。',
          },
          {
            role: 'user',
            content: toolInstruction(stateLane, initialSummary(stateLane)),
          },
        ],
        tools: toolContract(stateLane),
        maxNewTokens: 64,
        seed: 102_000 + index,
      })
    )
  );
  const mainBaselines = sequenceLengths(mainInitial.events);
  const mainPendingCalls = pendingCalls(mainInitial.events);

  const cancelledLane = requireArrayItem(auxiliaryLanes, 0);
  const auxiliaryUpdateCommands = auxiliaryLanes.map((stateLane, index) =>
    generation({
      requestId: `module-probe:aux-update:${index + 1}`,
      stateLane,
      stateTransition: 'continuation',
      input: [
        {
          type: 'tool_result',
          call_id: requirePendingCall(
            auxiliaryPendingCalls,
            stateLane.instanceId
          ).call_id,
          output: `${JSON.stringify({
            stored: true,
            next_observation: {
              source: 'main_thought_delta',
              echo: stateLane.echoId,
              module: stateLane.module,
              main_state_sequence_length: mainBaselines.get(
                mainStateId(stateLane.echoId)
              ),
            },
          })}\n保存に成功し、次の観測が届きました。${toolInstruction(
            stateLane,
            updateSummary(stateLane)
          )}`,
        },
      ],
      maxNewTokens: stateLane === cancelledLane ? 384 : 64,
      seed: 103_000 + index,
    })
  );
  const cancelledCommand = requireArrayItem(auxiliaryUpdateCommands, 0);
  const updateStarted = performance.now();
  const updatePromises = auxiliaryUpdateCommands.map(
    async (command) => await client.generate(command)
  );
  await delay(900);
  await client.cancel(cancelledCommand.request_id);
  const auxiliaryUpdateSettled = await Promise.allSettled(updatePromises);
  const auxiliaryUpdateElapsedMilliseconds = performance.now() - updateStarted;
  const cancelledResult = requireArrayItem(auxiliaryUpdateSettled, 0);
  const auxiliarySurvivors = auxiliaryUpdateSettled
    .slice(1)
    .map((result) => requireFulfilled(result));

  const cancelledBaseline = requireSequenceLength(
    auxiliaryBaselines,
    cancelledLane.instanceId
  );
  const retry = await client.generate(
    generation({
      requestId: 'module-probe:aux-update:retry',
      stateLane: cancelledLane,
      stateTransition: 'continuation',
      input: cancelledCommand.input,
      maxNewTokens: 64,
      seed: 103_000,
    })
  );

  const mainContinuation = await timedCohort(
    mainLanes.map((stateLane, index) =>
      generation({
        requestId: `module-probe:main-continuation:${index + 1}`,
        stateLane,
        stateTransition: 'continuation',
        input: [
          {
            type: 'tool_result',
            call_id: requirePendingCall(mainPendingCalls, stateLane.instanceId)
              .call_id,
            output: `${JSON.stringify({
              stored: true,
              memory: 'updated',
              emotion: 'updated',
            })}\n保存に成功し、補助モジュールの結果が届きました。${toolInstruction(
              stateLane,
              continuationSummary(stateLane)
            )}`,
          },
        ],
        maxNewTokens: 64,
        seed: 104_000 + index,
      })
    )
  );

  const mainSnapshots = await Promise.all(
    mainLanes.map(
      async (stateLane, index) =>
        await client.snapshot({
          type: 'snapshot',
          request_id: `module-probe:snapshot-main:${index + 1}`,
          instance_id: stateLane.instanceId,
        })
    )
  );
  const ephemeralSnapshotError = await client
    .snapshot({
      type: 'snapshot',
      request_id: 'module-probe:snapshot-auxiliary',
      instance_id: requireArrayItem(auxiliaryLanes, 1).instanceId,
    })
    .then(
      () => undefined,
      (error: unknown) => toError(error)
    );

  const checks = {
    protocolVersion: ready.protocol_version === 9,
    sixActiveRowsAdvertised: ready.max_active_batch_size === 6,
    mainOwnersAreDurable: openedMain.every(isDurableOpen),
    auxiliaryOwnersAreEphemeral: openedAuxiliary.every(isEphemeralOpen),
    auxiliaryInitialUsedWidthSix: auxiliaryInitial.events.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 6
    ),
    auxiliaryInitialStateExact: auxiliaryInitial.events.every((event) =>
      stateAccountingIsExact(event)
    ),
    auxiliaryInitialToolCallsAreExact: auxiliaryInitial.events.every((event) =>
      hasExpectedToolCall(
        event,
        laneFromInstanceId(event.response.instance_id),
        initialSummary(laneFromInstanceId(event.response.instance_id))
      )
    ),
    mainInitialUsedWidthThree: mainInitial.events.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 3
    ),
    mainInitialStateExact: mainInitial.events.every((event) =>
      stateAccountingIsExact(event)
    ),
    mainInitialToolCallsAreExact: mainInitial.events.every((event) =>
      hasExpectedToolCall(
        event,
        laneFromInstanceId(event.response.instance_id),
        initialSummary(laneFromInstanceId(event.response.instance_id))
      )
    ),
    onlySelectedAuxiliaryWasCancelled:
      cancelledResult.status === 'rejected' &&
      toError(cancelledResult.reason).message.includes(
        'cancelled before commit'
      ) &&
      auxiliarySurvivors.length === 5,
    auxiliarySurvivorsUsedWidthSix: auxiliarySurvivors.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 6
    ),
    auxiliarySurvivorStatesExact: auxiliarySurvivors.every((event) =>
      stateAccountingIsExact(
        event,
        requireSequenceLength(auxiliaryBaselines, event.response.instance_id)
      )
    ),
    auxiliarySurvivorToolCallsAreExact: auxiliarySurvivors.every((event) =>
      hasExpectedToolCall(
        event,
        laneFromInstanceId(event.response.instance_id),
        updateSummary(laneFromInstanceId(event.response.instance_id))
      )
    ),
    cancelledAuxiliaryRetriedFromPriorCommit:
      retry.response.metrics.cached_prefix_tokens === cancelledBaseline,
    retryStateExact: stateAccountingIsExact(retry, cancelledBaseline),
    retryToolCallIsExact: hasExpectedToolCall(
      retry,
      cancelledLane,
      updateSummary(cancelledLane)
    ),
    mainStateWasNotAdvancedByAuxiliaries: mainContinuation.events.every(
      (event) =>
        event.response.metrics.cached_prefix_tokens ===
        requireSequenceLength(mainBaselines, event.response.instance_id)
    ),
    mainContinuationUsedWidthThree: mainContinuation.events.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 3
    ),
    mainContinuationStateExact: mainContinuation.events.every((event) =>
      stateAccountingIsExact(
        event,
        requireSequenceLength(mainBaselines, event.response.instance_id)
      )
    ),
    mainContinuationToolCallsAreExact: mainContinuation.events.every((event) =>
      hasExpectedToolCall(
        event,
        laneFromInstanceId(event.response.instance_id),
        continuationSummary(laneFromInstanceId(event.response.instance_id))
      )
    ),
    onlyMainLanesPublished:
      mainSnapshots.length === 3 &&
      mainSnapshots.every((snapshot) =>
        snapshot.path.endsWith('/current.safetensors')
      ) &&
      ephemeralSnapshotError?.message.includes(
        'process-local and cannot publish a snapshot'
      ) === true,
  };

  const report = {
    schemaVersion: 1,
    protocolVersion: ready.protocol_version,
    lanes: {
      main: mainLanes.map((stateLane) => stateLane.instanceId),
      auxiliary: auxiliaryLanes.map((stateLane) => stateLane.instanceId),
    },
    auxiliaryInitial: cohortSummary(auxiliaryInitial),
    mainInitial: cohortSummary(mainInitial),
    auxiliaryUpdate: {
      elapsedMilliseconds: auxiliaryUpdateElapsedMilliseconds,
      survivor: eventSummary(auxiliarySurvivors),
      cancelledInstanceId: cancelledLane.instanceId,
      cancelledBaseline,
      retry: eventSummary([retry]),
    },
    mainContinuation: cohortSummary(mainContinuation),
    snapshot: {
      mainPaths: mainSnapshots.map((snapshot) => snapshot.path),
      ephemeralRejection: ephemeralSnapshotError?.message,
    },
    checks,
  };
  console.log(JSON.stringify(report, undefined, 2));
  requirePassingChecks(checks);
} finally {
  await client.shutdown();
  await rm(stateRoot, { force: true, recursive: true });
}

function lane(echoId: EchoId, module: StateLane['module']): StateLane {
  return {
    echoId,
    module,
    instanceId: `${echoId}.${module}`,
  };
}

function mainStateId(echoId: EchoId): string {
  return `${echoId}.main`;
}

function laneFromInstanceId(instanceId: string): StateLane {
  const found = [...mainLanes, ...auxiliaryLanes].find(
    (stateLane) => stateLane.instanceId === instanceId
  );
  if (found === undefined) {
    throw new Error(`unknown module state lane ${instanceId}`);
  }
  return found;
}

function toolName(stateLane: StateLane): string {
  return `publish_${stateLane.module}_update`;
}

function initialSummary(stateLane: StateLane): string {
  return `${stateLane.echoId}-${stateLane.module}-initial`;
}

function updateSummary(stateLane: StateLane): string {
  return `${stateLane.echoId}-${stateLane.module}-after-main-delta`;
}

function continuationSummary(stateLane: StateLane): string {
  return `${stateLane.echoId}-${stateLane.module}-after-auxiliary-summary`;
}

function toolInstruction(stateLane: StateLane, summary: string): string {
  return `${toolName(stateLane)}を一度だけ呼び出し、summaryを${summary}として保存してください。説明文は不要です。`;
}

function toolContract(stateLane: StateLane): NativeGenerateCommand['tools'] {
  return [
    {
      name: toolName(stateLane),
      description:
        '指定されたsummaryをモジュール更新として保存します。保存を求められた場合は必ず一度だけ呼び出してください。',
      input_schema: {
        type: 'object',
        properties: { summary: { type: 'string' } },
        required: ['summary'],
        additionalProperties: false,
      },
      strict: true,
    },
  ];
}

function generation(input: {
  requestId: string;
  stateLane: StateLane;
  stateTransition: NativeGenerateCommand['state_transition'];
  input: NativeGenerateCommand['input'];
  tools?: NativeGenerateCommand['tools'];
  maxNewTokens: number;
  seed: number;
}): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: input.requestId,
    instance_id: input.stateLane.instanceId,
    state_transition: input.stateTransition,
    stream_tokens: false,
    input: input.input,
    tools: input.tools ?? [],
    max_new_tokens: input.maxNewTokens,
    sampling: { ...SAMPLING, seed: input.seed },
  };
}

async function timedCohort(
  commands: readonly NativeGenerateCommand[]
): Promise<{ events: NativeCompletedEvent[]; elapsedMilliseconds: number }> {
  const started = performance.now();
  const events = await Promise.all(
    commands.map(async (command) => await client.generate(command))
  );
  return {
    events,
    elapsedMilliseconds: performance.now() - started,
  };
}

function sequenceLengths(
  events: readonly NativeCompletedEvent[]
): Map<string, number> {
  return new Map(
    events.map((event) => [
      event.response.instance_id,
      event.response.state_sequence_length,
    ])
  );
}

function pendingCalls(
  events: readonly NativeCompletedEvent[]
): Map<string, NativeToolCallOutput> {
  return new Map(
    events.map((event) => {
      const stateLane = laneFromInstanceId(event.response.instance_id);
      const call = event.output.find(
        (item): item is NativeToolCallOutput => item.type === 'tool_call'
      );
      if (
        call === undefined ||
        !hasExpectedToolCall(event, stateLane, initialSummary(stateLane))
      ) {
        throw new Error(
          `initial generation for ${stateLane.instanceId} did not produce its exact pending tool call: ${JSON.stringify({ finishReason: event.response.finish_reason, generatedTokens: event.response.metrics.generated_tokens, output: event.output, warning: event.tool_parse_warning })}`
        );
      }
      return [stateLane.instanceId, call];
    })
  );
}

function requirePendingCall(
  calls: ReadonlyMap<string, NativeToolCallOutput>,
  instanceId: string
): NativeToolCallOutput {
  const call = calls.get(instanceId);
  if (call === undefined) {
    throw new Error(`missing pending tool call for ${instanceId}`);
  }
  return call;
}

function hasExpectedToolCall(
  event: NativeCompletedEvent,
  stateLane: StateLane,
  summary: string
): boolean {
  if (
    event.response.finish_reason !== 'stop_token' ||
    event.tool_parse_warning !== undefined ||
    event.output.length !== 1
  ) {
    return false;
  }
  const [item] = event.output;
  if (item?.type !== 'tool_call' || item.tool_name !== toolName(stateLane)) {
    return false;
  }
  return parseOnlySummary(item.input) === summary;
}

function parseOnlySummary(input: string): string | undefined {
  try {
    const parsed: unknown = JSON.parse(input);
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed) ||
      Object.keys(parsed).length !== 1
    ) {
      return undefined;
    }
    const summary = (parsed as Record<string, unknown>).summary;
    return typeof summary === 'string' ? summary : undefined;
  } catch {
    return undefined;
  }
}

function requireSequenceLength(
  lengths: ReadonlyMap<string, number>,
  instanceId: string
): number {
  const length = lengths.get(instanceId);
  if (length === undefined) {
    throw new Error(`missing committed sequence length for ${instanceId}`);
  }
  return length;
}

function stateAccountingIsExact(
  event: NativeCompletedEvent,
  cachedPrefix = 0
): boolean {
  const metrics = event.response.metrics;
  const hiddenClosingToken = event.response.finish_reason === 'length' ? 1 : 0;
  return (
    event.response.state_sequence_length ===
    cachedPrefix +
      metrics.input_tokens_processed +
      metrics.generated_tokens +
      hiddenClosingToken
  );
}

function cohortSummary(cohort: {
  events: NativeCompletedEvent[];
  elapsedMilliseconds: number;
}): Record<string, unknown> {
  const generatedTokens = cohort.events.reduce(
    (total, event) => total + event.response.metrics.generated_tokens,
    0
  );
  return {
    elapsedMilliseconds: cohort.elapsedMilliseconds,
    generatedTokens,
    aggregateTokensPerSecond:
      generatedTokens / (cohort.elapsedMilliseconds / 1_000),
    rows: eventSummary(cohort.events),
  };
}

function eventSummary(
  events: readonly NativeCompletedEvent[]
): Record<string, unknown>[] {
  return events.map((event) => ({
    instanceId: event.response.instance_id,
    cachedPrefixTokens: event.response.metrics.cached_prefix_tokens,
    inputTokensProcessed: event.response.metrics.input_tokens_processed,
    generatedTokens: event.response.metrics.generated_tokens,
    stateSequenceLength: event.response.state_sequence_length,
    finishReason: event.response.finish_reason,
    maximumDecodeBatchSize: event.response.metrics.maximum_decode_batch_size,
    membershipChanges: event.response.metrics.decode_batch_membership_changes,
    output: event.output,
    ...(event.tool_parse_warning === undefined
      ? {}
      : { toolParseWarning: event.tool_parse_warning }),
  }));
}

function isDurableOpen(event: NativeStateOpenedEvent): boolean {
  return (
    event.persistence === 'durable' &&
    !event.restored &&
    event.current_path?.endsWith('/current.safetensors') === true
  );
}

function isEphemeralOpen(event: NativeStateOpenedEvent): boolean {
  return (
    event.persistence === 'ephemeral' &&
    !event.restored &&
    event.current_path === undefined
  );
}

function requireFulfilled(
  result: PromiseSettledResult<NativeCompletedEvent>
): NativeCompletedEvent {
  if (result.status === 'rejected') throw toError(result.reason);
  return result.value;
}

function requirePassingChecks(checks: Record<string, boolean>): void {
  const failed = Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(`module-workload probe failed: ${failed.join(', ')}`);
  }
}

async function delay(milliseconds: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function requireArrayItem<T>(values: readonly T[], index: number): T {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`missing array item at index ${index}`);
  }
  return value;
}
