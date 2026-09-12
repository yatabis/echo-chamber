import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';

import { NativeInferenceClient } from './native-inference-client';

import type { NativeCompletedEvent, NativeGenerateCommand } from './protocol';

const [binaryPath, modelDirectory] = process.argv.slice(2);
if (binaryPath === undefined || modelDirectory === undefined) {
  throw new Error(
    'usage: pnpm probe:continuous-batch <echo-inference-binary> <model-directory>'
  );
}

const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
} as const;
const INSTANCE_IDS = Array.from(
  { length: 6 },
  (_, index) => `continuous-probe-${index + 1}`
);
const LATE_PRIMARY_INSTANCE_ID = requireArrayItem(INSTANCE_IDS, 0);
const LATE_SECONDARY_INSTANCE_ID = requireArrayItem(INSTANCE_IDS, 1);
const CANCEL_SURVIVOR_INSTANCE_ID = requireArrayItem(INSTANCE_IDS, 2);
const CANCELLED_INSTANCE_ID = requireArrayItem(INSTANCE_IDS, 3);
const stateRoot = await mkdtemp(join(tmpdir(), 'echo-continuous-batch-'));
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
  await Promise.all(
    INSTANCE_IDS.map(async (instanceId, index) => {
      await client.openState({
        type: 'open_state',
        request_id: `probe:open:${index + 1}`,
        instance_id: instanceId,
        persistence: 'durable',
        snapshot_root: join(stateRoot, instanceId),
      });
    })
  );

  const widthSixStarted = performance.now();
  const widthSix = await Promise.all(
    INSTANCE_IDS.map(
      async (instanceId, index) =>
        await client.generate(
          generation({
            requestId: `probe:width6:${index + 1}`,
            instanceId,
            stateTransition: 'initial',
            maxNewTokens: 64 + index * 8,
            seed: 91_000 + index,
            prompt:
              'Return the positive integers in order, separated by commas, with no explanation. Continue until the output limit.',
          })
        )
    )
  );
  const widthSixElapsedMilliseconds = performance.now() - widthSixStarted;
  const widthSixBaselines = new Map(
    widthSix.map((event) => [
      event.response.instance_id,
      event.response.state_sequence_length,
    ])
  );

  const latePrimaryPromise = client.generate(
    generation({
      requestId: 'probe:late:primary',
      instanceId: LATE_PRIMARY_INSTANCE_ID,
      stateTransition: 'new_session',
      maxNewTokens: 256,
      seed: 92_001,
      prompt:
        'Return the positive integers in order, separated by commas, with no explanation. Continue until the output limit.',
    })
  );
  await delay(900);
  const lateSecondaryPromise = client.generate(
    generation({
      requestId: 'probe:late:secondary',
      instanceId: LATE_SECONDARY_INSTANCE_ID,
      stateTransition: 'new_session',
      maxNewTokens: 48,
      seed: 92_002,
      prompt:
        'Repeat the letters A through Z separated by spaces, with no explanation. Continue until the output limit.',
    })
  );
  const [latePrimary, lateSecondary] = await Promise.all([
    latePrimaryPromise,
    lateSecondaryPromise,
  ]);

  const cancelSurvivorPromise = client.generate(
    generation({
      requestId: 'probe:cancel:survivor',
      instanceId: CANCEL_SURVIVOR_INSTANCE_ID,
      stateTransition: 'new_session',
      maxNewTokens: 192,
      seed: 93_001,
      prompt:
        'Return the positive integers in order, separated by commas, with no explanation. Continue until the output limit.',
    })
  );
  const cancelledRequestId = 'probe:cancel:rollback';
  const cancelledPromise = client.generate(
    generation({
      requestId: cancelledRequestId,
      instanceId: CANCELLED_INSTANCE_ID,
      stateTransition: 'new_session',
      maxNewTokens: 384,
      seed: 93_002,
      prompt:
        'Return the positive integers in order, separated by commas, with no explanation. Continue until the output limit.',
    })
  );
  await delay(900);
  await client.cancel(cancelledRequestId);
  const cancelledError = await cancelledPromise.then(
    () => undefined,
    (error: unknown) => toError(error)
  );
  const cancelSurvivor = await cancelSurvivorPromise;

  const cancelledBaseline = widthSixBaselines.get(CANCELLED_INSTANCE_ID);
  if (cancelledBaseline === undefined) {
    throw new Error('cancelled lane has no width-six committed baseline');
  }
  const rollbackContinuation = await client.generate({
    type: 'generate',
    request_id: 'probe:cancel:verify',
    instance_id: CANCELLED_INSTANCE_ID,
    state_transition: 'continuation',
    stream_tokens: false,
    input: [
      {
        type: 'tool_result',
        call_id: 'continuous-probe-call',
        output: '{"ok":true}',
      },
    ],
    tools: [],
    max_new_tokens: 1,
    sampling: { ...GREEDY_SAMPLING, seed: 93_003 },
  });

  const checks = {
    readyAdvertisesWidthSix: ready.max_active_batch_size === 6,
    readyAdvertisesLateJoinFour: ready.max_late_join_batch_size === 4,
    allSixRequestsUsedOneWidthSixPath: widthSix.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 6
    ),
    widthSixRowsRemainStateExact: widthSix.every((event) =>
      stateAccountingIsExact(event)
    ),
    unequalRowsChangedMembership: widthSix.some(
      (event) => event.response.metrics.decode_batch_membership_changes > 0
    ),
    lateRequestJoinedRunningPrimary:
      latePrimary.response.metrics.maximum_decode_batch_size === 2 &&
      lateSecondary.response.metrics.maximum_decode_batch_size === 2,
    latePrimaryReturnedToWidthOne:
      latePrimary.response.metrics.decode_batch_membership_changes >= 2,
    cancelledOnlyRequestedRow:
      cancelledError?.message.includes('cancelled before commit') === true,
    cancellationSurvivorCommitted:
      cancelSurvivor.response.metrics.maximum_decode_batch_size === 2 &&
      stateAccountingIsExact(cancelSurvivor),
    cancelledLaneRolledBackToPriorCommit:
      rollbackContinuation.response.metrics.cached_prefix_tokens ===
      cancelledBaseline,
    rollbackLaneWasReusable: stateAccountingIsExact(
      rollbackContinuation,
      cancelledBaseline
    ),
  };
  const report = {
    schemaVersion: 1,
    protocolVersion: ready.protocol_version,
    advertised: {
      maxOutstandingRequests: ready.max_outstanding_requests,
      maxActiveBatchSize: ready.max_active_batch_size,
      maxLateJoinBatchSize: ready.max_late_join_batch_size,
    },
    widthSix: {
      elapsedMilliseconds: widthSixElapsedMilliseconds,
      generatedTokens: widthSix.reduce(
        (total, event) => total + event.response.metrics.generated_tokens,
        0
      ),
      maximumWidths: widthSix.map(
        (event) => event.response.metrics.maximum_decode_batch_size
      ),
      membershipChanges: widthSix.map(
        (event) => event.response.metrics.decode_batch_membership_changes
      ),
      stateSequenceLengths: widthSix.map(
        (event) => event.response.state_sequence_length
      ),
      rows: widthSix.map(summary),
    },
    lateJoin: {
      primary: summary(latePrimary),
      secondary: summary(lateSecondary),
    },
    cancellation: {
      error: cancelledError?.message,
      survivor: summary(cancelSurvivor),
      priorSequenceLength: cancelledBaseline,
      continuationCachedPrefix:
        rollbackContinuation.response.metrics.cached_prefix_tokens,
    },
    checks,
  };
  console.log(JSON.stringify(report, undefined, 2));
  requirePassingChecks(checks);
} finally {
  await client.shutdown();
  await rm(stateRoot, { force: true, recursive: true });
}

function generation(input: {
  requestId: string;
  instanceId: string;
  stateTransition: NativeGenerateCommand['state_transition'];
  maxNewTokens: number;
  seed: number;
  prompt: string;
}): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: input.requestId,
    instance_id: input.instanceId,
    state_transition: input.stateTransition,
    stream_tokens: false,
    input: [{ role: 'developer', content: input.prompt }],
    tools: [],
    max_new_tokens: input.maxNewTokens,
    sampling: { ...GREEDY_SAMPLING, seed: input.seed },
  };
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

function summary(event: NativeCompletedEvent): {
  cachedPrefixTokens: number;
  inputTokensProcessed: number;
  generatedTokens: number;
  stateSequenceLength: number;
  finishReason: NativeCompletedEvent['response']['finish_reason'];
  maximumDecodeBatchSize: number;
  membershipChanges: number;
  decodeTokensPerSecond: number;
} {
  const metrics = event.response.metrics;
  return {
    cachedPrefixTokens: metrics.cached_prefix_tokens,
    inputTokensProcessed: metrics.input_tokens_processed,
    generatedTokens: metrics.generated_tokens,
    stateSequenceLength: event.response.state_sequence_length,
    finishReason: event.response.finish_reason,
    maximumDecodeBatchSize: metrics.maximum_decode_batch_size,
    membershipChanges: metrics.decode_batch_membership_changes,
    decodeTokensPerSecond:
      metrics.generated_tokens / (metrics.decode_execution_nanos / 1e9),
  };
}

function requirePassingChecks(checks: Record<string, boolean>): void {
  const failed = Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(`continuous-batch probe failed: ${failed.join(', ')}`);
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
