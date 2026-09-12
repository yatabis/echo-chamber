/* eslint-disable no-await-in-loop -- One resident owner must advance one current state serially. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  writeFileSync,
} from 'node:fs';
import { arch, cpus, platform, release, totalmem } from 'node:os';
import { dirname, join } from 'node:path';

import { expect, test } from 'vitest';

import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import {
  requireNativeMetalMemory,
  type NativeCompletedEvent,
  type NativeGenerateCommand,
  type NativeRuntimeMetrics,
} from '@echo-chamber/native-inference-adapter/protocol';

import { EphemeralNativeStateRoots } from './ephemeral-state-roots';
import {
  LONG_SESSION_DEVELOPER_PROMPT,
  LONG_SESSION_FINAL_TEXT,
  LONG_SESSION_GREEDY_SAMPLING,
  LONG_SESSION_MAX_NEW_TOKENS,
  longSessionNativeContinuationCommand,
  longSessionNativePrefixCommand,
  longSessionNativeReplayCommand,
  requireLongSessionStepToolCall,
  serializeLongSessionToolResult,
} from './long-session-workload';

const LIVE_GATE_ENABLED =
  process.env.ECHO_NATIVE_LONG_SESSION_PERFORMANCE_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  outputPath: string;
  continuationSteps: number;
  paddingRepetitions: number;
  minimumFinalStateTokens: number;
}

interface AttemptRecord {
  phase: 'prefix' | 'cached-continuation' | 'stateless-replay';
  step: number;
  requestId: string;
  engineId: number;
  stateSequenceLength: number;
  finishReason: NativeCompletedEvent['response']['finish_reason'];
  visibleTtftMs: number;
  totalMs: number;
  externalDecodeTokensPerSecond: number;
  outputSha256: string;
  metrics: NativeRuntimeMetrics;
}

interface TimedAttempt {
  event: NativeCompletedEvent;
  record: AttemptRecord;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function parsePositiveInteger(
  name: string,
  fallback: number,
  minimum = 1
): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return value;
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_LONG_SESSION_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_LONG_SESSION_MODEL'),
    outputPath: requiredEnvironmentVariable('ECHO_LONG_SESSION_OUTPUT'),
    continuationSteps: parsePositiveInteger('ECHO_LONG_SESSION_STEPS', 8, 2),
    paddingRepetitions: parsePositiveInteger(
      'ECHO_LONG_SESSION_PADDING_REPETITIONS',
      256
    ),
    minimumFinalStateTokens: parsePositiveInteger(
      'ECHO_LONG_SESSION_MINIMUM_STATE_TOKENS',
      4_096
    ),
  };
}

function assertInputPaths(config: GateConfig): void {
  for (const [label, path] of [
    ['native binary', config.nativeBinaryPath],
    ['model directory', config.modelDirectory],
  ] as const) {
    if (!existsSync(path)) {
      throw new Error(`${label} does not exist: ${path}`);
    }
  }
}

function sha256Text(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest('hex');
}

function writeResult(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, undefined, 2)}\n`, 'utf8');
}

function gitState(workingDirectory: string): Record<string, unknown> {
  const command = (args: string[]): string =>
    execFileSync('git', args, {
      cwd: workingDirectory,
      encoding: 'utf8',
    }).trim();
  return {
    commit: command(['rev-parse', 'HEAD']),
    dirtyPaths: command(['status', '--short'])
      .split('\n')
      .filter((line) => line !== ''),
  };
}

function decodeRate(completionTokens: number, decodeWindowMs: number): number {
  if (completionTokens < 2 || decodeWindowMs <= 0) {
    return 0;
  }
  return (completionTokens - 1) / (decodeWindowMs / 1_000);
}

async function runAttempt(
  client: NativeInferenceClient,
  command: NativeGenerateCommand,
  phase: AttemptRecord['phase'],
  step: number
): Promise<TimedAttempt> {
  const startedAt = performance.now();
  let firstVisibleAt: number | undefined;
  const event = await client.generate(command, (token) => {
    if (token.text !== undefined && token.text !== '') {
      firstVisibleAt ??= performance.now();
    }
  });
  const completedAt = performance.now();
  if (firstVisibleAt === undefined) {
    throw new Error(`${command.request_id} emitted no visible token`);
  }
  const decodeWindowMs = completedAt - firstVisibleAt;
  return {
    event,
    record: {
      phase,
      step,
      requestId: command.request_id,
      engineId: event.response.engine_id,
      stateSequenceLength: event.response.state_sequence_length,
      finishReason: event.response.finish_reason,
      visibleTtftMs: firstVisibleAt - startedAt,
      totalMs: completedAt - startedAt,
      externalDecodeTokensPerSecond: decodeRate(
        event.response.metrics.generated_tokens,
        decodeWindowMs
      ),
      outputSha256: sha256Text(event.text),
      metrics: event.response.metrics,
    },
  };
}

function numberField(value: unknown, field: string): number {
  if (typeof value !== 'object' || value === null) {
    throw new Error(`native ready engine omitted numeric ${field}`);
  }
  const observed = (value as Record<string, unknown>)[field];
  if (typeof observed !== 'number') {
    throw new Error(`native ready engine omitted numeric ${field}`);
  }
  return observed;
}

liveTest(
  'advances one long-lived instance through repeated continuations',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
    const padding = 'state-cache-padding '.repeat(config.paddingRepetitions);
    const result: Record<string, unknown> = {
      schemaVersion: 2,
      generatedAt: new Date().toISOString(),
      conditions: {
        prompt: LONG_SESSION_DEVELOPER_PROMPT,
        promptSha256: sha256Text(LONG_SESSION_DEVELOPER_PROMPT),
        continuationSteps: config.continuationSteps,
        paddingRepetitionsPerStep: config.paddingRepetitions,
        paddingSha256: sha256Text(padding),
        minimumFinalStateTokens: config.minimumFinalStateTokens,
        maxNewTokens: LONG_SESSION_MAX_NEW_TOKENS,
        sampling: LONG_SESSION_GREEDY_SAMPLING,
        concurrency: 1,
      },
      machine: {
        platform: platform(),
        release: release(),
        architecture: arch(),
        cpu: cpus()[0]?.model ?? null,
        logicalCpuCount: cpus().length,
        unifiedMemoryBytes: totalmem(),
      },
      subjects: {
        echoChamber: gitState(process.cwd()),
        nativeBinaryPath: config.nativeBinaryPath,
        nativeBinarySha256: await sha256File(config.nativeBinaryPath),
        modelDirectory: config.modelDirectory,
        modelConfigSha256: await sha256File(
          join(config.modelDirectory, 'config.json')
        ),
        tokenizerConfigSha256: await sha256File(
          join(config.modelDirectory, 'tokenizer_config.json')
        ),
      },
      cachedAttempts: [],
    };
    writeResult(config.outputPath, result);

    const client = NativeInferenceClient.spawn({
      binaryPath: config.nativeBinaryPath,
      modelDirectory: config.modelDirectory,
      maxOutstandingRequests: 1,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH: libraryPath,
      },
    });
    const stateRoots = new EphemeralNativeStateRoots();
    try {
      const ready = await client.ready();
      const engineId = numberField(ready.engine, 'engine_id');
      result.engine = ready;
      writeResult(config.outputPath, result);

      const instanceId = 'long-session-cached';
      await stateRoots.open(client, instanceId);
      const prefix = await runAttempt(
        client,
        longSessionNativePrefixCommand(instanceId),
        'prefix',
        0
      );
      requireLongSessionStepToolCall(prefix.event, 1);
      const chain: NativeCompletedEvent[] = [prefix.event];
      const attempts: AttemptRecord[] = [prefix.record];
      const toolResults: string[] = [];
      let previous = prefix.event;

      for (let step = 1; step <= config.continuationSteps; step += 1) {
        const serializedResult = serializeLongSessionToolResult(
          step,
          config.continuationSteps,
          padding
        );
        toolResults.push(serializedResult);
        const attempt = await runAttempt(
          client,
          longSessionNativeContinuationCommand(
            instanceId,
            previous,
            step,
            serializedResult
          ),
          'cached-continuation',
          step
        );
        attempts.push(attempt.record);
        if (step < config.continuationSteps) {
          requireLongSessionStepToolCall(attempt.event, step + 1);
          chain.push(attempt.event);
        } else if (attempt.event.text.trim() !== LONG_SESSION_FINAL_TEXT) {
          throw new Error(
            `final cached output was ${JSON.stringify(attempt.event.text)}`
          );
        }
        previous = attempt.event;
        result.cachedAttempts = attempts;
        writeResult(config.outputPath, result);
      }

      const replayInstanceId = 'long-session-replay';
      await stateRoots.open(client, replayInstanceId);
      const replay = await runAttempt(
        client,
        longSessionNativeReplayCommand(replayInstanceId, chain, toolResults),
        'stateless-replay',
        config.continuationSteps
      );
      if (replay.event.text.trim() !== LONG_SESSION_FINAL_TEXT) {
        throw new Error(
          `final replay output was ${JSON.stringify(replay.event.text)}`
        );
      }

      const continuations = attempts.slice(1);
      const finalCached = attempts[attempts.length - 1];
      if (finalCached === undefined) {
        throw new Error('long-session benchmark omitted final cached attempt');
      }
      const checks = {
        oneResidentEngine: [...attempts, replay.record].every(
          (attempt) => attempt.engineId === engineId
        ),
        expectedStepCount: attempts.length === config.continuationSteps + 1,
        everyContinuationReusedCurrentState: continuations.every(
          (attempt, index) =>
            attempt.metrics.cached_prefix_tokens ===
            attempts[index]?.stateSequenceLength
        ),
        everyContinuationAdvancedOnlyItsSuffix: continuations.every(
          (attempt, index) => {
            const prior = attempts[index];
            return (
              prior !== undefined &&
              attempt.metrics.input_tokens_processed > 0 &&
              attempt.stateSequenceLength ===
                prior.stateSequenceLength +
                  attempt.metrics.input_tokens_processed +
                  attempt.metrics.generated_tokens +
                  (attempt.finishReason === 'length' ? 1 : 0)
            );
          }
        ),
        stateSequenceReachedMinimum:
          finalCached.stateSequenceLength >= config.minimumFinalStateTokens,
        finalGreedyOutputMatchesReplay:
          finalCached.outputSha256 === replay.record.outputSha256,
        replayProcessedSameFinalPrompt:
          replay.record.metrics.input_tokens_processed ===
          finalCached.stateSequenceLength -
            finalCached.metrics.generated_tokens -
            (finalCached.finishReason === 'length' ? 1 : 0),
        cachedFinalTtftFasterThanReplay:
          finalCached.visibleTtftMs < replay.record.visibleTtftMs,
      };
      result.statelessReplay = replay.record;
      result.summary = {
        initialStateSequenceLength: prefix.record.stateSequenceLength,
        finalStateSequenceLength: finalCached.stateSequenceLength,
        finalStep: config.continuationSteps,
        finalCachedVisibleTtftMs: finalCached.visibleTtftMs,
        statelessReplayVisibleTtftMs: replay.record.visibleTtftMs,
        cachedVsReplayTtftRatio:
          finalCached.visibleTtftMs / replay.record.visibleTtftMs,
        initialCommittedStateLogicalNbytes:
          prefix.record.metrics.committed_state_logical_nbytes,
        finalCommittedStateLogicalNbytes:
          finalCached.metrics.committed_state_logical_nbytes,
        activeMemoryGrowthNbytes:
          requireNativeMetalMemory(
            finalCached.metrics,
            'long-session final continuation'
          ).active_nbytes -
          requireNativeMetalMemory(prefix.record.metrics, 'long-session prefix')
            .active_nbytes,
      };
      result.checks = checks;
      writeResult(config.outputPath, result);

      expect(checks).toEqual({
        oneResidentEngine: true,
        expectedStepCount: true,
        everyContinuationReusedCurrentState: true,
        everyContinuationAdvancedOnlyItsSuffix: true,
        stateSequenceReachedMinimum: true,
        finalGreedyOutputMatchesReplay: true,
        replayProcessedSameFinalPrompt: true,
        cachedFinalTtftFasterThanReplay: true,
      });
    } finally {
      try {
        await client.shutdown();
      } finally {
        stateRoots.dispose();
      }
    }
  },
  10 * 60 * 1_000
);
