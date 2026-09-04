import { createHash } from 'node:crypto';
import {
  copyFileSync,
  createReadStream,
  existsSync,
  mkdirSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';

import { expect, test } from 'vitest';

import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import type {
  NativeGenerateCommand,
  NativeRuntimeMetrics,
} from '@echo-chamber/native-inference-adapter/protocol';

import { EphemeralNativeStateRoots } from './ephemeral-state-roots';

const LIVE_GATE_ENABLED =
  process.env.ECHO_NATIVE_CHUNKED_PREFILL_PARITY_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const TARGET_CONTEXT_TOKENS = 4_096;
const PREFILL_CHUNK_SIZE_TOKENS = 2_048;
const MAX_NEW_TOKENS = 8;
const NEW_SESSION_MAX_NEW_TOKENS = 128;
const NEW_SESSION_PROMPT =
  'Summarize what matters when preserving a long-lived internal state across a fresh session.';
const PROMPT_BASE_TOKENS = 108;
const PADDING_PHRASE_TOKENS = 3;
const PADDING_PHRASE = ' state-cache-padding';
const REMAINDER_TOKEN = ' x';
const BENCHMARK_PREFIX = [
  'This is a deterministic context-length performance benchmark.',
  'Ignore every padding token between <padding> and </padding>.',
  'After </padding>, write the integers from 1 through 400 on separate lines.',
  'Every line must use exactly this format: 0001: context-curve-performance',
  'Use four zero-padded digits, no preface, no explanation, and no closing text.',
  '<padding>',
].join('\n');
const BENCHMARK_SUFFIX = ['</padding>', 'Begin the numbered output now.'].join(
  '\n'
);
const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 1,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
  seed: 42,
} as const;

type PrefillMode = 'single_execution' | 'chunked_2k';

interface VariantResult {
  mode: PrefillMode;
  initialOutputSha256: string;
  newSessionOutputSha256: string;
  stateSha256: string;
  statePhysicalNbytes: number;
  stateSequenceLength: number;
  initialGeneratedTokens: number[];
  newSessionGeneratedTokens: number[];
  initialMetrics: NativeRuntimeMetrics;
  newSessionMetrics: NativeRuntimeMetrics;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function benchmarkPrompt(): string {
  const paddingTokens = TARGET_CONTEXT_TOKENS - PROMPT_BASE_TOKENS;
  const repetitions = Math.floor(paddingTokens / PADDING_PHRASE_TOKENS);
  const remainderTokens = paddingTokens % PADDING_PHRASE_TOKENS;
  return `${BENCHMARK_PREFIX}${PADDING_PHRASE.repeat(repetitions)}${REMAINDER_TOKEN.repeat(remainderTokens)}\n${BENCHMARK_SUFFIX}`;
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

function command(instanceId: string, mode: PrefillMode): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: `chunked-prefill-parity:${mode}`,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: false,
    input: [{ role: 'user', content: benchmarkPrompt() }],
    tools: [],
    max_new_tokens: MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

function newSessionCommand(
  instanceId: string,
  mode: PrefillMode
): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: `chunked-prefill-parity:${mode}:new-session`,
    instance_id: instanceId,
    state_transition: 'new_session',
    stream_tokens: false,
    input: [{ role: 'user', content: NEW_SESSION_PROMPT }],
    tools: [],
    max_new_tokens: NEW_SESSION_MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

async function runVariant(input: {
  mode: PrefillMode;
  binaryPath: string;
  modelDirectory: string;
  libraryPath: string;
}): Promise<VariantResult> {
  const instanceId = 'chunked-prefill-state-parity';
  const environment: NodeJS.ProcessEnv = {
    ...process.env,
    DYLD_LIBRARY_PATH: input.libraryPath,
    ECHO_NATIVE_PREFILL_CHUNK_SIZE_TOKENS:
      input.mode === 'single_execution'
        ? '0'
        : String(PREFILL_CHUNK_SIZE_TOKENS),
  };
  if (input.mode === 'chunked_2k') {
    environment.ECHO_NATIVE_PREFILL_CHUNK_AT_OR_ABOVE_TOKENS = '1';
  } else {
    delete environment.ECHO_NATIVE_PREFILL_CHUNK_AT_OR_ABOVE_TOKENS;
  }
  const client = NativeInferenceClient.spawn({
    binaryPath: input.binaryPath,
    modelDirectory: input.modelDirectory,
    maxOutstandingRequests: 1,
    environment,
  });
  const stateRoots = new EphemeralNativeStateRoots();
  try {
    await client.ready();
    await stateRoots.open(client, instanceId);
    const initial = await client.generate(command(instanceId, input.mode));
    const newSession = await client.generate(
      newSessionCommand(instanceId, input.mode)
    );
    const snapshot = await client.snapshot({
      type: 'snapshot',
      request_id: `chunked-prefill-parity:${input.mode}:snapshot`,
      instance_id: instanceId,
    });
    const retainedSnapshotDirectory =
      process.env.ECHO_CHUNKED_PREFILL_PARITY_SNAPSHOT_DIRECTORY;
    if (
      retainedSnapshotDirectory !== undefined &&
      retainedSnapshotDirectory.trim() !== ''
    ) {
      mkdirSync(retainedSnapshotDirectory, { recursive: true });
      copyFileSync(
        snapshot.path,
        join(retainedSnapshotDirectory, `${input.mode}.safetensors`)
      );
    }
    return {
      mode: input.mode,
      initialOutputSha256: sha256Text(initial.text),
      newSessionOutputSha256: sha256Text(newSession.text),
      stateSha256: await sha256File(snapshot.path),
      statePhysicalNbytes: snapshot.physical_nbytes,
      stateSequenceLength: newSession.response.state_sequence_length,
      initialGeneratedTokens: initial.response.generated_tokens,
      newSessionGeneratedTokens: newSession.response.generated_tokens,
      initialMetrics: initial.response.metrics,
      newSessionMetrics: newSession.response.metrics,
    };
  } finally {
    try {
      await client.shutdown();
    } finally {
      stateRoots.dispose();
    }
  }
}

function writeResult(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, undefined, 2)}\n`, 'utf8');
}

function firstTokenDivergence(left: number[], right: number[]): number | null {
  const sharedLength = Math.min(left.length, right.length);
  for (let index = 0; index < sharedLength; index += 1) {
    if (left[index] !== right[index]) {
      return index;
    }
  }
  return left.length === right.length ? null : sharedLength;
}

test('constructs an exactly calibrated 4K parity prompt', () => {
  expect(benchmarkPrompt()).toContain(PADDING_PHRASE);
  expect(TARGET_CONTEXT_TOKENS - PROMPT_BASE_TOKENS).toBe(
    Math.floor(
      (TARGET_CONTEXT_TOKENS - PROMPT_BASE_TOKENS) / PADDING_PHRASE_TOKENS
    ) *
      PADDING_PHRASE_TOKENS +
      ((TARGET_CONTEXT_TOKENS - PROMPT_BASE_TOKENS) % PADDING_PHRASE_TOKENS)
  );
});

liveTest(
  'characterizes 2K prefill chunks without assuming bit-exact carried GDN state',
  async () => {
    const binaryPath = requiredEnvironmentVariable(
      'ECHO_CHUNKED_PREFILL_NATIVE_INFERENCE_BIN'
    );
    const modelDirectory = requiredEnvironmentVariable(
      'ECHO_CHUNKED_PREFILL_MODEL'
    );
    const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
    for (const [label, path] of [
      ['native binary', binaryPath],
      ['model directory', modelDirectory],
    ] as const) {
      if (!existsSync(path)) {
        throw new Error(`${label} does not exist: ${path}`);
      }
    }

    const single = await runVariant({
      mode: 'single_execution',
      binaryPath,
      modelDirectory,
      libraryPath,
    });
    const chunked = await runVariant({
      mode: 'chunked_2k',
      binaryPath,
      modelDirectory,
      libraryPath,
    });
    const result = {
      generatedAt: new Date().toISOString(),
      conditions: {
        targetContextTokens: TARGET_CONTEXT_TOKENS,
        chunkSizeTokens: PREFILL_CHUNK_SIZE_TOKENS,
        maxNewTokens: MAX_NEW_TOKENS,
        newSessionMaxNewTokens: NEW_SESSION_MAX_NEW_TOKENS,
        sampling: GREEDY_SAMPLING,
      },
      variants: [single, chunked],
      checks: {
        exactInputTokenCounts:
          single.initialMetrics.input_tokens_processed ===
            TARGET_CONTEXT_TOKENS &&
          chunked.initialMetrics.input_tokens_processed ===
            TARGET_CONTEXT_TOKENS,
        expectedInputExecutionCounts:
          single.initialMetrics.input_model_execution_count === 1 &&
          chunked.initialMetrics.input_model_execution_count === 2,
        initialGeneratedTokensExact:
          JSON.stringify(single.initialGeneratedTokens) ===
          JSON.stringify(chunked.initialGeneratedTokens),
        initialOutputExact:
          single.initialOutputSha256 === chunked.initialOutputSha256,
        newSessionGeneratedTokensExact:
          JSON.stringify(single.newSessionGeneratedTokens) ===
          JSON.stringify(chunked.newSessionGeneratedTokens),
        newSessionOutputExact:
          single.newSessionOutputSha256 === chunked.newSessionOutputSha256,
        stateSequenceLengthExact:
          single.stateSequenceLength === chunked.stateSequenceLength,
        statePhysicalSizeExact:
          single.statePhysicalNbytes === chunked.statePhysicalNbytes,
        stateTokenAccountingExact:
          single.stateSequenceLength ===
            single.newSessionMetrics.input_tokens_processed +
              single.newSessionMetrics.generated_tokens +
              1 &&
          chunked.stateSequenceLength ===
            chunked.newSessionMetrics.input_tokens_processed +
              chunked.newSessionMetrics.generated_tokens +
              1,
        completeStateExact: single.stateSha256 === chunked.stateSha256,
      },
      observations: {
        newSessionFirstTokenDivergence: firstTokenDivergence(
          single.newSessionGeneratedTokens,
          chunked.newSessionGeneratedTokens
        ),
      },
    };
    const outputPath = process.env.ECHO_CHUNKED_PREFILL_PARITY_OUTPUT;
    if (outputPath !== undefined && outputPath.trim() !== '') {
      writeResult(outputPath, result);
    }
    expect(result.checks).toMatchObject({
      exactInputTokenCounts: true,
      expectedInputExecutionCounts: true,
      initialGeneratedTokensExact: true,
      initialOutputExact: true,
      stateSequenceLengthExact: true,
      statePhysicalSizeExact: true,
      stateTokenAccountingExact: true,
    });
    expect(single.initialGeneratedTokens).toHaveLength(MAX_NEW_TOKENS);
    expect(chunked.initialGeneratedTokens).toHaveLength(MAX_NEW_TOKENS);
    expect(single.newSessionGeneratedTokens).toHaveLength(
      NEW_SESSION_MAX_NEW_TOKENS
    );
    expect(chunked.newSessionGeneratedTokens).toHaveLength(
      NEW_SESSION_MAX_NEW_TOKENS
    );
  },
  10 * 60_000
);
