import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { expect, test } from 'vitest';

import type { ThinkingEngineResult } from '@echo-chamber/core/agent/thinking-engine';
import { ModelGenerationError } from '@echo-chamber/core/ports/model';
import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import { ECHO_NATIVE_PRODUCTION_SAMPLING } from '@echo-chamber/native-inference-adapter/native-inference-model';
import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
} from '@echo-chamber/native-inference-adapter/protocol';

import { LocalNativeInferenceRuntime } from './local-native-inference-runtime';
import {
  COGNITIVE_FIXTURE_PROMPT,
  COGNITIVE_REVIEW_PROMPT,
  CognitiveFixtureDomain,
  createCognitiveFixtureTools,
} from './testing/cognitive-fixture';

const binaryPath = process.env.ECHO_NATIVE_BINARY;
const modelDirectory = process.env.ECHO_NATIVE_MODEL_DIRECTORY;
const GREEDY = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
};

test
  .skipIf(binaryPath === undefined || modelDirectory === undefined)
  .each(['greedy', 'production'] as const)(
  'real Core Cognitive workflow with %s sampling, retry and next-session domain continuity',
  async (profile) => {
    if (binaryPath === undefined || modelDirectory === undefined)
      throw new Error('explicit Native model environment required');
    const snapshotDirectory = await mkdtemp(
      join(tmpdir(), 'echo-cognitive-workflow-')
    );
    const domain = new CognitiveFixtureDomain();
    const { tools, calls, finishedReasons } = createCognitiveFixtureTools();
    const commands: NativeGenerateCommand[] = [];
    const completed: NativeCompletedEvent[] = [];
    const cancelled: CancelledRequest[] = [];
    const sessions: ThinkingEngineResult[] = [];
    const sampling =
      profile === 'greedy' ? GREEDY : ECHO_NATIVE_PRODUCTION_SAMPLING;
    let armCancellation = false;
    let interruption: AbortController | undefined;
    let observedTokens = 0;
    let runtime: LocalNativeInferenceRuntime | undefined;
    try {
      runtime = await LocalNativeInferenceRuntime.start(
        {
          binaryPath,
          modelDirectory,
          snapshotDirectory,
          environment: {
            ...process.env,
            DYLD_LIBRARY_PATH:
              process.env.ECHO_NATIVE_LIBRARY_PATH ??
              process.env.DYLD_LIBRARY_PATH,
          },
          modelOptions: {
            rin: { sampling, maxTokens: 512, seedSource: (): number => 42 },
          },
          moduleOptions: {
            rin: {
              memory: {
                sampling,
                maxTokens: 512,
                seedSource: (): number => 42,
                onToken: (): void => {
                  if (interruption === undefined) return;
                  observedTokens += 1;
                  if (observedTokens === 3)
                    interruption.abort('injected Cognitive deadline');
                },
              },
              emotion: {
                sampling,
                maxTokens: 512,
                seedSource: (): number => 42,
              },
            },
          },
        },
        {
          spawnClient: (options): NativeInferenceClient => {
            const client = NativeInferenceClient.spawn(options);
            const generate = client.generate.bind(client);
            client.generate = async (
              command,
              onToken,
              signal
            ): Promise<NativeCompletedEvent> => {
              commands.push(structuredClone(command));
              try {
                const event = await generate(command, onToken, signal);
                completed.push(event);
                return event;
              } catch (error) {
                if (
                  error instanceof ModelGenerationError &&
                  error.name === 'AbortError'
                )
                  cancelled.push({ command, usage: error.usage });
                throw error;
              }
            };
            return client;
          },
        }
      );
      const options = {
        tools,
        systemPrompt: COGNITIVE_FIXTURE_PROMPT,
        cognitive: {
          domain,
          maxOutputTokens: 512,
          retryPolicy: {
            maxAttempts: 2,
            shouldRetry: ({ error }: { error: unknown }): boolean =>
              error instanceof Error && error.name === 'AbortError',
          },
          createRequestSignal: (): AbortSignal => {
            if (armCancellation) {
              armCancellation = false;
              interruption = new AbortController();
              return interruption.signal;
            }
            return AbortSignal.timeout(120_000);
          },
        },
      };
      sessions.push(await runtime.think('rin', options));
      const previousMemory = structuredClone(
        domain.state.previousSessionMemory
      );
      expect(previousMemory).not.toBeNull();
      armCancellation = true;
      sessions.push(
        await runtime.think('rin', {
          ...options,
          systemPrompt: COGNITIVE_REVIEW_PROMPT,
        })
      );
      assertRealWorkflow({
        sessions,
        calls,
        domain,
        commands,
        completed,
        cancelled,
        observedTokens,
        previousMemoryContent: previousMemory?.content,
      });
      expect(finishedReasons).toHaveLength(2);
      expect(finishedReasons[0]).toContain('KITE-47');
      assertUsage(sessions, completed, cancelled);
      expect(runtime.state('rin').snapshotDirty).toBe(false);
      expect(runtime.state('rin', 'memory').persistence).toBe('ephemeral');
    } finally {
      const reportDirectory =
        process.env.ECHO_NATIVE_COGNITIVE_REPORT_DIRECTORY;
      if (reportDirectory !== undefined)
        await writeFile(
          join(reportDirectory, `workflow-${profile}.json`),
          `${JSON.stringify({ profile, sessions, commits: domain.commits, failures: domain.failures, calls, commands, completed, cancelled }, null, 2)}\n`
        );
      await runtime?.shutdown();
      await rm(snapshotDirectory, { recursive: true, force: true });
    }
  },
  300_000
);

interface CancelledRequest {
  command: NativeGenerateCommand;
  usage: ModelGenerationError['usage'];
}

/** Verify actual dispatch/output, including the unchanged Main lineage transitions. */
function assertRealWorkflow(input: {
  sessions: ThinkingEngineResult[];
  calls: string[];
  domain: CognitiveFixtureDomain;
  commands: NativeGenerateCommand[];
  completed: NativeCompletedEvent[];
  cancelled: CancelledRequest[];
  observedTokens: number;
  previousMemoryContent: string | undefined;
}): void {
  const {
    sessions,
    calls,
    domain,
    commands,
    completed,
    cancelled,
    observedTokens,
    previousMemoryContent,
  } = input;
  expect(observedTokens).toBeGreaterThanOrEqual(3);
  expect(cancelled).toHaveLength(1);
  expect(cancelled[0]).toMatchObject({
    command: { state_transition: 'reset' },
  });
  expect(cancelled[0]?.usage?.outputTokens).toBeGreaterThanOrEqual(3);
  for (const command of commands) {
    expect(
      command.input.some((item) => 'role' in item && item.role === 'developer')
    ).toBe(false);
    if (command.state_transition !== 'continuation') {
      expect(command.input[0]).toMatchObject({ role: 'system' });
      expect(command.input[0]).toHaveProperty(
        'content',
        expect.stringContaining('<runtime_context>')
      );
      expect(
        command.input.filter((item) => 'role' in item && item.role === 'system')
      ).toHaveLength(1);
    }
  }
  assertPhaseFlow(sessions);
  expect(calls.filter((name) => name === 'check_notifications')).toHaveLength(
    2
  );
  expect(
    calls.filter((name) => name === 'inspect_note').length
  ).toBeGreaterThanOrEqual(1);
  const phaseCount = sessions.reduce(
    (total, session) => total + session.cognitiveModules.phases.length,
    0
  );
  expect(domain.commits).toHaveLength(phaseCount);
  const memory = commands.filter(
    (command) => command.instance_id === 'rin.memory'
  );
  expect(memory).toHaveLength(phaseCount + 1);
  expect(memory[0]?.state_transition).toBe('initial');
  expect(
    memory.slice(1).every((command) => command.state_transition === 'reset')
  ).toBe(true);
  const secondSessionOffset = sessions[0]?.cognitiveModules.phases.length;
  if (secondSessionOffset === undefined)
    throw new Error('missing first session');
  expect(memory[secondSessionOffset]?.input).toEqual(
    memory[secondSessionOffset + 1]?.input
  );
  assertPreviousMemory(memory[secondSessionOffset], previousMemoryContent);
  const auxiliary = completed.filter(
    (event) => event.response.instance_id !== 'rin'
  );
  expect(auxiliary).toHaveLength(phaseCount * 2);
  assertInputCacheReuse(commands, completed, cancelled);
  expect(
    Math.max(
      ...auxiliary.map(
        (event) => event.response.metrics.maximum_decode_batch_size
      )
    )
  ).toBeGreaterThan(1);
  expect(
    commands
      .filter((command) => command.instance_id === 'rin')
      .map((command) => command.state_transition)
  ).toEqual(
    sessions.flatMap((session, index) => [
      index === 0 ? 'initial' : 'new_session',
      ...Array<string>(session.cognitiveModules.phases.length - 2).fill(
        'continuation'
      ),
    ])
  );
}

/** Reconcile Core usage with every completed and rolled-back native request. */
function assertUsage(
  sessions: ThinkingEngineResult[],
  completed: NativeCompletedEvent[],
  cancelled: CancelledRequest[]
): void {
  const observedUsage = [
    ...completed.map((event) => ({
      input:
        event.response.metrics.input_tokens_processed +
        event.response.metrics.cached_prefix_tokens,
      output: event.response.metrics.generated_tokens,
    })),
    ...cancelled.map((event) => ({
      input: event.usage?.totalInputTokens ?? 0,
      output: event.usage?.outputTokens ?? 0,
    })),
  ];
  expect(
    sessions.reduce((total, session) => total + session.usage.totalTokens, 0)
  ).toBe(
    observedUsage.reduce(
      (total, usage) => total + usage.input + usage.output,
      0
    )
  );
}

/** One pre-main commit per Main turn and one terminal commit, with only Memory retried. */
function assertPhaseFlow(sessions: ThinkingEngineResult[]): void {
  expect(sessions[0]?.cognitiveModules.phases.length).toBeGreaterThanOrEqual(3);
  for (const session of sessions) {
    const phases = session.cognitiveModules.phases;
    expect(phases.length).toBeGreaterThanOrEqual(2);
    expect(
      phases.slice(0, -1).every((phase) => phase.phase === 'pre_main')
    ).toBe(true);
    expect(phases[phases.length - 1]?.phase).toBe('post_main');
  }
  expect(sessions[1]?.cognitiveModules.phases[0]).toMatchObject({
    memory: { attempts: 2 },
    emotion: { attempts: 1 },
  });
}

/** The committed domain memory is explicitly present in the next activation input. */
function assertPreviousMemory(
  command: NativeGenerateCommand | undefined,
  content: string | undefined
): void {
  if (command === undefined || content === undefined)
    throw new Error('missing prior session input');
  const observations = command.input.flatMap((item) =>
    'role' in item && typeof item.content === 'string' ? [item.content] : []
  );
  expect(
    observations.some((text) =>
      text.includes(JSON.stringify(content).slice(1, -1))
    )
  ).toBe(true);
}

/** Full-input reuse follows session and prompt identity, including cancelled prefill. */
function assertInputCacheReuse(
  commands: NativeGenerateCommand[],
  completed: NativeCompletedEvent[],
  cancelled: CancelledRequest[]
): void {
  const previous = new Map<string, NativeGenerateCommand>();
  for (const command of commands) {
    if (command.instance_id === 'rin') {
      expect(command.input_cache_scope).toBeUndefined();
      continue;
    }
    expect(command.input_cache_scope).toBeTypeOf('string');
    const before = previous.get(command.instance_id);
    const cached =
      completed.find((event) => event.request_id === command.request_id)
        ?.response.metrics.cached_prefix_tokens ??
      cancelled.find((event) => event.command.request_id === command.request_id)
        ?.usage?.cachedInputTokens;
    if (
      before !== undefined &&
      before.input_cache_scope === command.input_cache_scope &&
      JSON.stringify(before.input[0]) === JSON.stringify(command.input[0])
    ) {
      expect(cached).toBeGreaterThan(0);
    } else {
      expect(cached).toBe(0);
    }
    previous.set(command.instance_id, command);
  }
}
