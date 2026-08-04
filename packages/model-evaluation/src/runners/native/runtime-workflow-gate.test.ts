/* eslint-disable complexity, no-await-in-loop -- The live gate serializes one resident GPU owner and checkpoints evidence after each dependent workflow. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { arch, cpus, platform, release, tmpdir, totalmem } from 'node:os';
import { basename, dirname, join } from 'node:path';

import { expect, test } from 'vitest';

import systemPromptRin from '@echo-chamber/core/llm/prompts/rin';
import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import { NativeInferenceModel } from '@echo-chamber/native-inference-adapter/native-inference-model';
import type {
  NativeRuntimeMetrics,
  NativeStateTransition,
} from '@echo-chamber/native-inference-adapter/protocol';

import {
  CONTROLLED_GREEDY_PROFILE,
  PRODUCTION_SAMPLING_PROFILE,
} from '../../qwen36-eat-readiness/runtime-profiles';
import { runRuntimeWorkflow } from '../../qwen36-eat-readiness/runtime-workflow-harness';
import { RUNTIME_WORKFLOWS } from '../../qwen36-eat-readiness/runtime-workflows';

import type { RuntimeModelFactory } from '../../qwen36-eat-readiness/runtime-harness';
import type { RuntimeWorkflowFixture } from '../../qwen36-eat-readiness/runtime-workflows';
import type {
  RuntimeGenerationProfile,
  RuntimeWorkflowResult,
  TraceEvent,
} from '../../qwen36-eat-readiness/types';

const LIVE_GATE_ENABLED = process.env.ECHO_NATIVE_RUNTIME_WORKFLOW_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const ADAPTIVE_PREFILL_THRESHOLD_TOKENS = 8_192;

type WorkflowStateMode =
  | 'carried'
  | 'recurrent-only-ablation'
  | 'convolution-only-ablation'
  | 'fresh-session-ablation';

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  outputPath: string;
  libraryPath: string;
  profile: RuntimeGenerationProfile;
  maxTurns: number;
  seed: number;
  workflowFilter: RegExp | null;
  stateMode: WorkflowStateMode;
}

interface RequestEvidence {
  sessionId: string;
  sessionIndex: number;
  requestIndexWithinSession: number;
  turnIndex: number;
  stateTransition: NativeStateTransition;
  stateSequenceLength: number;
  finishReason: string;
  metrics: NativeRuntimeMetrics;
  derived: {
    inputExecutionMs: number;
    decodeExecutionMs: number;
    modelExecutionMs: number;
    requestMs: number;
    generatedTokensPerDecodeSecond: number;
  };
}

interface SnapshotEvidence {
  published: boolean;
  elapsedMs: number;
  ownerCount: number;
  physicalNbytes: number;
  owners: {
    instanceId: string;
    published: boolean;
    physicalNbytes?: number;
    fileName?: string;
    stateSequenceLength?: number;
  }[];
}

interface WorkflowEvidence {
  fixtureSessionCount: number;
  modelFactoryCallCount: number;
  result: RuntimeWorkflowResult;
  requests: RequestEvidence[];
  snapshot: SnapshotEvidence;
}

interface GateChecks {
  completedEverySelectedWorkflow: boolean;
  everyBehaviorCheckPassed: boolean;
  everySessionFinishedThinking: boolean;
  everySessionRecordedNativeExchange: boolean;
  noNativeProviderWarnings: boolean;
  transitionContractMatched: boolean;
  initialTransitionObserved: boolean;
  continuationTransitionObserved: boolean;
  newSessionTransitionObservedWhenRequired: boolean;
  freshSessionRequestsDidNotClaimTokenPrefixReuse: boolean;
  continuationRequestsReusedCommittedTokenPrefix: boolean;
  adaptivePrefillObservedAtOrAboveEightKiTokens: boolean;
  everyWorkflowPublishedCurrentState: boolean;
  engineReportedExpectedNewSessionGdnPolicy: boolean;
}

class RebindableEventPort implements EchoEventPort {
  private target: EchoEventPort | undefined;

  bind(target: EchoEventPort): void {
    this.target = target;
  }

  async emit(event: EchoEvent): Promise<void> {
    if (this.target === undefined) {
      throw new Error('native workflow event port is not bound to a session');
    }
    await this.target.emit(event);
  }
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function parsePositiveInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive safe integer`);
  }
  return value;
}

function parseNonNegativeInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function parseProfile(value: string | undefined): RuntimeGenerationProfile {
  if (value === undefined || value === CONTROLLED_GREEDY_PROFILE.id) {
    return CONTROLLED_GREEDY_PROFILE;
  }
  if (value === PRODUCTION_SAMPLING_PROFILE.id) {
    return PRODUCTION_SAMPLING_PROFILE;
  }
  throw new Error(
    `ECHO_NATIVE_WORKFLOW_PROFILE must be ${CONTROLLED_GREEDY_PROFILE.id} or ${PRODUCTION_SAMPLING_PROFILE.id}`
  );
}

function parseStateMode(value: string | undefined): WorkflowStateMode {
  if (value === undefined || value === 'carried') return 'carried';
  if (
    value === 'recurrent-only-ablation' ||
    value === 'convolution-only-ablation' ||
    value === 'fresh-session-ablation'
  ) {
    return value;
  }
  throw new Error(
    'ECHO_NATIVE_WORKFLOW_STATE_MODE must be carried, recurrent-only-ablation, convolution-only-ablation, or fresh-session-ablation'
  );
}

function carriesStateAcrossSessions(stateMode: WorkflowStateMode): boolean {
  return stateMode !== 'fresh-session-ablation';
}

function nativeNewSessionGdnPolicy(stateMode: WorkflowStateMode): string {
  if (stateMode === 'recurrent-only-ablation') return 'carry_recurrent_only';
  if (stateMode === 'convolution-only-ablation') {
    return 'carry_convolution_only';
  }
  return 'carry_all';
}

function stateLifetimeDescription(stateMode: WorkflowStateMode): string {
  switch (stateMode) {
    case 'carried':
      return 'one stable NativeInferenceModel instance per workflow; new_session retains GDN convolution and recurrent state';
    case 'recurrent-only-ablation':
      return 'diagnostic ablation only: one stable NativeInferenceModel instance per workflow; new_session clears GDN convolution and retains recurrent state';
    case 'convolution-only-ablation':
      return 'diagnostic ablation only: one stable NativeInferenceModel instance per workflow; new_session retains GDN convolution and clears recurrent state';
    case 'fresh-session-ablation':
      return 'diagnostic ablation only: one fresh NativeInferenceModel instance per harness session; no cross-session GDN carry';
  }
}

function loadConfig(): GateConfig {
  const filter = process.env.ECHO_NATIVE_WORKFLOW_FILTER;
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_NATIVE_WORKFLOW_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_NATIVE_WORKFLOW_MODEL'),
    outputPath: requiredEnvironmentVariable('ECHO_NATIVE_WORKFLOW_OUTPUT'),
    libraryPath: requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH'),
    profile: parseProfile(process.env.ECHO_NATIVE_WORKFLOW_PROFILE),
    maxTurns: parsePositiveInteger('ECHO_NATIVE_WORKFLOW_MAX_TURNS', 10),
    seed: parseNonNegativeInteger('ECHO_NATIVE_WORKFLOW_SEED', 42),
    workflowFilter:
      filter === undefined || filter === '' ? null : new RegExp(filter),
    stateMode: parseStateMode(process.env.ECHO_NATIVE_WORKFLOW_STATE_MODE),
  };
}

function selectWorkflows(config: GateConfig): RuntimeWorkflowFixture[] {
  const filter = config.workflowFilter;
  const selected =
    filter === null
      ? [...RUNTIME_WORKFLOWS]
      : RUNTIME_WORKFLOWS.filter((fixture) => filter.test(fixture.id));
  if (selected.length === 0) {
    throw new Error('ECHO_NATIVE_WORKFLOW_FILTER selected no workflows');
  }
  return selected;
}

function assertInputPaths(config: GateConfig): void {
  for (const [label, path] of [
    ['native binary', config.nativeBinaryPath],
    ['model directory', config.modelDirectory],
    ['model config', join(config.modelDirectory, 'config.json')],
    ['tokenizer config', join(config.modelDirectory, 'tokenizer_config.json')],
  ] as const) {
    if (!existsSync(path)) {
      throw new Error(`${label} does not exist: ${path}`);
    }
  }
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest('hex');
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function gitRepositoryRoot(workingDirectory: string): string {
  return execFileSync('git', ['rev-parse', '--show-toplevel'], {
    cwd: workingDirectory,
    encoding: 'utf8',
  }).trim();
}

function gitState(workingDirectory: string): Record<string, unknown> {
  const command = (arguments_: string[]): string =>
    execFileSync('git', arguments_, {
      cwd: workingDirectory,
      encoding: 'utf8',
    }).trim();
  return {
    commit: command(['rev-parse', 'HEAD']),
    branch: command(['branch', '--show-current']),
    dirtyPaths: command(['status', '--short'])
      .split('\n')
      .filter((line) => line !== ''),
  };
}

function writeResult(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, undefined, 2)}\n`, 'utf8');
}

function toError(value: unknown): Error {
  if (value instanceof Error) return value;
  return new Error(JSON.stringify(value));
}

function nativeSampling(
  profile: RuntimeGenerationProfile
): ConstructorParameters<typeof NativeInferenceModel>[0]['sampling'] {
  return {
    temperature: profile.temperature,
    top_p: profile.topP,
    top_k: profile.topK,
    min_p: profile.minP,
    repetition_penalty: profile.repetitionPenalty,
    presence_penalty: profile.presencePenalty,
  };
}

function createSeedSource(seed: number): () => number {
  let requestIndex = 0;
  return (): number => {
    const requestSeed = seed + requestIndex;
    if (!Number.isSafeInteger(requestSeed)) {
      throw new Error('native workflow request seed exceeded safe integer');
    }
    requestIndex += 1;
    return requestSeed;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireNumber(
  record: Record<string, unknown>,
  key: string,
  label: string
): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label}.${key} must be a finite non-negative number`);
  }
  return value;
}

function requireNativeMetrics(value: unknown): NativeRuntimeMetrics {
  if (!isRecord(value) || !isRecord(value.metal_memory)) {
    throw new Error('native workflow exchange did not retain runtime metrics');
  }
  for (const key of [
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
  ]) {
    requireNumber(value, key, 'metrics');
  }
  for (const key of ['active_nbytes', 'cache_nbytes', 'peak_nbytes']) {
    requireNumber(value.metal_memory, key, 'metrics.metal_memory');
  }
  if (value.first_generated_token_nanos !== undefined) {
    requireNumber(value, 'first_generated_token_nanos', 'metrics');
  }
  return value as unknown as NativeRuntimeMetrics;
}

function requireStateTransition(value: unknown): NativeStateTransition {
  if (
    value !== 'initial' &&
    value !== 'continuation' &&
    value !== 'new_session'
  ) {
    throw new Error('native workflow exchange has no valid stateTransition');
  }
  return value;
}

function nanosToMilliseconds(nanos: number): number {
  return nanos / 1_000_000;
}

function decodeRate(metrics: NativeRuntimeMetrics): number {
  if (metrics.decode_execution_nanos === 0) return 0;
  return metrics.generated_tokens / (metrics.decode_execution_nanos / 1e9);
}

function nativeExchangeEvent(event: TraceEvent): boolean {
  return (
    event.type === 'model.exchange.recorded' &&
    event.payload?.provider === 'echo.native_inference'
  );
}

function requestEvidence(
  event: TraceEvent,
  sessionId: string,
  sessionIndex: number,
  requestIndexWithinSession: number
): RequestEvidence {
  const payload = event.payload;
  if (payload === undefined) {
    throw new Error('native workflow exchange has no payload');
  }
  const metrics = requireNativeMetrics(payload.metrics);
  const turnIndex = requireNumber(payload, 'turnIndex', 'exchange');
  const stateSequenceLength = requireNumber(
    payload,
    'stateSequenceLength',
    'exchange'
  );
  if (
    !Number.isSafeInteger(turnIndex) ||
    !Number.isSafeInteger(stateSequenceLength)
  ) {
    throw new Error('native workflow exchange indices must be safe integers');
  }
  return {
    sessionId,
    sessionIndex,
    requestIndexWithinSession,
    turnIndex,
    stateTransition: requireStateTransition(payload.stateTransition),
    stateSequenceLength,
    finishReason:
      typeof payload.finishReason === 'string'
        ? payload.finishReason
        : 'unknown',
    metrics,
    derived: {
      inputExecutionMs: nanosToMilliseconds(metrics.input_execution_nanos),
      decodeExecutionMs: nanosToMilliseconds(metrics.decode_execution_nanos),
      modelExecutionMs: nanosToMilliseconds(metrics.model_execution_nanos),
      requestMs: nanosToMilliseconds(metrics.request_nanos),
      generatedTokensPerDecodeSecond: decodeRate(metrics),
    },
  };
}

function collectRequests(result: RuntimeWorkflowResult): RequestEvidence[] {
  return result.sessions.flatMap((session, sessionIndex) =>
    session.events
      .filter(nativeExchangeEvent)
      .map((event, requestIndexWithinSession) =>
        requestEvidence(
          event,
          session.sessionId,
          sessionIndex,
          requestIndexWithinSession
        )
      )
  );
}

function expectedTransition(
  request: RequestEvidence,
  stateMode: WorkflowStateMode
): NativeStateTransition {
  if (request.requestIndexWithinSession > 0) return 'continuation';
  if (stateMode === 'fresh-session-ablation') return 'initial';
  return request.sessionIndex === 0 ? 'initial' : 'new_session';
}

function sumMetrics(
  requests: readonly RequestEvidence[],
  select: (metrics: NativeRuntimeMetrics) => number
): number {
  return requests.reduce(
    (total, request) => total + select(request.metrics),
    0
  );
}

function summarize(
  workflows: readonly WorkflowEvidence[]
): Record<string, unknown> {
  const requests = workflows.flatMap((workflow) => workflow.requests);
  const generatedTokens = sumMetrics(
    requests,
    (metrics) => metrics.generated_tokens
  );
  const decodeExecutionNanos = sumMetrics(
    requests,
    (metrics) => metrics.decode_execution_nanos
  );
  return {
    workflowCount: workflows.length,
    sessionCount: workflows.reduce(
      (total, workflow) => total + workflow.result.sessions.length,
      0
    ),
    requestCount: requests.length,
    cachedPrefixTokens: sumMetrics(
      requests,
      (metrics) => metrics.cached_prefix_tokens
    ),
    newlyProcessedInputTokens: sumMetrics(
      requests,
      (metrics) => metrics.input_tokens_processed
    ),
    generatedTokens,
    inputModelExecutionCount: sumMetrics(
      requests,
      (metrics) => metrics.input_model_execution_count
    ),
    inputExecutionMs: nanosToMilliseconds(
      sumMetrics(requests, (metrics) => metrics.input_execution_nanos)
    ),
    decodeExecutionMs: nanosToMilliseconds(decodeExecutionNanos),
    modelExecutionMs: nanosToMilliseconds(
      sumMetrics(requests, (metrics) => metrics.model_execution_nanos)
    ),
    nativeRequestMs: nanosToMilliseconds(
      sumMetrics(requests, (metrics) => metrics.request_nanos)
    ),
    generatedTokensPerDecodeSecond:
      decodeExecutionNanos === 0
        ? 0
        : generatedTokens / (decodeExecutionNanos / 1e9),
    maximumNewlyProcessedInputTokens: Math.max(
      0,
      ...requests.map((request) => request.metrics.input_tokens_processed)
    ),
    maximumInputModelExecutionCount: Math.max(
      0,
      ...requests.map((request) => request.metrics.input_model_execution_count)
    ),
    maximumMetalPeakNbytes: Math.max(
      0,
      ...requests.map((request) => request.metrics.metal_memory.peak_nbytes)
    ),
  };
}

function evaluateChecks(
  selected: readonly RuntimeWorkflowFixture[],
  workflows: readonly WorkflowEvidence[],
  stateMode: WorkflowStateMode,
  engine: Record<string, unknown>
): GateChecks {
  const requests = workflows.flatMap((workflow) => workflow.requests);
  const continuationRequests = requests.filter(
    (request) => request.stateTransition === 'continuation'
  );
  const requiresNewSession =
    carriesStateAcrossSessions(stateMode) &&
    selected.some((workflow) => workflow.sessions.length > 1);
  return {
    completedEverySelectedWorkflow: workflows.length === selected.length,
    everyBehaviorCheckPassed: workflows.every(
      (workflow) =>
        workflow.result.score.earned === workflow.result.score.possible
    ),
    everySessionFinishedThinking: workflows.every((workflow) =>
      workflow.result.sessions.every(
        (session) => session.terminationReason === 'finish_thinking'
      )
    ),
    everySessionRecordedNativeExchange: workflows.every((workflow) =>
      workflow.result.sessions.every((session) =>
        session.events.some(nativeExchangeEvent)
      )
    ),
    noNativeProviderWarnings: workflows.every((workflow) =>
      workflow.result.sessions.every((session) =>
        session.events.every((event) => event.type !== 'model.provider.warning')
      )
    ),
    transitionContractMatched: requests.every(
      (request) =>
        request.stateTransition === expectedTransition(request, stateMode)
    ),
    initialTransitionObserved: requests.some(
      (request) => request.stateTransition === 'initial'
    ),
    continuationTransitionObserved: continuationRequests.length > 0,
    newSessionTransitionObservedWhenRequired:
      !requiresNewSession ||
      requests.some((request) => request.stateTransition === 'new_session'),
    freshSessionRequestsDidNotClaimTokenPrefixReuse: requests
      .filter((request) => request.stateTransition !== 'continuation')
      .every((request) => request.metrics.cached_prefix_tokens === 0),
    continuationRequestsReusedCommittedTokenPrefix:
      continuationRequests.length > 0 &&
      continuationRequests.every(
        (request) => request.metrics.cached_prefix_tokens > 0
      ),
    adaptivePrefillObservedAtOrAboveEightKiTokens: requests.some(
      (request) =>
        request.metrics.input_tokens_processed >=
          ADAPTIVE_PREFILL_THRESHOLD_TOKENS &&
        request.metrics.input_model_execution_count > 1
    ),
    everyWorkflowPublishedCurrentState: workflows.every(
      (workflow) =>
        workflow.snapshot.published && workflow.snapshot.physicalNbytes > 0
    ),
    engineReportedExpectedNewSessionGdnPolicy:
      engine.new_session_gdn_policy === nativeNewSessionGdnPolicy(stateMode),
  };
}

interface WorkflowModelOwner {
  instanceId: string;
  model: NativeInferenceModel;
  events: RebindableEventPort;
}

async function createWorkflowModelOwners(input: {
  client: NativeInferenceClient;
  fixture: RuntimeWorkflowFixture;
  config: GateConfig;
  stateRoot: string;
  workflowIndex: number;
}): Promise<WorkflowModelOwner[]> {
  const ownerCount = carriesStateAcrossSessions(input.config.stateMode)
    ? 1
    : input.fixture.sessions.length;
  const seedSource = createSeedSource(
    input.config.seed + input.workflowIndex * 100_000
  );
  const owners: WorkflowModelOwner[] = [];
  for (let ownerIndex = 0; ownerIndex < ownerCount; ownerIndex += 1) {
    const suffix = ownerCount === 1 ? '' : `-session-${ownerIndex + 1}`;
    const instanceId = `runtime-workflow-${input.fixture.id}${suffix}`;
    const events = new RebindableEventPort();
    const model = new NativeInferenceModel({
      client: input.client,
      instanceId,
      maxTokens: input.config.profile.maxTokensPerTurn,
      sampling: nativeSampling(input.config.profile),
      seedSource,
      events,
    });
    await model.openState(join(input.stateRoot, instanceId));
    owners.push({ instanceId, model, events });
  }
  return owners;
}

async function publishWorkflowOwners(
  owners: readonly WorkflowModelOwner[]
): Promise<SnapshotEvidence> {
  const startedAt = performance.now();
  const snapshots: SnapshotEvidence['owners'] = [];
  for (const owner of owners) {
    if (!owner.model.needsSnapshot()) {
      snapshots.push({ instanceId: owner.instanceId, published: false });
      continue;
    }
    const published = await owner.model.snapshot();
    snapshots.push({
      instanceId: owner.instanceId,
      published: true,
      physicalNbytes: published.physical_nbytes,
      fileName: basename(published.path),
      stateSequenceLength: owner.model.state().stateSequenceLength,
    });
  }
  return {
    published:
      snapshots.length > 0 && snapshots.every((snapshot) => snapshot.published),
    elapsedMs: performance.now() - startedAt,
    ownerCount: owners.length,
    physicalNbytes: snapshots.reduce(
      (total, snapshot) => total + (snapshot.physicalNbytes ?? 0),
      0
    ),
    owners: snapshots,
  };
}

async function runWorkflow(input: {
  client: NativeInferenceClient;
  fixture: RuntimeWorkflowFixture;
  config: GateConfig;
  stateRoot: string;
  workflowIndex: number;
}): Promise<WorkflowEvidence> {
  const owners = await createWorkflowModelOwners(input);
  let modelFactoryCallCount = 0;
  const createModel: RuntimeModelFactory = ({ events, generationProfile }) => {
    if (generationProfile.id !== input.config.profile.id) {
      throw new Error('workflow harness changed the configured profile');
    }
    const ownerIndex = carriesStateAcrossSessions(input.config.stateMode)
      ? 0
      : modelFactoryCallCount;
    const owner = owners[ownerIndex];
    if (owner === undefined) {
      throw new Error(
        'workflow harness requested more model owners than sessions'
      );
    }
    owner.events.bind(events);
    modelFactoryCallCount += 1;
    return owner.model;
  };
  const result = await runRuntimeWorkflow(input.fixture, {
    createModel,
    maxTurns: input.config.maxTurns,
    systemPrompt: systemPromptRin,
    generationProfile: input.config.profile,
    repetition: 1,
  });
  const snapshot = await publishWorkflowOwners(owners);
  return {
    fixtureSessionCount: input.fixture.sessions.length,
    modelFactoryCallCount,
    result,
    requests: collectRequests(result),
    snapshot,
  };
}

test('maps the two workflow profiles to native sampling exactly', () => {
  expect(nativeSampling(CONTROLLED_GREEDY_PROFILE)).toEqual({
    temperature: 0,
    top_p: 1,
    top_k: 1,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
  });
  expect(nativeSampling(PRODUCTION_SAMPLING_PROFILE)).toEqual({
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 1.5,
  });
  expect(parseStateMode(undefined)).toBe('carried');
  expect(parseStateMode('recurrent-only-ablation')).toBe(
    'recurrent-only-ablation'
  );
  expect(parseStateMode('convolution-only-ablation')).toBe(
    'convolution-only-ablation'
  );
  expect(parseStateMode('fresh-session-ablation')).toBe(
    'fresh-session-ablation'
  );
  expect(() => parseStateMode('unsupported')).toThrow(
    'ECHO_NATIVE_WORKFLOW_STATE_MODE must be carried, recurrent-only-ablation, convolution-only-ablation, or fresh-session-ablation'
  );
  expect(nativeNewSessionGdnPolicy('carried')).toBe('carry_all');
  expect(nativeNewSessionGdnPolicy('recurrent-only-ablation')).toBe(
    'carry_recurrent_only'
  );
  expect(nativeNewSessionGdnPolicy('convolution-only-ablation')).toBe(
    'carry_convolution_only'
  );
  expect(nativeNewSessionGdnPolicy('fresh-session-ablation')).toBe('carry_all');
});

liveTest(
  'runs production E.C.H.O. workflows through one stateful native owner',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    const selected = selectWorkflows(config);
    const repositoryRoot = gitRepositoryRoot(process.cwd());
    const workflows: WorkflowEvidence[] = [];
    const stateRoot = mkdtempSync(join(tmpdir(), 'echo-native-workflows-'));
    const artifact: Record<string, unknown> = {
      schemaVersion: 2,
      status: 'running',
      startedAt: new Date().toISOString(),
      conditions: {
        scope:
          'production Rin prompt, canonical runtime tools, stateful synthetic workflow ports, and native model-owned GDN/KV state',
        selectedWorkflowIds: selected.map((workflow) => workflow.id),
        generationProfile: config.profile,
        maxTurns: config.maxTurns,
        baseSeed: config.seed,
        seedSchedule:
          'base + workflow_index * 100000 + zero-based request index',
        streamTokens: false,
        concurrency: 1,
        stateMode: config.stateMode,
        nativeNewSessionGdnPolicy: nativeNewSessionGdnPolicy(config.stateMode),
        stateLifetime: stateLifetimeDescription(config.stateMode),
        snapshotPolicy:
          'publish every model owner once after each completed workflow, outside workflow elapsed time',
        decodeRateDenominator: 'generated_tokens / decode_execution_nanos',
        adaptivePrefillAdmission:
          'at least one request with input_tokens_processed >= 8192 and input_model_execution_count > 1',
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
        echoChamber: gitState(repositoryRoot),
        rinPromptSha256: sha256Text(systemPromptRin),
        runtimeHarnessSha256: await sha256File(
          join(
            repositoryRoot,
            'packages/model-evaluation/src/qwen36-eat-readiness/runtime-harness.ts'
          )
        ),
        runtimeWorkflowHarnessSha256: await sha256File(
          join(
            repositoryRoot,
            'packages/model-evaluation/src/qwen36-eat-readiness/runtime-workflow-harness.ts'
          )
        ),
        runtimeWorkflowFixturesSha256: await sha256File(
          join(
            repositoryRoot,
            'packages/model-evaluation/src/qwen36-eat-readiness/runtime-workflows.ts'
          )
        ),
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
      workflows,
    };
    writeResult(config.outputPath, artifact);

    const spawnedAt = performance.now();
    const client = NativeInferenceClient.spawn({
      binaryPath: config.nativeBinaryPath,
      modelDirectory: config.modelDirectory,
      maxOutstandingRequests: 1,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH: config.libraryPath,
        ECHO_NATIVE_NEW_SESSION_GDN_POLICY: nativeNewSessionGdnPolicy(
          config.stateMode
        ),
      },
    });
    let primaryError: Error | undefined;
    let shutdownError: Error | undefined;
    try {
      const ready = await client.ready();
      artifact.engine = ready;
      artifact.readyElapsedMs = performance.now() - spawnedAt;
      for (const [workflowIndex, fixture] of selected.entries()) {
        workflows.push(
          await runWorkflow({
            client,
            fixture,
            config,
            stateRoot,
            workflowIndex,
          })
        );
        artifact.completedWorkflowCount = workflows.length;
        writeResult(config.outputPath, artifact);
      }
      const checks = evaluateChecks(
        selected,
        workflows,
        config.stateMode,
        ready.engine
      );
      artifact.summary = summarize(workflows);
      artifact.checks = checks;
      artifact.status = 'completed';
      writeResult(config.outputPath, artifact);
      const failedChecks = (Object.keys(checks) as (keyof GateChecks)[]).filter(
        (key) => !checks[key]
      );
      expect(failedChecks).toEqual([]);
    } catch (error) {
      primaryError = toError(error);
      artifact.status = 'failed';
      artifact.error = primaryError.stack ?? primaryError.message;
    } finally {
      try {
        await client.shutdown();
      } catch (error) {
        shutdownError = toError(error);
      }
      rmSync(stateRoot, { recursive: true, force: true });
      artifact.cleanup = {
        nativeShutdown: shutdownError === undefined,
        ephemeralStateRemoved: !existsSync(stateRoot),
        ...(shutdownError === undefined
          ? {}
          : {
              shutdownError:
                shutdownError instanceof Error
                  ? shutdownError.message
                  : String(shutdownError),
            }),
      };
      artifact.completedAt = new Date().toISOString();
      writeResult(config.outputPath, artifact);
    }
    if (primaryError !== undefined) throw primaryError;
    if (shutdownError !== undefined) throw shutdownError;
  },
  45 * 60_000
);
