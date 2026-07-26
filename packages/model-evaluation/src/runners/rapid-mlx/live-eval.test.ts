/* eslint-disable no-await-in-loop -- Evaluation-target servers, dependent workflow sessions, and thermal ordering are intentionally sequential. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { expect, test } from 'vitest';

import systemPromptRin from '@echo-chamber/core/llm/prompts/rin';
import { createEchoSessionCacheRequestBodyExtension } from '@echo-chamber/openai-adapter/echo-session-cache-v1';

import {
  aggregateRuntimeResults,
  aggregateWorkflowResults,
} from '../../qwen36-eat-readiness/aggregate';
import { loadEvaluationTargets } from '../../qwen36-eat-readiness/evaluation-targets';
import { loadProductionSamplingOverride } from '../../qwen36-eat-readiness/production-sampling-profile';
import { runRuntimeScenario } from '../../qwen36-eat-readiness/runtime-harness';
import { IMPLICIT_RUNTIME_SCENARIOS } from '../../qwen36-eat-readiness/runtime-implicit-scenarios';
import {
  CONTROLLED_GREEDY_PROFILE,
  PRODUCTION_SAMPLING_PROFILE,
} from '../../qwen36-eat-readiness/runtime-profiles';
import { RUNTIME_SCENARIOS } from '../../qwen36-eat-readiness/runtime-scenarios';
import { runRuntimeWorkflow } from '../../qwen36-eat-readiness/runtime-workflow-harness';
import { RUNTIME_WORKFLOWS } from '../../qwen36-eat-readiness/runtime-workflows';
import { createOpenAICompatibleModelFactory } from '../openai-compatible-model';

import { startLocalModelServer } from './server-controller';

import type { RuntimeModelFactory } from '../../qwen36-eat-readiness/runtime-harness';
import type { RuntimeScenarioFixture } from '../../qwen36-eat-readiness/runtime-scenarios';
import type { RuntimeWorkflowFixture } from '../../qwen36-eat-readiness/runtime-workflows';
import type {
  LocalEvaluationTarget,
  RuntimeGenerationProfile,
  RuntimeScenarioResult,
  RuntimeWorkflowResult,
} from '../../qwen36-eat-readiness/types';

const LIVE_EVALUATION_ENABLED = process.env.ECHO_LOCAL_MODEL_EVAL === '1';
const liveTest = LIVE_EVALUATION_ENABLED ? test : test.skip;

interface ScenarioCellDefinition {
  id: string;
  purpose: string;
  generationProfile: RuntimeGenerationProfile;
  repetitions: number;
  fixtures: readonly RuntimeScenarioFixture[];
}

interface WorkflowCellDefinition {
  id: string;
  purpose: string;
  generationProfile: RuntimeGenerationProfile;
  repetitions: number;
  fixtures: readonly RuntimeWorkflowFixture[];
}

interface ScenarioCellResult {
  id: string;
  purpose: string;
  results: RuntimeScenarioResult[];
  aggregate?: ReturnType<typeof aggregateRuntimeResults>;
}

interface WorkflowCellResult {
  id: string;
  purpose: string;
  results: RuntimeWorkflowResult[];
  aggregate?: ReturnType<typeof aggregateWorkflowResults>;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function readNonNegativeInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return parsed;
}

function commandOutput(command: string, args: string[], cwd: string): string {
  return execFileSync(command, args, { cwd, encoding: 'utf8' }).trim();
}

function sha256File(path: string): string {
  return sha256Text(readFileSync(path, 'utf8'));
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function readModelArchitecture(modelPath: string): Record<string, unknown> {
  const config = JSON.parse(
    readFileSync(join(modelPath, 'config.json'), 'utf8')
  ) as Record<string, unknown>;
  const textConfig =
    typeof config.text_config === 'object' && config.text_config !== null
      ? (config.text_config as Record<string, unknown>)
      : {};
  const quantization =
    typeof config.quantization === 'object' && config.quantization !== null
      ? (config.quantization as Record<string, unknown>)
      : {};

  return {
    modelType: config.model_type,
    architecture: config.architectures,
    hiddenSize: textConfig.hidden_size,
    hiddenLayers: textConfig.num_hidden_layers,
    experts: textConfig.num_experts,
    expertsPerToken: textConfig.num_experts_per_tok,
    denseIntermediateSize: textConfig.intermediate_size,
    mixtureOfExpertsIntermediateSize: textConfig.moe_intermediate_size,
    sharedExpertIntermediateSize: textConfig.shared_expert_intermediate_size,
    fullAttentionInterval: textConfig.full_attention_interval,
    weightQuantizationBits: quantization.bits,
    weightQuantizationGroupSize: quantization.group_size,
  };
}

function writeResult(path: string, result: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
}

async function clearPrefixCache(baseURL: string): Promise<void> {
  const response = await fetch(`${baseURL}/cache/clear`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(
      `Failed to clear prefix cache: ${response.status} ${await response.text()}`
    );
  }
}

function createDefaultTargets(): LocalEvaluationTarget[] {
  return [
    {
      id: 'qwen36-35b-a3b-base',
      displayName: 'Qwen3.6-35B-A3B MLX 4-bit',
      modelPath: requiredEnvironmentVariable('ECHO_EVAL_35B_MODEL'),
      servedModelName: 'qwen36-35b-a3b-eval',
    },
    {
      id: 'qwen36-27b-base',
      displayName: 'Qwen3.6-27B MLX 4-bit',
      modelPath: requiredEnvironmentVariable('ECHO_EVAL_27B_MODEL'),
      servedModelName: 'qwen36-27b-eval',
    },
  ];
}

function createEvaluationTargets(): LocalEvaluationTarget[] {
  const targetsPath = process.env.ECHO_EVAL_TARGETS_FILE;
  return targetsPath === undefined
    ? createDefaultTargets()
    : loadEvaluationTargets(targetsPath);
}

function readKvCacheDtype(): 'int4' | 'int8' | 'bf16' {
  const value = process.env.ECHO_EVAL_KV_CACHE_DTYPE ?? 'int4';
  if (value !== 'int4' && value !== 'int8' && value !== 'bf16') {
    throw new Error(`Unsupported ECHO_EVAL_KV_CACHE_DTYPE: ${value}`);
  }
  return value;
}

function createScenarioCells(input: {
  smokeMode: boolean;
  productionRepetitions: number;
  productionSamplingProfile: RuntimeGenerationProfile;
}): ScenarioCellDefinition[] {
  const explicit = input.smokeMode
    ? RUNTIME_SCENARIOS.slice(0, 1)
    : RUNTIME_SCENARIOS;
  const implicit = input.smokeMode
    ? IMPLICIT_RUNTIME_SCENARIOS.slice(0, 1)
    : IMPLICIT_RUNTIME_SCENARIOS;
  const deploymentSentinels = input.smokeMode
    ? IMPLICIT_RUNTIME_SCENARIOS.slice(0, 1)
    : IMPLICIT_RUNTIME_SCENARIOS.filter((fixture) =>
        [
          'implicit_private_schedule_change',
          'implicit_multi_channel_priority',
        ].includes(fixture.id)
      );
  const cells: ScenarioCellDefinition[] = [
    {
      id: 'controlled-explicit-production-prompt',
      purpose:
        'Production prompt with user messages that explicitly name the required procedure.',
      generationProfile: CONTROLLED_GREEDY_PROFILE,
      repetitions: 1,
      fixtures: explicit,
    },
    {
      id: 'controlled-implicit-production-prompt',
      purpose:
        'Production prompt with procedural instructions removed from user messages.',
      generationProfile: CONTROLLED_GREEDY_PROFILE,
      repetitions: 1,
      fixtures: implicit,
    },
  ];
  if (input.productionRepetitions > 0) {
    cells.push({
      id: 'deployment-sampling-single-session-sentinels',
      purpose:
        'Small repeated sample under the current production sampling values.',
      generationProfile: input.productionSamplingProfile,
      repetitions: input.productionRepetitions,
      fixtures: deploymentSentinels,
    });
  }
  return cells;
}

function createWorkflowCells(input: {
  smokeMode: boolean;
  productionRepetitions: number;
  productionSamplingProfile: RuntimeGenerationProfile;
}): WorkflowCellDefinition[] {
  const controlledProduction = input.smokeMode
    ? RUNTIME_WORKFLOWS.filter((fixture) =>
        [
          'state_revision_across_cold_start',
          'transient_note_update_failure',
        ].includes(fixture.id)
      )
    : RUNTIME_WORKFLOWS;
  const deploymentSentinels = RUNTIME_WORKFLOWS.filter((fixture) =>
    [
      'queued_priority_after_session_boundary',
      'transient_note_update_failure',
    ].includes(fixture.id)
  );
  const cells: WorkflowCellDefinition[] = [
    {
      id: 'controlled-stateful-production-prompt',
      purpose:
        'Persistent memory/context, next-session priority, and injected-failure recovery.',
      generationProfile: CONTROLLED_GREEDY_PROFILE,
      repetitions: 1,
      fixtures: controlledProduction,
    },
  ];
  if (input.productionRepetitions > 0) {
    cells.push({
      id: 'deployment-sampling-stateful-sentinels',
      purpose:
        'Repeated state-boundary and failure-recovery sentinels under production sampling.',
      generationProfile: input.productionSamplingProfile,
      repetitions: input.productionRepetitions,
      fixtures: input.smokeMode
        ? deploymentSentinels.slice(0, 1)
        : deploymentSentinels,
    });
  }
  return cells;
}

function filterScenarioCells(
  cells: readonly ScenarioCellDefinition[],
  caseFilter: RegExp | null
): ScenarioCellDefinition[] {
  if (caseFilter === null) {
    return [...cells];
  }
  return cells
    .map((cell) => ({
      ...cell,
      fixtures: cell.fixtures.filter((fixture) => caseFilter.test(fixture.id)),
    }))
    .filter((cell) => cell.fixtures.length > 0);
}

function filterWorkflowCells(
  cells: readonly WorkflowCellDefinition[],
  caseFilter: RegExp | null
): WorkflowCellDefinition[] {
  if (caseFilter === null) {
    return [...cells];
  }
  return cells
    .map((cell) => ({
      ...cell,
      fixtures: cell.fixtures.filter((fixture) => caseFilter.test(fixture.id)),
    }))
    .filter((cell) => cell.fixtures.length > 0);
}

interface CellExecutionTarget {
  targetId: string;
  createModel: RuntimeModelFactory;
  systemPrompt: string;
  maxTurns: number;
  reverseFixtures: boolean;
  clearPrefixCache(): Promise<void>;
  checkpoint(): void;
}

function echoSessionCacheId(input: {
  targetId: string;
  cellId: string;
  fixtureId: string;
  repetition: number;
}): string {
  return [
    'eval',
    input.targetId,
    input.cellId,
    input.fixtureId,
    String(input.repetition),
  ].join(':');
}

async function executeScenarioCell(
  cell: ScenarioCellDefinition,
  target: CellExecutionTarget,
  cellResult: ScenarioCellResult
): Promise<void> {
  const fixtures = target.reverseFixtures
    ? [...cell.fixtures].reverse()
    : [...cell.fixtures];
  for (let repetition = 1; repetition <= cell.repetitions; repetition += 1) {
    for (const fixture of fixtures) {
      await target.clearPrefixCache();
      process.stdout.write(
        `[local-eval] ${target.targetId} ${cell.id} ${fixture.id} repetition=${repetition}\n`
      );
      cellResult.results.push(
        await runRuntimeScenario(fixture, {
          createModel: target.createModel,
          maxTurns: target.maxTurns,
          systemPrompt: target.systemPrompt,
          generationProfile: cell.generationProfile,
          repetition,
          sessionId: echoSessionCacheId({
            targetId: target.targetId,
            cellId: cell.id,
            fixtureId: fixture.id,
            repetition,
          }),
        })
      );
      target.checkpoint();
    }
  }
  cellResult.aggregate = aggregateRuntimeResults(cellResult.results);
}

async function executeWorkflowCell(
  cell: WorkflowCellDefinition,
  target: CellExecutionTarget,
  cellResult: WorkflowCellResult
): Promise<void> {
  const fixtures = target.reverseFixtures
    ? [...cell.fixtures].reverse()
    : [...cell.fixtures];
  for (let repetition = 1; repetition <= cell.repetitions; repetition += 1) {
    for (const fixture of fixtures) {
      await target.clearPrefixCache();
      process.stdout.write(
        `[local-eval] ${target.targetId} ${cell.id} ${fixture.id} repetition=${repetition}\n`
      );
      cellResult.results.push(
        await runRuntimeWorkflow(fixture, {
          createModel: target.createModel,
          maxTurns: target.maxTurns,
          systemPrompt: target.systemPrompt,
          generationProfile: cell.generationProfile,
          repetition,
          sessionId: echoSessionCacheId({
            targetId: target.targetId,
            cellId: cell.id,
            fixtureId: fixture.id,
            repetition,
          }),
        })
      );
      target.checkpoint();
    }
  }
  cellResult.aggregate = aggregateWorkflowResults(cellResult.results);
}

liveTest(
  'evaluates local models on E.C.H.O. runtime behavior',
  // Live setup branches are recorded explicitly in the result protocol.
  // eslint-disable-next-line complexity
  async () => {
    const repositoryRoot = requiredEnvironmentVariable('ECHO_EVAL_REPOSITORY');
    const rapidMlxBin = requiredEnvironmentVariable('ECHO_EVAL_RAPID_MLX_BIN');
    const rapidMlxWorkingDirectory = requiredEnvironmentVariable(
      'ECHO_EVAL_RAPID_MLX_CWD'
    );
    const outputPath = requiredEnvironmentVariable('ECHO_EVAL_OUTPUT');
    const outputDirectory = dirname(outputPath);
    const smokeMode = process.env.ECHO_EVAL_SMOKE === '1';
    const productionRepetitions = readNonNegativeInteger(
      'ECHO_EVAL_PRODUCTION_REPETITIONS',
      smokeMode ? 1 : 2
    );
    const productionSamplingOverridePath =
      process.env.ECHO_EVAL_PRODUCTION_SAMPLING_FILE;
    const productionSamplingProfile =
      productionSamplingOverridePath === undefined
        ? PRODUCTION_SAMPLING_PROFILE
        : loadProductionSamplingOverride(productionSamplingOverridePath);
    const allTargets = createEvaluationTargets();
    const targets = smokeMode ? allTargets.slice(0, 1) : allTargets;
    const caseFilterSource = process.env.ECHO_EVAL_CASE_FILTER;
    const caseFilter =
      caseFilterSource === undefined ? null : new RegExp(caseFilterSource);
    const cellFilterSource = process.env.ECHO_EVAL_CELL_FILTER;
    const cellFilter =
      cellFilterSource === undefined ? null : new RegExp(cellFilterSource);
    const scenarioCells = filterScenarioCells(
      createScenarioCells({
        smokeMode,
        productionRepetitions,
        productionSamplingProfile,
      }),
      caseFilter
    ).filter((cell) => cellFilter === null || cellFilter.test(cell.id));
    const workflowCells = filterWorkflowCells(
      createWorkflowCells({
        smokeMode,
        productionRepetitions,
        productionSamplingProfile,
      }),
      caseFilter
    ).filter((cell) => cellFilter === null || cellFilter.test(cell.id));
    if (scenarioCells.length === 0 && workflowCells.length === 0) {
      throw new Error('ECHO_EVAL_CASE_FILTER matched no runtime cases');
    }
    const runtimeMaxTurns = smokeMode ? 4 : 8;
    const port = Number(process.env.ECHO_EVAL_PORT ?? '8134');
    const kvCacheDtype = readKvCacheDtype();
    const promptPath = join(
      repositoryRoot,
      'packages/core/src/llm/prompts/rin.ts'
    );
    const mainLlmConfigPath = join(
      repositoryRoot,
      'apps/cloudflare-workers/src/config/main-llm-config.ts'
    );
    const sourceSnapshot = {
      repositoryHead: commandOutput(
        'git',
        ['rev-parse', 'HEAD'],
        repositoryRoot
      ),
      repositoryStatus: commandOutput(
        'git',
        ['status', '--short'],
        repositoryRoot
      ),
      repositoryDiffNames: commandOutput(
        'git',
        ['diff', '--name-status'],
        repositoryRoot
      ),
      repositoryStagedDiffNames: commandOutput(
        'git',
        ['diff', '--cached', '--name-status'],
        repositoryRoot
      ),
      rinPromptSha256: sha256File(promptPath),
      mainLlmConfigSha256: sha256File(mainLlmConfigPath),
      rapidMlxVersion: commandOutput(
        rapidMlxBin,
        ['--version'],
        rapidMlxWorkingDirectory
      ),
      rapidMlxHead: commandOutput(
        'git',
        ['rev-parse', 'HEAD'],
        rapidMlxWorkingDirectory
      ),
      rapidMlxStatus: commandOutput(
        'git',
        ['status', '--short'],
        rapidMlxWorkingDirectory
      ),
    };
    const result: {
      schemaVersion: number;
      startedAt: string;
      completedAt?: string;
      sourceSnapshot: typeof sourceSnapshot;
      protocol: Record<string, unknown>;
      candidates: unknown[];
    } = {
      schemaVersion: 4,
      startedAt: new Date().toISOString(),
      sourceSnapshot,
      protocol: {
        purpose:
          'Primary E.C.H.O. Chamber runtime behavior evaluation and reusable pre/post-EAT comparison matrix.',
        smokeMode,
        caseFilter: caseFilterSource ?? null,
        cellFilter: cellFilterSource ?? null,
        scenarioCells: scenarioCells.map((cell) => ({
          id: cell.id,
          purpose: cell.purpose,
          generationProfile: cell.generationProfile,
          repetitions: cell.repetitions,
          scenarioIds: cell.fixtures.map((fixture) => fixture.id),
        })),
        workflowCells: workflowCells.map((cell) => ({
          id: cell.id,
          purpose: cell.purpose,
          generationProfile: cell.generationProfile,
          repetitions: cell.repetitions,
          workflowIds: cell.fixtures.map((fixture) => fixture.id),
        })),
        runtimeMaxTurns,
        multiTokenPrediction: false,
        prefixCache:
          'enabled: each fixture gets an isolated session ID and a cleared process-local cache; Rapid-MLX reuses exact token prefixes at natural prefill boundaries',
        gdnRecurrentStateCarryover:
          'not measured: the current Chat Completions adapter has no token-boundary-safe recurrent-state continuation contract',
        midGenerationExternalInterrupt:
          'not supported by the current runtime; queued_priority_after_session_boundary measures the implemented next-session behavior instead',
        externalPorts:
          'stateful synthetic implementations of production port contracts; no Discord, KV, SQLite, note store, or Zenn side effect',
        productionSamplingOverridePath: productionSamplingOverridePath ?? null,
        productionSamplingCaveat: productionSamplingProfile.description,
        pflash: false,
        kvCacheDtype,
      },
      candidates: [],
    };

    mkdirSync(outputDirectory, { recursive: true });
    for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
      const target = targets[targetIndex];
      if (target === undefined) {
        throw new Error(`Missing evaluation target at index ${targetIndex}`);
      }
      process.stdout.write(`\n[local-eval] loading ${target.displayName}\n`);
      const logPath = join(outputDirectory, `${target.id}.server.log`);
      const server = await startLocalModelServer({
        target,
        rapidMlxBin,
        rapidMlxWorkingDirectory,
        port,
        logPath,
        kvCacheDtype,
        prefixCacheMode: 'enabled',
      });
      let stopResult: { exitCode: number | null; signalCode: string | null } = {
        exitCode: null,
        signalCode: null,
      };
      const candidateResult = {
        candidate: target,
        architecture: readModelArchitecture(target.modelPath),
        server: {
          startupElapsedMs: server.startupElapsedMs,
          temporaryModelPath: server.temporaryModelPath,
          temporaryPrefixCachePath: server.temporaryPrefixCachePath,
          logPath,
        },
        scenarioCells: [] as ScenarioCellResult[],
        workflowCells: [] as WorkflowCellResult[],
      };
      result.candidates.push(candidateResult);
      writeResult(outputPath, result);

      try {
        process.stdout.write(
          `[local-eval] ready ${target.displayName} in ${server.startupElapsedMs}ms\n`
        );
        const cellTarget: CellExecutionTarget = {
          targetId: target.id,
          systemPrompt: systemPromptRin,
          createModel: createOpenAICompatibleModelFactory({
            apiKey: 'local-evaluation',
            baseURL: server.baseURL,
            servedModelName: target.servedModelName,
            createSessionRequestBodyExtension:
              createEchoSessionCacheRequestBodyExtension,
          }),
          maxTurns: runtimeMaxTurns,
          reverseFixtures: targetIndex % 2 !== 0,
          clearPrefixCache: async (): Promise<void> => {
            await clearPrefixCache(server.baseURL);
          },
          checkpoint: (): void => {
            writeResult(outputPath, result);
          },
        };
        for (const cell of scenarioCells) {
          const cellResult: ScenarioCellResult = {
            id: cell.id,
            purpose: cell.purpose,
            results: [],
          };
          candidateResult.scenarioCells.push(cellResult);
          await executeScenarioCell(cell, cellTarget, cellResult);
          writeResult(outputPath, result);
        }

        for (const cell of workflowCells) {
          const cellResult: WorkflowCellResult = {
            id: cell.id,
            purpose: cell.purpose,
            results: [],
          };
          candidateResult.workflowCells.push(cellResult);
          await executeWorkflowCell(cell, cellTarget, cellResult);
          writeResult(outputPath, result);
        }
        writeResult(outputPath, result);
      } finally {
        stopResult = await server.stop();
        server.cleanup();
      }

      Object.assign(candidateResult, { serverExit: stopResult });
      writeResult(outputPath, result);
    }

    result.completedAt = new Date().toISOString();
    writeResult(outputPath, result);
    expect(result.candidates).toHaveLength(targets.length);
    expect(existsSync(outputPath)).toBe(true);
  },
  4 * 60 * 60_000
);
