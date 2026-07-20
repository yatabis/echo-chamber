import { aggregateRuntimeResults, aggregateWorkflowResults } from './aggregate';
import { IMPLICIT_RUNTIME_SCENARIOS } from './runtime-implicit-scenarios';
import { RUNTIME_SCENARIOS } from './runtime-scenarios';
import { RUNTIME_WORKFLOWS } from './runtime-workflows';
import { summarizeChecks } from './scoring';

import type { RuntimeScenarioFixture } from './runtime-scenarios';
import type { RuntimeWorkflowFixture } from './runtime-workflows';
import type { RuntimeScenarioResult, RuntimeWorkflowResult } from './types';

export interface RescorableScenarioCell {
  id: string;
  purpose: string;
  results: RuntimeScenarioResult[];
  aggregate?: ReturnType<typeof aggregateRuntimeResults>;
}

export interface RescorableWorkflowCell {
  id: string;
  purpose: string;
  results: RuntimeWorkflowResult[];
  aggregate?: ReturnType<typeof aggregateWorkflowResults>;
}

export interface RescorableCandidateResult {
  scenarioCells: RescorableScenarioCell[];
  workflowCells: RescorableWorkflowCell[];
  [key: string]: unknown;
}

const SCENARIOS: readonly RuntimeScenarioFixture[] = [
  ...RUNTIME_SCENARIOS,
  ...IMPLICIT_RUNTIME_SCENARIOS,
];

function requireScenario(id: string): RuntimeScenarioFixture {
  const fixture = SCENARIOS.find((candidate) => candidate.id === id);
  if (fixture === undefined) {
    throw new Error(`Cannot rescore unknown runtime scenario: ${id}`);
  }
  return fixture;
}

function requireWorkflow(id: string): RuntimeWorkflowFixture {
  const fixture = RUNTIME_WORKFLOWS.find((candidate) => candidate.id === id);
  if (fixture === undefined) {
    throw new Error(`Cannot rescore unknown runtime workflow: ${id}`);
  }
  return fixture;
}

function rescoreScenarioResult(
  result: RuntimeScenarioResult
): RuntimeScenarioResult {
  const checks = requireScenario(result.scenarioId).evaluate({
    calls: result.calls,
    events: result.events,
    terminationReason: result.terminationReason,
  });
  return { ...result, checks, score: summarizeChecks(checks) };
}

function rescoreWorkflowResult(
  result: RuntimeWorkflowResult
): RuntimeWorkflowResult {
  const checks = requireWorkflow(result.workflowId).evaluate({
    sessions: result.sessions,
    finalMemories: result.finalMemories,
    finalNotes: result.finalNotes,
  });
  return { ...result, checks, score: summarizeChecks(checks) };
}

/**
 * 保存済みの観測traceを現在の採点器で再評価する。
 * モデル出力、tool trace、時間、usageは変更しない。
 */
export function rescoreCandidateResult(
  candidate: RescorableCandidateResult
): RescorableCandidateResult {
  const scenarioCells = candidate.scenarioCells.map((cell) => {
    const results = cell.results.map(rescoreScenarioResult);
    return { ...cell, results, aggregate: aggregateRuntimeResults(results) };
  });
  const workflowCells = candidate.workflowCells.map((cell) => {
    const results = cell.results.map(rescoreWorkflowResult);
    return { ...cell, results, aggregate: aggregateWorkflowResults(results) };
  });

  const rescored: RescorableCandidateResult = {
    ...candidate,
    scenarioCells,
    workflowCells,
  };
  delete rescored.promptAblationComparison;
  return rescored;
}
