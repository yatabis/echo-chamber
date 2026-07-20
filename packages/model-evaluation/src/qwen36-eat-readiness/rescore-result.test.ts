import { readFileSync, writeFileSync } from 'node:fs';

import { expect, test } from 'vitest';

import { rescoreCandidateResult } from './rescore';

import type { RescorableCandidateResult } from './rescore';

const ENABLED = process.env.ECHO_EVAL_RESCORE_RESULT === '1';
const rescoreTest = ENABLED ? test : test.skip;

interface EvaluationArtifact {
  candidates: RescorableCandidateResult[];
  rescoreHistory?: {
    rescoredAt: string;
    scorer: string;
    reason: string;
  }[];
  [key: string]: unknown;
}

rescoreTest(
  'rescoring a saved evaluation artifact from its immutable traces',
  () => {
    const path = process.env.ECHO_EVAL_RESCORE_PATH;
    if (path === undefined || path.trim() === '') {
      throw new Error('ECHO_EVAL_RESCORE_PATH is required');
    }

    const artifact = JSON.parse(
      readFileSync(path, 'utf8')
    ) as EvaluationArtifact;
    expect(Array.isArray(artifact.candidates)).toBe(true);
    const rescoredAt = new Date().toISOString();
    const rescored: EvaluationArtifact = {
      ...artifact,
      candidates: artifact.candidates.map(rescoreCandidateResult),
      rescoreHistory: [
        ...(artifact.rescoreHistory ?? []),
        {
          rescoredAt,
          scorer: 'qwen36-eat-readiness/current',
          reason:
            process.env.ECHO_EVAL_RESCORE_REASON ??
            'Rebuilt checks and aggregates from saved traces.',
        },
      ],
    };

    writeFileSync(path, `${JSON.stringify(rescored, null, 2)}\n`, 'utf8');
    expect(rescored.candidates.length).toBeGreaterThan(0);
  }
);
