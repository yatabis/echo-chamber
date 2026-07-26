import { readFileSync } from 'node:fs';

import { PRODUCTION_SAMPLING_PROFILE } from './runtime-profiles';

import type { RuntimeGenerationProfile } from './types';

const OVERRIDE_KEYS = new Set([
  'description',
  'temperature',
  'topP',
  'topK',
  'minP',
  'repetitionPenalty',
  'presencePenalty',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requiredDescription(record: Record<string, unknown>): string {
  const value = record.description;
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error('Production sampling override has invalid description');
  }
  return value;
}

function requiredNumber(
  record: Record<string, unknown>,
  key: string,
  isValid: (value: number) => boolean
): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value) || !isValid(value)) {
    throw new Error(`Production sampling override has invalid ${key}`);
  }
  return value;
}

/**
 * 外部JSON由来のproduction sampling上書きを検証する。
 *
 * 評価コストと実運用modeを変えないため、出力上限とthinking設定は
 * ハーネス既定値から継承し、sampling値だけを上書きする。
 *
 * @param value JSONから復元した未検証の値。
 * @returns 検証済みのproduction sampling生成条件。
 * @throws 必須値、不明なキー、または許容範囲外の値がある場合。
 */
export function parseProductionSamplingOverride(
  value: unknown
): RuntimeGenerationProfile {
  if (!isRecord(value)) {
    throw new Error('Production sampling override must be an object');
  }

  for (const key of Object.keys(value)) {
    if (!OVERRIDE_KEYS.has(key)) {
      throw new Error(`Production sampling override has unknown key: ${key}`);
    }
  }

  return {
    id: 'production-sampling',
    description: requiredDescription(value),
    temperature: requiredNumber(value, 'temperature', (number) => number >= 0),
    topP: requiredNumber(value, 'topP', (number) => number >= 0 && number <= 1),
    topK: requiredNumber(
      value,
      'topK',
      (number) => Number.isInteger(number) && number >= 1
    ),
    minP: requiredNumber(value, 'minP', (number) => number >= 0 && number <= 1),
    repetitionPenalty: requiredNumber(
      value,
      'repetitionPenalty',
      (number) => number > 0
    ),
    presencePenalty: requiredNumber(
      value,
      'presencePenalty',
      (number) => number >= -2 && number <= 2
    ),
    maxTokensPerTurn: PRODUCTION_SAMPLING_PROFILE.maxTokensPerTurn,
    enableThinking: false,
  };
}

/**
 * 指定したJSONファイルからproduction sampling上書きを読み込む。
 *
 * @param path sampling上書きJSONの絶対パスまたは作業ディレクトリ相対パス。
 * @returns 検証済みのproduction sampling生成条件。
 * @throws ファイルを読めない、JSONが不正、または値の検証に失敗した場合。
 */
export function loadProductionSamplingOverride(
  path: string
): RuntimeGenerationProfile {
  return parseProductionSamplingOverride(
    JSON.parse(readFileSync(path, 'utf8')) as unknown
  );
}
