import { readFileSync } from 'node:fs';

import type { LocalEvaluationTarget } from './types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requiredString(
  record: Record<string, unknown>,
  key: string,
  index: number
): string {
  const value = record[key];
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Evaluation target ${index} has invalid ${key}`);
  }
  return value;
}

/**
 * 外部JSONから受け取った評価対象モデル一覧を検証する。
 *
 * 各対象はRapid-MLXが直接起動できるモデルディレクトリを指す。この関数は
 * ファイルシステムやモデル形式を検査せず、一覧の構造と識別子の一意性だけを
 * 保証する。
 *
 * @param value JSONから復元した未検証の値。
 * @returns 入力順を維持した評価対象モデル一覧。
 * @throws 一覧が空、必須文字列が不正、またはIDが重複している場合。
 */
export function parseEvaluationTargets(
  value: unknown
): LocalEvaluationTarget[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('Evaluation targets must be a non-empty array');
  }

  const targets = value.map((target, index): LocalEvaluationTarget => {
    if (!isRecord(target)) {
      throw new Error(`Evaluation target ${index} must be an object`);
    }
    return {
      id: requiredString(target, 'id', index),
      displayName: requiredString(target, 'displayName', index),
      modelPath: requiredString(target, 'modelPath', index),
      servedModelName: requiredString(target, 'servedModelName', index),
    };
  });

  const ids = new Set(targets.map((target) => target.id));
  if (ids.size !== targets.length) {
    throw new Error('Evaluation targets contain duplicate ids');
  }
  return targets;
}

/**
 * 指定したJSONファイルを読み込み、評価対象モデル一覧として検証する。
 *
 * @param path 評価対象一覧JSONの絶対パスまたは作業ディレクトリ相対パス。
 * @returns 入力ファイルに記載された順序の評価対象モデル一覧。
 * @throws ファイルを読めない、JSONが不正、または一覧の検証に失敗した場合。
 */
export function loadEvaluationTargets(path: string): LocalEvaluationTarget[] {
  return parseEvaluationTargets(
    JSON.parse(readFileSync(path, 'utf8')) as unknown
  );
}
