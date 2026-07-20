import type { RuntimeGenerationProfile } from './types';

/** 再現可能なモデル差・学習前後差を見るための決定的条件。 */
export const CONTROLLED_GREEDY_PROFILE: RuntimeGenerationProfile = {
  id: 'controlled-greedy',
  description:
    'Deterministic comparison profile: temperature=0, top-p=1, top-k=1.',
  temperature: 0,
  topP: 1,
  topK: 1,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 0,
  maxTokensPerTurn: 1_024,
  enableThinking: false,
};

/**
 * 2026-07-20時点のQwen3.6公式non-thinking推奨値かつ
 * E.C.H.O.本番のQwen3.6 non-thinking sampling値を写した条件。
 * max tokensだけは暴走時の評価コストを制限するため1,024に抑える。
 */
export const PRODUCTION_SAMPLING_PROFILE: RuntimeGenerationProfile = {
  id: 'production-sampling',
  description:
    'Qwen3.6 official non-thinking and deployment sampling: temperature=0.7, top-p=0.8, top-k=20, min-p=0, presence-penalty=1.5, repetition-penalty=1; evaluation output cap is 1,024 tokens per turn.',
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 1.5,
  maxTokensPerTurn: 1_024,
  enableThinking: false,
};
