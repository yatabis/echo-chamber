import { describe, it, expect } from 'vitest';

import { cosineSimilarity } from './vector';

describe('cosineSimilarity', () => {
  describe('基本的なケース', () => {
    it('同一ベクトルの場合は1.0を返す', () => {
      const a = [1, 2, 3];
      const b = [1, 2, 3];
      expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
    });

    it('直交ベクトルの場合は0.0を返す', () => {
      const a = [1, 0];
      const b = [0, 1];
      expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
    });

    it('逆方向ベクトルの場合は-1.0を返す', () => {
      const a = [1, 2, 3];
      const b = [-1, -2, -3];
      expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
    });
  });

  describe('正規化されていないベクトル', () => {
    it('スケールが異なっても同一方向なら1.0を返す', () => {
      const a = [1, 2, 3];
      const b = [2, 4, 6];
      expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
    });

    it('スケールが異なっても類似度は同じ', () => {
      const a = [1, 0];
      const b = [1, 1];
      const c = [10, 10];
      expect(cosineSimilarity(a, b)).toBeCloseTo(cosineSimilarity(a, c), 5);
    });
  });

  describe('高次元ベクトル（1536次元）', () => {
    it('1536次元のベクトルで正しく計算できる', () => {
      // すべて同じ値のベクトル同士は同一方向
      const a = new Array<number>(1536).fill(0.1);
      const b = new Array<number>(1536).fill(0.1);
      expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
    });

    it('1536次元の異なるベクトルで類似度が計算できる', () => {
      const a = new Array<number>(1536)
        .fill(0)
        .map((_, i) => (i % 2 === 0 ? 1 : 0));
      const b = new Array<number>(1536)
        .fill(0)
        .map((_, i) => (i % 2 === 1 ? 1 : 0));
      // 完全に直交（768個の1が交互に配置）
      expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
    });
  });

  describe('エッジケース', () => {
    it('ゼロベクトルの場合は0.0を返す', () => {
      const a = [0, 0, 0];
      const b = [1, 2, 3];
      expect(cosineSimilarity(a, b)).toBe(0.0);
    });

    it('両方ゼロベクトルの場合は0.0を返す', () => {
      const a = [0, 0, 0];
      const b = [0, 0, 0];
      expect(cosineSimilarity(a, b)).toBe(0.0);
    });

    it('空配列の場合は0.0を返す', () => {
      const a: number[] = [];
      const b: number[] = [];
      expect(cosineSimilarity(a, b)).toBe(0.0);
    });
  });
});
