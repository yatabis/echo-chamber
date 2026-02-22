/**
 * 2つのベクトル間のコサイン類似度を計算する
 *
 * @param a - 1つ目のベクトル
 * @param b - 2つ目のベクトル
 * @returns 類似度 (-1.0 ~ 1.0)。ゼロベクトルまたは空配列の場合は0.0を返す
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) {
    return 0.0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  // ゼロベクトルの場合
  if (normA === 0 || normB === 0) {
    return 0.0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
