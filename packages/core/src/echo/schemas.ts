import { z } from 'zod';

export const emotionSchema = z.object({
  valence: z
    .number()
    .min(-1.0)
    .max(1.0)
    .describe('感情価（-1.0：ネガティブ 〜 1.0：ポジティブ）'),
  arousal: z
    .number()
    .min(0.0)
    .max(1.0)
    .describe('覚醒度（0.0：穏やか 〜 1.0：興奮）'),
  labels: z
    .array(z.string())
    .describe('感情ラベル（例: "楽しい", "悲しい", "驚き", "知的好奇心"）'),
});
