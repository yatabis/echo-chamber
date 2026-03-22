import { z } from 'zod';

import { emotionSchema } from '../../echo/schemas';

import { createToolResultSchema, defineToolSpecification } from './shared';

const MAX_SESSION_RECORD_CONTENT_LENGTH = 500;

export const thinkDeeplyToolSpec = defineToolSpecification({
  name: 'think_deeply',
  description:
    'あるトピックについて深く考え、洞察を得る。新しい情報を取得したりデータベースを変更したりはせず、単に思考をログに追加するだけである。複雑な推論やキャッシュメモリが必要な場合に使用せよ。',
  parameters: {
    thought: z.string().describe('深く考える思考内容'),
  },
  outputSchema: createToolResultSchema({}),
});

export const finishThinkingToolSpec = defineToolSpecification({
  name: 'finish_thinking',
  description:
    '思考プロセスを終了し、エージェントループを停止する。現在のサイクルを正常終了する唯一の方法であり、十分な情報を得た、または現在のサイクルで行うべきことが完了したと判断した場合に必ず呼び出せ。次回起動時の再開に必要な要点を session_record に残せ。',
  parameters: {
    reason: z.string().describe('思考を終了する理由'),
    next_wake_at: z
      .string()
      .optional()
      .describe(
        '次回起動の目安時刻（ISO 8601形式、例: 2025-01-01T12:00:00Z）。特定のイベントを待つ場合や、一定時間後に再開したい場合に指定する。（ただしこれは目安であり、必ずしもその時刻に起動されることを保証するものではない）'
      ),
    session_record: z
      .object({
        content: z
          .string()
          .min(1)
          .max(MAX_SESSION_RECORD_CONTENT_LENGTH)
          .trim()
          .describe(
            `今回のセッションで判明したこと、行ったこと、次回再開時に参照すべき要点の記録。最大${MAX_SESSION_RECORD_CONTENT_LENGTH}文字。`
          ),
        emotion: emotionSchema.describe('このセッション記録に付随する感情'),
      })
      .describe(
        '次回起動時の冒頭コンテキストに注入するためのセッション記録。メモリーに近い形式で、内容と感情を簡潔にまとめること。'
      ),
  },
  outputSchema: createToolResultSchema({}),
});
