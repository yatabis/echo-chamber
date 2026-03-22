import { z } from 'zod';

import { createToolResultSchema, defineToolSpecification } from './shared';

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
    '思考プロセスを終了し、エージェントループを停止する。現在のサイクルを正常終了する唯一の方法であり、十分な情報を得た、または現在のサイクルで行うべきことが完了したと判断した場合に必ず呼び出せ。',
  parameters: {
    reason: z.string().describe('思考を終了する理由'),
    next_wake_at: z
      .string()
      .optional()
      .describe(
        '次回起動の目安時刻（ISO 8601形式、例: 2025-01-01T12:00:00Z）。特定のイベントを待つ場合や、一定時間後に再開したい場合に指定する。（ただしこれは目安であり、必ずしもその時刻に起動されることを保証するものではない）'
      ),
  },
  outputSchema: createToolResultSchema({}),
});
