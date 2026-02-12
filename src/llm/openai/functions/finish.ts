import { z } from 'zod';

import { Tool } from '.';

export const finishThinkingFunction = new Tool(
  'finish_thinking',
  '思考プロセスを終了し、エージェントループを停止する。十分な情報を得た、または現在のサイクルで行うべきことが完了したと判断した場合に呼び出せ。',
  {
    reason: z.string().describe('思考を終了する理由'),
    next_wake_at: z
      .string()
      .optional()
      .describe(
        '次回起動の目安時刻（ISO 8601形式、例: 2025-01-01T12:00:00Z）。特定のイベントを待つ場合や、一定時間後に再開したい場合に指定する。（ただしこれは目安であり、必ずしもその時刻に起動されることを保証するものではない）'
      ),
  },
  () => {
    return {
      success: true,
    };
  }
);
