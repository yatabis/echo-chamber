import { z } from 'zod';

import { Tool } from '.';

export const thinkDeeplyFunction = new Tool(
  'think_deeply',
  'あるトピックについて深く考え、洞察を得る。新しい情報を取得したりデータベースを変更したりはせず、単に思考をログに追加するだけである。複雑な推論やキャッシュメモリが必要な場合に使用せよ。',
  {
    thought: z.string().describe('深く考える思考内容'),
  },
  () => {
    // LLM が考えるだけで処理は何もない
    return {
      success: true,
    };
  }
);
