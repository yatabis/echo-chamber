import { thinkDeeplyToolSpec } from '../tools/thinking';

import { Tool } from './tool';

export const thinkDeeplyTool = new Tool(thinkDeeplyToolSpec, () => {
  // LLM が考えるだけで処理は何もない
  return {
    success: true,
  };
});
