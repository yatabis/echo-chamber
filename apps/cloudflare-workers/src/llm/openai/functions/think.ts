import { thinkDeeplyToolSpec } from '@echo-chamber/core/agent/tools/thinking';

import { Tool } from '.';

export const thinkDeeplyFunction = new Tool(
  thinkDeeplyToolSpec.name,
  thinkDeeplyToolSpec.description,
  thinkDeeplyToolSpec.parameters,
  () => {
    // LLM が考えるだけで処理は何もない
    return {
      success: true,
    };
  }
);
