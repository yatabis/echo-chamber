import { finishThinkingToolSpec } from '@echo-chamber/core/agent/tools/thinking';

import { Tool } from '.';

export const finishThinkingFunction = new Tool(
  finishThinkingToolSpec.name,
  finishThinkingToolSpec.description,
  finishThinkingToolSpec.parameters,
  () => {
    return {
      success: true,
    };
  }
);
