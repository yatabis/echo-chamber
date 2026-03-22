import { finishThinkingToolSpec } from '../tools/thinking';

import { Tool } from './tool';

export const finishThinkingTool = new Tool(finishThinkingToolSpec, () => {
  return {
    success: true,
  };
});
