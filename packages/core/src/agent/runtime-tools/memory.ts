import { searchMemoryToolSpec, storeMemoryToolSpec } from '../tools/memory';

import { createToolErrorResult, Tool } from './tool';

/** Main が明示的に選択した記憶を永続化する runtime tool。 */
export const storeMemoryTool = new Tool(
  storeMemoryToolSpec,
  async ({ content, type }, ctx) => {
    try {
      await ctx.memory.store(content, type);
      return { success: true };
    } catch (error) {
      return createToolErrorResult('Failed to store memory', error);
    }
  }
);

/** Main が明示的に必要とした記憶を検索する runtime tool。 */
export const searchMemoryTool = new Tool(
  searchMemoryToolSpec,
  async ({ query, type }, ctx) => {
    try {
      const results = await ctx.memory.search(query, type);
      return {
        success: true,
        results: results.map(({ content, type, emotion, createdAt }) => ({
          content,
          type,
          emotion,
          createdAt,
        })),
      };
    } catch (error) {
      return createToolErrorResult('Failed to search memory', error);
    }
  }
);
