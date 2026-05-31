import { searchMemoryToolSpec, storeMemoryToolSpec } from '../tools/memory';

import { createToolErrorResult, Tool } from './tool';

export const storeMemoryTool = new Tool(
  storeMemoryToolSpec,
  async ({ content, type, emotion }, ctx) => {
    try {
      await ctx.memory.store(content, emotion, type);
      return { success: true };
    } catch (error) {
      return createToolErrorResult('Failed to store memory', error);
    }
  }
);

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
