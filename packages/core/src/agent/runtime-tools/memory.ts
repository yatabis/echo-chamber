import { getErrorMessage } from '../../utils/error';
import { searchMemoryToolSpec, storeMemoryToolSpec } from '../tools/memory';

import { Tool } from './tool';

export const storeMemoryTool = new Tool(
  storeMemoryToolSpec,
  async ({ content, type, emotion }, ctx) => {
    try {
      await ctx.memory.store(content, emotion, type);
      return { success: true };
    } catch (error) {
      await ctx.logger.error(`Error storing memory: ${getErrorMessage(error)}`);
      return {
        success: false,
        error: 'Failed to store memory',
      };
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
      await ctx.logger.error(
        `Error searching memory: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to search memory',
      };
    }
  }
);
