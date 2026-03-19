import {
  searchMemoryToolSpec,
  storeMemoryToolSpec,
} from '@echo-chamber/core/agent/tools/memory';
import { getErrorMessage } from '@echo-chamber/core/utils/error';

import { Tool } from './index';

export const storeMemoryFunction = new Tool(
  storeMemoryToolSpec.name,
  storeMemoryToolSpec.description,
  storeMemoryToolSpec.parameters,
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

export const searchMemoryFunction = new Tool(
  searchMemoryToolSpec.name,
  searchMemoryToolSpec.description,
  searchMemoryToolSpec.parameters,
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
