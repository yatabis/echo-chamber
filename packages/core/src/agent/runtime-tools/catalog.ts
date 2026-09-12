import {
  addReactionToChatMessageTool,
  checkNotificationsTool,
  readChatMessagesTool,
  sendChatMessageTool,
} from './chat';
import { finishThinkingTool } from './finish';
import { searchMemoryTool, storeMemoryTool } from './memory';
import {
  createNoteTool,
  deleteNoteTool,
  getNoteTool,
  listNotesTool,
  searchNotesTool,
  updateNoteTool,
} from './note';
import { thinkDeeplyTool } from './think';
import { readWebPageTool } from './web';
import { getZennArticleTool, listTrendingZennArticlesTool } from './zenn';

export const canonicalRuntimeTools = [
  checkNotificationsTool,
  readChatMessagesTool,
  sendChatMessageTool,
  addReactionToChatMessageTool,
  storeMemoryTool,
  searchMemoryTool,
  createNoteTool,
  listNotesTool,
  getNoteTool,
  searchNotesTool,
  updateNoteTool,
  deleteNoteTool,
  listTrendingZennArticlesTool,
  getZennArticleTool,
  readWebPageTool,
  thinkDeeplyTool,
  finishThinkingTool,
] as const;
