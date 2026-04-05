import {
  addReactionToChatMessageToolSpec,
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
} from './chat';
import { searchMemoryToolSpec, storeMemoryToolSpec } from './memory';
import {
  createNoteToolSpec,
  deleteNoteToolSpec,
  getNoteToolSpec,
  listNotesToolSpec,
  searchNotesToolSpec,
  updateNoteToolSpec,
} from './note';
import { finishThinkingToolSpec, thinkDeeplyToolSpec } from './thinking';
import {
  getZennArticleToolSpec,
  listTrendingZennArticlesToolSpec,
} from './zenn';

export const canonicalToolSpecifications = [
  checkNotificationsToolSpec,
  readChatMessagesToolSpec,
  sendChatMessageToolSpec,
  addReactionToChatMessageToolSpec,
  storeMemoryToolSpec,
  searchMemoryToolSpec,
  createNoteToolSpec,
  listNotesToolSpec,
  getNoteToolSpec,
  searchNotesToolSpec,
  updateNoteToolSpec,
  deleteNoteToolSpec,
  listTrendingZennArticlesToolSpec,
  getZennArticleToolSpec,
  thinkDeeplyToolSpec,
  finishThinkingToolSpec,
] as const;
