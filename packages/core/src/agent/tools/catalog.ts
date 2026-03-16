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
  thinkDeeplyToolSpec,
  finishThinkingToolSpec,
] as const;
