import { z } from 'zod';

import {
  MAX_NOTE_CONTENT_LENGTH,
  MAX_NOTE_QUERY_LENGTH,
  MAX_NOTE_TITLE_LENGTH,
} from '../../../echo/note-system';
import { getErrorMessage } from '../../../utils/error';

import { Tool } from './index';

import type { Note } from '../../../echo/types';

const noteIdSchema = z.string().min(1).trim().describe('ノートID。例: note-12');

const noteTitleSchema = z
  .string()
  .min(1)
  .max(MAX_NOTE_TITLE_LENGTH)
  .trim()
  .describe(`ノートタイトル。最大${MAX_NOTE_TITLE_LENGTH}文字。`);

const noteContentSchema = z
  .string()
  .min(1)
  .max(MAX_NOTE_CONTENT_LENGTH)
  .trim()
  .describe(`ノート本文。最大${MAX_NOTE_CONTENT_LENGTH}文字。`);

const noteQuerySchema = z
  .string()
  .min(1)
  .max(MAX_NOTE_QUERY_LENGTH)
  .trim()
  .describe(
    `検索クエリ。title/contentに対して部分一致で検索する。最大${MAX_NOTE_QUERY_LENGTH}文字。`
  );

type NoteSummary = Pick<Note, 'id' | 'title' | 'createdAt' | 'updatedAt'>;

export const createNoteFunction = new Tool(
  'create_note',
  '新しいノートを作成する。タイトルと本文を保存し、作成日時と更新日時を自動で設定する。',
  {
    title: noteTitleSchema,
    content: noteContentSchema,
  },
  async ({ title, content }, ctx) => {
    try {
      const note = await ctx.noteSystem.createNote({ title, content });
      return {
        success: true,
        note,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error creating note: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);

export const listNotesFunction = new Tool(
  'list_notes',
  '保存済みノートを一覧取得する。更新日時の降順で返す。本文は含めずメタ情報のみ返す。',
  {},
  async (_, ctx) => {
    try {
      const notes = await ctx.noteSystem.listNotes();
      const summaries: NoteSummary[] = notes.map((note) => ({
        id: note.id,
        title: note.title,
        createdAt: note.createdAt,
        updatedAt: note.updatedAt,
      }));
      return {
        success: true,
        notes: summaries,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error listing notes: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);

export const getNoteFunction = new Tool(
  'get_note',
  'IDを指定してノート1件を取得する。本文を含む完全なノート情報を返す。',
  {
    id: noteIdSchema,
  },
  async ({ id }, ctx) => {
    try {
      const note = await ctx.noteSystem.getNote(id);
      if (note === null) {
        return {
          success: false,
          error: 'Note not found',
        };
      }
      return {
        success: true,
        note,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error getting note: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);

export const searchNotesFunction = new Tool(
  'search_notes',
  'ノートを検索する。タイトルと本文に対して大文字小文字を区別しない部分一致で検索する。',
  {
    query: noteQuerySchema,
  },
  async ({ query }, ctx) => {
    try {
      const notes = await ctx.noteSystem.searchNotes(query);
      return {
        success: true,
        notes,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error searching notes: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);

export const updateNoteFunction = new Tool(
  'update_note',
  '既存ノートを更新する。title または content の少なくとも一方を指定する。',
  {
    id: noteIdSchema,
    title: noteTitleSchema.optional(),
    content: noteContentSchema.optional(),
  },
  async ({ id, title, content }, ctx) => {
    if (title === undefined && content === undefined) {
      return {
        success: false,
        error: 'Either title or content is required',
      };
    }

    try {
      const note = await ctx.noteSystem.updateNote(id, { title, content });
      if (note === null) {
        return {
          success: false,
          error: 'Note not found',
        };
      }
      return {
        success: true,
        note,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error updating note: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);

export const deleteNoteFunction = new Tool(
  'delete_note',
  '既存ノートを削除する。',
  {
    id: noteIdSchema,
  },
  async ({ id }, ctx) => {
    try {
      const deleted = await ctx.noteSystem.deleteNote(id);
      if (!deleted) {
        return {
          success: false,
          error: 'Note not found',
        };
      }
      return {
        success: true,
      };
    } catch (error) {
      const message = getErrorMessage(error);
      await ctx.logger.error(`Error deleting note: ${message}`);
      return {
        success: false,
        error: message,
      };
    }
  }
);
