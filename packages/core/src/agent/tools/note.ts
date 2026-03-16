import { z } from 'zod';

import {
  MAX_NOTE_CONTENT_LENGTH,
  MAX_NOTE_QUERY_LENGTH,
  MAX_NOTE_TITLE_LENGTH,
} from '../../echo/note-constraints';

import { createToolResultSchema, defineToolSpecification } from './shared';

const noteSchema = z.object({
  id: z.string(),
  title: z.string(),
  content: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

const noteSummarySchema = noteSchema.pick({
  id: true,
  title: true,
  createdAt: true,
  updatedAt: true,
});

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

export const createNoteToolSpec = defineToolSpecification({
  name: 'create_note',
  description:
    '新しいノートを作成する。タイトルと本文を保存し、作成日時と更新日時を自動で設定する。',
  parameters: {
    title: noteTitleSchema,
    content: noteContentSchema,
  },
  outputSchema: createToolResultSchema({
    note: noteSchema,
  }),
});

export const listNotesToolSpec = defineToolSpecification({
  name: 'list_notes',
  description:
    '保存済みノートを一覧取得する。更新日時の降順で返す。本文は含めずメタ情報のみ返す。',
  parameters: {},
  outputSchema: createToolResultSchema({
    notes: z.array(noteSummarySchema),
  }),
});

export const getNoteToolSpec = defineToolSpecification({
  name: 'get_note',
  description:
    'IDを指定してノート1件を取得する。本文を含む完全なノート情報を返す。',
  parameters: {
    id: noteIdSchema,
  },
  outputSchema: createToolResultSchema({
    note: noteSchema,
  }),
});

export const searchNotesToolSpec = defineToolSpecification({
  name: 'search_notes',
  description:
    'ノートを検索する。タイトルと本文に対して大文字小文字を区別しない部分一致で検索する。',
  parameters: {
    query: noteQuerySchema,
  },
  outputSchema: createToolResultSchema({
    notes: z.array(noteSchema),
  }),
});

export const updateNoteToolSpec = defineToolSpecification({
  name: 'update_note',
  description:
    '既存ノートを更新する。title または content の少なくとも一方を指定する。',
  parameters: {
    id: noteIdSchema,
    title: noteTitleSchema.optional(),
    content: noteContentSchema.optional(),
  },
  outputSchema: createToolResultSchema({
    note: noteSchema,
  }),
});

export const deleteNoteToolSpec = defineToolSpecification({
  name: 'delete_note',
  description: '既存ノートを削除する。',
  parameters: {
    id: noteIdSchema,
  },
  outputSchema: createToolResultSchema({}),
});
