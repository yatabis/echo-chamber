import { getErrorMessage } from '../../utils/error';
import {
  createNoteToolSpec,
  deleteNoteToolSpec,
  getNoteToolSpec,
  listNotesToolSpec,
  searchNotesToolSpec,
  updateNoteToolSpec,
} from '../tools/note';

import { Tool } from './tool';

import type { Note } from '../../echo/types';

type NoteSummary = Pick<Note, 'id' | 'title' | 'createdAt' | 'updatedAt'>;

export const createNoteTool = new Tool(
  createNoteToolSpec,
  async ({ title, content }, ctx) => {
    try {
      const note = await ctx.notes.create({ title, content });
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

export const listNotesTool = new Tool(listNotesToolSpec, async (_, ctx) => {
  try {
    const notes = await ctx.notes.list();
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
});

export const getNoteTool = new Tool(getNoteToolSpec, async ({ id }, ctx) => {
  try {
    const note = await ctx.notes.get(id);
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
});

export const searchNotesTool = new Tool(
  searchNotesToolSpec,
  async ({ query }, ctx) => {
    try {
      const notes = await ctx.notes.search(query);
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

export const updateNoteTool = new Tool(
  updateNoteToolSpec,
  async ({ id, title, content }, ctx) => {
    if (title === undefined && content === undefined) {
      return {
        success: false,
        error: 'Either title or content is required',
      };
    }

    try {
      const note = await ctx.notes.update(id, { title, content });
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

export const deleteNoteTool = new Tool(
  deleteNoteToolSpec,
  async ({ id }, ctx) => {
    try {
      const deleted = await ctx.notes.delete(id);
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
