import type { Note } from '@echo-chamber/core';
import { getErrorMessage } from '@echo-chamber/core';
import {
  createNoteToolSpec,
  deleteNoteToolSpec,
  getNoteToolSpec,
  listNotesToolSpec,
  searchNotesToolSpec,
  updateNoteToolSpec,
} from '@echo-chamber/core/agent/tools/note';

import { Tool } from './index';

type NoteSummary = Pick<Note, 'id' | 'title' | 'createdAt' | 'updatedAt'>;

export const createNoteFunction = new Tool(
  createNoteToolSpec.name,
  createNoteToolSpec.description,
  createNoteToolSpec.parameters,
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
  listNotesToolSpec.name,
  listNotesToolSpec.description,
  listNotesToolSpec.parameters,
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
  getNoteToolSpec.name,
  getNoteToolSpec.description,
  getNoteToolSpec.parameters,
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
  searchNotesToolSpec.name,
  searchNotesToolSpec.description,
  searchNotesToolSpec.parameters,
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
  updateNoteToolSpec.name,
  updateNoteToolSpec.description,
  updateNoteToolSpec.parameters,
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
  deleteNoteToolSpec.name,
  deleteNoteToolSpec.description,
  deleteNoteToolSpec.parameters,
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
