import type { Note } from '../echo/types';

export interface NotePort {
  list(): Promise<Note[]>;
  get(id: string): Promise<Note | null>;
  search(query: string): Promise<Note[]>;
  create(input: { title: string; content: string }): Promise<Note>;
  update(
    id: string,
    patch: { title?: string; content?: string }
  ): Promise<Note | null>;
  delete(id: string): Promise<boolean>;
}
