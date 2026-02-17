import { formatDatetime } from '../../../utils/datetime';

import type { Note } from '../../types';
import type { FC } from 'hono/jsx';

export const NoteList: FC<{
  id: string;
  notes: Note[];
  noteQuery: string;
}> = async ({ id, notes, noteQuery }) => {
  const hasQuery = noteQuery.length > 0;

  return (
    <div>
      <form
        method="get"
        action={`/${id}`}
        style={{
          display: 'flex',
          gap: '8px',
          marginBottom: '10px',
        }}
      >
        <input
          type="search"
          name="noteQuery"
          value={noteQuery}
          placeholder="ノート検索"
          style={{
            flex: 1,
            minWidth: 0,
            borderRadius: '10px',
            border: '1px solid #1f2a37',
            background: '#0f1520',
            color: '#e6edf3',
            padding: '10px 12px',
          }}
        />
        <button className="btn" type="submit">
          検索
        </button>
      </form>

      {hasQuery && (
        <div className="small muted" style={{ marginBottom: '8px' }}>
          query: <span className="mono">{noteQuery}</span>
        </div>
      )}

      {notes.length === 0 ? (
        <p className="muted small">
          {hasQuery ? 'No notes matched.' : 'No notes stored.'}
        </p>
      ) : (
        <div className="list">
          {notes.map(async (note) => (
            <article className="item" key={note.id}>
              <div>
                <h3>{note.title}</h3>
                <div
                  className="small muted mono"
                  style={{ marginBottom: '6px' }}
                >
                  updated: {formatDatetime(new Date(note.updatedAt))}
                </div>
                <div className="small" style={{ whiteSpace: 'pre-wrap' }}>
                  {note.content}
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
};
