import {
  MAX_NOTE_CONTENT_LENGTH,
  MAX_NOTE_QUERY_LENGTH,
  MAX_NOTE_TITLE_LENGTH,
} from '@echo-chamber/core/echo/note-constraints';

/**
 * ノート title を trim し、必須・長さ制約を検証する。
 *
 * @param title 入力 title
 * @returns 正規化後の title
 */
export function validateTitle(title: string): string {
  const normalizedTitle = title.trim();
  if (normalizedTitle.length === 0) {
    throw new Error('Title is required');
  }
  if (normalizedTitle.length > MAX_NOTE_TITLE_LENGTH) {
    throw new Error(
      `Title must be at most ${MAX_NOTE_TITLE_LENGTH} characters`
    );
  }
  return normalizedTitle;
}

/**
 * ノート content を trim し、必須・長さ制約を検証する。
 *
 * @param content 入力 content
 * @returns 正規化後の content
 */
export function validateContent(content: string): string {
  const normalizedContent = content.trim();
  if (normalizedContent.length === 0) {
    throw new Error('Content is required');
  }
  if (normalizedContent.length > MAX_NOTE_CONTENT_LENGTH) {
    throw new Error(
      `Content must be at most ${MAX_NOTE_CONTENT_LENGTH} characters`
    );
  }
  return normalizedContent;
}

/**
 * 検索クエリを trim し、必須・長さ制約を検証する。
 *
 * @param query 入力 query
 * @returns 正規化後の query
 */
export function validateQuery(query: string): string {
  const normalizedQuery = query.trim();
  if (normalizedQuery.length === 0) {
    throw new Error('Query is required');
  }
  if (normalizedQuery.length > MAX_NOTE_QUERY_LENGTH) {
    throw new Error(
      `Query must be at most ${MAX_NOTE_QUERY_LENGTH} characters`
    );
  }
  return normalizedQuery;
}
