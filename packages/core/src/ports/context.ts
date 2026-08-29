import type { Emotion } from '../echo/types';

/** Dashboard が表示する既存の永続 Context snapshot。 */
export interface ContextSnapshot {
  content: string;
  emotion: Emotion;
  createdAt: string;
  updatedAt: string;
}
