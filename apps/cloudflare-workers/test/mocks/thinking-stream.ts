import { vi } from 'vitest';

import type { ThinkingStream } from '@echo-chamber/core/utils/thinking-stream';

export const mockThinkingStream = {
  send: vi.fn().mockResolvedValue(undefined),
} as unknown as ThinkingStream;
