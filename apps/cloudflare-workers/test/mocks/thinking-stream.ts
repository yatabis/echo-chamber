import { vi } from 'vitest';

import type { ThinkingStream } from '@echo-chamber/core/utils/thinking-stream';

export const mockThinkingStreamSend = vi.fn().mockResolvedValue(undefined);

export const mockThinkingStream = {
  send: mockThinkingStreamSend,
} as unknown as ThinkingStream;
