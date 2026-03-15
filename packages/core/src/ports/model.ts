export type ModelMessageRole =
  | 'system'
  | 'developer'
  | 'user'
  | 'assistant'
  | 'tool';

export interface ModelMessage {
  role: ModelMessageRole;
  content: string;
}

export interface ModelToolContract {
  name: string;
  description: string;
  inputSchema: unknown;
  outputSchema?: unknown;
  strict?: boolean;
}

export interface ModelToolCall {
  type: 'tool_call';
  callId: string;
  toolName: string;
  input: string;
}

export interface ModelOutputMessage {
  type: 'message';
  role: 'assistant';
  content: string;
}

export type ModelOutputItem = ModelToolCall | ModelOutputMessage;

export interface ModelUsage {
  cachedInputTokens: number;
  uncachedInputTokens: number;
  totalInputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  totalTokens: number;
}

export interface ModelRequest {
  messages: ModelMessage[];
  tools: ModelToolContract[];
  previousResponseToken?: string;
}

export interface ModelResponse {
  output: ModelOutputItem[];
  usage: ModelUsage;
  responseToken?: string;
}

export interface ModelPort {
  generate(request: ModelRequest): Promise<ModelResponse>;
}
