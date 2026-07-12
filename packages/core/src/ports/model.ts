/**
 * Provider 非依存の会話メッセージ role。
 * 現時点では agent 起動時 prompt と通常会話に必要な role だけを持つ。
 */
export type ModelMessageRole = 'system' | 'developer' | 'user' | 'assistant';

/**
 * 画像入力の detail 指定。
 * provider ごとの差異は adapter 側で吸収する。
 */
export type ModelImageDetail = 'low' | 'high' | 'auto' | 'original';

/**
 * モデルへ渡すテキスト part。
 */
export interface ModelTextContentPart {
  type: 'text';
  text: string;
}

/**
 * モデルへ渡す画像 part。
 * URL は provider が取得できる fully qualified URL または data URL を想定する。
 */
export interface ModelImageContentPart {
  type: 'image';
  imageUrl: string;
  detail?: ModelImageDetail;
}

/**
 * provider 非依存の message content part。
 */
export type ModelMessageContentPart =
  | ModelTextContentPart
  | ModelImageContentPart;

/**
 * モデルへそのまま渡せるメッセージ content。
 * 既存のテキスト専用経路は string のまま扱い、画像を含む場合だけ part 配列にする。
 */
export type ModelMessageContent = string | readonly ModelMessageContentPart[];

/**
 * モデルへそのまま渡せる会話メッセージ。
 */
export interface ModelMessage {
  role: ModelMessageRole;
  content: ModelMessageContent;
}

/**
 * provider adapter が理解できるツール契約。
 * `inputSchema` / `outputSchema` の具体型は provider ごとに異なるため、
 * `core` では unknown として保持する。
 */
export interface ModelToolContract {
  name: string;
  description: string;
  inputSchema: unknown;
  outputSchema?: unknown;
  strict?: boolean;
}

/**
 * モデルが要求した tool call。
 * `input` は tool 実行前の生 JSON 文字列をそのまま保持する。
 */
export interface ModelToolCall {
  type: 'tool_call';
  callId: string;
  toolName: string;
  input: string;
}

/**
 * tool 実行結果を次ターンへ返すための input item。
 */
export interface ModelToolResult {
  type: 'tool_result';
  callId: string;
  output: string;
}

/**
 * モデルの自然言語出力。
 */
export interface ModelOutputMessage {
  type: 'message';
  role: 'assistant';
  content: string;
}

/**
 * provider から返る output item の正規化表現。
 */
export type ModelOutputItem = ModelToolCall | ModelOutputMessage;

/**
 * provider へ渡す input item の正規化表現。
 */
export type ModelInputItem = ModelMessage | ModelToolCall | ModelToolResult;

/**
 * provider 固有の usage を `core` で扱いやすい形に正規化した集計値。
 */
export interface ModelUsage {
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  uncachedInputTokens: number;
  totalInputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  totalTokens: number;
}

/**
 * 1ターンぶんのモデル呼び出し要求。
 */
export interface ModelRequest {
  input: ModelInputItem[];
  tools: ModelToolContract[];
  previousResponseToken?: string;
  turnIndex?: number;
}

/**
 * 1ターンぶんのモデル応答。
 */
export interface ModelResponse {
  output: ModelOutputItem[];
  usage: ModelUsage;
  responseToken?: string;
}

/**
 * 実際の LLM provider を隠蔽する port。
 * session loop はこの interface だけに依存して進行する。
 */
export interface ModelPort {
  generate(request: ModelRequest): Promise<ModelResponse>;
}
