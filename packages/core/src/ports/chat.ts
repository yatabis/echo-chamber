export interface ChatChannel {
  key: string;
  displayName: string;
  description?: string;
}

export interface ChatMessageReaction {
  emoji: string | null;
  me: boolean;
}

/**
 * チャットメッセージに添付された画像。
 */
export interface ChatMessageImage {
  url: string;
  filename: string | null;
  contentType: string | null;
  width: number | null;
  height: number | null;
  size: number | null;
  description: string | null;
}

export interface ChatMessage {
  messageId: string;
  user: string;
  message: string;
  createdAt: string;
  reactions: ChatMessageReaction[];
  images: ChatMessageImage[];
}

export interface ChatPort {
  readMessages(channelKey: string, limit: number): Promise<ChatMessage[]>;
  sendMessage(channelKey: string, message: string): Promise<void>;
  addReaction(
    channelKey: string,
    messageId: string,
    reaction: string
  ): Promise<void>;
}
