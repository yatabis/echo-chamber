export interface ChatChannel {
  key: string;
  displayName: string;
  description?: string;
}

export interface ChatMessageReaction {
  emoji: string | null;
  me: boolean;
}

export interface ChatMessage {
  messageId: string;
  user: string;
  message: string;
  createdAt: string;
  reactions: ChatMessageReaction[];
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
