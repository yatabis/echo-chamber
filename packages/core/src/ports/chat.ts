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
  readMessages(limit: number): Promise<ChatMessage[]>;
  sendMessage(message: string): Promise<void>;
  addReaction(messageId: string, reaction: string): Promise<void>;
}
