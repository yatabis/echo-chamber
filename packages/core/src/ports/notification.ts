import type { ChatChannel } from './chat';

export interface NotificationPreview {
  messageId: string;
  user: string;
  message: string;
  createdAt: string;
}

export interface ChannelNotificationSummary {
  channel: ChatChannel;
  unreadCount: number;
  latestMessagePreview: NotificationPreview | null;
}

export interface NotificationPort {
  getNotificationSummary(): Promise<ChannelNotificationSummary[]>;
}
