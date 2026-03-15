export interface NotificationPreview {
  messageId: string;
  user: string;
  message: string;
  createdAt: string;
}

export interface NotificationSummary {
  unreadCount: number;
  latestMessagePreview: NotificationPreview | null;
}

export interface NotificationPort {
  getNotificationSummary(): Promise<NotificationSummary>;
}
