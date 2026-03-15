export interface ThoughtLogPort {
  send(content: string): Promise<void>;
}
