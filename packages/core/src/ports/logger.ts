import type { LogContext, LogLevel } from '../types/logger';

export interface LoggerPort {
  log(level: LogLevel, message: string, context?: LogContext): Promise<void>;
  debug(message: string, context?: LogContext): Promise<void>;
  info(message: string, context?: LogContext): Promise<void>;
  warn(message: string, context?: LogContext): Promise<void>;
  error(message: string, error?: Error): Promise<void>;
}
