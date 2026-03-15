export interface ClockPort {
  now(): Date;
}

export interface SchedulerPort {
  getNextWakeAt(): Promise<Date | null>;
  scheduleWakeAt(at: Date): Promise<void>;
  clearWake(): Promise<void>;
}
