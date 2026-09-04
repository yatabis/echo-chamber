import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

import {
  NATIVE_INFERENCE_PROTOCOL_VERSION,
  parseNativeWireEvent,
  type NativeCompletedEvent,
  type NativeGenerateCommand,
  type NativeOpenStateCommand,
  type NativeSnapshotCommand,
  type NativeSnapshotPublishedEvent,
  type NativeStateOpenedEvent,
  type NativeWireCommand,
  type NativeWireEvent,
} from './protocol';

import type { ChildProcessWithoutNullStreams } from 'node:child_process';

export { NATIVE_INFERENCE_PROTOCOL_VERSION } from './protocol';

const MAX_STDERR_CHARACTERS = 65_536;

type EventListener = (event: NativeWireEvent) => void;
type ErrorListener = (error: Error) => void;

/** Low-level command/event transport used by the native client. */
export interface NativeInferenceTransport {
  send(command: NativeWireCommand): Promise<void>;
  onEvent(listener: EventListener): () => void;
  onError(listener: ErrorListener): () => void;
  close(): Promise<void>;
}

/** Options for spawning the Rust native inference owner. */
export interface SpawnNativeInferenceOptions {
  binaryPath: string;
  modelDirectory: string;
  maxOutstandingRequests?: number;
  environment?: NodeJS.ProcessEnv;
}

/** Per-token callback for local UI or interruption integration. */
export type NativeTokenListener = (
  event: Extract<NativeWireEvent, { event: 'token' }>
) => void;

interface PendingGeneration {
  resolve(event: NativeCompletedEvent): void;
  reject(error: Error): void;
  onToken?: NativeTokenListener;
  listenerError?: Error;
}

/**
 * A token listener failed, but Native committed and reported the terminal
 * completion before cancellation took effect.
 *
 * The model adapter must accept `completedEvent` before surfacing
 * `listenerError`, keeping its process-local continuation state synchronized
 * with the resident owner.
 */
export class NativeTokenListenerCompletionError extends Error {
  /** Listener failure that requested cancellation. */
  readonly listenerError: Error;

  /** Authoritative completion emitted after Native committed state. */
  readonly completedEvent: NativeCompletedEvent;

  constructor(listenerError: Error, completedEvent: NativeCompletedEvent) {
    super(
      `native token listener failed after generation committed: ${listenerError.message}`
    );
    this.name = 'NativeTokenListenerCompletionError';
    this.listenerError = listenerError;
    this.completedEvent = completedEvent;
  }
}

type NativeLifecycleEvent =
  | NativeStateOpenedEvent
  | NativeSnapshotPublishedEvent;

interface PendingLifecycle {
  expectedEvent: NativeLifecycleEvent['event'];
  resolve(event: NativeLifecycleEvent): void;
  reject(error: Error): void;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: Error): void;
}

/**
 * Multiplexes typed requests over one resident native owner.
 *
 * Model state-transition policy remains in `NativeInferenceModel`; this
 * client owns only process readiness, request identity, token routing, and
 * terminal protocol failure.
 */
export class NativeInferenceClient {
  private readonly pending = new Map<string, PendingGeneration>();
  private readonly pendingLifecycle = new Map<string, PendingLifecycle>();
  private readonly readyDeferred =
    createDeferred<Extract<NativeWireEvent, { event: 'ready' }>>();
  private terminalError: Error | undefined;

  /**
   * @param transport NDJSON command/event transport
   */
  constructor(private readonly transport: NativeInferenceTransport) {
    transport.onEvent((event) => {
      this.handleEvent(event);
    });
    transport.onError((error) => {
      this.fail(error);
    });
  }

  /** Spawns `echo-inference serve-stdio` and returns its typed client. */
  static spawn(options: SpawnNativeInferenceOptions): NativeInferenceClient {
    return new NativeInferenceClient(
      StdioNativeInferenceTransport.spawn(options)
    );
  }

  /** Resolves after model/tokenizer admission and the native ready event. */
  async ready(): Promise<Extract<NativeWireEvent, { event: 'ready' }>> {
    return await this.readyDeferred.promise;
  }

  /** Sends one generation and resolves only after its state commits. */
  async generate(
    command: NativeGenerateCommand,
    onToken?: NativeTokenListener
  ): Promise<NativeCompletedEvent> {
    await this.ready();
    this.throwIfFailed();
    if (this.hasPendingRequest(command.request_id)) {
      throw new Error(
        `native request is already pending: ${command.request_id}`
      );
    }
    const deferred = createDeferred<NativeCompletedEvent>();
    this.pending.set(command.request_id, {
      resolve: (event): void => {
        deferred.resolve(event);
      },
      reject: (error): void => {
        deferred.reject(error);
      },
      ...(onToken === undefined ? {} : { onToken }),
    });
    try {
      await this.transport.send(command);
    } catch (error) {
      this.pending.delete(command.request_id);
      throw toError(error);
    }
    return await deferred.promise;
  }

  /** Opens one durable state root and restores its current payload if present. */
  async openState(
    command: NativeOpenStateCommand
  ): Promise<NativeStateOpenedEvent> {
    return await this.requestLifecycle(command, 'state_opened');
  }

  /** Atomically replaces one opened instance's current state payload. */
  async snapshot(
    command: NativeSnapshotCommand
  ): Promise<NativeSnapshotPublishedEvent> {
    return await this.requestLifecycle(command, 'snapshot_published');
  }

  /** Requests cancellation at the next native generated-token boundary. */
  async cancel(requestId: string): Promise<void> {
    this.throwIfFailed();
    await this.transport.send({ type: 'cancel', request_id: requestId });
  }

  /** Gracefully closes the resident owner after all prior commands. */
  async shutdown(): Promise<void> {
    if (this.terminalError !== undefined) {
      await this.transport.close();
      return;
    }
    await this.transport.send({ type: 'shutdown' });
    await this.transport.close();
  }

  private handleEvent(event: NativeWireEvent): void {
    if (event.event === 'ready') {
      this.handleReady(event);
      return;
    }
    if (event.event === 'token') {
      this.handleToken(event);
      return;
    }
    if (event.event === 'completed') {
      this.handleCompleted(event);
      return;
    }
    if (isLifecycleEvent(event)) {
      this.handleLifecycleEvent(event);
      return;
    }
    if (event.event === 'failed') {
      this.handleRequestFailure(event);
      return;
    }
    if (event.event === 'shutdown') {
      return;
    }
    this.handleCancellation(event);
  }

  private handleReady(
    event: Extract<NativeWireEvent, { event: 'ready' }>
  ): void {
    if (event.protocol_version !== NATIVE_INFERENCE_PROTOCOL_VERSION) {
      this.fail(
        new Error(
          `native inference protocol ${event.protocol_version} is incompatible with required ${NATIVE_INFERENCE_PROTOCOL_VERSION}`
        )
      );
      return;
    }
    this.readyDeferred.resolve(event);
  }

  private handleCompleted(event: NativeCompletedEvent): void {
    const pending = this.pending.get(event.request_id);
    this.pending.delete(event.request_id);
    if (pending?.listenerError !== undefined) {
      pending.reject(
        new NativeTokenListenerCompletionError(pending.listenerError, event)
      );
      return;
    }
    pending?.resolve(event);
  }

  private handleCancellation(event: NativeWireEvent): void {
    if (event.event !== 'cancelled') {
      return;
    }
    const pending = this.pending.get(event.request_id);
    this.pending.delete(event.request_id);
    pending?.reject(
      pending.listenerError ??
        new Error(`native request cancelled before commit: ${event.request_id}`)
    );
  }

  private handleToken(
    event: Extract<NativeWireEvent, { event: 'token' }>
  ): void {
    const pending = this.pending.get(event.request_id);
    if (pending?.onToken === undefined || pending.listenerError !== undefined) {
      return;
    }
    try {
      pending.onToken(event);
    } catch (error) {
      pending.listenerError = toError(error);
      delete pending.onToken;
      void this.cancel(event.request_id).catch((cancelError: unknown) => {
        this.fail(toError(cancelError));
      });
    }
  }

  private handleRequestFailure(
    event: Extract<NativeWireEvent, { event: 'failed' }>
  ): void {
    const error = new Error(
      `native inference ${event.phase} failed: ${event.error}`
    );
    if (event.request_id === undefined) {
      this.fail(error);
      return;
    }
    const pending = this.pending.get(event.request_id);
    this.pending.delete(event.request_id);
    pending?.reject(
      pending.listenerError === undefined
        ? error
        : new AggregateError(
            [pending.listenerError, error],
            `native request failed after token listener error: ${event.request_id}`
          )
    );
    const pendingLifecycle = this.pendingLifecycle.get(event.request_id);
    this.pendingLifecycle.delete(event.request_id);
    pendingLifecycle?.reject(error);
  }

  private handleLifecycleEvent(event: NativeLifecycleEvent): void {
    const pending = this.pendingLifecycle.get(event.request_id);
    this.pendingLifecycle.delete(event.request_id);
    if (pending === undefined) {
      return;
    }
    if (pending.expectedEvent !== event.event) {
      pending.reject(
        new Error(
          `native request ${event.request_id} returned ${event.event}, expected ${pending.expectedEvent}`
        )
      );
      return;
    }
    pending.resolve(event);
  }

  private async requestLifecycle<E extends NativeLifecycleEvent['event']>(
    command: NativeOpenStateCommand | NativeSnapshotCommand,
    expectedEvent: E
  ): Promise<Extract<NativeLifecycleEvent, { event: E }>> {
    await this.ready();
    this.throwIfFailed();
    if (this.hasPendingRequest(command.request_id)) {
      throw new Error(
        `native request is already pending: ${command.request_id}`
      );
    }
    const deferred =
      createDeferred<Extract<NativeLifecycleEvent, { event: E }>>();
    this.pendingLifecycle.set(command.request_id, {
      expectedEvent,
      resolve: (event): void => {
        deferred.resolve(event as Extract<NativeLifecycleEvent, { event: E }>);
      },
      reject: (error): void => {
        deferred.reject(error);
      },
    });
    try {
      await this.transport.send(command);
    } catch (error) {
      this.pendingLifecycle.delete(command.request_id);
      throw toError(error);
    }
    return await deferred.promise;
  }

  private hasPendingRequest(requestId: string): boolean {
    return this.pending.has(requestId) || this.pendingLifecycle.has(requestId);
  }

  private fail(error: Error): void {
    if (this.terminalError !== undefined) {
      return;
    }
    this.terminalError = error;
    this.readyDeferred.reject(error);
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
    for (const pending of this.pendingLifecycle.values()) {
      pending.reject(error);
    }
    this.pendingLifecycle.clear();
  }

  private throwIfFailed(): void {
    if (this.terminalError !== undefined) {
      throw this.terminalError;
    }
  }
}

class StdioNativeInferenceTransport implements NativeInferenceTransport {
  private readonly eventListeners = new Set<EventListener>();
  private readonly errorListeners = new Set<ErrorListener>();
  private readonly exitDeferred = createDeferred<undefined>();
  private stderr = '';
  private closed = false;
  private shutdownObserved = false;
  private exitSettled = false;

  private constructor(private readonly child: ChildProcessWithoutNullStreams) {
    const lines = createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      try {
        const event = parseNativeWireEvent(line);
        if (event.event === 'shutdown') {
          this.shutdownObserved = true;
        }
        for (const listener of this.eventListeners) {
          listener(event);
        }
      } catch (error) {
        this.emitError(toError(error));
      }
    });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-MAX_STDERR_CHARACTERS);
    });
    child.once('error', (error) => {
      this.failExit(error);
    });
    child.once('close', (code, signal) => {
      if (this.exitSettled) {
        return;
      }

      const exitDetails = `code ${String(code)} signal ${String(signal)}${this.stderr === '' ? '' : `: ${this.stderr.trim()}`}`;
      if (code !== 0 || signal !== null) {
        this.failExit(new Error(`native inference exited with ${exitDetails}`));
        return;
      }
      if (!this.closed) {
        this.failExit(
          new Error(
            `native inference exited before close was requested (${exitDetails})`
          )
        );
        return;
      }
      if (!this.shutdownObserved) {
        this.failExit(
          new Error(
            `native inference exited normally without a protocol shutdown event (${exitDetails})`
          )
        );
        return;
      }

      this.exitSettled = true;
      this.exitDeferred.resolve(undefined);
    });
  }

  static spawn(
    options: SpawnNativeInferenceOptions
  ): StdioNativeInferenceTransport {
    const maxOutstandingRequests = options.maxOutstandingRequests ?? 8;
    if (
      !Number.isSafeInteger(maxOutstandingRequests) ||
      maxOutstandingRequests < 1
    ) {
      throw new Error('maxOutstandingRequests must be a positive safe integer');
    }
    return new StdioNativeInferenceTransport(
      spawn(
        options.binaryPath,
        ['serve-stdio', options.modelDirectory, String(maxOutstandingRequests)],
        {
          env: options.environment ?? process.env,
          stdio: ['pipe', 'pipe', 'pipe'],
        }
      )
    );
  }

  async send(command: NativeWireCommand): Promise<void> {
    if (this.closed || this.child.stdin.destroyed) {
      throw new Error('native inference stdin is closed');
    }
    const line = `${JSON.stringify(command)}\n`;
    await new Promise<void>((resolve, reject) => {
      this.child.stdin.write(line, 'utf8', (error) => {
        if (error === null || error === undefined) {
          resolve();
        } else {
          reject(error);
        }
      });
    });
  }

  onEvent(listener: EventListener): () => void {
    this.eventListeners.add(listener);
    return (): void => {
      this.eventListeners.delete(listener);
    };
  }

  onError(listener: ErrorListener): () => void {
    this.errorListeners.add(listener);
    return (): void => {
      this.errorListeners.delete(listener);
    };
  }

  async close(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      this.child.stdin.end();
    }
    await this.exitDeferred.promise;
  }

  private emitError(error: Error): void {
    for (const listener of this.errorListeners) {
      listener(error);
    }
  }

  private failExit(error: Error): void {
    if (this.exitSettled) {
      return;
    }
    this.exitSettled = true;
    this.emitError(error);
    this.exitDeferred.reject(error);
  }
}

function createDeferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined;
  let rejectPromise: ((error: Error) => void) | undefined;
  const promise = new Promise<T>((resolve, reject) => {
    resolvePromise = resolve;
    rejectPromise = reject;
  });
  return {
    promise,
    resolve: (value): void => {
      resolvePromise?.(value);
    },
    reject: (error): void => {
      rejectPromise?.(error);
    },
  };
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isLifecycleEvent(
  event: NativeWireEvent
): event is NativeLifecycleEvent {
  return event.event === 'state_opened' || event.event === 'snapshot_published';
}
