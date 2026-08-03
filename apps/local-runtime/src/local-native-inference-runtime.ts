import { join, resolve } from 'node:path';

import { ECHO_INSTANCE_IDS } from '@echo-chamber/core/types/echo-config';
import type { EchoInstanceId } from '@echo-chamber/core/types/echo-config';
import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import type { SpawnNativeInferenceOptions } from '@echo-chamber/native-inference-adapter/native-inference-client';
import { NativeInferenceModel } from '@echo-chamber/native-inference-adapter/native-inference-model';
import type {
  NativeInferenceModelOptions,
  NativeInferenceModelState,
} from '@echo-chamber/native-inference-adapter/native-inference-model';

type RuntimeLifecycle = 'starting' | 'running' | 'stopping' | 'stopped';
type InstanceModelOptions = Omit<
  NativeInferenceModelOptions,
  'client' | 'instanceId'
>;

/** Configuration for one local process-wide Native inference lifecycle. */
export interface LocalNativeInferenceRuntimeOptions {
  /** Executable that provides the Native protocol over stdio. */
  binaryPath: string;
  /** Admitted Qwen model directory shared by every E.C.H.O. instance. */
  modelDirectory: string;
  /**
   * Parent directory for per-instance durable snapshots.
   *
   * Each instance uses `<snapshotDirectory>/<instanceId>`, preventing one
   * instance's authoritative `current.safetensors` from replacing another.
   */
  snapshotDirectory: string;
  /** Bound for active plus queued Native generation requests. */
  maxOutstandingRequests?: number;
  /** Environment inherited by the Native child process. */
  environment?: NodeJS.ProcessEnv;
  /** Optional adapter controls and event sinks for each instance. */
  modelOptions?: Partial<Record<EchoInstanceId, InstanceModelOptions>>;
}

/**
 * Narrow process-creation seam used by lifecycle integration tests.
 *
 * Production callers should omit this argument and use the default Native
 * stdio owner.
 */
export interface LocalNativeInferenceRuntimeDependencies {
  spawnClient(options: SpawnNativeInferenceOptions): NativeInferenceClient;
}

/** Work performed while one E.C.H.O. instance exclusively owns its model. */
export type NativeThinkingSession<T> = (
  model: NativeInferenceModel
) => Promise<T>;

/**
 * Owns one resident Native process and one stable model state per existence.
 *
 * A thinking session is one `ThinkingEngine.think()` invocation, including
 * any internal model/tool iterations. The runtime snapshots after that
 * boundary instead of after every generation, keeping durable serialization
 * outside the decode hot path while bounding ordinary restart loss to the
 * currently active thinking session.
 */
export class LocalNativeInferenceRuntime {
  private readonly models = new Map<EchoInstanceId, NativeInferenceModel>();
  private readonly activeSessions = new Map<
    EchoInstanceId,
    Deferred<undefined>
  >();
  private readonly executingSessions = new Set<EchoInstanceId>();
  private lifecycle: RuntimeLifecycle = 'starting';
  private shutdownPromise: Promise<void> | undefined;

  private constructor(
    private readonly client: NativeInferenceClient,
    private readonly snapshotDirectory: string,
    modelOptions: LocalNativeInferenceRuntimeOptions['modelOptions']
  ) {
    for (const instanceId of ECHO_INSTANCE_IDS) {
      this.models.set(
        instanceId,
        new NativeInferenceModel({
          ...modelOptions?.[instanceId],
          client,
          instanceId,
        })
      );
    }
  }

  /**
   * Starts the resident owner and opens every per-instance state directory.
   *
   * A missing current.safetensors means that instance has no durable state
   * yet. Existing invalid state fails the whole startup closed. Opening every
   * directory also retains each owner lock for the process lifetime.
   */
  static async start(
    options: LocalNativeInferenceRuntimeOptions,
    dependencies: LocalNativeInferenceRuntimeDependencies = DEFAULT_DEPENDENCIES
  ): Promise<LocalNativeInferenceRuntime> {
    const snapshotDirectory = resolve(
      requireNonEmpty(options.snapshotDirectory, 'snapshotDirectory')
    );
    const spawnOptions: SpawnNativeInferenceOptions = {
      binaryPath: requireNonEmpty(options.binaryPath, 'binaryPath'),
      modelDirectory: requireNonEmpty(options.modelDirectory, 'modelDirectory'),
      ...(options.maxOutstandingRequests === undefined
        ? {}
        : { maxOutstandingRequests: options.maxOutstandingRequests }),
      ...(options.environment === undefined
        ? {}
        : { environment: options.environment }),
    };
    const client = dependencies.spawnClient(spawnOptions);
    try {
      const runtime = new LocalNativeInferenceRuntime(
        client,
        snapshotDirectory,
        options.modelOptions
      );
      await client.ready();
      await runtime.openStateOwners();
      runtime.lifecycle = 'running';
      return runtime;
    } catch (error) {
      const startupError = toError(error);
      try {
        await client.shutdown();
      } catch (shutdownError) {
        throw new AggregateError(
          [startupError, toError(shutdownError)],
          'local Native inference startup and cleanup both failed'
        );
      }
      throw startupError;
    }
  }

  /** Returns the adapter-owned lifecycle state for one existence. */
  state(instanceId: EchoInstanceId): NativeInferenceModelState {
    return this.requireModel(instanceId).state();
  }

  /**
   * Runs one full thinking session with exclusive access to the stable model.
   *
   * Any state committed before a later tool or session failure is still
   * snapshotted. If both the session and its checkpoint fail, both errors are
   * retained in an `AggregateError`.
   */
  async runThinkingSession<T>(
    instanceId: EchoInstanceId,
    operation: NativeThinkingSession<T>
  ): Promise<T> {
    this.requireRunning();
    if (this.activeSessions.has(instanceId)) {
      throw new Error(
        `native instance ${instanceId} already has an active thinking session`
      );
    }
    const completion = deferred<undefined>();
    this.activeSessions.set(instanceId, completion);
    this.executingSessions.add(instanceId);
    const model = this.requireModel(instanceId);
    let outcome: OperationOutcome<T>;
    try {
      outcome = { ok: true, value: await operation(model) };
    } catch (error) {
      outcome = { ok: false, error };
    }
    this.executingSessions.delete(instanceId);

    let checkpointFailure: { error: unknown } | undefined;
    try {
      await this.checkpointIfDirty(instanceId);
    } catch (error) {
      checkpointFailure = { error };
    }
    this.activeSessions.delete(instanceId);
    completion.resolve(undefined);

    if (!outcome.ok && checkpointFailure !== undefined) {
      throw new AggregateError(
        [toError(outcome.error), toError(checkpointFailure.error)],
        `native instance ${instanceId} thinking session and checkpoint both failed`
      );
    }
    if (!outcome.ok) {
      throw toError(outcome.error);
    }
    if (checkpointFailure !== undefined) {
      throw toError(checkpointFailure.error);
    }
    return outcome.value;
  }

  /**
   * Stops accepting sessions, cancels active generation, persists dirty state,
   * and closes the shared Native owner.
   *
   * Repeated calls share the same shutdown operation.
   */
  async shutdown(): Promise<void> {
    if (this.shutdownPromise !== undefined) {
      await this.shutdownPromise;
      return;
    }
    this.lifecycle = 'stopping';
    this.shutdownPromise = this.performShutdown();
    await this.shutdownPromise;
  }

  private async openStateOwners(): Promise<void> {
    for (const instanceId of ECHO_INSTANCE_IDS) {
      const snapshotRoot = this.snapshotRoot(instanceId);
      // Opening is deliberately sequential so one failed authority check can
      // close the owner without racing another in-flight lifecycle request.
      const model = this.requireModel(instanceId);
      // See the serialization reason above.
      // eslint-disable-next-line no-await-in-loop
      await model.openState(snapshotRoot);
    }
  }

  private async checkpointIfDirty(instanceId: EchoInstanceId): Promise<void> {
    const model = this.requireModel(instanceId);
    if (!model.needsSnapshot()) {
      return;
    }
    await model.snapshot();
  }

  private async performShutdown(): Promise<void> {
    const failures: Error[] = [];
    const activeInstanceIds = [...this.activeSessions.keys()];
    const cancellableInstanceIds = activeInstanceIds.filter((instanceId) =>
      this.executingSessions.has(instanceId)
    );
    await Promise.all(
      cancellableInstanceIds.map(async (instanceId) => {
        try {
          await this.requireModel(instanceId).cancelActive();
        } catch (error) {
          failures.push(toError(error));
        }
      })
    );
    await Promise.all(
      activeInstanceIds.map(async (instanceId) => {
        await this.activeSessions.get(instanceId)?.promise;
      })
    );
    await Promise.all(
      ECHO_INSTANCE_IDS.map(async (instanceId) => {
        try {
          await this.checkpointIfDirty(instanceId);
        } catch (error) {
          failures.push(toError(error));
        }
      })
    );
    try {
      await this.client.shutdown();
    } catch (error) {
      failures.push(toError(error));
    } finally {
      this.lifecycle = 'stopped';
    }
    const [singleFailure] = failures;
    if (failures.length === 1 && singleFailure !== undefined) {
      throw singleFailure;
    }
    if (failures.length > 1) {
      throw new AggregateError(
        failures,
        'local Native inference shutdown encountered multiple failures'
      );
    }
  }

  private requireRunning(): void {
    if (this.lifecycle !== 'running') {
      throw new Error(
        `local Native inference runtime is ${this.lifecycle}, not running`
      );
    }
  }

  private requireModel(instanceId: EchoInstanceId): NativeInferenceModel {
    const model = this.models.get(instanceId);
    if (model === undefined) {
      throw new Error(`native model is missing for instance ${instanceId}`);
    }
    return model;
  }

  private snapshotRoot(instanceId: EchoInstanceId): string {
    return join(this.snapshotDirectory, instanceId);
  }
}

const DEFAULT_DEPENDENCIES: LocalNativeInferenceRuntimeDependencies = {
  spawnClient: (options): NativeInferenceClient =>
    NativeInferenceClient.spawn(options),
};

type OperationOutcome<T> =
  | { ok: true; value: T }
  | { ok: false; error: unknown };

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

function deferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolvePromiseCallback) => {
    resolvePromise = resolvePromiseCallback;
  });
  return {
    promise,
    resolve: (value): void => {
      resolvePromise?.(value);
    },
  };
}

function requireNonEmpty(value: string, label: string): string {
  if (value.trim() === '') {
    throw new Error(`${label} must not be empty`);
  }
  return value;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
