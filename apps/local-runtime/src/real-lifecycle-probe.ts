import { access, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import type {
  ModelRequest,
  ModelResponse,
} from '@echo-chamber/core/ports/model';
import type { NativeInferenceModelState } from '@echo-chamber/native-inference-adapter/native-inference-model';

import { LocalNativeInferenceRuntime } from './local-native-inference-runtime';

const TOOL = {
  name: 'lookup_probe_code',
  description: 'Returns the integration probe code for the requested key.',
  inputSchema: {
    type: 'object',
    properties: { key: { type: 'string' } },
    required: ['key'],
    additionalProperties: false,
  },
  strict: true,
};
const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
} as const;

const probeArguments = process.argv.slice(2);
const binaryPath = requireArgument(probeArguments[0]);
const modelDirectory = requireArgument(probeArguments[1]);
const snapshotDirectory = requireArgument(probeArguments[2]);
const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const nativeEnvironment =
  nativeLibraryPath === undefined
    ? process.env
    : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath };

const producer = await runProducer();
const cleanupFixture = await injectCleanupFixture();
const restarter = await runRestarter();
const instanceRoot = join(snapshotDirectory, 'rin');
const currentPath = join(instanceRoot, 'current.safetensors');
const entries = (await readdir(instanceRoot)).sort();
const cleanupObservations = {
  managedStagingRemoved: !(await pathExists(cleanupFixture.managedStaging)),
  unknownOperatorFilePreserved: await pathExists(
    cleanupFixture.unknownOperatorFile
  ),
};
const checks = {
  producerPublishedCurrentState:
    producer.state.hasState && (await pathExists(currentPath)),
  restarterAutomaticallyRestoredCurrentState: restarter.restoredState.hasState,
  restoredStateHasNoLiveResponseToken:
    restarter.restoredState.responseToken === undefined,
  restartSelectedNewSession: restarter.response.usage.cachedInputTokens === 0,
  replacementStatePublished:
    restarter.finalState.hasState && !restarter.finalState.snapshotDirty,
  oneFixedCurrentPayload:
    entries.filter((entry) => entry === 'current.safetensors').length === 1 &&
    entries.every((entry) => !/^revision-\d+$/.test(entry)),
  startupRemovedManagedCrashRemainder:
    cleanupObservations.managedStagingRemoved,
  startupPreservedUnknownOperatorEntry:
    cleanupObservations.unknownOperatorFilePreserved,
};
if (Object.values(checks).some((passed) => !passed)) {
  throw new Error(
    `local Native lifecycle probe failed: ${JSON.stringify(checks)}`
  );
}
console.log(
  JSON.stringify(
    {
      schemaVersion: 2,
      producer,
      restarter,
      currentPath,
      entries,
      cleanupObservations,
      checks,
    },
    undefined,
    2
  )
);

/** Runs one fresh local runtime and relies on its session checkpoint. */
async function runProducer(): Promise<{
  state: NativeInferenceModelState;
  response: ModelResponse;
}> {
  const runtime = await startRuntime(42);
  try {
    const response = await runtime.runThinkingSession(
      'rin',
      async (model) => await model.generate(probeRequest())
    );
    return { state: runtime.state('rin'), response };
  } finally {
    await runtime.shutdown();
  }
}

interface CleanupFixture {
  managedStaging: string;
  unknownOperatorFile: string;
}

/** Seeds one managed crash remainder plus one unknown operator file. */
async function injectCleanupFixture(): Promise<CleanupFixture> {
  const root = join(snapshotDirectory, 'rin');
  const fixture = {
    managedStaging: join(root, '.current.safetensors.tmp-4242-1.safetensors'),
    unknownOperatorFile: join(root, 'operator-note.txt'),
  };
  await writeFile(fixture.managedStaging, 'incomplete\n', 'utf8');
  await writeFile(fixture.unknownOperatorFile, 'preserve\n', 'utf8');
  return fixture;
}

/** Starts a distinct owner, observes restore, and begins a new session. */
async function runRestarter(): Promise<{
  restoredState: NativeInferenceModelState;
  finalState: NativeInferenceModelState;
  response: ModelResponse;
}> {
  const runtime = await startRuntime(43);
  try {
    const restoredState = runtime.state('rin');
    const response = await runtime.runThinkingSession(
      'rin',
      async (model) => await model.generate(probeRequest())
    );
    return {
      restoredState,
      finalState: runtime.state('rin'),
      response,
    };
  } finally {
    await runtime.shutdown();
  }
}

/** Starts the production local composition with deterministic controls. */
async function startRuntime(
  seed: number
): Promise<LocalNativeInferenceRuntime> {
  return await LocalNativeInferenceRuntime.start({
    binaryPath,
    modelDirectory,
    snapshotDirectory,
    maxOutstandingRequests: 2,
    environment: nativeEnvironment,
    modelOptions: {
      rin: {
        maxTokens: 128,
        sampling: GREEDY_SAMPLING,
        seedSource: (): number => seed,
      },
    },
  });
}

/** Returns the complete prompt used at each independent thinking-session start. */
function probeRequest(): ModelRequest {
  return {
    input: [
      {
        role: 'developer',
        content:
          'Your entire reply must be exactly this function call, with no prefix or suffix:\n\n<tool_call>\n<function=lookup_probe_code>\n<parameter=key>\necho_lifecycle\n</parameter>\n</function>\n</tool_call>',
      },
    ],
    tools: [TOOL],
  };
}

/** Reports path existence without weakening any other filesystem error. */
async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (
      typeof error === 'object' &&
      error !== null &&
      'code' in error &&
      error.code === 'ENOENT'
    ) {
      return false;
    }
    throw error;
  }
}

/** Returns one required positional argument or reports this probe's usage. */
function requireArgument(value: string | undefined): string {
  if (value === undefined) {
    throw new Error(
      'usage: pnpm probe:real-lifecycle <echo-inference-binary> <model-directory> <empty-snapshot-directory>'
    );
  }
  return value;
}
