import type { ModelOutputItem } from '@echo-chamber/core/ports/model';

import { NativeInferenceClient } from './native-inference-client';
import { NativeInferenceModel } from './native-inference-model';
import { NATIVE_INFERENCE_PROTOCOL_VERSION } from './protocol';

const INSTANCE_ID = 'echo-native-recovery-probe';
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
const snapshotRoot = requireArgument(probeArguments[2]);

const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const nativeEnvironment =
  nativeLibraryPath === undefined
    ? process.env
    : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath };

const producer = await produceCurrentState();
const restorer = await restoreIntoNewSession();

console.log(
  JSON.stringify(
    {
      schemaVersion: 2,
      producer,
      restorer,
      checks: {
        protocolV7InBothProcesses:
          producer.protocolVersion === NATIVE_INFERENCE_PROTOCOL_VERSION &&
          restorer.protocolVersion === NATIVE_INFERENCE_PROTOCOL_VERSION,
        separateNativeOwners: true,
        restoredAdapterIdle:
          restorer.restoredState.activeRequestId === undefined,
        restoredStateHasNoLiveResponseToken:
          restorer.restoredState.responseToken === undefined,
        restoredStateSelectedNewSession:
          restorer.newSessionUsage.cachedInputTokens === 0,
        liveContinuationReusedNewSessionState:
          restorer.continuationUsage.cachedInputTokens ===
          restorer.newSessionState.stateSequenceLength,
        continuationConsumedToolResult: restorer.output.includes('7391'),
        oneFixedCurrentPath:
          producer.snapshot.path === restorer.snapshot.path &&
          producer.snapshot.path.endsWith('/current.safetensors'),
      },
    },
    undefined,
    2
  )
);

/** Produces and durably publishes one completed native state. */
async function produceCurrentState(): Promise<{
  protocolVersion: number;
  state: ReturnType<NativeInferenceModel['state']>;
  snapshot: Awaited<ReturnType<NativeInferenceModel['snapshot']>>;
}> {
  const client = spawnClient();
  try {
    const ready = await client.ready();
    const model = createModel(client, 42);
    await model.openState(snapshotRoot);
    await model.generate({
      input: [
        {
          role: 'developer',
          content: 'Reply with a short acknowledgement for the recovery probe.',
        },
      ],
      tools: [],
      turnIndex: 1,
    });
    const snapshot = await model.snapshot();
    return {
      protocolVersion: ready.protocol_version,
      state: model.state(),
      snapshot,
    };
  } finally {
    await client.shutdown();
  }
}

/** Restores current state, starts a new session, and continues it live. */
async function restoreIntoNewSession(): Promise<{
  protocolVersion: number;
  restoredState: ReturnType<NativeInferenceModel['state']>;
  newSessionState: ReturnType<NativeInferenceModel['state']>;
  newSessionUsage: Awaited<
    ReturnType<NativeInferenceModel['generate']>
  >['usage'];
  continuationUsage: Awaited<
    ReturnType<NativeInferenceModel['generate']>
  >['usage'];
  output: string;
  snapshot: Awaited<ReturnType<NativeInferenceModel['snapshot']>>;
}> {
  const client = spawnClient();
  try {
    const ready = await client.ready();
    const model = createModel(client, 43);
    const restoredState = await model.openState(snapshotRoot);
    const newSession = await model.generate({
      input: [
        {
          role: 'developer',
          content:
            'Your entire reply must be exactly this function call, with no prefix or suffix:\n\n<tool_call>\n<function=lookup_probe_code>\n<parameter=key>\necho_recovered_session\n</parameter>\n</function>\n</tool_call>\n\nAfter its result arrives, reply with only the returned code.',
        },
      ],
      tools: [TOOL],
      turnIndex: 1,
    });
    const responseToken = newSession.responseToken;
    if (responseToken === undefined) {
      throw new Error('new session returned no live response token');
    }
    const toolCall = requireToolCall(newSession.output);
    const newSessionState = model.state();
    const continuation = await model.generate({
      input: [
        {
          type: 'tool_result',
          callId: toolCall.callId,
          output: '{"code":"7391"}',
        },
      ],
      tools: [TOOL],
      previousResponseToken: responseToken,
      turnIndex: 2,
    });
    const output = continuation.output
      .filter(
        (item): item is Extract<ModelOutputItem, { type: 'message' }> =>
          item.type === 'message'
      )
      .map((item) => item.content)
      .join('\n');
    if (!output.includes('7391')) {
      throw new Error(
        `restored new-session continuation did not consume its tool result: ${JSON.stringify(continuation.output)}`
      );
    }
    const snapshot = await model.snapshot();
    return {
      protocolVersion: ready.protocol_version,
      restoredState,
      newSessionState,
      newSessionUsage: newSession.usage,
      continuationUsage: continuation.usage,
      output,
      snapshot,
    };
  } finally {
    await client.shutdown();
  }
}

function spawnClient(): NativeInferenceClient {
  return NativeInferenceClient.spawn({
    binaryPath,
    modelDirectory,
    maxOutstandingRequests: 2,
    environment: nativeEnvironment,
  });
}

function createModel(
  client: NativeInferenceClient,
  seed: number
): NativeInferenceModel {
  return new NativeInferenceModel({
    client,
    instanceId: INSTANCE_ID,
    maxTokens: 128,
    sampling: GREEDY_SAMPLING,
    seedSource: (): number => seed,
  });
}

/** Returns the single parsed tool call required by this recovery probe. */
function requireToolCall(
  output: readonly ModelOutputItem[]
): Extract<ModelOutputItem, { type: 'tool_call' }> {
  const toolCall = output.find(
    (item): item is Extract<ModelOutputItem, { type: 'tool_call' }> =>
      item.type === 'tool_call'
  );
  if (toolCall === undefined || toolCall.toolName !== TOOL.name) {
    throw new Error(
      `new session returned no expected tool call: ${JSON.stringify(output)}`
    );
  }
  return toolCall;
}

/** Returns one required positional argument or reports this probe's usage. */
function requireArgument(value: string | undefined): string {
  if (value === undefined) {
    throw new Error(
      'usage: pnpm probe:real-recovery <echo-inference-binary> <model-directory> <empty-snapshot-root>'
    );
  }
  return value;
}
