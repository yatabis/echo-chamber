import { NativeInferenceClient } from './native-inference-client';
import { NATIVE_INFERENCE_PROTOCOL_VERSION } from './protocol';

const [binaryPath, modelDirectory, stateRoot] = process.argv.slice(2);
if (
  binaryPath === undefined ||
  modelDirectory === undefined ||
  stateRoot === undefined
) {
  throw new Error(
    'usage: pnpm probe:real-length <echo-inference-binary> <model-directory> <empty-state-root>'
  );
}

const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const client = NativeInferenceClient.spawn({
  binaryPath,
  modelDirectory,
  maxOutstandingRequests: 1,
  environment:
    nativeLibraryPath === undefined
      ? process.env
      : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath },
});

try {
  const ready = await client.ready();
  const opened = await client.openState({
    type: 'open_state',
    request_id: 'length-probe:open',
    instance_id: 'echo-native-length-probe',
    persistence: 'durable',
    snapshot_root: stateRoot,
  });
  if (opened.restored) {
    throw new Error('length probe requires an empty state root');
  }

  const completed = await client.generate({
    type: 'generate',
    request_id: 'length-probe:generate',
    instance_id: 'echo-native-length-probe',
    state_transition: 'initial',
    stream_tokens: false,
    input: [
      {
        role: 'developer',
        content:
          'Your entire reply must be exactly this function call, with no prefix or suffix:\n\n<tool_call>\n<function=lookup_probe_code>\n<parameter=key>\nlength_probe\n</parameter>\n</function>\n</tool_call>',
      },
    ],
    tools: [
      {
        name: 'lookup_probe_code',
        description: 'Returns the code for a length-limit integration probe.',
        input_schema: {
          type: 'object',
          properties: { key: { type: 'string' } },
          required: ['key'],
          additionalProperties: false,
        },
        strict: true,
      },
    ],
    max_new_tokens: 1,
    sampling: {
      temperature: 0,
      top_p: 0,
      top_k: 0,
      min_p: 0,
      repetition_penalty: 1,
      presence_penalty: 0,
      seed: 42,
    },
  });

  const response = completed.response;
  const expectedStateSequenceLength =
    response.metrics.input_tokens_processed +
    response.generated_tokens.length +
    1;
  const checks = {
    protocolV7: ready.protocol_version === NATIVE_INFERENCE_PROTOCOL_VERSION,
    reportedLength: response.finish_reason === 'length',
    oneVisibleToken: response.generated_tokens.length === 1,
    closingEosAdvancedOnlyInState:
      response.state_sequence_length === expectedStateSequenceLength,
    threeModelExecutions: response.metrics.model_step_count === 3,
    metricsExcludeClosingEos:
      response.metrics.generated_tokens === response.generated_tokens.length,
  };
  if (Object.values(checks).some((passed) => !passed)) {
    throw new Error(
      `native length-close probe failed: ${JSON.stringify(checks)}`
    );
  }

  const snapshot = await client.snapshot({
    type: 'snapshot',
    request_id: 'length-probe:snapshot',
    instance_id: 'echo-native-length-probe',
  });
  console.log(
    JSON.stringify(
      {
        schemaVersion: 1,
        response,
        expectedStateSequenceLength,
        snapshot,
        checks,
      },
      undefined,
      2
    )
  );
} finally {
  await client.shutdown();
}
