import { writeFile } from 'node:fs/promises';

import { expect, test } from 'vitest';

import {
  createEmotionCognitiveModuleOutputFormat,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  parseEmotionCognitiveModuleOutput,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
} from '@echo-chamber/core/agent/cognitive-module-schema';
import { ModelGenerationError } from '@echo-chamber/core/ports/model';
import type { ModelStructuredOutputFormat } from '@echo-chamber/core/ports/model';

import { NativeInferenceClient } from './native-inference-client';
import {
  NativeInferenceIncompleteGenerationError,
  NativeInferenceModel,
  ECHO_NATIVE_PRODUCTION_SAMPLING,
} from './native-inference-model';

const modelDirectory = process.env.ECHO_NATIVE_MODEL_DIRECTORY;
const binaryPath = process.env.ECHO_NATIVE_BINARY;
const GREEDY = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
};

test.skipIf(modelDirectory === undefined || binaryPath === undefined)(
  'real Native request controls validate Cognitive schemas, output caps, and abort/retry',
  async () => {
    if (modelDirectory === undefined || binaryPath === undefined)
      throw new Error('missing explicit Native test environment');
    const client = NativeInferenceClient.spawn({
      binaryPath,
      modelDirectory,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH:
          process.env.ECHO_NATIVE_LIBRARY_PATH ?? process.env.DYLD_LIBRARY_PATH,
      },
    });
    const results: unknown[] = [];
    try {
      await client.ready();
      results.push(...(await checkStructuredResponses(client)));
      results.push(await checkConstrainedBatch(client));
      results.push(await checkAbortAndLimit(client));
      const report = JSON.stringify({ requestControls: results }, null, 2);
      const reportPath = process.env.ECHO_NATIVE_REQUEST_CONTROLS_REPORT;
      if (reportPath !== undefined) await writeFile(reportPath, `${report}\n`);
      console.log(report);
    } finally {
      await client.shutdown();
    }
  },
  240_000
);

/** Exercise each real Cognitive JSON schema with both sampling profiles. */
async function checkStructuredResponses(
  client: NativeInferenceClient
): Promise<unknown[]> {
  const results: unknown[] = [];
  const cases = [
    {
      format: createMemoryRecallCognitiveModuleOutputFormat(),
      value: { query: '明日の予定' },
      parse: parseMemoryRecallCognitiveModuleOutput,
    },
    {
      format: createMemoryStoreCognitiveModuleOutputFormat(),
      value: { content: '明日は図書館へ行く予定がある。', type: 'episode' },
      parse: parseMemoryStoreCognitiveModuleOutput,
    },
    {
      format: createEmotionCognitiveModuleOutputFormat(),
      value: { valence: 0.2, arousal: 0.1, labels: ['平静'] },
      parse: parseEmotionCognitiveModuleOutput,
    },
  ];
  for (const profile of ['greedy', 'production']) {
    for (const { format, value, parse } of cases) {
      const model = new NativeInferenceModel({
        client,
        instanceId: `request-controls.${profile}.${format.name}`,
        maxTokens: 256,
        seedSource: (): number => 42,
        ...(profile === 'greedy' ? { sampling: GREEDY } : {}),
      });
      // Independent lanes isolate schema/request behavior from phase lifetime policy.
      // eslint-disable-next-line no-await-in-loop
      await model.openState({ persistence: 'ephemeral' });
      // eslint-disable-next-line no-await-in-loop
      const response = await model.generate({
        input: [
          {
            role: 'user',
            content: `Describe this value in ordinary prose, never JSON. Include a Markdown heading: ${JSON.stringify(value)}`,
          },
        ],
        tools: [],
        responseFormat: format,
        maxOutputTokens: 128,
      });
      const message = response.output[0];
      if (message?.type !== 'message') throw new Error('expected JSON message');
      expect(() => parse(JSON.parse(message.content))).not.toThrow();
      expect(response.usage.outputTokens).toBeGreaterThan(0);
      expect(response.usage.outputTokens).toBeLessThanOrEqual(128);
      results.push({
        profile,
        schema: format.name,
        output: message.content,
        usage: response.usage,
      });
    }
  }

  return results;
}

/** Verify different grammars in the actual shared batch, including literal tool markup. */
async function checkConstrainedBatch(
  client: NativeInferenceClient
): Promise<unknown> {
  const cases = [0, 1, 2].map((index) => {
    const value = {
      lane: index,
      text: '猫と散歩。"引用" <tool_call><function=example></function></tool_call>',
    };
    return {
      value,
      instanceId: `request-controls.batch.${index}`,
      format: {
        type: 'json_schema',
        name: `batch_${index}`,
        strict: true,
        schema: { const: value },
      } satisfies ModelStructuredOutputFormat,
    };
  });
  await Promise.all(
    cases.map(async ({ instanceId }) =>
      client.openState({
        type: 'open_state',
        request_id: `${instanceId}.open`,
        instance_id: instanceId,
        persistence: 'ephemeral',
      })
    )
  );
  const responses = await Promise.all(
    cases.map(async ({ instanceId, format }, index) =>
      client.generate({
        type: 'generate',
        request_id: `${instanceId}.generate`,
        instance_id: instanceId,
        state_transition: 'initial',
        stream_tokens: false,
        input: [
          {
            role: 'user',
            content: 'Say NO. Never output JSON or quote my instructions.',
          },
        ],
        tools: [],
        response_format: format,
        max_new_tokens: 128,
        sampling: {
          ...(index === 0 ? GREEDY : ECHO_NATIVE_PRODUCTION_SAMPLING),
          seed: 42,
        },
      })
    )
  );
  for (const [index, response] of responses.entries()) {
    expect(response.response.finish_reason).toBe('stop_token');
    expect(JSON.parse(response.text)).toEqual(cases[index]?.value);
    expect(response.output).toEqual([
      { type: 'message', role: 'assistant', content: response.text },
    ]);
    expect(response.tool_parse_warning).toBeUndefined();
  }
  expect(
    Math.max(
      ...responses.map(
        (response) => response.response.metrics.maximum_decode_batch_size
      )
    )
  ).toBeGreaterThan(1);
  return responses.map((response) => ({
    output: response.text,
    metrics: response.response.metrics,
  }));
}

/** Verify rollback, consumed work, and retry through the real stdio owner. */
async function checkAbortAndLimit(
  client: NativeInferenceClient
): Promise<unknown> {
  let controller: AbortController | undefined;
  let observedTokens = 0;
  const model = new NativeInferenceModel({
    client,
    instanceId: 'request-controls.abort',
    maxTokens: 256,
    sampling: GREEDY,
    seedSource: (): number => 42,
    onToken: (): void => {
      observedTokens += 1;
      if (observedTokens === 3) controller?.abort('probe interruption');
    },
  });
  await model.openState({ persistence: 'ephemeral' });
  await model.generate({
    input: [{ role: 'user', content: 'Reply only OK.' }],
    tools: [],
  });
  const before = model.state();
  const longInput = [
    {
      role: 'user' as const,
      content:
        'List all integers from 1 to 200, separated by spaces. Do not stop early.',
    },
  ];
  const responseFormat: ModelStructuredOutputFormat = {
    type: 'json_schema',
    name: 'long_string',
    strict: true,
    schema: {
      type: 'string',
      const: Array.from({ length: 100 }, (_, index) => index + 1).join(' '),
    },
  };
  controller = new AbortController();
  observedTokens = 0;
  const cancelled = await model
    .generate({
      input: longInput,
      tools: [],
      responseFormat,
      signal: controller.signal,
    })
    .catch((error: unknown) => error);
  expect(cancelled).toBeInstanceOf(ModelGenerationError);
  if (!(cancelled instanceof ModelGenerationError))
    throw new Error('expected a cancellation error');
  expect(cancelled.name).toBe('AbortError');
  expect(cancelled.usage?.outputTokens).toBeGreaterThanOrEqual(3);
  expect(cancelled.usage?.uncachedInputTokens).toBeGreaterThan(0);
  expect(model.state()).toEqual(before);
  controller = undefined;
  const truncated = await model
    .generate({
      input: longInput,
      tools: [],
      responseFormat,
      maxOutputTokens: 1,
    })
    .catch((error: unknown) => error);
  expect(truncated).toBeInstanceOf(NativeInferenceIncompleteGenerationError);
  if (!(truncated instanceof NativeInferenceIncompleteGenerationError))
    throw new Error('expected length limit');
  expect(truncated.usage?.outputTokens).toBe(1);
  expect(model.state().responseToken).toBe(truncated.responseToken);
  const retry = await model.generate({
    input: [{ role: 'user', content: 'Reply only OK.' }],
    tools: [],
  });
  const message = retry.output[0];
  if (message?.type !== 'message') throw new Error('expected retry message');
  expect(message.content).toMatch(/^OK\.?$/);
  return {
    cancellationUsage: cancelled.usage,
    lengthUsage: truncated.usage,
    retryUsage: retry.usage,
  };
}
