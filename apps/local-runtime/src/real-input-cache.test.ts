import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { expect, test } from 'vitest';

import type {
  ModelInputItem,
  ModelRequest,
} from '@echo-chamber/core/ports/model';
import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import { NativeInferenceModel } from '@echo-chamber/native-inference-adapter/native-inference-model';
import type {
  NativeRuntimeMetrics,
  NativeCompletedEvent,
} from '@echo-chamber/native-inference-adapter/protocol';

const binaryPath = process.env.ECHO_NATIVE_BINARY;
const modelDirectory = process.env.ECHO_NATIVE_MODEL_DIRECTORY;

test.skipIf(binaryPath === undefined || modelDirectory === undefined)(
  'real input cache bounds prefill to appended context on a growing cognitive session',
  async () => {
    if (binaryPath === undefined || modelDirectory === undefined)
      throw new Error('explicit model required');
    const client = NativeInferenceClient.spawn({
      binaryPath,
      modelDirectory,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH:
          process.env.ECHO_NATIVE_LIBRARY_PATH ?? process.env.DYLD_LIBRARY_PATH,
      },
    });
    const observations: { label: string; metrics: NativeRuntimeMetrics }[] = [];
    let label = '';
    const generate = client.generate.bind(client);
    client.generate = async (...args): Promise<NativeCompletedEvent> => {
      const result = await generate(...args);
      observations.push({ label, metrics: result.response.metrics });
      return result;
    };
    const model = new NativeInferenceModel({
      client,
      instanceId: 'cache.probe',
      statePolicy: 'independent',
      maxTokens: 32,
      sampling: {
        temperature: 0,
        top_p: 0,
        top_k: 0,
        min_p: 0,
        repetition_penalty: 1,
        presence_penalty: 0,
      },
    });
    try {
      await model.openState({ persistence: 'ephemeral' });
      model.beginInputCacheSession('growing-context');
      const input: ModelInputItem[] = [
        {
          role: 'system',
          content:
            'Acknowledge these recorded observations with the required JSON object.',
        },
      ];
      for (const [index, count] of [512, 512, 1].entries()) {
        input.push({
          role: 'user',
          content: Array.from(
            { length: count },
            (_, line) =>
              `Observation ${index}-${line}: The session continues and the previous results remain available.`
          ).join('\n'),
        });
        label = `cached-${index}`;
        // Each request extends the exact input whose checkpoint the previous one produced.
        // eslint-disable-next-line no-await-in-loop
        const response = await model.generate(cacheRequest(input));
        expect(response.output).toHaveLength(1);
        expect(response.output[0]).toMatchObject({
          type: 'message',
          role: 'assistant',
        });
      }
      await model.endInputCacheSession();
      label = 'cold-final-input';
      await model.generate(cacheRequest(input));
      assertPrefillWork(observations);
    } finally {
      await model.endInputCacheSession();
      await client.shutdown();
      const directory = process.env.ECHO_NATIVE_COGNITIVE_REPORT_DIRECTORY;
      if (directory !== undefined)
        await writeFile(
          join(directory, 'input-cache-prefill.json'),
          `${JSON.stringify(observations, null, 2)}\n`
        );
    }
  },
  300_000
);

/** A fixed structured result keeps the probe focused on input work. */
function cacheRequest(input: ModelInputItem[]): ModelRequest {
  return {
    input,
    tools: [],
    maxOutputTokens: 32,
    responseFormat: {
      type: 'json_schema',
      name: 'acknowledgement',
      strict: true,
      schema: {
        type: 'object',
        properties: { ok: { type: 'boolean', enum: [true] } },
        required: ['ok'],
        additionalProperties: false,
      },
    },
  };
}

/** Verify reused tokens plus executed suffix equal the full input. */
function assertPrefillWork(
  observations: { metrics: NativeRuntimeMetrics }[]
): void {
  const first = observations[0]?.metrics;
  const extended = observations[1]?.metrics;
  const last = observations[2]?.metrics;
  const cold = observations[3]?.metrics;
  if (!first || !extended || !last || !cold)
    throw new Error('missing measurements');
  expect(first.input_tokens_processed).toBeGreaterThan(4_000);
  expect(extended.cached_prefix_tokens).toBeGreaterThan(4_000);
  expect(last.cached_prefix_tokens).toBeGreaterThan(8_000);
  expect(last.input_tokens_processed).toBeLessThan(100);
  expect(last.cached_prefix_tokens + last.input_tokens_processed).toBe(
    cold.input_tokens_processed
  );
  expect(cold.cached_prefix_tokens).toBe(0);
}
