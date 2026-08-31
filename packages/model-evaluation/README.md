# E.C.H.O. Chamber runtime model evaluation

This package contains two separate evaluation lanes. It is not a general base-model benchmark.

- The Qwen / Rapid-MLX lane evaluates tool selection, ordering, persistence, safety, termination, and session-prefix caching. It does not exercise the Cognitive Module path.
- The Hosted Cognitive lane exercises structured Memory / Emotion output and two-phase orchestration against the real OpenAI Responses API with synthetic input.

The package owns model-evaluation scenarios, scoring, and local runner composition. The Qwen / Rapid-MLX harness accepts a provider-neutral `ModelPort` factory. OpenAI-compatible connection code lives under `src/runners`, and Rapid-MLX process/cache control is isolated under `src/runners/rapid-mlx`.

No Environment Aware Training (EAT) artifact exists yet. The current run therefore establishes base-model results and a reusable comparison matrix. A future EAT model can use the same cases and scoring after it has been exported as a self-contained model directory that Rapid-MLX can serve directly.

## Qwen evaluator terms and conditions

- **Production prompt**: the current Rin prompt followed by the evaluator's generated tool catalog and runtime context.
- **Explicit message**: the user message names the procedure, such as reading the full message, searching memory, storing a decision, or updating an existing note.
- **Implicit message**: the environmental fact or desired outcome is present, but those procedural instructions are absent.
- **Controlled-greedy generation**: temperature `0`, top-p `1`, and top-k `1`. This is the primary reproducible runtime-behavior and pre/post-training comparison. It is not the model-native recommended generation profile and must not be presented as a measurement of the model's maximum response quality.
- **Production-sampling generation**: temperature `0.7`, top-p `0.8`, top-k `20`, and presence penalty `1.5`, matching both the current E.C.H.O. Chamber Rapid-MLX application configuration and Qwen's official recommendation for instruct/non-thinking Qwen3.6. Qwen also specifies min-p `0.0` and repetition penalty `1.0`; Rapid-MLX uses those neutral defaults when the request omits them. This profile is repeated on a small sentinel set because individual samples are stochastic.
- **Mode-specific Qwen recommendations**: both local artifacts' `generation_config.json` files declare the generic sampling defaults temperature `1.0`, top-p `0.95`, and top-k `20`, but the official model cards provide separate recommendations for thinking, precise coding, and instruct/non-thinking modes. The thinking profiles are not the operating mode of the current E.C.H.O. Chamber runtime and are not exercised by this matrix.
- **Runtime score**: the sum of predefined outcome, protocol, completion, and safety checks. Every check stores its weight, pass/fail state, first satisfaction time when applicable, and concrete trace evidence.

The two evaluator generation profiles disable model thinking, matching the current Rapid-MLX application configuration. Each request is capped at 1,024 output tokens for evaluation safety; the deployed application and Qwen's general output-length recommendation both permit a much larger 32,768-token ceiling. That difference is recorded in every result and prevents this from being described as an exact production replay. The stochastic profile uses Qwen's official instruct/non-thinking sampling values; the separate thinking-mode recommendations have not been run by this evaluator.

The controlled cells execute one deterministic repetition per case. They are suitable for reproducing tool-selection, ordering, persistence, safety, and termination failures. Estimating the probability of a desirable answer under either stochastic profile requires repeated trials and is explicitly outside a one-repetition controlled result.

## Primary single-session cases

Six explicit cases run with the production prompt:

1. Read a private schedule change, acknowledge the new time, persist it, and avoid another channel.
2. Retrieve a fact that was not injected into the prompt from external memory before answering.
3. Let a current cancellation supersede older context and memory.
4. Switch from a persisted technical task to the current practical request.
5. Locate and update an existing note without duplicate creation or deletion.
6. Prioritize an urgent private message over a non-urgent public notification without leaking private details.

Four matching implicit cases remove procedural wording from the schedule, memory, note, and multi-channel messages. They run with the same production prompt as the explicit cases so the result isolates whether the user must prescribe the procedure. It does not by itself measure EAT.

## Stateful workflows

The stateful fixture ports mutate real in-memory state across independent model conversations. Stored memories, notes, chat history, and session state are therefore load-bearing inputs to later sessions rather than prewritten expected outputs.

### Latest-state recovery after a cold start

Three sessions establish an 18:00 deployment, cancel it later, then ask for the final status after the short session context has been cleared and the earlier messages have fallen outside the simulated chat-history window. The evaluator checks that both state changes were persisted, long-term memory was searched, the cancellation was answered, and the obsolete 18:00 plan was not revived.

### Priority switch at a session boundary

One session receives a non-urgent article task. The next session starts with that persisted context plus a new urgent private battery message. The evaluator checks that the urgent message is read and answered before deferred Zenn work and that private details remain in the private channel.

This is next-session interruption behavior. It must not be described as a mid-generation interrupt: while an Echo instance is in `Running` state, the current runtime has no mechanism to inject a newly arrived event into an active model request or agent session.

### Recovery from a transient tool failure

The first `update_note` operation fails with a synthetic timeout. The same stateful note port then permits a retry. The evaluator checks that the failure was observed, the operation was retried, the final persisted note is correct, completion is acknowledged only after success, and the model does not create or delete a note as a fallback.

## Default evaluation size

For each model, the controlled comparison executes:

- 6 explicit single-session cases with the production prompt;
- 4 implicit single-session cases with the production prompt;
- 3 stateful workflows with the production prompt.

Those three workflow invocations contain six actual model sessions in total. The default production-sampling sentinel adds 10 sessions: two single-session cases and two stateful workflows, each repeated twice. This bounded matrix focuses evaluation cost on behavior exercised by the E.C.H.O. runtime.

Set `ECHO_EVAL_PRODUCTION_REPETITIONS=0` to skip the stochastic sentinels, or another non-negative integer to change their repetition count.

Set `ECHO_EVAL_CELL_FILTER` to a JavaScript regular-expression string to select whole evaluation cells. For example, `^deployment-sampling-single-session-sentinels$` runs only the repeated non-thinking production-sampling single-session cell and avoids rerunning controlled or stateful cells that are not part of the current question.

Set `ECHO_EVAL_PRODUCTION_SAMPLING_FILE` to apply another model's documented sampling values to the production-sampling cells without changing the controlled-greedy cells. The default remains the Qwen3.6/E.C.H.O. profile above. The override is strict JSON with camel-case field names; the evaluator keeps non-thinking mode and the 1,024-token safety cap fixed:

```json
{
  "description": "Agents-A1 official sampling values in the E.C.H.O. non-thinking evaluator; output remains capped at 1,024 tokens per turn.",
  "temperature": 0.85,
  "topP": 0.95,
  "topK": 20,
  "minP": 0.0,
  "repetitionPenalty": 1.0,
  "presencePenalty": 1.1
}
```

The result protocol stores both the resolved generation profile and the override-file path. This makes the run reproducible but does not make a sampling profile model-specific automatically; the caller must supply the profile that belongs to the evaluated model.

## Artifact retention

Store durable local evaluation results, server logs, smoke runs, and machine-local evaluation-target files under `.artifacts/model-evaluation/`. Use `/private/tmp` instead for disposable runs. The repository-local artifact directory is ignored because these generated files contain machine-specific paths and can be large. Commit only a deliberately curated, human-readable report under `docs/` after its evidence and limitations are stable; do not place raw evaluation artifacts there.

## What remains outside the measured score

- External services use stateful synthetic implementations of the production TypeScript port contracts. No real Discord, Cloudflare Durable Object storage, embedding search, note database, or Zenn network request occurs.
- True mid-generation external interruption is not supported by the current product runtime, so the evaluator records that capability gap and measures only the implemented next-session behavior.
- Gated Delta Network recurrent-state continuation is not measured. The current Chat Completions adapter has no contract that binds recurrent state to the exact replayed token and key/value-cache boundary.
- Multi-token prediction (MTP speculative decoding) and PFlash prompt compression are disabled. Token-prefix caching is enabled: every fixture/repetition receives a unique cache session ID, the process-local cache is cleared before that fixture starts, and only the growing exact prefix inside that one session can be reused. This makes the timing closer to the current E.C.H.O. runtime while preventing state leakage between scored fixtures.
- No EAT improvement can be claimed until a trained model is evaluated against its matching base model and the resulting observations are compared explicitly.

## Hosted Cognitive live smoke

Set `OPENAI_API_KEY` in the command environment without placing its value in the command line, then run the explicit live lane:

```sh
pnpm eval:cognitive-hosted
```

The smoke runs Memory / Emotion for `pre_main` and `post_main` with synthetic input. It checks the dedicated module system prompts, recall, store, and emotion schemas, the system-owned `search_memory` / `update_emotion` handoff, shared chronological context, non-empty usage, and local model events that preserve the same payload fields as Main while adding module attribution. The command is excluded from `pnpm test:run` so ordinary tests never spend API quota. The execution design is documented in [Cognitive Module Architecture](../../docs/cognitive-module-architecture.md).

This is an integration smoke. It does not by itself establish model quality or persistence correctness.

## Running the Qwen / Rapid-MLX evaluator

Run the deterministic evaluator checks without starting a model server:

```sh
pnpm eval:check
```

Run the full Rapid-MLX model evaluation with the following environment variables. The `pnpm eval` script explicitly enables the live evaluator; it is not part of `pnpm test:run`.

```sh
ECHO_EVAL_REPOSITORY=/absolute/path/to/echo-chamber \
ECHO_EVAL_RAPID_MLX_BIN=/absolute/path/to/rapid-mlx \
ECHO_EVAL_RAPID_MLX_CWD=/absolute/path/to/rapid-mlx-repository \
ECHO_EVAL_27B_MODEL=/absolute/path/to/Qwen3.6-27B-MLX-4bit \
ECHO_EVAL_35B_MODEL=/absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
ECHO_EVAL_OUTPUT=/absolute/path/to/result.json \
pnpm eval
```

`ECHO_EVAL_SMOKE=1` loads only the first evaluation target and runs a reduced but stateful path through explicit and implicit cases, transient-failure handling, and one production-sampling sentinel.

Set `ECHO_EVAL_CASE_FILTER` to a JavaScript regular-expression string to rerun only matching scenario or workflow IDs. This is intended for validator fixes, disagreements between models, and higher-repetition confirmation without rerunning the entire suite.

To replace the two default model variables with an explicit list, set `ECHO_EVAL_TARGETS_FILE` to a JSON file. A normal run evaluates every listed model sequentially in file order. Each `modelPath` must be a self-contained model directory that Rapid-MLX can serve directly:

```json
[
  {
    "id": "qwen36-35b-a3b-base",
    "displayName": "Qwen3.6-35B-A3B base",
    "modelPath": "/models/base",
    "servedModelName": "base-eval"
  },
  {
    "id": "qwen36-35b-a3b-eat-v1",
    "displayName": "Qwen3.6-35B-A3B EAT v1",
    "modelPath": "/models/eat-v1",
    "servedModelName": "eat-v1-eval"
  }
]
```

The target file does not label, pair, or automatically compare base and EAT models, and it does not attach a separate adapter to a base model. Those capabilities must be implemented together with the eventual EAT artifact format rather than inferred from target names.

The output JSON checkpoints after every completed case. It records source commits and dirty paths, a hash of the current Rin prompt and the application LLM configuration, model architecture fields, exact cell definitions, every tool call and model exchange, persisted final state, timings, token usage, scoring evidence, and server cleanup status.

## Session prefix-cache probe

The primary behavior evaluation uses the dedicated session-scoped token-prefix cache. This gated integration probe compares the same growing E.C.H.O. histories with prefix caching disabled, with a cold session cache, and with one continuing session. It verifies the pinned and rolling entry counts, increasing cached-token counts, output equality against the cache-disabled baseline, and two 512-token long-output cases.

This is the relevant contract for Qwen3.6 because its linear-attention layers carry Gated Delta Network recurrent state that cannot be treated as an ordinary trimmable key/value cache. The probe tests the E.C.H.O.-specific session boundary instead of the superseded generic exact-prompt cache experiment.

```sh
ECHO_EVAL_RAPID_MLX_BIN=/absolute/path/to/rapid-mlx \
ECHO_EVAL_RAPID_MLX_CWD=/absolute/path/to/rapid-mlx-repository \
ECHO_EVAL_TARGETS_FILE=/absolute/path/to/evaluation-targets.json \
ECHO_SESSION_PREFIX_CACHE_PROBE_OUTPUT=/absolute/path/to/session-prefix-cache-result.json \
pnpm eval:session-prefix-cache
```

## Rescoring a saved result

When only scoring rules change, reuse the recorded model exchanges and tool traces instead of rerunning inference. This command overwrites the specified result JSON after appending a rescore-history entry. It cannot evaluate evidence that the original run did not record. Rescoring an older result removes its legacy `promptAblationComparison` summary because the current evaluator no longer recomputes that provisional-prompt experiment.

```sh
ECHO_EVAL_RESCORE_PATH=/absolute/path/to/result.json \
ECHO_EVAL_RESCORE_REASON="Describe the scorer change" \
pnpm eval:rescore
```
