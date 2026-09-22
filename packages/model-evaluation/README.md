# E.C.H.O. Chamber runtime model evaluation

This package owns reusable behavior scenarios, scoring, saved-result rescoring,
and explicit live evaluation runners. It is not a general base-model benchmark.

- Native gates measure inference-state continuation, long-context execution,
  memory use, and existing runtime workflow scenarios.
- The Hosted Cognitive smoke exercises structured Memory / Emotion output and
  two-phase orchestration against the real OpenAI Responses API with synthetic
  input.
- Provider-neutral fixtures and scorers live in `src/qwen36-eat-readiness`.
  Their harness accepts a `ModelPort` factory.

## Evaluation boundaries

The existing behavior harness calls `runAgentSession` directly and uses an
assessment-specific `finish_thinking` tool with a `session_record`. It does not
connect the Cognitive coordinator. The current Rin prompt expects Cognitive
exchanges, so this harness does not represent the complete current runtime flow.
The single-session cases are retained as reusable fixtures; the Native workflow
runner executes the three multi-session workflows below.

Connecting those scenarios to the actual ThinkingEngine / Cognitive path remains
work described in [Native runtime integration readiness](../../docs/native-runtime-integration-readiness.md#残る実装要件).
A successful harness run must not be presented as acceptance of that full path.
The local Native [Cognitive integration](../../apps/local-runtime/src/real-cognitive-workflow.test.ts)
and [input-cache](../../apps/local-runtime/src/real-input-cache.test.ts) tests live
in `apps/local-runtime`. Inference-state checks live in
[`native/echo-inference`](../../native/echo-inference/README.md).

External services are stateful synthetic implementations of the TypeScript port
contracts. No real Discord, Cloudflare Durable Object storage, embedding search,
note database, or Zenn network request occurs. The fixtures validate observable
behavior and persisted in-memory state, not real storage or application restart.

## Reusable behavior scenarios

Six explicit single-session cases cover:

1. Read a private schedule change, acknowledge the new time, persist it, and avoid another channel.
2. Retrieve a fact that was not injected into the prompt from external memory before answering.
3. Let a current cancellation supersede older context and memory.
4. Switch from a persisted technical task to the current practical request.
5. Locate and update an existing note without duplicate creation or deletion.
6. Prioritize an urgent private message over a non-urgent public notification without leaking private details.

Four matching implicit cases remove procedural wording from the schedule, memory,
note, and multi-channel messages. They preserve the desired outcome without
instructing the model which tools to call.

The three stateful workflows mutate in-memory fixture ports across independent
model conversations. Later sessions consume the state that earlier sessions
actually saved:

- **Latest-state recovery after a cold start:** establish an 18:00 deployment,
  cancel it, then ask for the final status after clearing short session context
  and aging earlier messages out of the simulated chat-history window. Checks
  cover persistence, memory retrieval, and not reviving the obsolete plan.
- **Priority switch at a session boundary:** follow a non-urgent article task
  with an urgent private battery message. Checks cover responding to the urgent
  message before deferred work and keeping private details in the private channel.
  This measures the next session, not interruption of an active generation.
- **Recovery from a transient tool failure:** fail the first `update_note` with
  a synthetic timeout, allow a retry, and check the final stored note. Completion
  must follow success, without creating or deleting a note as a fallback.

Scoring combines predefined outcome, protocol, completion, and safety checks.
Each check retains its weight, result, concrete trace evidence, and first
satisfaction time where applicable.

## Offline checks

Run scenario, scoring, aggregation, and rescoring tests without inference:

```sh
pnpm eval:check
```

Native runner helper tests are included in `pnpm test:run`. Live gates skip unless
their explicit enable flag is set; ordinary tests do not load model weights or
spend API quota.

## Native live gates

Build the Native binary and configure its MLX libraries as described in the
[Native README](../../native/echo-inference/README.md). The following commands
explicitly enable live inference and require an output path for the result JSON:

| Command                                     | Purpose                                                  | Required environment variables                                                                                                |
| ------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `pnpm eval:native-stateful-performance`     | State continuation, owner switching, and resident memory | `ECHO_STATEFUL_NATIVE_INFERENCE_BIN`, `ECHO_STATEFUL_MODEL`, `ECHO_STATEFUL_OUTPUT`                                           |
| `pnpm eval:native-long-session-performance` | Long-context and repeated tool-result continuation       | `ECHO_LONG_SESSION_NATIVE_INFERENCE_BIN`, `ECHO_LONG_SESSION_MODEL`, `ECHO_LONG_SESSION_OUTPUT`                               |
| `pnpm eval:native-runtime-workflow`         | The three existing behavior workflows                    | `ECHO_NATIVE_WORKFLOW_INFERENCE_BIN`, `ECHO_NATIVE_WORKFLOW_MODEL`, `ECHO_NATIVE_WORKFLOW_OUTPUT`, `ECHO_NATIVE_LIBRARY_PATH` |

Additional opt-in tests under `src/runners/native` cover context-length curves,
sustained execution, and chunked-prefill parity. Each file declares its enable
flag and required environment variables.

### Native workflow runner

One resident Native process hosts a stable model/state owner per workflow. Tool
results within a harness session use `continuation`; subsequent harness sessions
use `new_session`. Token streaming is disabled. After each workflow, the runner
publishes a `current.safetensors` snapshot outside the workflow elapsed time and
removes the evaluation state during cleanup.

Artifacts retain scored traces, new and cached token counts, input/decode/request
timings, throughput, Metal memory, state transitions, and snapshot size. Admission
requires all behavior checks and session completions to pass, continuation cache
reuse, and a request with at least 8,192 newly processed tokens using multiple
adaptive-prefill model calls. These are the runner's criteria, not a statement
that the current fixtures or models pass them.

```sh
ECHO_NATIVE_WORKFLOW_INFERENCE_BIN=/absolute/path/to/echo-inference \
ECHO_NATIVE_WORKFLOW_MODEL=/absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
ECHO_NATIVE_WORKFLOW_OUTPUT=/absolute/path/to/native-runtime-workflows.json \
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm eval:native-runtime-workflow
```

The default `controlled-greedy` profile uses temperature `0`, top-p `1`, and top-k
`1`. Set `ECHO_NATIVE_WORKFLOW_PROFILE=production-sampling` for temperature `0.7`,
top-p `0.8`, top-k `20`, min-p `0`, presence penalty `1.5`, and repetition penalty
`1`. Both profiles disable thinking and cap output at 1,024 tokens per turn.
These fixed evaluation settings do not constitute an exact production replay;
a single trial also does not estimate success probability under sampling.

`ECHO_NATIVE_WORKFLOW_FILTER` accepts a JavaScript regular expression over workflow
IDs. `ECHO_NATIVE_WORKFLOW_SEED` and `ECHO_NATIVE_WORKFLOW_MAX_TURNS` control the
recorded seed schedule and per-session agent-loop ceiling.

`ECHO_NATIVE_WORKFLOW_STATE_MODE` defaults to carrying complete GDN state. Its
optional diagnostic modes isolate cross-session state effects:

- `fresh-session-ablation`: create a fresh Native state owner for every session.
- `recurrent-only-ablation`: retain the recurrent matrix and clear convolution history.
- `convolution-only-ablation`: retain convolution history and clear the recurrent matrix.

## Hosted Cognitive live smoke

Set `OPENAI_API_KEY` in the command environment without placing its value in the
command line, then run:

```sh
pnpm eval:cognitive-hosted
```

The smoke runs Memory / Emotion for `pre_main` and `post_main` with synthetic
input. It checks module system prompts, recall/store/emotion schemas, the
system-owned `search_memory` / `update_emotion` handoff, shared chronological
context, non-empty usage, and model-event attribution. It is excluded from
`pnpm test:run` and uses the real API only when explicitly enabled.
See [Cognitive Module Architecture](../../docs/cognitive-module-architecture.md)
for the execution design. This smoke does not establish model quality or real
persistence correctness.

## Saved results

Keep generated result JSON and logs under the ignored
`.artifacts/model-evaluation/` directory, or use `/private/tmp` for disposable
runs. Curated reports may live under `docs/`; raw machine-specific artifacts do not.

When only scoring rules change, reuse saved model exchanges and tool traces.
The following command overwrites the specified result JSON after appending a
rescore-history entry. It expects the saved artifact's `candidates` structure;
it does not accept arbitrary Native diagnostic reports. It cannot evaluate
missing evidence or add Cognitive coverage to an older trace. Rescoring removes
the legacy `promptAblationComparison` summary, which is no longer recomputed.

```sh
ECHO_EVAL_RESCORE_PATH=/absolute/path/to/result.json \
ECHO_EVAL_RESCORE_REASON="Describe the scorer change" \
pnpm eval:rescore
```
