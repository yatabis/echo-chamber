# Native inference architecture

## Status

This is the current implementation boundary for E.C.H.O. Chamber's local
Qwen3.5-family MoE inference path. The primary admitted artifact is
Qwen3.6-35B-A3B-MLX-4bit. The numerical model, resident execution, composite
KV/GDN state, atomic durable publication, protocol-v7 adapter, and local
thinking-session lifecycle are implemented.

Dated reports under `../evidence/` record the exact numerical and performance
conditions tested at each milestone. Some reports predate this state design
and use older protocol terminology; they are evidence of those runs, not the
current contract.

## Repository and component boundaries

The native engine is kept in the E.C.H.O. Chamber monorepo so its protocol and
harness evolve together, but Cargo remains independent from pnpm.

- `echo-mlx-sys` generates an allow-listed Rust FFI from a caller-pinned
  official MLX C checkout.
- `echo-mlx` owns safe MLX handles, stream boundaries, quantized operations,
  safetensors, and custom Metal-kernel dispatch.
- `echo-inference-state` owns instance identity, one-current-state storage,
  exclusive transaction leases, atomic process-local replacement, and
  rollback-on-drop. It knows nothing about model tensors.
- `echo-inference` admits the specialized Qwen layout, binds model weights,
  executes the model, owns composite KV/GDN state, renders the admitted chat
  template, parses Qwen tool output, persists current state, and serves the
  local NDJSON protocol.
- `@echo-chamber/native-inference-adapter` maps protocol version 7 to the
  provider-neutral `ModelPort`. It owns only process-local continuation
  capability and lifecycle metadata, never model tensors.
- `@echo-chamber/local-runtime` owns one native child process, one stable
  adapter object per E.C.H.O. existence, per-instance session exclusion, and
  snapshotting at thinking-session boundaries.

There is deliberately no generic backend registry, interchangeable model
plugin layer, OpenAI-compatible local server, or JavaScript model runtime in
this path.

## Process and request flow

```text
ThinkingEngine session
  -> LocalNativeInferenceRuntime
     -> stable NativeInferenceModel for one existence
        -> protocol-v7 command over NDJSON
           -> one resident Rust model owner
              -> exclusive state transaction
                 -> MLX/Metal execution
                 -> commit or rollback
        -> completed provider-neutral output
  -> session-boundary snapshot when state is dirty
     -> atomic current.safetensors replacement
```

The resident engine loads the model plan, GPU owner, tokenizer, bound weights,
and custom kernels once. Requests for different existences share those static
resources but never share mutable inference state. Execution requires mutable
access to the resident owner, so generations are serialized. The bounded
request queue controls memory and backpressure; it is not a parallel batching
scheduler.

Parallel generation across different existences is planned but deliberately
deferred. The current process can keep multiple isolated states resident and
accept multiple outstanding requests, yet one mutable engine executes them in
FIFO order. A future implementation must compare continuous batching with
multiple execution contexts and preserve per-instance exclusion, cancellation,
fair scheduling, bounded memory, and exact KV/GDN state isolation before this
contract can claim parallel generation.

## Composite state invariant

For each E.C.H.O. existence, the process-local store contains either no state
or one current `CommittedState<MlxInferenceState>`. A committed state binds:

- the stable E.C.H.O. instance ID;
- the complete model identity: architecture plus config, weight, tokenizer,
  and chat-template digests;
- both convolution and recurrent tensors for every GDN layer;
- both key and value tensors for every full-attention layer.

The attention tensors themselves provide the current sequence length. Every
attention layer must report the same length, and every state is validated
against the admitted hybrid layer schedule, shape, dtype, and batch size.

The store does not keep generations, revisions, text history, decoded
assistant output, or a full token-ID history. State replacement is safe
without a stale-write cursor because:

1. the local runtime excludes overlapping thinking sessions for one instance;
2. the adapter excludes overlapping requests for its stable model object;
3. the Rust store permits only one active lease per instance;
4. the native process serializes model execution;
5. the durable directory is held by one process-lifetime advisory owner lock.

A pending transaction is never visible through `current()`. Successful commit
replaces the one current value atomically. Cancellation, observer failure,
model failure, validation failure, or a dropped lease leaves the previous
committed value unchanged.

## Request state transitions

Every generation explicitly selects one of three transitions.

### `initial`

The instance must not yet have state. The runtime constructs empty GDN and
attention state, processes a complete prompt, and commits the result.

### `continuation`

The instance must have state. The runtime reuses the complete current KV/GDN
payload and processes only the newly encoded suffix. The current admitted chat
path limits this suffix to tool-result items, because ordinary user/developer
history can cause Qwen's template to rewrite earlier thinking and is not
necessarily append-only.

Production completions always advance a Qwen end-of-message token into state.
Consequently a tool-result continuation can encode only the new Qwen tool
envelope; it does not need the preceding text or a separately persisted full
token sequence.

### `new_session`

The instance must have state. The transition preserves every GDN convolution
and recurrent tensor, replaces every attention key/value tensor with an empty
cache, then processes a complete fresh prompt. The old current state remains
intact until this whole request commits, so a failure cannot leave a
half-cleared existence.

The adapter chooses the transition as follows:

| Adapter condition                              | Transition     |
| ---------------------------------------------- | -------------- |
| no live state                                  | `initial`      |
| live response token is supplied                | `continuation` |
| state exists and no response token is supplied | `new_session`  |

`previousResponseToken` is a process-local continuation capability. Its
presence, combined with a token previously issued by that live adapter object,
is the signal; its string contents are intentionally opaque and are not
compared with a durable cursor. It is never stored in safetensors. After a
restart, restored state exists but no live response token does, so the first
request is `new_session`.

## Generation and transaction boundaries

The runtime executes the complete newly encoded input once, then repeatedly:

1. samples one token from the current logits;
2. builds and schedules the model step that advances that token;
3. reads the sampled scalar so the observer can classify terminal EOS;
4. when diagnostic streaming is enabled, incrementally decodes and emits it;
5. continues until EOS, cancellation, failure, or the visible token limit.

A streamed token is provisional. Only a terminal `completed` event means the
new state committed. Production disables per-token events; diagnostics and
external-TTFT gates enable them explicitly. If cancellation becomes visible at
any boundary, or if an enabled stream consumer fails, the request returns
`cancelled`/`failed` and the lease rolls back even if provisional token events
were already emitted. The caller must discard that request's partial output.

On normal Qwen EOS, the EOS token is advanced into model state and the request
commits with `stop_token`. If the visible `max_new_tokens` limit is reached,
production reserves one extra KV slot and advances an internal EOS model step
before committing. The closing EOS is neither streamed nor included in
`generated_tokens`; the response still reports `length`. This gives the next
tool-result continuation the same closed assistant boundary as an ordinary
completion while preserving honest truncation reporting.

The TypeScript model accepts the committed lifecycle state before raising
`NativeInferenceIncompleteGenerationError`. That error carries the new opaque
response token, allowing a higher layer to accept the truncated semantic result
explicitly. Selective rollback to the beginning of a partial tool call is not
implemented; a cancelled or failed request rolls back in full.

## Durable publication and restart

Each instance owns one directory:

```text
<snapshot-directory>/<instance-id>/
├── .owner.lock
└── current.safetensors
```

`open_state` creates the directory if needed, obtains a non-blocking exclusive
advisory lock, rejects legacy `current.json`, removes only managed hidden
staging remnants, and restores `current.safetensors` when present. The native
engine retains the owner object, and therefore the lock, until process exit.

The current safetensors file contains all model-state tensors plus exactly
three metadata entries:

- schema version;
- instance ID;
- serialized composite model identity.

Restore validates exact metadata equality, tensor names, tensor count, layer
kind, shape, dtype, batch size, and consistent attention sequence length before
installing state into an empty process-local slot.

`snapshot` follows this order:

1. validate the committed state and model identity;
2. evaluate and synchronize every MLX state tensor;
3. save a hidden, uniquely named safetensors staging file;
4. synchronize that file;
5. atomically rename it over `current.safetensors`;
6. synchronize the instance directory;
7. acknowledge `snapshot_published`.

The fixed file is therefore the authority by name; no `current.json`, sidecar
manifest, digest journal, or old generation is needed. A crash exposes the old
complete file, the new complete file, or an ignored managed staging remainder.
The startup cleanup preserves unknown files. Legacy roots require an explicit
migration or archive operation rather than silent deletion.

The restart contract is intentionally session-boundary recovery. The local
runtime snapshots dirty state after one complete `ThinkingEngine.think()`
invocation and on shutdown. If a later tool/session action fails after an
earlier model request committed, that earlier current state is still
snapshotted. A process crash can lose commits made since the last successful
snapshot; resuming in the middle of that thinking session is unsupported.

## Protocol version 7

The local child-process protocol accepts only:

- `open_state { request_id, instance_id, snapshot_root }`;
- `generate { request_id, instance_id, state_transition, stream_tokens, input,
tools, max_new_tokens, sampling }`;
- `cancel { request_id }`;
- `snapshot { request_id, instance_id }`;
- `shutdown`.

State roots must be opened before generation or snapshot. A root is bound to
one instance and one resident owner; later snapshot commands cannot redirect
publication to a caller-selected path.

The main thread owns MLX and mutable inference. A reader thread remains able to
mark cancellation while MLX executes, and a bounded writer channel provides
backpressure for enabled diagnostic streams. `stream_tokens: false` is the
production path and omits incremental decoding plus provisional token events.
Active plus queued generation requests are bounded and duplicate request IDs
fail admission.

Terminal events have strict meanings:

- `completed`: state committed;
- `cancelled`: active request rolled back;
- `failed`: active request did not commit;
- `state_opened`: owner lock acquired and optional current state restored;
- `snapshot_published`: fixed current file atomically replaced and synced.

The protocol never sends model state, full prior prompts, revisions, or token
history back to TypeScript. `state_sequence_length` is an observation derived
from committed attention tensors, used for metrics and verification rather
than as a caller-owned precondition.

## Specialized model path

The current model plan admits the local affine-Q4 `qwen3_5_moe` artifact:

- 40 decoder layers;
- 30 GDN and 10 full-attention layers;
- 256 routed experts with top-8 selection;
- checkpoint-specific Q4/Q8 quantization overrides.

The model path includes embedding, RMS normalization, GDN depthwise
convolution and GatedDelta kernel, partial RoPE, grouped-query causal
attention, routed and shared experts, final normalization, and the untied
quantized language head. Static weights are bound once into typed handles.
Request-local attention storage grows in capacity blocks and is compacted to
the exact logical state at commit.

Batch-one decode uses specialized GDN and MoE Metal paths where retained
evidence showed dispatch and graph-construction overhead mattered. The fused
router preserves MLX's BF16 softmax and selection semantics while retaining
the 256-way intermediate inside threadgroup memory. Generic shapes and prefill
remain on MLX operations. Numerical parity, not approximate similarity, is the
admission rule for the retained oracle fixtures.

The production sampler follows the admitted Rapid-MLX operation order:
generated-output-only presence penalty, log-softmax, top-p, top-k,
temperature, and categorical sampling with a request-local MLX key. E.C.H.O.'s
current profile is temperature `0.7`, top-p `0.8`, top-k `20`, neutral min-p
and repetition penalty, and presence penalty `1.5`.

## Validation boundary and remaining work

Unit tests cover state ownership, rollback, transition mapping, protocol
validation, fixed-path publication metadata, owner locking, staging cleanup,
adapter cancellation, length completion handling, and local lifecycle
coordination. MLX-linked Rust tests cover the existing operator and numerical
fixtures. Real-model probes cover the full adapter flow and process restart;
they must be rerun whenever the durable format or finish semantics change.

Still outside the admitted production boundary:

- partial-tool-call rollback;
- recovery in the middle of one thinking session;
- multiple simultaneous native owners for one instance directory;
- arbitrary Qwen chat-template changes or vision input;
- Qwen3.5-122B-A10B and REAP-pruned model admission;
- proof that every broader prompt, context length, and thermal state retains
  the earlier performance envelope.
