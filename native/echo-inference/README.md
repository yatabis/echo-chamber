# E.C.H.O. inference engine

This directory contains E.C.H.O. Chamber's specialized native inference
engine. It is a Cargo workspace inside the polyglot monorepo; it does not use
pnpm. The narrow TypeScript boundary lives in
`packages/native-inference-adapter`, and `apps/local-runtime` owns the child
process plus per-existence lifecycle.

The admitted model family is Qwen3.5-style hybrid MoE, with
Qwen3.6-35B-A3B-MLX-4bit as the current primary artifact. The implementation
contains the complete batch-one model path: embeddings, GDN and full-attention
layers, Q4/Q8 projections, sparse routed and shared experts, final
normalization, sampling, chat rendering, tool parsing, and KV/GDN state carry.
The numerical path and retained performance work are described in
`evidence/`; those dated reports remain historical evidence and may describe
the older state protocol used when they were recorded.

## Current state contract

One E.C.H.O. existence owns exactly one current composite inference state:

- all GDN convolution and recurrent tensors;
- all full-attention key/value tensors;
- the exact composite model identity: architecture plus config, weights,
  tokenizer, and chat-template digests.

There is no revision history, rollback generation, current pointer, sidecar
manifest, or separately persisted token sequence. Same-instance work is
serialized, so the resident owner can replace the single current state
directly. Different existences remain isolated while sharing one loaded model.

The three request transitions are:

- `initial`: no current state exists; start from empty GDN and attention state;
- `continuation`: reuse the complete current KV/GDN state and process only the
  newly supplied suffix;
- `new_session`: retain the current GDN state, clear every attention KV cache,
  and process a complete fresh prompt.

The TypeScript adapter derives that transition from two facts. A
`previousResponseToken` supplied after a successful response from the same
live adapter process selects `continuation`. If it is absent, an existing state
selects `new_session`, otherwise the request is `initial`. The token is an
opaque, process-local continuation capability, not an LLM token or durable
cursor. Its contents are not decoded or persisted. Restoring a process starts
with state but no live response token, so its first request is necessarily a
`new_session`.

Normal EOS completion commits. Cancellation, diagnostic-stream delivery
failure, and model or protocol errors roll back the active transaction and
leave the preceding current state untouched. If the visible output limit is reached, production
reserves one additional state slot, advances an internal Qwen EOS through the
model, commits that closed state, and still reports `length`; the EOS is not
included in streamed or returned visible tokens. The adapter raises
`NativeInferenceIncompleteGenerationError` with the new response token so a
caller must explicitly decide whether to accept that incomplete semantic
result.

Selective rollback of only a partial tool call is intentionally deferred. A
cancelled or failed request currently rolls back the whole request. Restarting
in the middle of one live thinking session is also outside the contract; only
the last published session boundary is recoverable.

## Durable state

Each instance directory contains one authoritative payload:

```text
<snapshot-directory>/<instance-id>/
├── .owner.lock
└── current.safetensors
```

The native process holds the advisory owner lock for its lifetime. Startup
loads `current.safetensors` when present and validates every tensor name,
shape, dtype, hybrid-layer position, instance identity, and model identity.
The safetensors metadata contains only:

- `echo_schema_version`;
- `echo_instance_id`;
- `echo_model_identity`.

Publication writes a uniquely named hidden staging file, evaluates and
synchronizes the tensors, synchronizes the file, atomically renames it over
`current.safetensors`, and synchronizes the instance directory. A crash
therefore exposes either the preceding complete current file or the complete
replacement. On startup, only managed `.current.safetensors.tmp-*` remnants
are removed; unknown operator files are preserved.

Legacy `current.json` roots fail closed and are never silently deleted or
interpreted. They must be archived or migrated explicitly before this engine
opens the instance.

## Build

The workspace deliberately does not vendor MLX. Point it at a pinned official
`mlx-c` checkout/build and the matching MLX library:

```sh
MLX_C_INCLUDE_DIR=/absolute/path/to/mlx-c \
MLX_C_LIB_DIR=/absolute/path/to/mlx-c/build \
MLX_LIB_DIR=/absolute/path/to/python/site-packages/mlx/lib \
DYLD_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/python/site-packages/mlx/lib \
CARGO_TARGET_DIR=/tmp/echo-inference-target \
cargo test --workspace
```

The build fails closed when any path is absent. Generated bindings are limited
to the MLX surface used by the engine. `DYLD_LIBRARY_PATH` is needed for local
development when those libraries are not installed in a loader-visible path;
production packaging should use a stable loader-relative layout.

## Commands

```sh
cargo run -p echo-inference -- probe-mlx
cargo run -p echo-inference -- inspect-model /absolute/path/to/model
cargo run -p echo-inference -- inspect-checkpoint \
  /absolute/path/to/current.safetensors
cargo run -p echo-inference -- run-gdn-layer-parity \
  /absolute/path/to/model \
  /absolute/path/to/gdn-layer.safetensors \
  /absolute/path/to/gdn-layer.manifest.json
cargo run -p echo-inference -- run-decoder-layer-parity \
  /absolute/path/to/model \
  /absolute/path/to/decoder-layer.safetensors \
  /absolute/path/to/decoder-layer.manifest.json
cargo run -p echo-inference -- run-attention-layer-parity \
  /absolute/path/to/model \
  /absolute/path/to/attention-layer.safetensors \
  /absolute/path/to/attention-layer.manifest.json
cargo run -p echo-inference -- run-hybrid-block-parity \
  /absolute/path/to/model \
  /absolute/path/to/hybrid-block.safetensors \
  /absolute/path/to/hybrid-block.manifest.json
cargo run -p echo-inference -- run-full-model-parity \
  /absolute/path/to/model \
  /absolute/path/to/full-model.safetensors \
  /absolute/path/to/full-model.manifest.json
cargo run -p echo-inference -- run-live-state-parity \
  /absolute/path/to/model \
  /absolute/path/to/full-model.safetensors \
  /absolute/path/to/full-model.manifest.json
cargo run --release -p echo-inference -- run-resident-runtime-parity \
  /absolute/path/to/model \
  /absolute/path/to/full-model.safetensors \
  /absolute/path/to/full-model.manifest.json
cargo run --release -p echo-inference -- run-new-session-parity \
  /absolute/path/to/model \
  /absolute/path/to/full-model.safetensors \
  /absolute/path/to/full-model.manifest.json
cargo run --release -p echo-inference -- run-durable-state-parity \
  /absolute/path/to/model \
  /absolute/path/to/full-model.safetensors \
  /absolute/path/to/full-model.manifest.json \
  /absolute/path/to/instance-state-root
cargo run -p echo-inference -- run-chat-template-parity \
  /absolute/path/to/model \
  /absolute/path/to/chat-template.manifest.json
cargo run -p echo-inference -- run-sampling-parity \
  /absolute/path/to/qwen35-production-sampling.fixture.json
cargo run --release -p echo-inference \
  --features moe-performance-diagnostics -- \
  run-moe-performance-diagnostic \
  /absolute/path/to/model 1 3 128 \
  /absolute/path/to/local-result.json
cargo run --release -p echo-inference -- serve-stdio \
  /absolute/path/to/model \
  8
```

`run-moe-performance-diagnostic` is excluded from ordinary builds. Set
`ECHO_MOE_PERFORMANCE_MODE` to `full`, `none`, `router_only`, `routed_only`,
or `shared_only`. Every mode except `full` deliberately changes model output
and is valid only for fixed-length component-cost diagnosis.

The parity manifests above describe oracle fixtures; they are unrelated to the
production durable-state layout.

## Local protocol

`serve-stdio` reads one JSON command per stdin line and writes one typed event
per stdout line. The second argument bounds active plus waiting generation
requests. Protocol version 7 admits:

- `open_state`: bind an instance to a durable directory for this process
  lifetime and restore `current.safetensors` if present;
- `generate`: process one `initial`, `continuation`, or `new_session` request;
- `cancel`: request rollback at the next cancellation boundary;
- `snapshot`: atomically replace the opened instance's fixed current payload;
- `shutdown`: close the resident owner.

Every `generate` command must set `stream_tokens`. Production requests use
`false`, avoiding incremental text decoding and per-token writer events. Tests
that need external TTFT or token-delivery diagnostics explicitly use `true`.
Those `token` events are provisional; only `completed` acknowledges a committed
state. `cancelled` and `failed` never commit the active request. A
`snapshot_published` event acknowledges durable replacement only after file and
directory synchronization.

The native protocol is a trusted-local child-process contract, not an
OpenAI-compatible HTTP API. E.C.H.O.'s provider-neutral `ModelPort` mapping is
owned by `packages/native-inference-adapter`.

## Real-model probes

After building the release binary, the live four-request adapter flow can be
run with:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:real-model \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
  42 \
  greedy
```

The cross-process recovery probe is:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:real-recovery \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
  /absolute/path/to/empty-instance-state-root
```

The length-close probe limits visible output to one token and verifies that an
additional unstreamed EOS step is present in committed state and metrics:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:real-length \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
  /absolute/path/to/empty-instance-state-root
```

The matched stream-overhead probe alternates both modes in one resident process
and verifies identical output, state length, and finish reason:

```sh
ECHO_STREAM_OVERHEAD_ROUNDS=21 \
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:stream-overhead \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit
```

The local composition probe starts two native owners in sequence, verifies
automatic restore, begins a new session, replaces the fixed current payload,
removes a managed crash remainder, and preserves an unknown file:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/local-runtime probe:real-lifecycle \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
  /absolute/path/to/empty-snapshot-directory
```

## Evidence

The Python/MLX oracle scripts under `oracles/` generate numerical fixtures for
GDN, attention, decoder, hybrid-block, full-model, chat-template, and sampling
parity. They deliberately retain fixture manifests and sometimes complete
token sequences because those artifacts authenticate an offline comparison;
that does not make them part of the production state contract.

See [architecture.md](docs/architecture.md) for the current runtime design,
[evidence/README.md](evidence/README.md) for retention policy, and the dated
evidence directories for the exact conditions and limits of earlier numerical
and performance measurements.
