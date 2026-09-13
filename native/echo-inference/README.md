# E.C.H.O. inference engine

This directory contains E.C.H.O. Chamber's specialized native inference
engine. Build and run the Rust workspace with Cargo. The
[TypeScript adapter](../../packages/native-inference-adapter) maps requests
to the local protocol, and [local runtime](../../apps/local-runtime) owns the
child process and per-existence lifecycle.

The admitted model family is Qwen3.5-style hybrid MoE, with
Qwen3.6-35B-A3B-MLX-4bit as the current primary artifact. The implementation
contains the complete model path plus variable-width continuous batching:
embeddings, GDN and full-attention layers, Q4/Q8 projections, sparse routed
and shared experts, final normalization, sampling, chat rendering, tool
parsing, and KV/GDN state carry.
See [architecture.md](docs/architecture.md) for component responsibilities and
state invariants. The [evidence archive](evidence/README.md) records numerical
and performance measurements with their model, runtime and workload conditions.

## Application integration boundary

E.C.H.O.'s [Cognitive Module workflow](../../docs/cognitive-module-architecture.md)
runs Memory and Emotion before each Main turn and at session completion. Native
accepts their committed results in Main's continuation input.

Running Memory/Emotion themselves on Native requires request-level structured
output, output limits, abort propagation, and state boundaries for each module
activation. Local domain storage and the application entry point also require
integration. See [Native runtime integration](../../docs/native-runtime-integration-readiness.md)
for these requirements and the validation coverage of each boundary.

## State contract

One independently named state lane owns exactly one current composite
inference state:

- all GDN convolution and recurrent tensors;
- all full-attention key/value tensors;
- the exact composite model identity: architecture plus config, weights,
  tokenizer, and chat-template digests.

There is no revision history, rollback generation, current pointer, sidecar
manifest, or separately persisted token sequence. Same-lane work is
serialized, so the resident owner can replace the single current state
directly. Different lanes remain isolated while sharing one loaded model.

The local E.C.H.O. composition opens three stable lanes per existence:

- `main`: durable and eligible to publish `current.safetensors`;
- `memory`: process-local and ephemeral;
- `emotion`: process-local and ephemeral.

Memory and emotion may generate in parallel from their own KV/GDN states.
They never commit into `main`, never snapshot, and do not consume each other's
same-turn result. The module-workload probe exercises a main thought path
that consumes both outputs.

The three request transitions are:

- `initial`: no current state exists; start from empty GDN and attention state;
- `continuation`: reuse the complete current KV/GDN state and process only the
  newly supplied suffix;
- `new_session`: by default retain the current GDN convolution and recurrent
  state, clear every attention KV cache, and process a complete fresh prompt.

The TypeScript adapter derives that transition from two facts. A
`previousResponseToken` supplied after a successful response from the same
live adapter process selects `continuation`. If it is absent, an existing state
selects `new_session`, otherwise the request is `initial`. The token is an
opaque, process-local continuation capability, not an LLM token or durable
cursor. Its contents are not decoded or persisted. Restoring a process starts
with state but no live response token, so its first request is necessarily a
`new_session`.

Exact `continuation` accepts the ordered results for Main's pending calls,
followed optionally by complete runtime-owned tool call/result pairs. The
runtime marks each appended call with `origin: "runtime"` only after its
Cognitive phase is committed. Model output cannot supply this provenance,
and matching a function name or call-ID prefix is insufficient. Runtime
exchanges are already-resolved input history; they do not add executable
Main tools.

The adapter checks Main's pending result IDs, order and count before any
runtime exchange. Both adapter and Rust renderer reject unmarked calls,
missing/mismatched results and duplicate IDs within the suffix. With no
pending call, an empty retry or complete runtime exchanges are admitted;
arbitrary user/developer/assistant messages still require `new_session`.
This preserves the official Qwen template and the committed EOS boundary
without reconstructing prior output or replaying the token history.

`ECHO_NATIVE_NEW_SESSION_GDN_POLICY` selects the GDN components retained by
`new_session`. The production policy is `carry_all`. For component-isolation
experiments, `carry_recurrent_only` clears the three-position convolution
history, and `carry_convolution_only` clears the recurrent matrix. The selected
policy is reported in `ready.engine`; unsupported values fail at startup.

Normal EOS completion commits. Cancellation, diagnostic-stream delivery
failure, and model or protocol errors roll back the active transaction and
leave the preceding current state untouched. If the visible output limit is
reached, production reserves one additional state slot, advances an internal Qwen EOS through the
model, commits that closed state, and still reports `length`; the EOS is not
included in streamed or returned visible tokens. The adapter raises
`NativeInferenceIncompleteGenerationError` with the new response token so a
caller must explicitly decide whether to accept that incomplete semantic
result.

Selective rollback of only a partial tool call is intentionally deferred. A
cancelled or failed request currently rolls back the whole request. Restarting
in the middle of one live thinking session is also outside the contract; only
the last published session boundary is recoverable.

## Continuous batching

The production stdio scheduler starts a lone request immediately. Before its
first decode step it may admit up to six already-ready state lanes; after
decode begins, late joining is capped at width four. Membership changes occur
only after a sampled token has advanced model state. Completed or cancelled
rows are split back to compact, independently owned KV/GDN states while their
survivors continue. Work beyond the active limit remains in the bounded queue.

Batch widths are not required to match each other bit-for-bit because the
floating-point execution shape changes. Admission instead requires exact
official-MLX parity within each shape, co-tenant and row-position invariance,
independent state ownership, and valid output. The hard capacity is six;
choosing a narrower cohort depends on the workload's latency and fairness
requirements.

## Durable state

Each durable lane directory contains one authoritative payload. The auxiliary
memory and emotion lanes have no durable directory:

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

Publication evaluates and synchronizes the tensors, writes a uniquely named
hidden staging file, synchronizes the file, atomically renames it over
`current.safetensors`, and synchronizes the instance directory. A crash
therefore exposes either the preceding complete current file or the complete
replacement. On startup, only managed `.current.safetensors.tmp-*` remnants
are removed; unknown operator files are preserved.

Legacy `current.json` roots fail closed and are never silently deleted or
interpreted. They must be archived or migrated explicitly before this engine
opens the instance.

## Build

Run the Cargo commands in this README from `native/echo-inference`. The
workspace uses an external, pinned official `mlx-c` checkout/build and the
matching MLX library. Export their paths before building or running commands:

```sh
export MLX_C_INCLUDE_DIR=/absolute/path/to/mlx-c
export MLX_C_LIB_DIR=/absolute/path/to/mlx-c/build
export MLX_LIB_DIR=/absolute/path/to/python/site-packages/mlx/lib
export DYLD_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/python/site-packages/mlx/lib
export CARGO_TARGET_DIR=/tmp/echo-inference-target

cargo build --release -p echo-inference
cargo test --workspace --all-features -- --test-threads=1
```

The build fails closed when any path is absent. Generated bindings are limited
to the MLX surface used by the engine. `DYLD_LIBRARY_PATH` is needed for local
development when those libraries are not installed in a loader-visible path;
production packaging should use a stable loader-relative layout.

MLX-linked tests require access to a Metal device. Ordinary tests do not load
model weights; explicitly invoked real-model probes and the state-integrity
test below require a local model directory as well.

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
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-parallel-generation-diagnostic \
  /absolute/path/to/model 1 3 128 \
  /absolute/path/to/local-result.json
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-resident-batch-oracle-parity \
  /absolute/path/to/model \
  /absolute/path/to/resident-batch-oracle
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-resident-batch-context-diagnostic \
  /absolute/path/to/model 1 2 64 \
  /absolute/path/to/local-context-result.json
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-production-batch-quality-diagnostic \
  /absolute/path/to/model 1 2 64 4096 3 \
  /absolute/path/to/local-quality-result.json
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-batch-width-scaling-diagnostic \
  /absolute/path/to/model 6 1 2 64 \
  /absolute/path/to/local-width-result.json
cargo run --release -p echo-inference \
  --features parallel-generation-diagnostics -- \
  run-production-batch-width-scaling-diagnostic \
  /absolute/path/to/model 6 1 2 64 \
  /absolute/path/to/local-production-width-result.json
cargo run --release -p echo-inference -- serve-stdio \
  /absolute/path/to/model \
  8
```

### Long-input prefill

The resident runtime keeps the single-execution path below 8,192 newly
executed input tokens. Inputs at or above that boundary are processed as
sequential 2,048-token model executions while carrying the complete in-memory
KV and GDN state between executions. Previously committed prefix tokens are
reported separately as `cached_prefix_tokens` and do not count toward this
boundary. This bounds long-prefill intermediate memory without changing decode
or the short-input graph.

Both boundaries can be overridden at process startup:

```bash
ECHO_NATIVE_PREFILL_CHUNK_SIZE_TOKENS=4096 \
ECHO_NATIVE_PREFILL_CHUNK_AT_OR_ABOVE_TOKENS=8192 \
cargo run --release -p echo-inference -- serve-stdio \
  /absolute/path/to/model \
  8
```

Setting `ECHO_NATIVE_PREFILL_CHUNK_SIZE_TOKENS=0` disables chunking. A response
reports `input_model_execution_count`, so a caller can distinguish one logical
input from the number of model executions used to process it.

Chunking is mathematically equivalent to one full prefill, but it is not a
bit-exact transformation of BF16 hybrid-model state: GDN scans and downstream
layers accumulate floating-point operations in a different execution shape.
The 2,048-token path matches MLX-LM's corresponding default prefill shape.
E.C.H.O. therefore treats the chunked result as the canonical state for long
inputs instead of comparing it bit-for-bit with the single-execution state.

`run-moe-performance-diagnostic` is excluded from ordinary builds. Set
`ECHO_MOE_PERFORMANCE_MODE` to `full`, `none`, `router_only`, `routed_only`,
or `shared_only`. Every mode except `full` deliberately changes model output
and is valid only for fixed-length component-cost diagnosis.

`run-parallel-generation-diagnostic` is also excluded from ordinary builds.
It compares equal-length, simultaneous-arrival greedy execution using
production FIFO, two independent MLX streams, and a fixed batch of two.
`run-resident-batch-oracle-parity` checks unequal resident caches against an
official MLX-LM fixture. `run-resident-batch-context-diagnostic` compares FIFO
and fixed batch at 4K, 16K, and 32K resident lengths. The production-quality
diagnostic uses request-owned production sampling, fixed-row isolation, a
mixed EOS/length boundary, and sampled two-turn tool workflows. Its arguments
are warmup rounds, measured rounds, generated tokens, resident context tokens,
and workflow seed-pair count.

`run-batch-width-scaling-diagnostic` measures every fixed width from one
through the requested maximum (currently at most six) at 4K, 16K, and 32K.
It rotates execution order, reports aggregate and per-request decode rates plus
Metal allocation, verifies same-shape co-tenant and row-permutation isolation
at the maximum width, and exercises exact state accounting while membership
shrinks one row at a time from the maximum to one. Its arguments are maximum
batch width, warmup rounds, measured rounds, and generated tokens per row.
`run-production-batch-width-scaling-diagnostic` applies the same width sweep
at 4K with each row's current production sampling configuration, seed, and
generated-token presence history kept request-owned. It repeats maximum-width
co-tenant and row-permutation isolation under sampling.

Use these diagnostics when changing model execution or scheduling. Record the
model, runtime, sampling and batch shape with each comparison. Store raw JSON
in the ignored local artifact directory; retain reproducible conclusions under
`evidence/` with their measurement conditions.

The parity manifests above describe oracle fixtures; they are unrelated to the
production durable-state layout.

## Local protocol

`serve-stdio` reads one JSON command per stdin line and writes one typed event
per stdout line. The second argument bounds active plus waiting generation
requests. Protocol version 11 admits:

- `open_state`: register either a durable lane with a fixed snapshot root or
  an ephemeral process-local lane;
- `generate`: process one `initial`, `continuation`, or `new_session` request;
- `cancel`: request rollback at the next cancellation boundary;
- `snapshot`: atomically replace a durable lane's fixed current payload;
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

Adapter and engine must both use protocol version 11; mismatched versions are
rejected at startup. The snapshot format has its own schema version, validated
when opening durable state.

## Cognitive continuation validation

`oracles/qwen35_cognitive_continuation_parity.py` derives suffix fixtures from
the installed model's official Transformers/Jinja template. It asserts that
the committed prefix ends at EOS and that prefix tokens plus suffix tokens
exactly equal the complete prompt tokens. The checked-in
`fixtures/cognitive-continuation.json` covers a tool result or plain Main
completion, each with and without Cognitive exchanges. Rust unit tests check
the rendered bytes; `run-chat-template-parity` also
checks the actual tokenizer IDs:

```sh
python oracles/qwen35_cognitive_continuation_parity.py \
  --model /absolute/path/to/model \
  --output /absolute/path/to/cognitive-continuation.json
cargo run --release -p echo-inference -- run-chat-template-parity \
  /absolute/path/to/model /absolute/path/to/cognitive-continuation.json
```

The fixture schema is version 2; version 1 full-prompt fixtures remain supported.

## Real-model probes

A probe is an explicitly invoked integration-check script that starts the
Native process and runs real model inference through the TypeScript adapter.
It records outputs, usage, timing and state observations as JSON. Run the pnpm
commands in this section from the repository root after building the release
binary and configuring the MLX libraries.

`probe:real-model` exercises two sessions with two generations each: a model
tool call, a supplied tool result, and a continuation that answers with the
returned code. It checks tool-call parsing and arguments, result consumption,
resident-prefix accounting, and reports state-length advancement.

Its final argument selects `plain` (tool results only, the default) or
`cognitive` (also include committed Memory/Emotion exchanges). The latter uses
the Core handoff formatter before every Main turn, including the first turn of
each session. Module responses and domain state are fixtures; Main generation
uses the real model. Initial recall is empty, and continuation recall contains
the observed lookup result. The sampling profile is `greedy` or `production`.

For example:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:real-model \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit \
  42 \
  greedy \
  cognitive
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

The production-scheduler probe covers six-row admission, late joining,
independent cancellation, survivor commit, and retry from the prior state:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:continuous-batch \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit
```

The three-E.C.H.O. module probe uses a synthetic one-call-per-observation tool
loop to exercise durable main lanes, ephemeral memory
and emotion lanes, exact pending-tool continuations, cancellation retry, and
main-only publication. The 16K soak reuses that valid tool loop while sweeping
active widths three through six:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:module-workload \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit

ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:16k-soak \
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

## Real-model state integrity test

The Rust `state_integrity` test compares every KV/GDN tensor after interrupted
generation and retry, including cancellation of one row in a six-row batch.
It freezes independent tensor references in temporary safetensors files so
shared GPU buffers cannot hide a mutation. See the
[validation contract](docs/architecture.md#validation) for the comparison
conditions and coverage.

Ordinary `cargo test` skips this model-dependent test. With the MLX environment
configured, run it from `native/echo-inference`:

```sh
ECHO_NATIVE_TEST_MODEL=/absolute/path/to/model \
  cargo test --release -p echo-inference --all-features state_integrity:: \
  -- --ignored --nocapture --test-threads=1
```

Successful runs remove their temporary reference directory. Failed runs retain
the directory printed in the log for diagnosis.

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
