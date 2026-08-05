# E.C.H.O. inference engine

This directory contains E.C.H.O. Chamber's specialized native inference
engine. It is a Cargo workspace inside the polyglot monorepo; it does not use
pnpm. The narrow TypeScript boundary lives in
`packages/native-inference-adapter`, and `apps/local-runtime` owns the child
process plus per-existence lifecycle.

The admitted model family is Qwen3.5-style hybrid MoE, with
Qwen3.6-35B-A3B-MLX-4bit as the current primary artifact. The implementation
contains the complete model path plus variable-width continuous batching:
embeddings, GDN and full-attention layers, Q4/Q8 projections, sparse routed
and shared experts, final normalization, sampling, chat rendering, tool
parsing, and KV/GDN state carry.
The numerical path and retained performance work are described in
`evidence/`; those dated reports remain historical evidence and may describe
the older state protocol used when they were recorded.

## Current state contract

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
same-turn result. The main thought path integrates both outputs.

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

Exact `continuation` currently accepts only tool-result input. In particular,
a persistent memory or emotion lane is a lightweight tool loop: each
invocation ends in one valid update tool call, and the next observation is
returned as the result of that exact pending call. A fabricated tool result
after a normal assistant message is not an admitted semantic boundary. This
constraint preserves the official Qwen chat-template prefix without replaying
or separately storing the prior token sequence. The adapter rejects a
continuation before native execution unless its ordered tool-result call IDs
exactly match the preceding completion's pending calls.

For the bounded cross-session quality experiment only,
`ECHO_NATIVE_NEW_SESSION_GDN_POLICY=carry_recurrent_only` clears each GDN
layer's three-position convolution history while retaining its recurrent
matrix. `carry_convolution_only` retains that short-range history while
clearing the recurrent matrix to complete the component ablation. The default
and only production behavior remains `carry_all`; the selected policy is
reported in the Native `ready.engine` payload. Unsupported values fail at
startup.

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
independent state ownership, and valid output. The retained 16K integrated
probe found aggregate wall throughput effectively saturated at widths four
through six. The hard capacity remains six; choosing a narrower cohort for a
particular latency/fairness workload is an admission-policy decision rather
than a state-layout change.

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
It is the initial equal-length, simultaneous-arrival greedy comparison of
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

These diagnostics supplied the numerical and performance gates used by the
production continuous scheduler. Different batch-width floating-point paths
are not required to be bit-exact with each other; official MLX-LM exhibits the
same shape dependence. Admission instead requires official-oracle parity
within each shape, exact co-tenant and row-placement invariance, correct
per-lane state ownership, and representative output quality. Raw JSON belongs
in the ignored local diagnostic archive; retain conclusions under `evidence/`
instead of committing repeated machine-local attempts.

The parity manifests above describe oracle fixtures; they are unrelated to the
production durable-state layout.

## Local protocol

`serve-stdio` reads one JSON command per stdin line and writes one typed event
per stdout line. The second argument bounds active plus waiting generation
requests. Protocol version 9 admits:

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

The production-scheduler probe covers six-row admission, late joining,
independent cancellation, survivor commit, and retry from the prior state:

```sh
ECHO_NATIVE_LIBRARY_PATH=/absolute/path/to/mlx-c/build:/absolute/path/to/mlx/lib \
pnpm --filter @echo-chamber/native-inference-adapter probe:continuous-batch \
  /absolute/path/to/echo-inference \
  /absolute/path/to/Qwen3.6-35B-A3B-MLX-4bit
```

The three-E.C.H.O. module probe exercises durable main lanes, ephemeral memory
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
