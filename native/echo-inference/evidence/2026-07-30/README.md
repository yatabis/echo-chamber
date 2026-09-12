# Bootstrap evidence — 2026-07-30

## Retention

This page is the Git-candidate narrative record. The adjacent JSON set is limited
to the reusable production-sampling and chat-template fixtures plus the final
complete-model parity contract and result. Stage-level checkpoints, layer
probes, runtime-operation dumps, and process-local results were moved without
deletion to the ignored local diagnostic archive described in
[`../README.md`](../README.md).

## Environment

- Apple M4 Pro Mac mini, 64 GB unified memory
- MLX 0.32.0
- MLX-LM 0.31.3
- official MLX C commit `fba4470b89073180056c9ea46c443051375f7399`
- Rust 1.93.1

Rust reported `mlx_version=0.32.0` and `metal_available=true`.

## Rust model admission

The generated MLX C binding inspected both existing local artifacts:

| Artifact                  | Config architecture | Layers | GDN / full attention |    Experts | Shards | Tensors | Logical weight bytes |
| ------------------------- | ------------------- | -----: | -------------------: | ---------: | -----: | ------: | -------------------: |
| Qwen3.5-4B MLX 4-bit      | `qwen3_5`           |     32 |               24 / 8 |      dense |      1 |   1,221 |        3,034,147,328 |
| Qwen3.6-35B-A3B MLX 4-bit | `qwen3_5_moe`       |     40 |              30 / 10 | 256, top-8 |      4 |   2,090 |       20,401,929,952 |

Both contained
`language_model.model.layers.0.linear_attn.in_proj_qkv.weight`.

## Complete-cache continuation oracle

The oracle prefills 19 tokens, saves the complete MLX-LM cache, continues with
12 suffix tokens, reloads the saved cache into a fresh cache object, and
continues with the same suffix.

| Artifact        | Cache tensors | GDN / KV layers | Logical cache bytes | Logits max abs diff | Final cache max abs diff | Greedy token |
| --------------- | ------------: | --------------: | ------------------: | ------------------: | -----------------------: | -----------: |
| Qwen3.5-4B      |            64 |          24 / 8 |          52,133,888 |                 0.0 |                      0.0 |    271 / 271 |
| Qwen3.6-35B-A3B |            80 |         30 / 10 |          64,778,240 |                 0.0 |                      0.0 |    271 / 271 |

Rust then loaded and validated the 35B-A3B checkpoint as one hybrid payload,
including its instance, revision, token boundary, GDN/KV counts, logical bytes,
and payload SHA-256.

The generated checkpoint files were 52,140,532 and 64,786,398 physical bytes.
They are not retained in Git. Their exact SHA-256 digests and model identities
are retained here; the path-bearing manifests/results are local diagnostics:

- 4B checkpoint:
  `cb847b0240df985735400d87095c0bec083a1fa2fb024ed623940f8c23376bc7`
- 35B-A3B checkpoint:
  `06a1d551fe7ae08884af7953b5d107851146f0c41937dd914ad1a0ea5dee1401`

## Real GDN layer from Rust

The second oracle advances the official MLX-LM GDN layer for three
deterministic BF16 hidden-state positions, retains the resulting non-zero conv
and F32 recurrent states, and evaluates a four-position continuation. A traced
oracle implementation first had to match the direct official call exactly.

Rust then loaded the same real weights and states and executed affine Q4
projections, BF16 depthwise convolution, q/k normalization, beta/decay, the
official GatedDelta Metal kernel, gated RMS normalization, and the output
projection.

| Artifact        | Hidden | Output max abs diff | Conv state max abs diff | Recurrent state max abs diff | Max across 16 trace points |
| --------------- | -----: | ------------------: | ----------------------: | ---------------------------: | -------------------------: |
| Qwen3.5-4B      |  2,560 |                 0.0 |                     0.0 |                          0.0 |                        0.0 |
| Qwen3.6-35B-A3B |  2,048 |                 0.0 |                     0.0 |                          0.0 |                        0.0 |

The uncommitted binary fixtures are 6,928,607 bytes (4B) and 6,913,247 bytes
(35B-A3B). Their retained identities are:

- 4B GDN fixture:
  `9505faa3917fc9747361f953db44cb6717a7f11a5a65b295f215b99161b07546`
- 35B-A3B GDN fixture:
  `4776e9d716f7a52271448c9a9ff27ec429b6eb9b7ab15aa471a9b1ec33310908`

## Real decoder layer 0 from Rust

The third oracle advances the official Qwen3.6-35B-A3B decoder layer 0 for
three deterministic BF16 hidden-state positions, retaining its non-zero GDN
states, and then evaluates a four-position continuation. Its independent trace
reconstructs both RMS normalizations and residuals, the full GDN path, precise
Q8 router softmax and top-8 selection, Q4 routed experts, score normalization,
the Q4 shared expert, and its Q8 gate.

The direct official layer call and traced Python decomposition first matched
exactly. Rust then produced:

| Artifact        | Layer | Output max abs diff | Conv state max abs diff | Recurrent state max abs diff | Max across 38 trace points |
| --------------- | ----: | ------------------: | ----------------------: | ---------------------------: | -------------------------: |
| Qwen3.6-35B-A3B |     0 |                 0.0 |                     0.0 |                          0.0 |                        0.0 |

The binary fixture is 7,308,743 bytes and is not retained in Git. Its retained
SHA-256 identity is
`26bf38ff40668b24f438b8fe088a2c37fbd1394e788b4ac67af3afc773c56d93`.

The four-position fixture selects 32 routed experts, below MLX-LM's
64-selection threshold for sorting by expert. This proves the unsorted decode
path and stateful short continuation.

## Sorted expert dispatch

An eight-position continuation selects exactly 64 routed experts and therefore
enters MLX-LM's sorted dispatch branch. Rust reproduces the flattened indices,
sort order, inverse order, token grouping, sorted inputs/indices, three sorted
Q4 expert projections, and inverse scatter.

| Artifact        | Selections | Output max abs diff | State max abs diff | Max across retained trace points |
| --------------- | ---------: | ------------------: | -----------------: | -------------------------------: |
| Qwen3.6-35B-A3B |         64 |                 0.0 |                0.0 |                              0.0 |

The 8,638,575-byte binary fixture is not retained in Git. Its SHA-256 identity
is `c61f84939827e670d0a82c966961c4f24d38f5619d60f6a70557487b9818827f`.

## Full-attention decoder layer 3

The attention oracle advances layer 3 through a three-position prefix, then
evaluates an eight-position continuation with a causal mask. Its independent
trace covers partial `RoPE`, a non-zero offset, 16-query/2-KV-head grouped
attention, KV append, sigmoid output gating, both decoder residuals, and the
sorted sparse `MoE`.

| Artifact        | Layer | Output max abs diff | Keys max abs diff | Values max abs diff | Max across 45 trace points |
| --------------- | ----: | ------------------: | ----------------: | ------------------: | -------------------------: |
| Qwen3.6-35B-A3B |     3 |                 0.0 |               0.0 |                 0.0 |                        0.0 |

The 2,117,434-byte binary fixture is not retained in Git. Its SHA-256 identity
is `403d93a6a5d25c7e792f42f0267565c5f63ea3a5ffb8caa8ee8a39a897b69d28`.

## First hybrid block

The block oracle advances layers 0–3 through the same prefix and then evaluates
the same continuation while retaining a separate state pair for each layer.
Rust executes GDN ×3 followed by full attention ×1, with sorted sparse `MoE` in
all four layers.

| Artifact        | Layers | Layer input max abs diff | Layer output max abs diff | Eight state tensors max abs diff | Final output max abs diff |
| --------------- | -----: | -----------------------: | ------------------------: | -------------------------------: | ------------------------: |
| Qwen3.6-35B-A3B |    0–3 |                      0.0 |                       0.0 |                              0.0 |                       0.0 |

The 13,249,071-byte binary fixture is not retained in Git. Its SHA-256 identity
is `b6f5db909c09399d40a62ce6d93844730d993346cc2e653ccffc5956d016f398`.

## Complete 40-layer model and greedy generation

The full-model oracle applies the target tokenizer and chat template to
`1 + 1 はいくつですか？`, producing 19 prompt tokens. It executes a
17-token empty-state prefix and a two-token stateful continuation, retaining
embedding, every decoder-layer output, final normalized hidden state, logits,
and all 80 cache tensors. It then executes two single-token greedy steps.

The official MLX-LM model and independent Python decomposition first matched at
every retained output and state boundary. Rust loaded all four weight shards
and 2,090 tensors, then produced:

| Artifact        | Layers | Prefix / continuation | Layer outputs max abs diff | 80 state tensors max abs diff | Logits max abs diff | Generated tokens |
| --------------- | -----: | --------------------: | -------------------------: | ----------------------------: | ------------------: | ---------------- |
| Qwen3.6-35B-A3B |     40 |                17 / 2 |                        0.0 |                           0.0 |                 0.0 | `1596`, `1144`   |

Both generated-token logits and the final post-generation GDN/KV state also
matched with maximum absolute difference `0.0`. The 208,070,919-byte binary
fixture contains 331 tensors and is not retained in Git. Its retained SHA-256
identity is
`600bfc53c5fbb4404e040088136c968826962fbe691d800c99f351568dce801c`.

The exact per-boundary CLI result and a path-neutral fixture contract are
retained beside this page. Every named comparison in the retained result is
`0.0`.

## Live instance-state commit and restore

Rust now binds the full 40-layer cache to one instance and exact token
lineage. It verifies the actual fixture input tensors against the manifest
tokens, computes the runtime model identity from the real config, four weight
shards, tokenizer, and tokenizer config, then performs two atomic commits:

| Commit       | Revision | Token boundary | State tensors | Logical state bytes |
| ------------ | -------: | -------------: | ------------: | ------------------: |
| Prefix       |        1 |             17 |            80 |          64,737,280 |
| Continuation |        2 |             19 |            80 |          64,778,240 |

The direct continuation is materialized before the first model owner is
released. Rust then drops the original four-shard/2,090-tensor weight owner and
GPU handle, reloads them, obtains revision 1 only through the state store, and
runs the same two-token continuation from that committed payload.

| Comparison                                               | Maximum absolute difference |
| -------------------------------------------------------- | --------------------------: |
| Fixture token tensors vs manifest token sequence         |                         0.0 |
| Prefix execution vs official oracle                      |                         0.0 |
| Direct continuation vs official oracle                   |                         0.0 |
| State-store-restored continuation vs official oracle     |                         0.0 |
| State-store-restored continuation vs direct continuation |                         0.0 |

Each execution comparison covers embedding, all 40 layer outputs, final
normalized hidden state, logits, and all 80 state tensors. The model identity
also exactly matches the retained checkpoint identity. The process-local raw
result was captured as `qwen36-35b-a3b-live-state-result.json` and is preserved
only in the ignored diagnostic archive; the exact comparisons remain in this
page.

This proves in-process ownership transfer and exact revision advancement. The
new execution owner shares the process-wide MLX runtime/default Metal stream;
the test does not serialize state, start a second process, or measure a
production latency metric.

## Durable atomic publication and process restart

The durable command runs the model only in sequential child processes. The
producer evaluates the same prefix and direct continuation, commits revision
1, serializes its 80 tensors, publishes the revision, and exits. The restorer
starts afterward, authenticates revision 1 from disk, restores it into a new
process-local state store, executes the continuation, commits revision 2, and
publishes that revision through the same path.

| Phase    |   PID | Revision | Token boundary | Logical bytes | Physical bytes | Payload SHA-256                                                    |
| -------- | ----: | -------: | -------------: | ------------: | -------------: | ------------------------------------------------------------------ |
| Producer | 20255 |        1 |             17 |    64,737,280 |     64,745,933 | `241481fb47358a026a59edbcb32ec6f9263b93ff4cbcc2e1c4c8ed5fea0ef35d` |
| Restorer | 20354 |        2 |             19 |    64,778,240 |     64,786,893 | `a8ec952b21013def5190c0b4f10b81128fc8cd6f4bf7c88a8293ade432114ad5` |

Each immutable revision was created as a hidden sibling directory. Rust
materialized and synchronized the MLX arrays, wrote and synchronized the
safetensors payload, computed its digest, wrote and synchronized the binding
manifest, synchronized the staging directory, published it with an exclusive
atomic directory rename, and synchronized the revision root. The exclusive
rename is covered by a regression test that verifies an existing revision is
not replaced.

Schema 2 binds the instance, exact revision and token digest, model/config/
weights/tokenizer/template identity, batch size, hybrid schedule, tensor
layout, tensor count, logical bytes, and payload digest. The same identity is
embedded in safetensors metadata and checked during restoration. The restored
state then passed all 80 shape/dtype checks, including exact attention KV
length.

| Comparison                                           | Maximum absolute difference |
| ---------------------------------------------------- | --------------------------: |
| Fixture token tensors vs manifest token sequence     |                         0.0 |
| Producer prefix vs official oracle                   |                         0.0 |
| Producer direct continuation vs official oracle      |                         0.0 |
| New-process restored continuation vs official oracle |                         0.0 |
| New-process restored logits vs official oracle       |                         0.0 |
| New-process restored 80-state payload vs oracle      |                         0.0 |

The direct and restored continuations are therefore exactly equal through
their common bit-identical oracle. The orchestrator also loaded revision 2
after the restorer exited and revalidated its payload digest and binding
manifest. A second complete run from the final source produced the same two
payload digests, despite using different process IDs and a new revision root.
The final-run process dump was captured as
`qwen36-35b-a3b-durable-state-result.json` and moved to the ignored diagnostic
archive. The 129.5 MB of binary checkpoint payloads also remain outside Git;
the durable assertions and exact differences are summarized here.

## Resident model owner and FIFO runtime

The resident-runtime command loaded the four shards and 2,090 tensor handles
once, then placed six typed requests into one bounded FIFO:

| Ticket | Instance  | Operation                        | Result                         |
| -----: | --------- | -------------------------------- | ------------------------------ |
|      1 | Rin       | 17-token new lineage             | revision 1                     |
|      2 | Marie     | 17-token new lineage             | revision 1                     |
|      3 | Rin       | 2-token suffix + 2 greedy tokens | revision 2, `1596`, `1144`     |
|      4 | Rin       | deliberately wrong cached prefix | rejected; revision stayed at 2 |
|      5 | Marie     | 2-token suffix                   | revision 2                     |
|      6 | cancelled | waiting new lineage              | removed before execution       |

Tickets 1–5 executed in that order through resident engine 1. Rin and Marie
held distinct committed state objects. Ticket 4 acquired the exact revision-2
lease, detected a full-lineage digest mismatch, and dropped the transaction;
the committed object and revision were unchanged. Ticket 5 then ran
successfully, establishing that one failed request does not poison the FIFO or
another instance.

The runtime path evaluates logits plus the 80 GDN/KV tensors, but does not
retain or evaluate the 40 parity-only intermediate layer outputs. Each
successful state was still compared with the retained official oracle:

| State boundary                  | Maximum absolute difference |
| ------------------------------- | --------------------------: |
| Rin prefix, 17 tokens           |                         0.0 |
| Marie prefix, 17 tokens         |                         0.0 |
| Rin after generation, 21 tokens |                         0.0 |
| Marie continuation, 19 tokens   |                         0.0 |

The final run recorded the following single-sample wall-clock observations:

| Phase                              | New input | Cached prefix | First token | Model execution |
| ---------------------------------- | --------: | ------------: | ----------: | --------------: |
| Rin first/cold prefix              |        17 |             0 |           — |    1,243.839 ms |
| Marie warm prefix                  |        17 |             0 |           — |      124.178 ms |
| Rin continuation + two generations |         2 |            17 |   24.174 ms |       63.288 ms |
| Marie continuation                 |         2 |            17 |           — |       19.637 ms |

Model admission plus identity and weight loading took 35.535 seconds. The
first request includes cold MLX/Metal compilation; the second same-shape
prefix is the within-process warm observation. Queue-wait values are retained
in the raw result but are not latency claims because the scenario
intentionally enqueued all requests before draining them.

These timings prove that the intended instrumentation surrounds materialized
work; they are not a Rapid-MLX comparison. No distributions, matched warmup,
peak unified-memory sample, streaming cadence, or alternative sampling policy
were measured. The raw `qwen36-35b-a3b-resident-runtime-result.json` is kept in
the ignored diagnostic archive rather than committed.

## Local chat composition and stdio owner

`qwen36-35b-a3b-chat-template.manifest.json` was generated by the official
Python/Transformers template and tokenizer. The native specialized renderer
and official Rust tokenizer matched all rendered bytes and token IDs exactly:

| Case                                    | Tokens | Rendered bytes | Token IDs |
| --------------------------------------- | -----: | -------------- | --------- |
| plain Japanese, non-thinking            |     21 | exact          | exact     |
| E.C.H.O. startup tool round trip        |    427 | exact          | exact     |
| assistant plus consecutive tool results |    441 | exact          | exact     |

The admitted raw template SHA-256 is
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`;
Qwen EOS is token 248046. Any different template fails startup.

The local diagnostic
`qwen36-35b-a3b-local-composition-result.json` captured the real two-process
scenario summarized here:

- process one encoded and executed a 346-token E.C.H.O. startup/tool prompt,
  committed revision 1, and published an 80-tensor snapshot whose
  71,483,859-byte payload digest was
  `9f30313a2781dd8da630efabf85ae6eff3b54dce89a96c5d0f56bf8959a91964`;
- a short request streamed `1 + 1 = 2 です。`, emitted EOS as a terminal event,
  and committed the EOS-advanced boundary;
- a long generation accepted an in-flight cancellation after 252 emitted
  tokens and committed exactly those 252 tokens rather than rolling visible
  output back;
- a second process authenticated and restored revision 1 at the same 346-token
  digest, rendered a 386-token extended tool loop, executed only its 40-token
  suffix, generated two tokens, and committed revision 2 at 388 tokens.

The interactive NDJSON transcript was inspected but is summarized rather than
retained verbatim. The exact aggregate values and snapshot digest are retained.
These are single-run composition observations under greedy decoding, not
matched Rapid-MLX performance evidence.

## Scope

This proves that the actual Qwen3.5-family MLX cache format can serialize and
restore KV plus GDN at one exact boundary without changing the measured
continuation. It also proves that Rust can execute the complete target model,
carry every GDN and KV state across prefix, continuation, and greedy decode,
reproduce the official logits and selected tokens without numerical drift, and
atomically transfer the live 80-tensor state between revisioned execution
owners and across a producer-process exit without numerical drift. The
resident core now defines exact lineage admission, single-generation FIFO
scheduling, per-instance cache reuse, waiting cancellation, and transactional
rollback. The native composition root now adds exact E.C.H.O. chat
tokenization, UTF-8-safe token streaming, EOS termination, in-flight
token-boundary cancellation, bounded local stdio transport, and authenticated
snapshot publish/restore. At this evidence date it did not yet provide
production sampling, Qwen XML tool-output parsing, the TypeScript `ModelPort`
child-process adapter, cross-instance shared-prefix reuse, revision
retention/current-pointer policy, exhaustive crash injection, or Rapid-MLX
performance parity. The first three of those boundaries were added and
evaluated on 2026-07-31; see
[the sampler and ModelPort evidence](../2026-07-31/README.md). The remaining
limitations still apply.
