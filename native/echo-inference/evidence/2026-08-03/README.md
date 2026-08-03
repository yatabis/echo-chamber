# Protocol v7 performance evidence — 2026-08-03

## Retention

This README is the Git-candidate source of truth for the dated conclusions.
Repeated attempts, complete protocol event streams, and Rapid-MLX logs remain
only in the ignored local diagnostic archive described in
[`../README.md`](../README.md). They are not build or test inputs.

The local source files for this report are:

- `protocol-v7-short-native-first.json`;
- `protocol-v7-short-rapid-first.json`;
- `protocol-v7-native-stateful.json`; and
- `protocol-v7-native-rapid-long-session.json`;
- `protocol-v7-moe-profile-{full,none,router-only,routed-only,shared-only}.json`;
- `moe-gate-up-final-ab-order1-{baseline,candidate}.json`; and
- `moe-gate-up-final-ab-order2-{candidate,baseline}.json`;
- `routed-down-reduce-final-ab-order1-{baseline,candidate}.json`; and
- `routed-down-reduce-final-ab-order2-{candidate,baseline}.json`.

## Scope and subjects

The measurements used the local affine-Q4
`Qwen3.6-35B-A3B-MLX-4bit` artifact on an Apple M4 Pro Mac mini with 64 GiB
unified memory. Both runtimes used MLX 0.32.0. The Native release binary had
SHA-256
`820507988eb852ae96f3863c6c8fff2f4419eb41d7c678468d1248bb7429cbc9`;
Rapid-MLX was version 0.10.10 at commit
`fd23c2a8f1f416bddb976b2394aaa014672914ad`.

All gates fixed concurrency at one, BF16 KV state, greedy sampling with seed
42, and no speculative decoding or PFlash. Short requests also disabled both
prefix caches. One warmup preceded each measured block.

The short and long comparisons enabled diagnostic token events in Native so
that external visible TTFT and decode rate used the same observable boundary
as Rapid-MLX and the retained historical gates. Production requests normally
disable those events. A separate same-binary 21-pair check found a small
median total-time benefit of 0.183% from the production no-stream path; that
value is not arithmetically added to the cross-engine results below.

## Matched short generation

Each engine processed the same 112-token prompt and generated 128 tokens.
The table reports medians over seven measured requests in each engine order.
External decode rate is `(completion tokens - 1) / (completion time - first
visible token time)`.

| Engine order | Metric               |      Native |   Rapid-MLX | Native / Rapid |
| ------------ | -------------------- | ----------: | ----------: | -------------: |
| Native first | Visible TTFT         |   327.60 ms |   415.18 ms |         78.91% |
| Native first | Total request time   | 1,811.89 ms | 1,978.38 ms |         91.58% |
| Native first | External decode rate | 85.54 tok/s | 81.51 tok/s |        104.95% |
| Rapid first  | Visible TTFT         |   327.87 ms |   422.84 ms |         77.54% |
| Rapid first  | Total request time   | 1,815.22 ms | 1,996.75 ms |         90.91% |
| Rapid first  | External decode rate | 85.39 tok/s | 80.93 tok/s |        105.52% |

Native therefore reduced median visible TTFT by 21.09–22.46%, reduced total
request time by 8.42–9.09%, and raised external decode rate by 4.95–5.52%.
All 14 measured output hashes matched their paired Rapid-MLX outputs, and both
orders passed the prespecified five-percent non-inferiority gate.

Compared with the retained 2026-07-31 final fused-GDN baseline, current Native
improved by approximately 0.65% in visible TTFT, 1.82–2.09% in total time, and
2.10–2.50% in decode rate under the corresponding engine order. This is a
cross-date comparison, not a same-process ablation.

## Native stateful continuation

Five measured rounds covered one direct instance, two interleaved instances,
and a stateless replay of the same logical context. The fifteen cached
continuations reused 436 tokens and processed only the 25-token tool-result
suffix. Five stateless controls processed all 461 context tokens again.

| Continuation path        | Count | Visible TTFT |  Total time | Decode rate | Processed | Cached |
| ------------------------ | ----: | -----------: | ----------: | ----------: | --------: | -----: |
| Direct current state     |     5 |     96.95 ms |   845.39 ms | 84.17 tok/s |        25 |    436 |
| Switched current states  |    10 |     96.67 ms |   843.57 ms | 84.37 tok/s |        25 |    436 |
| All cached continuations |    15 |     96.83 ms |   844.41 ms | 84.29 tok/s |        25 |    436 |
| Stateless replay         |     5 |    838.16 ms | 1,584.21 ms | 84.31 tok/s |       461 |      0 |

Cached continuation reached the first visible token 8.66 times sooner than
stateless replay, an 88.45% reduction, and reduced total time by 46.70%. The
switched-instance TTFT was 99.72% of the direct-instance TTFT, so this serial
gate observed no switching penalty. All output hashes and state/accounting
checks matched.

Across 20 measured resident instance states, active Metal memory grew by
83.45 MB per state. The median logical state payload was 75.16 MB, giving an
active-growth-to-logical-state ratio of 1.11. Active plus reusable allocator
cache grew by 128.08 MB per instance, or 1.70 times the logical payload. The
latter includes MLX allocator cache and is not established as a stable
per-instance residency cost; a sustained plateau was not measured.

## Native versus Rapid-MLX long session

The production-contract gate ran one initial tool call and eight
continuations. Each continuation added 825 logical tokens. Three measured
sessions produced matching final outputs and the expected cache trajectory.

| Final request median |      Native |   Rapid-MLX | Native / Rapid |
| -------------------- | ----------: | ----------: | -------------: |
| Prompt tokens        |       6,974 |       7,113 |              — |
| Reused prompt tokens |       6,176 |       4,096 |              — |
| Processed prompt     |         798 |       3,017 |              — |
| Visible TTFT         | 1,476.19 ms | 4,502.53 ms |         32.79% |
| Total request time   | 1,515.25 ms | 4,573.57 ms |         33.13% |

Native reached the first visible token 3.05 times sooner and completed the
request 3.02 times sooner. Its observed cache sequence was
`[0, 401, 1226, 2051, 2876, 3701, 4526, 5351, 6176]`; Rapid-MLX's was
`[0, 0, 0, 0, 2048, 2048, 2048, 4096, 4096]`.

This is deliberately a production-contract comparison, not a token-identical
kernel comparison. Native retains every exact committed output boundary,
whereas Rapid-MLX publishes 2,048-token-aligned checkpoints during a following
request. Rapid-MLX also rendered 139 more prompt tokens at every step because
the two production adapters add different fields and suffixes. Those policies
are load-bearing causes of the observed result.

Compared with the corresponding 2026-07-31 long-session gate, current Native
improved by approximately 4.25% in visible TTFT and 4.40% in total time;
Rapid-MLX also improved by approximately 3.47% and 3.41%. The current relative
Native advantage is therefore slightly stronger, but again this is a
cross-date observation.

## Current MoE decomposition

A Cargo-feature-gated diagnostic remeasured the final Protocol v7 decode path
instead of carrying forward the earlier component attribution. It used the
same 112-token prompt, 128 forced decode steps, one warmup, and three measured
runs. The deliberately invalid component modes retain dependencies needed to
execute the selected path, but their generated output has no quality meaning.

| Derived decode path                          | Median cost |
| -------------------------------------------- | ----------: |
| Common path with MoE bypassed                |    6.758 ms |
| Fused router increment                       |    0.896 ms |
| Routed-expert increment beyond the router    |    2.972 ms |
| Shared-expert increment in isolation         |    1.046 ms |
| Complete MoE increment                       |    4.850 ms |
| Non-additive interaction and measurement gap |   -0.064 ms |

The decomposition shows that the retained router fusion moved routing below
one millisecond per token. The routed experts are now the largest single MoE
subpath. Their selected Q4 gate/up/down tensors contain approximately 566 MB
of nominal weights, scales, and biases per token across 40 layers. Dividing
that nominal traffic by the 2.972 ms increment gives about 191 GB/s. This is a
proxy, not a hardware-counter bandwidth measurement, but it left a bounded
dispatch/locality experiment worth testing.

## Retained routed-expert gate/up fusion

The retained Metal kernel computes the independent affine-Q4 gate and up
projections for the eight selected experts in one dispatch. It reads the BF16
hidden input once per thread, preserves the existing weight tensors without a
concatenated copy, and emits the same two BF16 arrays consumed by SwiGLU. The
specialized path is limited to batch-one, sequence-one BF16 decode with affine
Q4 group size 64 and compatible dimensions. Prefill, trace capture, sorted
multi-token expert dispatch, and unsupported quantization shapes keep the MLX
path.

An initial prototype produced identical 128-token hashes but failed the
resident-state gate after two decode steps: the maximum recurrent/KV-state
difference was 0.8125, and a direct first-layer comparison found gate and up
projection differences of 0.03125 and 0.0078125. The prototype had promoted
four BF16 inputs to float before forming the bias-term input sum. MLX evaluates
that four-value expression in BF16 before accumulating it into float. Matching
that exact order removed the projection differences across all 40 real-model
decode layers and restored complete resident-state parity. This is direct
evidence that output-token equality alone was not an adequate admission gate.

The final regression test uses the production 2048-to-512 shape, eight
non-contiguous expert indices, nontrivial BF16 inputs, and independent gate/up
weights. Both fused projections match MLX's gather-QMV output with maximum
absolute difference 0.0. The regenerated MLX-LM 0.31.3 full-model fixture had
the same historical SHA-256,
`600bfc53c5fbb4404e040088136c968826962fbe691d800c99f351568dce801c`.
The bound-weight resident runtime then matched generated tokens and every GDN
and KV state exactly while preserving FIFO, cancellation, failure rollback,
and independent instance state.

The performance gate used one warmup and five measured 128-token attempts per
binary in both sequential orders. The feature-gated baseline binary SHA-256
was `df302c7d256aa2236ae867f9830ee1672d0ceab068dd139587959f15b9e49647`;
the final candidate was
`2b6cb6e29064a1fe278694f808be80399a51f89eeac49c739236b54a2fbe6049`.

| Sequential order    | Baseline decode | Candidate decode |                 Saved | Baseline rate | Candidate rate |
| ------------------- | --------------: | ---------------: | --------------------: | ------------: | -------------: |
| Baseline, candidate |  11.6490 ms/tok |   11.4405 ms/tok | 0.2085 ms/tok (1.79%) |   85.84 tok/s |    87.41 tok/s |
| Candidate, baseline |  11.6126 ms/tok |   11.4056 ms/tok | 0.2070 ms/tok (1.78%) |   86.11 tok/s |    87.68 tok/s |

All 20 measured attempts produced the same token digest,
`b7e1b41abe26bc47a2f79b6bb446442d0b7f609bf3ccb3755c82a3b05a69c3f7`.
The nearly identical both-order deltas admit the fusion. The rates are internal
fixed-length decode measurements and should not be substituted directly for
the external streamed rates in the Rapid-MLX table above.

## Rejected shared-expert gate/up fusion

The same exact affine-Q4 arithmetic was then applied to the shared expert's
independent 2048-to-512 gate and up projections. A focused rank-two-weight test
matched both MLX QMV outputs at maximum absolute difference `0.0`. The
two-step resident-runtime gate also remained exact, including tokens `1596`,
`1144`, all four retained state comparisons at `0.0`, FIFO execution,
cancellation, rollback, residency, and cross-instance isolation.

Performance nevertheless regressed in both sequential orders. Each binary ran
one warmup and five measured 128-token attempts:

| Sequential order    | Baseline decode | Candidate decode |                  Change | Baseline rate | Candidate rate |
| ------------------- | --------------: | ---------------: | ----------------------: | ------------: | -------------: |
| Baseline, candidate |  11.3772 ms/tok |   11.4577 ms/tok | +0.0805 ms/tok (+0.71%) |   87.89 tok/s |    87.28 tok/s |
| Candidate, baseline |  11.3964 ms/tok |   11.4899 ms/tok | +0.0935 ms/tok (+0.82%) |   87.75 tok/s |    87.03 tok/s |

The feature-gated baseline binary SHA-256 was
`69150c7ff63b563ba7daeb023028fb19884dbd06740d4162eb76a16cedbf4e5d`;
the candidate was
`1408a418ea338555df5da46019d071bf37b7151d1a31d9052922a9a883454d78`.
All 24 recorded attempts, including warmups, retained the same token digest as
the admitted routed-expert gate above. The candidate also increased median
graph-construction time by approximately 0.050–0.070 ms/token, while its total
decode result remained worse regardless of execution order. The likely local
explanation is that one small shared matrix does not provide enough selected-
expert parallelism to amortize this custom kernel as effectively as the eight
routed matrices; the timing establishes rejection even though that mechanism
has not been isolated with GPU counters.

The shared fusion was removed. A clean feature-gated rebuild after removal had
the exact baseline SHA-256 above, proving that the executable decode path was
restored rather than merely disabled through a runtime branch. The four raw
A/B JSON files remain in the ignored diagnostic archive; only this compiled
negative result is retained in the project evidence.

## Retained routed-expert down/reduction fusion

The next retained Metal specialization fuses each selected expert's affine-Q4
512-to-2048 down projection with the existing BF16 score multiplication and
selected-order eight-expert sum. One 256-thread group keeps the same eight
SIMD-group expert parallelism as the QMV work, stores only four weighted BF16
results per expert in threadgroup memory, and emits the final 2048-element
routed output. It therefore avoids materializing the 8-by-2048 BF16 expert
output, as well as the separate elementwise multiply and reduction dispatches.
It is limited to compatible batch-one, sequence-one, unsorted BF16 decode with
bound affine-Q4 group-64 weights; prefill, trace capture, sorted multi-token
dispatch, and unsupported shapes keep the ordinary MLX path.

A focused production-shape test compared the fused result with the exact MLX
chain: gathered Q4 down projection, reshape, BF16 score multiplication, and
sum over the selected-expert axis. The maximum absolute difference was `0.0`.
The bound-weight resident-runtime gate then generated tokens `1596`, `1144`
and matched all four retained GDN/KV state comparisons at `0.0`, while FIFO,
cancellation, failure rollback, residency, and cross-instance isolation all
passed. The fixture retained its historical SHA-256,
`600bfc53c5fbb4404e040088136c968826962fbe691d800c99f351568dce801c`.

The performance gate used one warmup and five measured 128-token attempts per
binary in both sequential orders. The feature-gated baseline binary SHA-256
was `69150c7ff63b563ba7daeb023028fb19884dbd06740d4162eb76a16cedbf4e5d`;
the candidate was
`b67ca35f7e3664b5db71e59bd0245c0a25ed1e1c8686fbdfb5de3800dabf7f08`.

| Sequential order    | Baseline decode | Candidate decode |                 Saved | Baseline rate | Candidate rate |
| ------------------- | --------------: | ---------------: | --------------------: | ------------: | -------------: |
| Baseline, candidate |  11.3085 ms/tok |   10.9772 ms/tok | 0.3313 ms/tok (2.93%) |   88.43 tok/s |    91.10 tok/s |
| Candidate, baseline |  11.3772 ms/tok |   10.9755 ms/tok | 0.4017 ms/tok (3.53%) |   87.90 tok/s |    91.11 tok/s |

All 24 recorded attempts, including warmups, produced the same token digest,
`b7e1b41abe26bc47a2f79b6bb446442d0b7f609bf3ccb3755c82a3b05a69c3f7`.
Both execution orders therefore admit the fusion. The ordinary feature-free
production binary SHA-256 was
`afbf658852c619cc9661ab876fc0794d7b00def4a9af88b5015b389701e302e7`.
Raw A/B JSON remains in the ignored diagnostic archive; the focused exactness
test and this compiled result are retained with the source.

Final verification passed Rust formatting and type checking, warning-denied
Clippy with default and all features, 55 default-feature and 56 all-feature
workspace tests on Metal, and the monorepo lint, format, typecheck, and 521
TypeScript tests without errors or warnings.

## Conclusion and limits

Protocol v7 clears the retained Rapid-MLX performance floor on the measured
short request and materially exceeds Rapid-MLX on the E.C.H.O.-specific
stateful long-session contract. The earlier external short decode gate remains
85.39–85.54 tok/s, while the latest retained expert fusion raises the separate
internal fixed-length gate to 91.10–91.11 tok/s. The exact current-state design
makes cached continuation useful without an observed serial switching penalty.

These runs do not establish parallel-generation scaling, sustained thermal or
allocator behavior, 16K/32K-context performance, stochastic structured-tool
reliability, Vision behavior, or performance on the planned larger model
tier. They also do not prove that the short-gate advantage comes from any one
kernel; only the long-session gap has an audited cache-policy mechanism.
