# Production-promotion evidence — 2026-08-04

## Retention

This README is the Git-candidate source of truth for the dated conclusions.
Repeated attempts and complete machine-local metrics remain in the ignored
diagnostic archive described in [`../README.md`](../README.md); they are not
build or test inputs.

The local source files for this report are:

- `production-promotion-streaming-{native,rapid}-first.json`;
- `production-promotion-buffered-{native,rapid}-first.json`;
- `production-context-curve-independent-initial.json`;
- `production-sustained-20m.json`;
- `production-multi-instance-stateful.json`;
- `parallel-generation-128x3-counterbalanced.json`;
- `echo-prefill-{cache-clear,chunk8k-32k,chunk4k-32k,chunk2k-32k}.json`;
- `echo-prefill-fastpath-16k.json`;
- `echo-prefill-{16k-single,16k-chunk4k,16k-chunk2k}-20260804.json`;
- `echo-prefill-4k-8k-{single,chunk2k}-20260804.json`;
- `echo-prefill-8k-order2-{chunk2k,single}-20260804.json`;
- `echo-prefill-8k-default-final-20260804.json`;
- `echo-chunked-prefill-state-parity.json`; and
- `echo-chunked-prefill-python-reference.json`;
- `gdn-boundary-production-seed{42,10042,20042}-{carried,recurrent-only,fresh}.json`;
  and
- `gdn-boundary-production-seed{42,10042,20042}-convolution-only.json`.

`production-context-curve.json` and `production-sustained-smoke-60s.json`
remain as local diagnostic records of corrected harness assumptions. Their
failed checks are described below and are not mixed into the admitted results.

## Scope and subjects

The measurements used the local affine-Q4
`Qwen3.6-35B-A3B-MLX-4bit` artifact on an Apple M4 Pro Mac mini with 64 GiB
unified memory. The Native release was rebuilt from E.C.H.O. Chamber commit
`4a300ded30c029f77ed9200eaef0f268a275c234`; its SHA-256 was
`afbf658852c619cc9661ab876fc0794d7b00def4a9af88b5015b389701e302e7`.
That digest exactly matched the retained routed-down/reduction candidate.
Rapid-MLX was version 0.10.10 at commit
`fd23c2a8f1f416bddb976b2394aaa014672914ad`. Both used MLX 0.32.0.

Unless a section says otherwise, the gates used concurrency one, BF16 state,
greedy sampling with seed 42, no speculative decoding or PFlash, and no
prefix cache. The runtime binary came from the committed source above; the
new evaluation-runner changes were still working-tree changes while the live
runs were collected. Unrelated pre-existing E.C.H.O. working-tree changes
were not included in the Native binary.

## Matched short generation

Each runtime received the same 112-token prompt and generated 128 tokens.
One warmup preceded seven measured requests in both sequential engine orders.
All 28 measured Native/Rapid output pairs across the streaming and buffered
blocks had identical SHA-256 hashes.

### Diagnostic token-event delivery

This mode explicitly enabled Native token events and Rapid-MLX HTTP streaming
so visible TTFT and external decode rate could be compared at the same
observable boundary. External decode rate is `(completion tokens - 1) /
(completion time - first visible token time)`.

| Engine order |               Metric |      Native |   Rapid-MLX | Native / Rapid |
| ------------ | -------------------: | ----------: | ----------: | -------------: |
| Native first |         Visible TTFT |   345.20 ms |   445.23 ms |         77.53% |
| Native first |   Total request time | 1,810.09 ms | 2,107.64 ms |         85.88% |
| Native first | External decode rate | 86.71 tok/s | 76.27 tok/s |        113.69% |
| Rapid first  |         Visible TTFT |   345.42 ms |   437.71 ms |         78.92% |
| Rapid first  |   Total request time | 1,822.43 ms | 2,043.75 ms |         89.17% |
| Rapid first  | External decode rate | 86.01 tok/s | 79.48 tok/s |        108.22% |

Native reduced visible TTFT by 21.08–22.47%, reduced total time by
10.83–14.12%, and raised external decode rate by 8.22–13.69%. Rapid-MLX was
slower than its 2026-08-03 block, so these absolute cross-date values should
not be used as an incremental Rapid-MLX regression claim. Reversing engine
order preserved the Native advantage.

### Production buffered delivery

This mode used `stream_tokens: false` in Native and `stream: false` in
Rapid-MLX. A complete response has no externally observable first-token
boundary, so visible TTFT and external decode rate are deliberately `null`;
the comparison is total request time plus exact output and token accounting.

| Engine order | Native total | Rapid-MLX total | Native / Rapid | Native saved |
| ------------ | -----------: | --------------: | -------------: | -----------: |
| Native first |  1,735.01 ms |     1,975.25 ms |         87.84% |       12.16% |
| Rapid first  |  1,746.83 ms |     1,987.76 ms |         87.88% |       12.12% |

Both orders passed the five-percent non-inferiority floor. The promotion gate
now makes all paired greedy output hashes a required admission condition,
rather than recording them without using them in the final decision.

## Independent-state context curve

Each target used a cold, complete initial prompt whose runtime-observed input
length exactly matched 2,048, 8,192, 16,384, or 32,768 tokens. Each target ran
in its own Native process, with one warmup and two measured requests. Every
request used a fresh instance and the `initial` transition, so every GDN and
KV state began empty. Production token streaming was disabled and every
measured request generated the same 128-token output.

The first-token and full-request columns include cold full-context prefill;
they do not estimate normal cached-continuation suffix prefill. Internal decode
rate divides 128 visible generated tokens by `decode_execution_nanos`, whose
interval also includes the hidden forced-EOS state advance.

Each target process retained its warmup instance and two measured instances
until exit. `Metal peak` is therefore that three-instance process high-water
mark, not the allocation of one context; `Logical state` and `Active growth /
instance` are the per-instance residency observations.

| Input context | First generated token | Total request | Internal decode | Logical state | Active growth / instance | Metal peak |
| ------------: | --------------------: | ------------: | --------------: | ------------: | -----------------------: | ---------: |
|         2,048 |               3.241 s |       4.731 s |     79.34 tok/s |    103.93 MiB |               106.64 MiB |  20.95 GiB |
|         8,192 |              14.092 s |      15.729 s |     72.28 tok/s |    223.93 MiB |               226.41 MiB |  26.18 GiB |
|        16,384 |              35.207 s |      37.028 s |     64.95 tok/s |    383.93 MiB |               386.42 MiB |  36.42 GiB |
|        32,768 |             156.353 s |     158.664 s |      4.38 tok/s |    703.93 MiB |               706.41 MiB |  69.64 GiB |

Decode retained 91.11% of the 2K rate at 8K and 81.87% at 16K. At 32K it
retained only 5.52%. The 32K state itself remained representable and its
incremental resident allocation tracked its logical payload, but cold prefill
raised the observed Metal high-water mark beyond the machine's 64 GiB physical
capacity and decode fell off a discontinuous cliff. On the current
implementation, 16K is therefore the pragmatic admitted operating cap. 32K
is a supported shape, not a production-performance tier.

### Corrected GDN-isolation assumption

The first curve attempt reused one instance and selected `new_session` between
identical prompts. Its input-length and cache checks passed, but later outputs
changed and two requests stopped after seven tokens. This was not cross-instance
leakage or random greedy execution: `new_session` intentionally clears
full-attention KV while retaining that existence's GDN recurrent state.
Identical text is therefore not an independent model trial on one existence.

The admitted curve replaced this invalid assumption with a separate process
per target and a fresh `initial` instance per attempt. All eight measured
outputs then had the same hash, generated exactly 128 tokens, advanced the
hidden length EOS, and passed exact state/token accounting. Future independent
quality or performance trials must not use same-instance `new_session` as a
reset-to-empty operation.

## Adaptive long-input prefill

The retained adaptive candidate was built from the same `4a300de` base plus
the working-tree chunked-prefill implementation. Its release-binary SHA-256
was `e7ac0d5a6d3394e81ee417b5d780e926f2cdc63a250dd9fbd63a2207016f1367`.
The 2K, 4K, 8K, and 16K measurements below used that one binary with only the
documented startup boundaries changed.

The 32K row above is the retained single-execution comparator. A control build
that cleared MLX's reusable allocator cache after the 32K prefill produced a
119.593-second input execution, 4.844 tok/s decode, and the same 67.84 GiB peak
as the first comparator attempt. Clearing reusable cache therefore did not
recover decode and rules it out as the sole cause of the cliff; it does not
rule out pressure from live sequence-shaped intermediates.

The candidate instead materialized complete KV/GDN state between bounded input
executions. The table below uses the first isolated attempt from each process,
so every row has one retained instance and the Metal peaks are directly
comparable. All rows generated the same 128-token output hash as the original
single-execution path.

| 32K input shape  | Model input executions | Input execution | Total request | Internal decode | Metal peak |
| ---------------- | ---------------------: | --------------: | ------------: | --------------: | ---------: |
| One 32K call     |                      1 |       120.054 s |     146.723 s |     4.806 tok/s |  67.84 GiB |
| Four 8K calls    |                      4 |        71.496 s |      73.899 s |    53.272 tok/s |  31.93 GiB |
| Eight 4K calls   |                      8 |        68.809 s |      71.283 s |    51.744 tok/s |  26.69 GiB |
| Sixteen 2K calls |                     16 |        67.311 s |      69.694 s |    53.717 tok/s |  22.88 GiB |

The 2K shape reduced isolated 32K input time by 43.9%, total request time by
52.5%, and peak Metal allocation by 66.3%. Its second attempt measured
68.382-second input execution, 70.667-second total request time, and
56.019 tok/s decode. The 4K and 8K second attempts measured 55.914 and
56.225 tok/s respectively, so 2K won mainly on prefill time and peak memory
rather than a material steady-decode difference.

The first retained production threshold was strictly above 16K. A fresh first
measured 16K attempt therefore used one input execution and produced the same
output hash as the retained comparator. Input time changed from 35.433 to
35.825 seconds (+1.1%), total time from 37.431 to 37.811 seconds (+1.0%), and
decode from 64.494 to 64.440 tok/s (-0.08%). Those differences are within run
noise and there was no alternate graph on that initial 16K path.

### Shorter-boundary recheck

A later same-binary comparison directly measured the single-execution and 2K
execution shapes below the first threshold. Each block used one warmup and
three measured independent `initial` instances; the 8K pair was repeated in
reverse process order. The boundary counts only tokens newly executed by the
request, not the already committed prefix reported as `cached_prefix_tokens`.

| Newly executed input | Single prefill median | 2K prefill median | Prefill saved | Total saved | Metal peak saved |
| -------------------: | --------------------: | ----------------: | ------------: | ----------: | ---------------: |
|                   4K |               6.403 s |           6.364 s |         0.62% |       0.56% |            8.09% |
|     8K, single first |              14.833 s |          13.924 s |         6.13% |       5.63% |           18.45% |
|         8K, 2K first |              14.556 s |          13.546 s |         6.94% |       6.37% |           18.64% |
|                  16K |              34.148 s |          29.703 s |        13.01% |      12.42% |           38.39% |

At 16K, four 4K executions measured 29.910 seconds of prefill, 12.41% shorter
than the single call. Eight 2K executions were another 0.69% faster and used
11.07% less peak Metal memory than the 4K execution shape. At 4K, the 0.62%
prefill difference is within ordinary run variation and is not retained as a
speed claim.

Every measured request passed exact input-token, independent-state,
completion-length, forced-EOS state-length, and stable-output checks. All
single and chunked requests produced the same immediate 128-token output hash.
This does not supersede the cross-shape BF16/GDN limitation below. Following
these measurements, the user approved making 2K executions the default when a
request must newly execute at least 8,192 input tokens. Shorter new input stays
on one execution even when the committed context itself is longer.

The post-change default release binary had SHA-256
`fd8576a02c85321b68686c3cc2b787811fdabc058537b7c16b090b5c44ce0f0c`.
With both current and former boundary overrides explicitly absent, its 8,192
token real-model gate reported four input model executions in both warmup and
measurement. The measured request took 13.233 seconds for input and 15.005
seconds total, decoded at 72.476 tok/s, produced the same retained output hash,
and passed every context-curve accounting check. This is a default-wiring
check, not an additional multi-sample performance comparison.

### Hybrid-state semantics

A forced 4K diagnostic compared one 4K model call with two 2K calls. The first
eight generated tokens were exact, as were state sequence length and physical
size, but the complete BF16 state payloads were not bit-identical. After
`new_session` retained GDN and cleared KV, greedy generation remained exact for
24 tokens and diverged at zero-based token index 24. Both decoded outputs were
coherent, but coherence alone is not a quality-equivalence proof.

The paired Python oracle then ran MLX-LM itself with the same two execution
shapes. Python's single-call initial and new-session token sequences exactly
matched Native's single-call sequences; Python's 2K initial and new-session
sequences exactly matched Native's 2K sequences. Python also diverged between
shapes at the same token and with the same two token IDs. This establishes that
the observed difference comes from BF16 execution shape rather than a Native
state-carry bug. It does not establish that arbitrary chunk sizes are
quality-identical, so the production runtime fixes 2K as its canonical long
input shape instead of varying it per request.

The retained default is consequently one model execution below 8,192 newly
executed input tokens and sequential 2,048-token executions at or above that
boundary. The 32K shape no longer exhibits the allocator/decode collapse,
although roughly 70.7 seconds of cold initial latency still makes 16K the
latency-preferred long-context tier. `ECHO_NATIVE_PREFILL_CHUNK_SIZE_TOKENS=0`
remains an explicit diagnostic escape hatch, not the production setting.

## Twenty-minute sustained resident load

One existence ran once with `initial` and then repeatedly with `new_session`.
This is intentionally the production state boundary: GDN state carried across
sessions while KV and token lineage restarted from the same 112-token prompt.
All requests disabled token events and allowed up to 128 generated tokens.

The measured interval lasted 1,200.38 seconds and completed 689 requests with
85,730 generated tokens. Of those requests, 666 (96.66%) reached 128 tokens;
total generated-token duty was 97.21% of the theoretical maximum. The 34
distinct output hashes and 23 early EOS stops reflect the intentionally
evolving GDN state, so this is a sustained execution and residency gate, not a
repeated-output quality gate.

| Sustained metric                |           Observation |
| ------------------------------- | --------------------: |
| First five-minute median decode |          87.868 tok/s |
| Last five-minute median decode  |          87.933 tok/s |
| Last / first                    |              100.074% |
| Whole-run median decode         |          87.927 tok/s |
| Whole-run p10 / p90 decode      | 87.282 / 88.407 tok/s |
| Full-length median request      |           1,779.28 ms |
| Active Metal growth             |                16 KiB |
| Allocator-cache growth          |               112 KiB |
| Maximum Metal peak              |             18.55 GiB |

The macOS `pmset -g therm` boundary reported no recorded thermal or performance
warning after the run. No privileged temperature sensor was sampled, so speed
retention is the primary thermal proxy. The gate proves a stable 20-minute
resident plateau for this short new-session workload; it does not prove idle
power, multi-hour behavior, or a sustained 16K workload.

## Serial multi-instance continuation

Immediately after the sustained run, five measured rounds compared one direct
continuation, two interleaved instance continuations, and a stateless replay.
This diagnostic enabled token events to observe external TTFT. The Native
process still executed one request at a time; this is state isolation and
switch-cost evidence, not parallel generation.

| Path                     | Count | Visible TTFT |  Total time | External decode | Processed | Cached |
| ------------------------ | ----: | -----------: | ----------: | --------------: | --------: | -----: |
| Direct continuation      |     5 |     98.89 ms |   824.51 ms |     86.83 tok/s |        25 |    436 |
| Interleaved continuation |    10 |     98.80 ms |   824.82 ms |     86.77 tok/s |        25 |    436 |
| All cached continuations |    15 |     98.84 ms |   824.57 ms |     86.81 tok/s |        25 |    436 |
| Stateless replay         |     5 |    855.43 ms | 1,587.18 ms |     86.85 tok/s |       461 |      0 |

Interleaving retained 99.91% of direct TTFT, so no serial switching penalty
was observed. Exact current-state reuse reached the first visible token 8.65
times sooner than replay. All cached/replay outputs, state advances, engine
ownership, and prefix counts matched.

Twenty measured instance states added 83.42 MB active Metal memory each,
against a 75.16 MB median logical payload (1.11 times). Active plus reusable
allocator cache grew by 123.82 MB per instance; allocator cache is not treated
as a permanent per-instance tax.

## Two-instance parallel-generation probe

A feature-gated Native diagnostic compared the current FIFO execution with two
same-process candidates: two independent MLX GPU streams sharing one bound
weight set, and one fixed batch of two. Production scheduling and the stdio
protocol remained unchanged. The release binary was built from commit
`1b6a1207b3061ce442298a266242e87b9eafc2d4` plus the uncommitted diagnostic
changes; its SHA-256 was
`0bcdea1149a0285bae696a3c90d7420a73f4aceafa20ff12f4a1363ce653aa7e`.

Both instances began with empty state and distinct, equal-length 74-token
prompts. Greedy generation was forced to 128 tokens. After one warmup per
mode, three measured rounds rotated every mode through first, second, and
third execution position. Aggregate decode rate is 256 generated tokens
divided by the pair's decode interval. TTFT and completion are measured from
the simultaneous arrival of both requests, so FIFO's second request includes
its queue wait.

| Mode                    | Aggregate decode | Pair total |        TTFT A / B |  Completion A / B | Peak above start | Exact output + state |
| ----------------------- | ---------------: | ---------: | ----------------: | ----------------: | ---------------: | -------------------: |
| FIFO serial             |      88.70 tok/s |    3.340 s | 238 ms / 1,911 ms | 1.671 s / 3.340 s |       309.08 MiB |                  3/3 |
| Independent MLX streams |      91.87 tok/s |    3.238 s |   475 ms / 478 ms | 3.238 s / 3.238 s |       344.15 MiB |                  3/3 |
| Fixed batch of two      |     115.01 tok/s |    2.542 s |   336 ms / 336 ms | 2.542 s / 2.542 s |       524.09 MiB |                  0/3 |

Independent streams raised median aggregate decode by 3.57%, reduced the time
until both requests completed by 3.07%, and used 35.06 MiB more peak Metal
allocation than FIFO. Both requests progressed together at about 45.93 tok/s
each. Every generated token and every compacted KV/GDN tensor matched its own
FIFO reference exactly in all three attempts, while the two instance states
remained distinct. The cost is scheduling semantics: the first request's TTFT
and completion time nearly doubled, while the second request's TTFT fell by
74.98%.

Fixed batch raised aggregate decode by 29.65%, but both 128-token outputs and
their states diverged deterministically from the batch-one references. At
eight generated tokens its outputs still matched while state already differed,
so the later divergence is not a timing artifact. The present comparison does
not prove that batch rows mixed; once the output paths diverge, closeness to a
batch-one terminal state is no longer a valid isolation test. It does prove
that the existing generic batch-two path fails the current exactness admission
gate and cannot be promoted as-is.

The retained decision is therefore to keep FIFO as the production default and
admit independent streams only as the candidate for an explicit two-active-
request path. The probe establishes feasibility, small aggregate benefit, and
exact state separation; it does not yet establish production behavior for
unequal arrivals or lengths, cancellation and rollback, production sampling,
committed continuation state, or representative 16K residency.

> **Superseded on 2026-08-05:** this was the correct bounded conclusion from
> the first probe, but batch-one versus batch-two bit identity is not a valid
> production admission rule. Official MLX-LM shows the same batch-shape
> dependence. The follow-up in [`../2026-08-05/README.md`](../2026-08-05/README.md)
> instead tested oracle parity within shape, row invariance, unequal resident
> state, lifecycle transitions, long contexts, production sampling, and tool
> continuations. It retains fixed batch of two—not independent streams—as the
> continuous-batching candidate while FIFO remains the current implementation.

## Production workflow integration and cross-session state finding

The Native adapter was connected to the existing stateful E.C.H.O. workflow
harness without adding the future SQLite/local-application layer. The runner
used the current Rin production prompt, canonical runtime tools, real agent
loop, stateful synthetic chat/memory/note/context ports, buffered delivery, and
the same local 35B artifact as the performance gates. The Rin prompt SHA-256
was `9335a7297d5c9af356edb600e354667eff5003c908111df86d5f56968f902f6a`.

All engine-side integration checks passed in both controlled-greedy and
production-sampling runs: six sessions were executed across three workflows,
every session emitted bounded Native metrics without a tool-parser warning,
the adapter selected the expected `initial`, `continuation`, and `new_session`
transitions, every continuation reused a committed prefix, every workflow
published `current.safetensors`, and a production-shaped request crossed the
8,192-token adaptive-prefill boundary. The controlled run processed at most
8,306 new tokens in five model calls; the production-sampling run processed at
most 8,382 in five calls.

| Carried-state workflow run | Requests | Generated | Aggregate decode rate | Behavior score | Finished sessions |
| -------------------------- | -------: | --------: | --------------------: | -------------: | ----------------: |
| controlled-greedy          |       44 |     4,987 |          70.916 tok/s |          19/32 |               4/6 |
| production-sampling        |       48 |     5,927 |          71.354 tok/s |          23/32 |               4/6 |

The decode rates use `generated_tokens / decode_execution_nanos`; they are not
end-to-end workflow throughput. The behavioral admission failed only because
the `state_revision_across_cold_start` workflow failed. The other production
sampling workflows, including injected note-update failure recovery, both
scored 10/10.

The failing workflow was then compared under three conditions. This is one
deterministic observation, not a population estimate:

| Controlled state-revision condition | Score | Session termination pattern    |
| ----------------------------------- | ----: | ------------------------------ |
| Native, GDN carried across sessions |  4/12 | finish / max-turns / max-turns |
| Native, fresh state each session    |  8/12 | finish / finish / max-turns    |
| Rapid-MLX, no recurrent carry       |  6/12 | max-turns / finish / finish    |

All three missed the final latest-state answer, establishing a model/prompt
weakness independent of Native. The Native-only ablation nevertheless matters:
with the same binary, prompt, fixture, controlled profile, and request-seed
schedule, removing only cross-session GDN carry restored cancellation-memory
storage, removed the unrequested note mutation, and avoided the carried run's
false claim that the deployment had completed. The current carried transition
therefore cannot yet be described as behaviorally safe.

This is consistent with the earlier Qwen3.6-27B recurrent-only lab probes in
the Rapid-MLX research tree: hidden recurrent state materially shifted the
same visible prompt, contaminated an engineering choice with a prior geology
instruction, and did not reliably recover a matching hidden task. An explicit
runtime policy reduced one contamination choice but did not turn hidden state
into a dependable continuity signal.

The default `carry_all` implementation matches its documented mechanics:
`new_session` clones both GDN convolution and recurrent tensors, clears
full-attention KV, and processes a complete fresh prompt. No state-transition,
cache-accounting, snapshot, or parser bug was observed. This first comparison
left an architectural question—whether to reset either component or require an
explicit policy/training strategy—which the follow-up below tested further.

### Four-condition production-sampling boundary ablation

A follow-up crossed the two GDN components independently: retain or clear the
three-position convolution history, and retain or clear the recurrent matrix.
Each of the four resulting conditions ran all three workflows with production
sampling at base seeds 42, 10,042, and 20,042. Thus each table mean is over
three complete 32-point runs; the completion denominator is 18 sessions.

| GDN state visible after the boundary  | Mean total (per-seed totals) | State revision mean /12 | Queued priority mean /10 | Injected failure mean /10 | Finished sessions | Median decode |
| ------------------------------------- | ---------------------------: | ----------------------: | -----------------------: | ------------------------: | ----------------: | ------------: |
| Convolution + recurrent retained      |        22.33/32 (23, 21, 23) |                    4.00 |                     8.33 |                     10.00 |             16/18 |  71.429 tok/s |
| Recurrent retained, convolution clear |        23.00/32 (23, 25, 21) |                    7.67 |                     5.33 |                     10.00 |             18/18 |  71.576 tok/s |
| Convolution retained, recurrent clear |        17.67/32 (15, 16, 22) |                    3.33 |                     4.33 |                     10.00 |             15/18 |  71.482 tok/s |
| Both clear                            |        18.67/32 (19, 17, 20) |                    2.00 |                     6.67 |                     10.00 |             16/18 |  71.599 tok/s |

No condition passed every behavior check in any seed. Recurrent-only carry was
the strongest state-revision condition: it stored the later cancellation in
3/3 runs, answered with the latest state in 2/3, and finished all 18 sessions.
Complete carry was stronger on the competing queued-priority workflow: it read
the new urgent DM in 3/3 runs and returned the exact adapter location in 2/3,
whereas recurrent-only carry returned the exact location in 0/3 and placed the
urgent work before deferred work in only 1/3. Convolution-only carry read the
new urgent DM in 0/3 and did not improve either workflow. Every condition
scored 10/10 on injected note-update failure recovery.

Two failures were independent of the GDN boundary choice under this production
sampling matrix: every condition missed initial-schedule persistence in 3/3
runs and made an unrequested note mutation in 3/3. One convolution-only run
also emitted four consecutive `no_tool_calls` provider warnings, ending one
queued-priority session with `error`; this was a model-output failure, not a
Native process crash. The four median aggregate decode rates span only 0.24%,
so no repeatable boundary-policy throughput cost was observed.

The empty-state condition allocated a fresh Native owner for each harness
session instead of clearing both GDN tensors in the same owner. Both variants
present empty KV and GDN state to the model, and transition metadata is not a
model input, so this is a model-state comparison; it does not compare
same-owner lifecycle overhead. The first three conditions used release binary
SHA-256 `3cd17dccf1c9e223cdc974a8612c62a71e99bbd050c480c17aaeb736282e21b0`.
Convolution-only support was then added without changing the existing policy
arithmetic, and its three runs used
`8339e85b749844cb7256ae4cbf1f2b2db0607430fe484bc6d8a03cc71223905c`.
The exact 12-run matrix was therefore not collected from one binary.

The result rejects a component-only production switch. Recurrent-only carry is
useful diagnostic evidence that the recurrent matrix can preserve prior state,
but its repeatable queued-priority regression prevents admission. Complete
carry remains the runtime default because it best preserved that workflow, not
because it passed the quality gate. A production change now requires an
explicit, model-visible session-boundary policy and/or boundary-aware training,
followed by a broader behavioral gate; further tensor-retention combinations
are not the next useful lever.

Raw Native controlled, Native production-sampling, all four boundary
conditions, and the current Rapid-MLX comparator JSON remain under ignored
`.artifacts/model-evaluation/native-inference/evidence/2026-08-04/`; they are
not added to the repository.

## Final verification

The isolated Native commit passed the monorepo lint and format checks, all
package typechecks, and all 521 ordinary TypeScript tests. The Native evaluation
runners also passed their focused non-live tests. The adaptive-prefill runner's
environment-gated real-model case passed both variants in 88.17 seconds, and
the paired Python oracle reported
`all_native_paths_match_python_reference: true`.

With the pinned MLX C/MLX paths configured, the Rust workspace passed
formatting, warning-denied Clippy with default and all features, 58
default-feature tests, and 59 all-feature tests. These counts include exact
MLX-array checks for each diagnostic GDN component policy and strict startup
policy parsing. The Python oracle also passed syntax compilation. No errors or
warnings remain in the checked scope.

## Conclusion and limits

The rebuilt production binary passes the matched Rapid-MLX short gate in both
delivery modes and both engine orders, the independent context curve through
16K, a 20-minute resident-load gate, and the serial multi-instance state gate.
The adaptive candidate additionally removes the measured 32K memory/decode
cliff. The boundary recheck supports 2K chunks for requests that must newly
execute at least 8K tokens: the gain reproduced in both 8K process orders and
grew at 16K, while 4K showed no material speed change. 16K remains the
latency-preferred long-context tier and 32K becomes an available slower tier
rather than an implicit hard failure boundary.

None of the four tested cross-session GDN component policies passed behavioral
admission. Complete carry remains the observable default while model-visible
boundary semantics or boundary-aware training is investigated; it is not a
quality-safe production invariant.

It does not establish interactive cold-prefill latency at 32K, broad semantic
quality equivalence across arbitrary chunk sizes, production simultaneous
parallel generation, complete local-application and OS-signal composition, varied
structured-tool reliability, Vision, multi-hour/idle power behavior, or the
planned larger Qwen3.5-122B-A10B tier.
