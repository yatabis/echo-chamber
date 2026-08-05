# Continuous-batching adoption evidence — 2026-08-05

## Status and retained decision

**Status:** continuous batching from one through six active state lanes is now
implemented in the production stdio scheduler. A lone request starts
immediately, up to six already-ready lanes may join before its first decode
step, and late joining after decode begins is capped at width four. Protocol 9
also distinguishes durable and ephemeral state lanes.

The earlier requirement that batch-two output match batch-one bit-for-bit is
superseded. Floating-point model execution changes with batch shape in both
Native and official MLX-LM. The retained admission rules are instead:

- exact agreement with official MLX-LM for the same execution shape;
- exact invariance for one request when its co-tenant or batch-row placement
  changes within that shape;
- independently owned, compact KV/GDN state after every merge and split;
- correct join, leave, cancellation, EOS, output-limit, and continuation
  boundaries;
- positive aggregate throughput under production sampling; and
- representative structured-tool output without parser or state-accounting
  regressions.

All bounded gates below passed. A lone request must continue immediately on
the batch-one fast path; the scheduler must not delay it merely to fill a
batch. Membership may change only at a sampled-token/state-advance boundary.
The hard limit of six provides the capacity envelope for three E.C.H.O.
instances with memory and emotion work running together. The integrated stdio,
module-lane, and 16K soak gates described below passed. The scheduler therefore
remains enabled; choosing a narrower active cohort for latency or fairness is a
future workload policy, not a rollback to FIFO architecture.

## Integrated production scheduler and module lanes

The production stdio probe sent six simultaneous initial requests, then late
joins and an independently cancelled continuation. All six initial rows
observed width six. The late-join case stayed within its configured bound, the
cancelled lane retained its preceding commit, the survivor committed, and the
retry reported that exact preceding sequence length as its cached prefix.
Every generated-token/state-length equation was exact. This probe uses the
real TypeScript client and protocol rather than the feature-gated batch
diagnostic.

The three-E.C.H.O. module probe then opened nine independent state lanes:
three durable `main` lanes plus six ephemeral `memory`/`emotion` lanes. Each
module invocation ended in one exact Qwen tool call. The next observation was
sent as the result of that actual pending call ID, so this was an admitted
append-only Qwen template boundary rather than a fabricated tool result after
plain text.

- all six auxiliary initial requests executed together at width six;
- all three main initial requests executed together at width three;
- five auxiliary updates committed while the selected sixth rolled back;
- the cancelled auxiliary retry reused cached prefix 393 exactly;
- all three later main continuations reused their own prefix 392, proving that
  auxiliary work had not advanced the main states;
- every initial, update, retry, and main-continuation output contained the
  expected tool name and exact `summary`, without parser warning; and
- only the three durable main lanes published `current.safetensors`; an
  auxiliary snapshot was rejected.

The successful run observed 192 generated auxiliary-initial tokens in
5,577.56 ms, 93 main-initial tokens in 2,789.47 ms, and 105 main-continuation
tokens in 1,479.67 ms. These are whole-request wall measurements and include
suffix prefill and finalization; they are not decode-only rates.

## Integrated 16K resident soak

The retained 16K probe used the production sampler
`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`,
`repetition_penalty=1`, and `presence_penalty=1.5`. Six ephemeral lanes each
processed 16,168 initial tokens in eight sequential model executions and
finished with an exact tool call. Initial prefill plus generation for all six
lanes took 175,251.34 ms and produced committed length 16,205 per lane.

Every measured continuation returned one matched tool call of 39 tokens
(40 once the decimal round number gained a token), stopped naturally on Qwen
EOS, and had zero membership changes within its cohort. This removes the
early-EOS and split/remerge noise from the earlier synthetic continuation
attempt. The rate below is generated tokens divided by complete cohort wall
time, including each request's sequential suffix prefill, sampling, decode,
and finalization. It is therefore intentionally lower than the earlier
decode-only width diagnostic.

| Active width | Median aggregate wall rate | Median cohort completion | Median TTFT across rows |
| -----------: | -------------------------: | -----------------------: | ----------------------: |
|            1 |                42.48 tok/s |                918.01 ms |               330.55 ms |
|            3 |                48.70 tok/s |              2,402.34 ms |               669.84 ms |
|            4 |                49.62 tok/s |              3,184.71 ms |               839.38 ms |
|            5 |                49.80 tok/s |              3,915.33 ms |             1,338.62 ms |
|            6 |                49.46 tok/s |              4,819.09 ms |             1,217.46 ms |

The width-three-through-six figures are medians of three rounds each. One
first width-six round took 9,783.20 ms and one later width-five round took
5,171.98 ms; the probe did not retain enough profiler detail to attribute
those outliers, so no compilation or thermal cause is claimed. Across all 12
rounds, 2,121 visible tokens completed in 49,109.81 ms, or 43.19 aggregate
wall tok/s including the outliers.

All cached prefixes, committed lengths, requested widths, tool outputs, and
six state owners passed. Final state lengths were 17,336–17,464. Maximum
observed active Metal allocation was 27,370,486,092 bytes (27.37 decimal GB)
on the 64 GiB machine. The measured throughput is effectively saturated from
width four through six: width six retains useful capacity for six simultaneous
requests, but this workload does not show a throughput advantage over width
four. Keeping the hard limit at six while leaving the 3–6 admission choice
adjustable is the retained decision.

The integrated runs used release binary SHA-256
`60c78ae639ea6aa9b40d8aaecccc92663cc8603f9abb6f4bae86c31c42e71bf4`,
built from the uncommitted working tree based on
`1b6a1207b3061ce442298a266242e87b9eafc2d4`. Raw stdout JSON was not retained;
this section is the compiled record. The reusable probe programs are retained
under `packages/native-inference-adapter/src/`.

After a lint-driven source refactor split the scheduler state machine into
smaller functions without changing its operation order, release binary
`757c091026e709dff1d8674fb293cd2ae3d77a03a49abb18689c29d16ae5d14f`
reran the production continuous-batch probe. Six unequal-length rows generated
504 tokens in 4,957.82 ms (101.66 aggregate wall tok/s), all observed width
six, and every state length remained exact. A late request joined an active
row at width two and the survivor returned to width one. The selected
cancelled row rolled back to sequence length 122, its survivor committed, and
the verification continuation reused cached prefix 122. All ten probe checks
passed.

## Retention and subjects

This README is the Git-candidate record. Repeated raw runs are not committed.
The final two-row production-quality output was written to
`/private/tmp/echo-production-batch-quality.json`; the context and lifecycle
outputs were likewise temporary local diagnostics. The width-one-through-six
outputs were `/private/tmp/echo-batch-width-6.json` and
`/private/tmp/echo-production-batch-width-6.json`. The reusable official
oracle generator is retained under `oracles/`.

The measurements used the local affine-Q4
`Qwen3.6-35B-A3B-MLX-4bit` artifact on an Apple M4 Pro Mac mini with 64 GiB
unified memory. The source base was commit
`1b6a1207b3061ce442298a266242e87b9eafc2d4` on
`feat/native-inference-engine`, plus the uncommitted continuous-batching
diagnostic changes. The greedy-width and official-parity release binary
SHA-256 was
`a4c08c212c9b583aec90d347e3599a53038d449e3dd053346196c48864159fca`;
the later production-width binary was
`6162dcffd639257d7c95a6d31253bbcaa2d4dfe3053563d8557e6ec2c00e9345`.
The model `config.json` SHA-256 was
`a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767`;
`tokenizer_config.json` was
`672488283cdbf3530ecd2e3f90da54f9998cbae6befb5b32877590f72c7a9b2c`.
The official Python oracle used MLX 0.32.0 and MLX-LM 0.31.3.

## Same-shape numerical semantics

The first full-model batch-two oracle used equal 19-token histories, a
17-token prefix plus a two-token continuation, and eight greedy generated
tokens. Native matched official MLX-LM exactly for embeddings, all 40 layer
outputs, final logits, both generated rows, and all 80 final KV/GDN state
tensors.

The unequal-cache oracle then prefetched independent 72- and 81-token
histories, appended token 198 to both rows, and generated 32 tokens. Native
matched both official output rows exactly. The maximum absolute difference
over each row's complete 80-tensor final state was `0.0`.

On the 128-token Native isolation probe:

- replacing row B with row C changed row A by zero tokens and `0.0` state;
- swapping A/B row positions changed both outputs by zero tokens and both
  states by `0.0`; and
- duplicating A into both rows produced identical outputs and `0.0` row-state
  difference.

These are the relevant state-isolation observations. Comparing batch-two to a
batch-one terminal state after their token paths diverge is not an isolation
test.

## Unequal resident state and lifecycle

The Native resident merge/split probe began with state lengths 74, 83, and 83.
Pairing the 74- and 83-token rows introduced nine positions of physical left
padding for the shorter attention cache. Splitting immediately recovered both
original states with maximum difference `0.0`. Co-tenant replacement, row
permutation, and a two-stage split/remerge continuation all retained exact
tokens and `0.0` final-state difference.

The dynamic membership transaction used two committed base states of length
74 and 83:

1. A processed one continuation token and generated three tokens alone,
   reaching lengths 78 and 84 before B joined;
2. A and B generated five tokens together, reaching lengths 83 and 89;
3. B was cancelled and rolled back to its original committed length 83;
4. A generated four more tokens alone and committed length 87.

The survivor commit was visible, the cancelled base remained unchanged, both
instance leases were released, and every emitted-token/state-length equation
was exact.

## Resident-context performance

The context probe built one immutable resident state per tier with sequential
2,048-token prefill chunks, then excluded that prefill from request timing.
Each request appended one token and generated 64 greedy tokens. One warmup and
two measured attempts per mode were counterbalanced. Aggregate decode is 128
generated tokens divided by the pair decode interval. TTFT and completion are
from simultaneous pair arrival; FIFO's second row therefore includes queue
wait.

| Resident context | Mode      | Aggregate decode | Pair completion |          TTFT A / B |  Per-request decode | Peak above start |
| ---------------: | --------- | ---------------: | --------------: | ------------------: | ------------------: | ---------------: |
|               4K | FIFO      |      82.81 tok/s |     1,571.80 ms |   25.06 / 811.07 ms | 82.78 / 82.83 tok/s |       315.67 MiB |
|               4K | batch two |     107.90 tok/s |     1,209.00 ms |    41.05 / 41.05 ms | 53.95 / 53.95 tok/s |       350.88 MiB |
|              16K | FIFO      |      70.03 tok/s |     1,866.12 ms |   33.90 / 966.18 ms | 70.06 / 69.99 tok/s |       793.15 MiB |
|              16K | batch two |      86.27 tok/s |     1,523.07 ms |    62.47 / 62.47 ms | 43.14 / 43.14 tok/s |       926.54 MiB |
|              32K | FIFO      |      59.41 tok/s |     2,210.87 ms | 46.91 / 1,146.32 ms | 59.64 / 59.19 tok/s |     1,439.19 MiB |
|              32K | batch two |      70.03 tok/s |     1,893.04 ms |    93.15 / 93.15 ms | 35.01 / 35.01 tok/s |     1,565.28 MiB |

Batch-two aggregate throughput improved by 30.30% at 4K, 23.20% at 16K, and
17.86% at 32K. Pair completion fell by 23.08%, 18.38%, and 14.38% respectively.
The cost is explicit: an already-running first request gets lower individual
decode rate and a slower TTFT when a co-tenant is admitted. The gain is total
capacity and second-request latency, not a claim that each row runs faster.
All 12 measured state lengths were exact. The additional peak allocation over
FIFO was 35.21 MiB at 4K, 133.38 MiB at 16K, and 126.09 MiB at 32K.

## Production sampling and sampled tool loop

The production probe used `temperature=0.7`, `top_p=0.8`, `top_k=20`,
`min_p=0`, `repetition_penalty=1`, and `presence_penalty=1.5`. Each logical
request owned its functional MLX seed and generated-output history; logits
were sliced per row before applying the already oracle-tested sampler.

At 4K resident context, 64 tokens per row, one warmup, and two counterbalanced
measured attempts:

| Mode      | Aggregate decode | Pair decode |  Per-request decode | Exact state lengths |
| --------- | ---------------: | ----------: | ------------------: | ------------------: |
| FIFO      |      77.32 tok/s | 1,655.52 ms | 77.33 / 77.31 tok/s |                 2/2 |
| batch two |      97.74 tok/s | 1,309.65 ms | 48.87 / 48.87 tok/s |                 2/2 |

The production-sampling aggregate gain was 26.41%. Across seeds 10,001,
20,003, and 30,007, replacing the co-tenant or swapping row positions changed
the retained row by zero tokens and `0.0` state. The two primary rows remained
distinct, so equality did not come from accidentally sharing a random stream.

Three seed pairs—50,021/50,022, 50,121/50,122, and 50,221/50,222—each ran two
independent exact-template workflows:

1. generate a structured `record_runtime_probe` call with instance-specific
   `instance` and `nonce` arguments;
2. split the states when EOS arrival differed;
3. append only the exact tool-result continuation suffix;
4. merge the unequal states again and generate the final assistant response.

All six tool calls used the expected name and exact arguments, all six parser
warnings were absent, all six tool-result continuations produced non-empty
assistant messages without another tool call, and every turn stopped on EOS
with exact state accounting. The model outputs included:

```text
<tool_call>
<function=record_runtime_probe>
<parameter=instance>
rin
</parameter>
<parameter=nonce>
alpha-17
</parameter>
</function>
</tool_call>
```

and final messages such as `成功しました。`, `完了`, and
`記録は正常に保存されました。`. All three tool-call pairs and two of the
three continuation pairs exercised a batch-two to batch-one survivor switch.

The output-limit boundary reused seed pair 50,021/50,022 with a 43-token cap.
Row A sampled EOS as token 43 while row B reached `length` on the same model
step. Row B alone received the internal, non-visible closing EOS. Final state
lengths were 412 and 414, exactly matching each row's independent accounting.

## Width-one-through-six extension

The later width gate tested the same local model and machine with six distinct
row-continuation tokens. Each context tier used one immutable resident state,
2,048-token prefill chunks, 64 generated tokens per row, one warmup, and two
measured rounds with rotated width order. Prefill was excluded from request
timing. The production column used the admitted
`0.7/0.8/top-20/presence-1.5` sampler and six request-owned seeds at 4K.

| Fixed width |    Greedy 4K |   Greedy 16K |  Greedy 32K | Production 4K |
| ----------: | -----------: | -----------: | ----------: | ------------: |
|           1 |  82.31 tok/s |  69.78 tok/s | 59.35 tok/s |   77.18 tok/s |
|           2 | 105.80 tok/s |  86.56 tok/s | 68.66 tok/s |   97.20 tok/s |
|           3 | 122.50 tok/s |  96.59 tok/s | 75.59 tok/s |  112.16 tok/s |
|           4 | 130.65 tok/s | 103.12 tok/s | 78.54 tok/s |  120.64 tok/s |
|           5 | 136.02 tok/s | 103.52 tok/s | 79.51 tok/s |  125.21 tok/s |
|           6 | 139.15 tok/s | 104.73 tok/s | 80.29 tok/s |  126.87 tok/s |

Width six improved aggregate throughput over width one by 69.06% at greedy
4K, 50.09% at 16K, 35.28% at 32K, and 64.39% under production sampling at
4K. Its per-row rates were 23.19, 17.46, 13.38, and 21.14 tok/s respectively.
Peak allocation above each attempt's starting active allocation was 1,039.75
MiB at 4K, 2,545.72 MiB at 16K, and 4,664.72 MiB at 32K. The 64 GiB target
had adequate bounded headroom in this diagnostic.

Scaling becomes shallow above width four. Greedy width-four to width-six
aggregate gains were 6.51% at 4K, 1.57% at 16K, and 2.23% at 32K;
production 4K gained 5.17%. That does not make width six useless when six
requests are already ready. For one 64-token six-request cohort, one width-six
execution reduced all-requests-complete time versus width four followed by
width two from 3.169 to 2.760 seconds at greedy 4K, 3.962 to 3.667 at 16K,
5.124 to 4.782 at 32K, and 3.439 to 3.027 under production sampling at 4K.
Those reductions are 12.93%, 7.45%, 6.66%, and 11.99%.

An official MLX-LM width-six fixture used six equal 19-token histories, a
17-token prefix, two-token continuation, and eight generated steps. Native
matched embeddings, every one of 40 layer outputs, logits, all six generated
rows, and every final KV/GDN tensor exactly; every reported maximum difference
was `0.0`. The temporary fixture SHA-256 was
`67463af9d97b50a2d38f63e0b4560961cdb57b89dafc8b63271341094995bd47`.

At width six, both greedy and production-sampling isolation retained row zero
exactly when the other five co-tenants changed. Reversing all row placements
retained all six outputs and states exactly, with `0.0` state difference. The
six baseline states were pairwise distinct. A separate cursor diagnostic then
visited widths 6, 5, 4, 3, 2, and 1, emitted two tokens per active row at each
stage, and retained exact state lengths 4,099 through 4,109 plus pairwise
distinct surviving states.

The admitted six rows are six independently owned request/state lanes. This
does not authorize two concurrent commits into the same lane. The later
integrated contract resolved the composition as one durable main lane plus
separate ephemeral memory and emotion lanes per E.C.H.O. existence.

## Interpretation and remaining work

The combined evidence supports the implemented hard maximum of six and
immediate width-one path. It does not support waiting to fill a batch, claiming
linear scaling, or assuming width six is always the best latency choice.
Independent streams offered only a small aggregate gain in the first probe
and are no longer the selected design. Rows with materially different context
lengths may waste full-attention work through left-padding and still warrant a
context-aware admission or bucketing policy.

The remaining gate is application-level rather than a missing Native
primitive: E.C.H.O. must supply the real memory/emotion prompts, update tools,
domain persistence, retry policy, and observed arrival distribution. That
workload should decide whether ordinary operation caps a cohort at three or
four while retaining six as burst capacity. Longer thermal soaks, mixed
context lengths, queue fairness/starvation, vision, and the planned
Qwen3.5-122B-A10B or REAP-pruned tier remain outside this evidence. The module
tool loop is a deterministic structural gate, not a broad semantic-quality
evaluation of memory or emotion behavior.
