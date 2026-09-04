# Exact fused decode router evidence

This directory follows the curated retention policy in
[`../README.md`](../README.md). Only the final portable validation and
both-order A/B summary remain as JSON; exploratory and redundant diagnostics
are local-only.

## Outcome

The retained batch-one BF16 decode path now fuses the complete routing work
after the Q8 router linear projection into one Metal dispatch:

1. reproduce MLX 0.32.0's precise 256-way BF16 softmax;
2. select the stable top eight in the same order as MLX's full-sort-backed
   `argpartition` tail;
3. reproduce MLX's input-ordered BF16 selected-score sum and normalization;
4. write only eight `uint32` expert indices and eight BF16 scores.

The 256 BF16 probabilities remain in threadgroup memory. For nonnegative BF16
probabilities, `(probability_bits << 16) | expert_index` has the same ascending
order as MLX's stable `(probability, original_index)` sort. Eight `simd_max`
reductions can therefore recover the exact tail without materializing a
256-element probability array or partition result in device memory.

The algebraic cancellation of the full-softmax denominator is valid in real
arithmetic, but it is not bit-for-bit equivalent to this MLX execution path:
MLX first rounds all probabilities to BF16, then performs stable selection and
an input-ordered BF16 reduction. The retained kernel deliberately reproduces
those rounding and tie boundaries rather than selecting directly from logits.

The specialization is used only for batch size 1, sequence length 1, BF16
input, and normalized top-k routing. Prefill, generic shapes, non-BF16 input,
non-normalized routing, and trace capture retain the original MLX graph.

## Direct-logits alternative check

When the exact kernel was selected, the non-equivalence of
`logits -> top-8 -> 8-way softmax` was a mechanism-based inference supported by
an earlier custom-top-k tie failure, not yet a retained direct comparison. A
dedicated MLX 0.32.0 comparison was subsequently run over 16,384 synthetic
BF16 router rows. It compared:

- reference: 256-way precise softmax, stable top eight, selected-probability
  BF16 sum, and division;
- alternative: stable top eight directly from logits, followed by an eight-way
  precise softmax.

Inputs were generated in the listed table order with
`numpy.random.default_rng(20260801)`: three `[4096, 256]` float32 normal arrays
at scales 0.25, 1, and 4, followed by integers in `[0, 4096)` transformed as
`value / 256 - 8`. Every array was cast to MLX BF16 before comparison.

| Input distribution |       Rows | Ordered-index mismatch | Selected-set mismatch | Score mismatch | Complete-output mismatch | Maximum score difference |
| ------------------ | ---------: | ---------------------: | --------------------: | -------------: | -----------------------: | -----------------------: |
| Normal, scale 0.25 |      4,096 |                    706 |                   220 |          3,945 |                    3,967 |              0.001953125 |
| Normal, scale 1    |      4,096 |                      0 |                     0 |          3,971 |                    3,971 |               0.00390625 |
| Normal, scale 4    |      4,096 |                      0 |                     0 |          3,880 |                    3,880 |               0.00390625 |
| Discrete tie-heavy |      4,096 |                      0 |                     0 |          3,877 |                    3,877 |              0.001953125 |
| **Total**          | **16,384** |                **706** |               **220** |     **15,673** |               **15,695** |           **0.00390625** |

Thus 95.79% of the tested rows differed in the complete ordered router output.
At ordinary and wide scales, selected indices remained exact in this sample,
but approximately 95-97% of score vectors differed because the reference
rounds the 256-way probabilities before selected-score normalization. At the
narrow scale, BF16 probability ties also changed the selected expert set in
220 rows (5.37% of that dataset).

This establishes that the two paths are not generally bit-exact. It does not
measure how often they differ on captured Qwen3.6 router logits or whether the
differences would change end-to-end model quality. Those are separate questions
that would require real-logit capture and an approximate-path quality gate.
The table and deterministic generation specification above are the
Git-candidate record. The generated logits, comparison script, and redundant
standalone summary JSON are local-only; the JSON remains in the ignored local
archive.

## Exactness checks

- The focused GPU test compares indices and scores exactly against MLX for 22
  BF16 rows: all-equal logits, a structured tie-heavy row, and 20 deterministic
  random tie-heavy rows.
- The final resident-runtime fixture generated tokens `[1596, 1144]` and
  reported zero maximum absolute difference at all four checked state
  boundaries.
- FIFO execution, continuation after a failed request, cancellation,
  per-instance state separation, stale-revision rejection, rollback, and a
  single resident model owner all passed.

The compact machine-readable record is
[`qwen36-fused-decode-router-validation.json`](qwen36-fused-decode-router-validation.json).
The complete terminal event stream was not retained; the file records the
assertions, fixture digest, generated tokens, and numeric differences needed
to interpret the check.

## GDN-preserving new-session transition

The release runtime was exercised on the same local
Qwen3.6-35B-A3B-MLX-4bit artifact and retained full-model fixture. The first
thinking session advanced through the oracle's 17-token prefix, two-token
continuation, and two greedy generated tokens to revision 2 at a 21-token
boundary. Its complete state remained exactly equal to the official fixture
(maximum absolute difference `0.0`).

From that exact revision, `new_session` retained both tensors in every GDN
layer and replaced all full-attention KV tensors with zero-length caches before
processing a fresh 17-token prompt. The transition-level GDN difference was
`0.0`; the carried GDN differed from an empty GDN state by `35.75`, proving
that the retained value was not merely zero state. One generated token then
committed revision 3 at an 18-token boundary, shorter than the preceding
21-token lineage. The request reported zero cached-prefix tokens and 17
processed input tokens. An otherwise identical empty-state ablation produced
the same greedy token but a model-state maximum absolute difference of
`19.951172`, so the carried state was load-bearing even though this particular
argmax did not change. A following exact request reused all 18 new-lineage
tokens, processed one token, and committed revision 4 at 19 tokens.

The protocol-v5 TypeScript `ModelPort` probe independently exercised the
composition boundary with four Native generation requests. Revisions 1 and 2
formed the original tool round trip at 383 and 413 tokens. The second request
reused the Native-owned 383-token prefix and supplied only a 25-token
tool-result suffix. Omitting `previousResponseToken` then started a fresh
session, accepted a changed tool description, processed 360 uncached input
tokens, parsed the requested tool call, and committed revision 3 at 389 tokens
with zero old-prefix reuse. Its tool-result request reused exactly those 389
tokens and committed revision 4 at 419 tokens. Both tool results (`7391` and
`8642`) were consumed successfully. The final release binary's four single-run
wall times were 1,203.7, 144.0, 1,024.2, and 142.9 milliseconds; they are functional evidence,
not a performance distribution.

The complete parity and protocol JSON outputs were inspected but intentionally
not added as redundant repository artifacts. The source-backed command,
assertions, fixture digest, and compact observations above are the retained
record.

## Durable state, token sequence, and restart recovery

Each schema-3 revision is an immutable, exclusively renamed directory
containing `state.safetensors`, `lineage.json`, and `manifest.json`. Here
`lineage.json` is the complete ordered sequence of tokenizer IDs whose model
execution produced the stored KV/GDN tensors. Its SHA-256 must equal the
committed token-boundary digest. A separate schema-1 `current.json` binds the
instance, selected revision, and exact manifest SHA-256. Pointer writers
serialize through an advisory file lock, reject instance changes and revision
regression, sync a hidden staging file, atomically replace the pointer, and
sync the root before acknowledging `snapshot` success. Startup ignores hidden
files and complete but unpointed revisions. Repeating an interrupted revision
is allowed only when fresh serialization has the same state, token sequence,
and durable identity; a different value at the same revision fails closed.

Focused filesystem tests cover:

- a revision-2 / 21-token old session remaining current when a complete
  revision-3 / 18-token `new_session` directory and a broken pointer staging
  file exist but pointer publication has not completed;
- the same revision 3 becoming current only after pointer replacement;
- idempotent pointer and identical-revision retry;
- concurrent revision-2 and revision-3 pointer publishers ending at revision
  3 without regression;
- rejection of revision regression, instance replacement, different state at
  one revision, target-manifest tampering, token-file tampering, and a token
  sequence whose recomputed boundary differs from the manifest.

The real Qwen3.6-35B-A3B durable parity path then used two distinct execution
owners. Producer PID 72707 published revision 1 at 17 tokens and exited.
Restorer PID 72764 selected that revision through `current.json`, restored all
80 tensors and the 17 token IDs, produced the admitted two-token continuation
with oracle, logits, and state maximum absolute difference `0.0`, published
revision 2 at 19 tokens, and advanced the pointer. Revision 1's 90-byte token
file had SHA-256
`950580e44b513dac76f809318b76738dadace93d72088a9df95e4b3c600d75e0`;
revision 2's 101-byte token file had SHA-256
`024dbc26824cdf8bbdb579d675f0302a50e24e6dbbb6d64056e4a1fc4c630b1f`.
Both exactly matched their committed boundary digests. The reported physical
state-plus-token sizes were 64,746,023 and 64,786,994 bytes.

The adapter-level recovery probe separately started one Native process,
generated a structured tool call, and published revision 1 at 369 tokens
(71,956,594 reported state-plus-token bytes). After that process exited, a new
Native process restored the current revision without receiving a prompt or any
caller-provided token IDs. The opaque response token remained stable across
the restart; the restored adapter was idle; the exact continuation reported
369 cached tokens, processed only the 25-token tool-result suffix, generated
`7391` in five tokens, committed revision 2 at 399 tokens, and published it at
72,571,138 reported bytes. All nine protocol, ownership, restore, reuse,
revision, output, and publication checks were true.

Together these observations prove that Native durably owns and restores the
complete token ID sequence with the KV/GDN state, and that TypeScript needs
only the revision cursor to continue. They do not prove retention/garbage
collection policy or power-loss behavior at every remaining syscall boundary.
The complete command outputs and temporary snapshot trees were inspected but
were not added as redundant repository artifacts.

## Node local-composition lifecycle

The local Node composition root now starts one typed Native client, constructs
one stable `NativeInferenceModel` for each of `rin` and `marie`, and maps
each instance to its own snapshot root. On startup it checks for the
authoritative `current.json` pointer and restores that instance when present.
An existing pointer whose restore fails aborts startup and closes the owner;
it is not treated as an empty state.

The checkpoint boundary is one complete E.C.H.O. thinking session, meaning one
`ThinkingEngine.think()` invocation and all of its internal model/tool
iterations. A changed revision is published after the session, including when
a later tool or session operation fails. This avoids serializing the roughly
70 MB KV/GDN payload after every internal model generation. Sessions for the
same instance are exclusive so their continuation cursors cannot interleave.
Shutdown stops admission, requests token-boundary cancellation for active
generation, waits for session cleanup, retries any dirty checkpoint, and then
closes the shared process.

Six Vitest integration cases use the real TypeScript
`NativeInferenceClient` and `NativeInferenceModel` with a simulated typed
transport. They cover existing-state restore, retained model identity,
`new_session` selection across two thinking sessions, success and
post-commit-failure checkpoints, same-instance exclusion, idempotent waiting
shutdown, active-generation cancellation with committed-prefix persistence,
an in-flight checkpoint that is awaited rather than mis-cancelled, and
fail-closed restore cleanup. These tests prove the Node lifecycle and adapter
protocol decisions.

The app-level real lifecycle probe then ran the local
Qwen3.6-35B-A3B-MLX-4bit artifact through two distinct process owners. A stale
temporary protocol-v4 binary was first rejected before generation, as required.
After rebuilding current protocol-v5 source, the producer committed and
automatically checkpointed revision 1 at 358 lineage tokens. The second local
runtime restored revision 1 without an explicit restore call from the probe,
reported the same opaque response key and token boundary, and started a new
thinking session without `previousResponseToken`. That session processed 329
fresh input tokens, generated the same 29-token tool call, replaced the
lineage with the same 358-token count and digest, committed revision 2, and
advanced `current.json` to revision 2. All five lifecycle checks passed.

Both immutable state files were 71,729,618 bytes; each complete
`lineage.json` was 1,622 bytes. The current-source release binary had SHA-256
`50c39e9e612c425e3892170582ba1023b634ede5bb5ea55f6901141cbd8adf55`.
The compact output and temporary snapshot tree were inspected and then
removed rather than retained as redundant artifacts. Together with the
separate recovery probe above, this proves the real Rust-process path beneath
the Node lifecycle. At that milestone, physical snapshot cleanup was still
unverified; the latest-only result below supersedes that storage behavior. OS
signal delivery and complete local Chat/Memory/Note composition remain
unverified.

## Latest-only physical checkpoint

Durable revision remains monotonic for stale-write rejection and exact state
identity, but it is not a rollback-generation policy. Publication now holds
the per-instance current-pointer lock across immutable checkpoint creation,
pointer replacement, and cleanup. It synchronizes the new revision and
`current.json` before removing older exact managed revision directories. An
ordinary advance preserves a higher complete revision that may belong to a
publisher waiting for the lock. Successful startup holds the same lock through
complete current-state authentication and reconstruction, then removes every
other exact managed revision and exact crash-staging entry. Unknown files,
directories, and merely similar names remain untouched.

Ten snapshot tests cover pointer ordering, concurrent revision-2/revision-3
publishers, exact retry, tamper rejection, ordinary older-only cleanup,
startup all-except-current cleanup, staging-name boundaries, and preservation
of unknown entries. The complete Rust workspace passed 60 unit tests plus doc
tests, and Clippy passed for every target with warnings denied.

The app-level lifecycle probe was then rerun against the local
Qwen3.6-35B-A3B-MLX-4bit artifact with two distinct Native process owners. The
producer committed revision 1 at 358 tokens. The restarter automatically
restored its exact opaque response token after deleting an injected unpointed
revision 9, an exact revision-staging directory, and an exact pointer-staging
file while preserving an unknown operator note. It then began a fresh session,
reproduced the same 29-token `lookup_probe_code` tool call from 329 input
tokens, and committed revision 2 at the same 358-token lineage boundary.
`current.json` selected revision 2 and the only managed directory was
`revision-00000000000000000002`; all eight probe checks were true and the
command exited 0. The retained state file was 71,729,618 bytes and
`lineage.json` was 1,622 bytes. The tested release binary SHA-256 was
`a50f676b6a9854521eb8206598d64b07f98994ce8f93d68fd5f2009f9dc901e8`.

This proves latest-only physical convergence for clean sequential publication,
competing pointer publication, and successful process restart. It does not
inject power loss at every filesystem syscall or prove recovery from media
failure. The raw JSON and temporary 72 MB snapshot were inspected but are not
retained as duplicate repository artifacts.

## Full production-path A/B

The comparison used Qwen3.6-35B-A3B-MLX-4bit, the same 112-token prompt and 128
forced greedy decode steps, prefix cache and speculative decode disabled, one
warmup, and five measured requests per engine. Both sequential engine orders
were run. The baseline was the preceding exact score-only fusion binary.

| Sequential order    | Baseline internal | Candidate internal |                   Saved | Baseline external |   Candidate external |
| ------------------- | ----------------: | -----------------: | ----------------------: | ----------------: | -------------------: |
| Baseline, candidate |  12.6127 ms/token |   12.2558 ms/token | 0.3569 ms/token (2.83%) |       79.57 tok/s | 81.62 tok/s (+2.58%) |
| Candidate, baseline |  12.6008 ms/token |   12.2189 ms/token | 0.3819 ms/token (3.03%) |       79.68 tok/s | 81.89 tok/s (+2.77%) |

All 20 measured outputs had the same SHA-256 digest,
`7ad36ec149c73fb6b130c9d67a740bdc235eefea3bd598fa40f51a29ca609acc`.
At 12.219-12.256 ms/token, reaching 100 tok/s still requires approximately
2.219-2.256 ms/token, or 18.2-18.4% of candidate latency. Absolute rates should
not be compared directly with measurements from a different thermal/session
state; the same-session both-order deltas are the evidence for retention.

## Router-dependency isolation

An invalid-output diagnostic returned immediately after routing, preserving
the score dependency through one sum and multiply while removing the experts.
It used the same prompt, decode length, warmup count, measured count, and both
orders.

| Sequential order    | Baseline internal | Candidate internal |                   Saved | Baseline external |    Candidate external |
| ------------------- | ----------------: | -----------------: | ----------------------: | ----------------: | --------------------: |
| Baseline, candidate |   8.3378 ms/token |    8.1529 ms/token | 0.1849 ms/token (2.22%) |      119.94 tok/s | 122.74 tok/s (+2.33%) |
| Candidate, baseline |   8.3398 ms/token |    8.1661 ms/token | 0.1737 ms/token (2.08%) |      119.98 tok/s | 122.57 tok/s (+2.16%) |

The candidate-first isolation block had two early measured transients at
10.2969 and 10.0963 ms/token before three measurements at 8.1139-8.1661
ms/token. The table reports the predeclared five-sample median. The clean
baseline-first block and the full production-path both-order result are the
stronger evidence; the isolation establishes direction and approximate
component locality, not a complete additive attribution.

Individual measured values, medians, binary hashes, and output hashes are in
[`qwen36-fused-decode-router-ab-summary.json`](qwen36-fused-decode-router-ab-summary.json).
Warmup timings and the full protocol responses were not retained in the
repository.

## Interpretation

The earlier score-only fusion saved approximately 0.069-0.074 ms/token in its
router-only comparison. Keeping the probability vector and stable selection
inside one dispatch recovers a materially larger 0.174-0.185 ms/token in the
same isolation shape, while the complete production path improves by
0.357-0.382 ms/token. The larger end-to-end delta includes scheduling and
interaction effects and must not be treated as direct router kernel time.

The remaining post-linear routing work still includes 256 exponentials, exact
MLX-compatible reductions, and one dispatch. Fusing the Q8 router matrix-vector
projection into this kernel is a separate, higher-risk optimization because it
would require reproducing MLX's quantized-linear semantics as well as routing.
