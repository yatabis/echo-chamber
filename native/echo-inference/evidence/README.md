# Native inference evidence retention

## Git retention policy

The Git-candidate `evidence/` tree is intentionally curated. It contains only:

- dated Markdown conclusions and explicit limitations;
- small deterministic fixtures or contracts that can be consumed again;
- final, portable parity or performance summaries that support the currently
  retained implementation.

It does not contain exploratory ablations, superseded candidate runs, repeated
raw attempts, server logs, temporary binary paths, worktree snapshots, or
machine-specific protocol dumps. A result already represented faithfully in a
dated README does not also need a standalone JSON summary unless automation or
exact per-run values materially depend on it.

## Current Git-candidate artifacts

The 2026-07-30 set retains reusable production-sampling and chat-template
fixtures plus the complete-model parity contract and result. The 2026-08-01
set retains the final fused-router validation and both-order A/B measurements.
The 2026-07-31 conclusions are fully compiled into its README. The 2026-08-03
README retains the protocol-v7 short, stateful, and long-session performance
conclusions. The 2026-08-04 README retains the production-buffered promotion,
independent 2K–32K context curve, 20-minute sustained-load, and serial
multi-instance conclusions, plus the retained adaptive-prefill decision and
its Native/MLX-LM execution-shape comparison. Raw performance and diagnostic
outputs remain local-only.

## Local diagnostic archive

Raw and superseded outputs are stored locally under:

```text
.artifacts/model-evaluation/native-inference/evidence/
```

The root `.gitignore` excludes this directory. It is a recoverable local
working archive, not durable project history, and may be regenerated from the
oracles and evaluation runners. Dated READMEs may name local artifacts for
provenance, but must not imply that those files will be tracked beside the page.

Before adding a new evidence file, prefer updating the relevant dated README.
Keep a JSON file only when it is a reusable fixture/contract, a final portable
validation result, or the exact measured sample is materially useful beyond
the prose summary.
