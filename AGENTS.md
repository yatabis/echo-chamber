# Repository Guidelines

## Working Agreement

- ユーザー向けの最終出力は日本語で行う
- unresolved な error / warning を残したまま完了しない
- code change 後は `pnpm lint:check`、`pnpm typecheck`、`pnpm format:check`、関連する test を実行し、問題があれば修正して再実行する
- 振る舞いを変える変更では、可能な限り test を先に追加または更新してから実装する
- マージ前の最低条件は `pnpm check` と `pnpm test:run` が通ること

## Architectural Invariants

- `packages/core` は Cloudflare 固有型や provider SDK に依存しない
- `packages/contracts` は API 境界の型 / schema を持ち、runtime 実装を持たない
- adapter / runtime package は `packages/core` に依存してよいが、`packages/core` から `apps/*` や adapter へ逆依存させない
- `apps/cloudflare-workers` は composition root として `core` / `contracts` / adapter / runtime を束ねる
- workspace package は root barrel ではなく subpath import で参照する

## Runtime Constraints

- Secret は Wrangler secrets を使い、ローカル開発時のみ `apps/cloudflare-workers/.dev.vars` を使う
- Worker / Durable Object では `setTimeout` / `setInterval` を避け、alarm を使う
