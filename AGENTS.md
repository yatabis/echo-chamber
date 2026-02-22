# Repository Guidelines

## Project Structure & Modules

- `apps/cloudflare-workers/`: Worker アプリ層（エントリ `src/index.ts`、`wrangler.jsonc`、`public/`）
- `apps/dashboard/`: React + Vite の Dashboard フロントエンド
- `packages/core/`: Cloudflare 非依存ロジック（型、usage、datetime、vector、error、Discord API ラッパ、共有 DTO）
- `packages/cloudflare-workers/`: Cloudflare 依存実装（Durable Object、KV、Workers AI、OpenAI クライアント、ルーティング実装）
- ルート設定: `pnpm-workspace.yaml`, `tsconfig.json`, `eslint.config.js`

## Dependency Rules

- `packages/core` は Cloudflare 固有型に依存しない
- `packages/cloudflare-workers` は `packages/core` に依存する
- `apps/cloudflare-workers` は配線層として `packages/cloudflare-workers` を利用する
- `apps/dashboard` は `packages/core` の型・変換ロジックを利用する
- 禁止: `packages/core -> packages/cloudflare-workers` の逆依存

## Build, Test, and Development

- Install: `pnpm install`
- Worker local dev: `pnpm dev` / `pnpm start`
- Worker typegen: `pnpm cf-typegen`
- Dashboard dev: `pnpm --filter @echo-chamber/dashboard dev`
- Dashboard build: `pnpm dashboard:build`
- Tests: `pnpm test`, `pnpm test:run`, `pnpm test:coverage`
- Lint/Format/Types: `pnpm lint`, `pnpm lint:check`, `pnpm format`, `pnpm typecheck`, `pnpm check`
- Deploy: `pnpm deploy`

## Coding Style & Naming

- Language: TypeScript (strict). JSX via `hono/jsx` (Worker 側)
- Format: Prettier（2-space, semicolon, single quote）
- Lint: ESLint + TypeScript rules（explicit return type, type-only import, import order, no unused vars）
- Naming: `kebab-case.ts`, component は `PascalCase.tsx`, tests は `*.test.ts`

## Testing Guidelines

- Framework: Vitest + `@cloudflare/vitest-pool-workers`
- Core tests: `packages/core/src/**/*.test.ts`
- Cloudflare tests: `packages/cloudflare-workers/src/**/*.test.ts`
- Test helpers/mocks:
  - `packages/core/test/**`
  - `packages/cloudflare-workers/test/**`
- Coverage は Cloudflare 側を中心に維持・改善する

## Commits & Pull Requests

- Commits: concise, present tense（英語/日本語可）
- PRs: 目的、変更概要、確認手順、必要ならスクリーンショットを記載
- マージ前に `pnpm check` と `pnpm test:run` を通す

## Security & Config

- Secrets は Wrangler secrets を使用（ローカルは `.dev.vars`）
- 例: `wrangler secret put OPENAI_API_KEY`, `wrangler secret put DISCORD_BOT_TOKEN_RIN`
- Worker では `setTimeout/Interval` を避け、Durable Object alarm を使用する
