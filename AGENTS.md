# Repository Guidelines

## Project Structure & Modules

- `apps/cloudflare-workers/`: composition root / HTTP entry / Durable Object entry（Cloudflare binding, wrangler, tests, `public/`）
- `apps/dashboard/`: React + Vite の Dashboard フロントエンド
- `packages/core/`: Cloudflare 非依存の agent domain / application（tool spec, prompt builder, session loop, ports, shared utils）
- `packages/contracts/`: Worker / Dashboard 間の API contract（DTO, zod schema, dashboard util）
- `packages/openai-adapter/`: `ModelPort` の OpenAI Responses 実装
- `packages/discord-adapter/`: `ChatPort` / `NotificationPort` / `ThoughtLogPort` の Discord 実装
- `packages/cloudflare-runtime/`: Memory / Note など Cloudflare runtime 実装
- ルート設定: `pnpm-workspace.yaml`, `tsconfig.json`, `eslint.config.js`

## Dependency Rules

- `packages/core` は Cloudflare 固有型や provider SDK に依存しない
- `packages/contracts` は API 境界の型 / schema を持ち、runtime 実装を持たない
- adapter / runtime package は `packages/core` に依存する
- `apps/cloudflare-workers` は `core` / `contracts` / adapter / runtime を束ねる
- `apps/dashboard` は `packages/core` ではなく `packages/contracts` を参照する
- workspace package は root barrel ではなく subpath import で参照する
- 禁止: `packages/core -> apps/cloudflare-workers` の逆依存

## Build, Test, and Development

- Install: `pnpm install`
- Worker local dev: `pnpm dev` / `pnpm start`
- Worker typegen: `pnpm cf-typegen`
- Dashboard dev: `pnpm --filter @echo-chamber/dashboard dev`
- Dashboard build: `pnpm dashboard:build`
- Tests: `pnpm test:run`, `pnpm test:coverage`
- Lint/Format/Types: `pnpm lint`, `pnpm lint:check`, `pnpm format`, `pnpm typecheck`, `pnpm check`
- Deploy: `pnpm run deploy`

## Coding Style & Naming

- Language: TypeScript (strict). JSX via `hono/jsx` (Worker 側)
- Format: Prettier（2-space, semicolon, single quote）
- Lint: ESLint + TypeScript rules（explicit return type, type-only import, import order, no unused vars）
- Naming: `kebab-case.ts`, component は `PascalCase.tsx`, tests は `*.test.ts`

## Testing Guidelines

- Framework: Vitest + `@cloudflare/vitest-pool-workers`
- Core tests: `packages/core/src/**/*.test.ts`
- Contracts tests: `packages/contracts/src/**/*.test.ts`
- Adapter tests: `packages/openai-adapter/src/**/*.test.ts`, `packages/discord-adapter/src/**/*.test.ts`
- Runtime tests: `packages/cloudflare-runtime/src/**/*.test.ts`
- Cloudflare tests: `apps/cloudflare-workers/src/**/*.test.ts`
- Dashboard は現状、専用 test script ではなく build / typecheck と contract parser で整合を保つ
- Test helpers/mocks:
  - `packages/core/test/**`
  - `apps/cloudflare-workers/test/**`
- `pnpm test:coverage` は `core` / `contracts` / adapter / `cloudflare-runtime` / worker の coverage を集約する
- `pnpm test:coverage` は `@cloudflare/vitest-pool-workers` の都合で sandbox 外で実行する

## Commits & Pull Requests

- Commits: concise, present tense（英語/日本語可）
- PRs: 目的、変更概要、確認手順、必要ならスクリーンショットを記載
- マージ前に `pnpm check` と `pnpm test:run` を通す

## Security & Config

- Secrets は Wrangler secrets を使用（ローカルは `apps/cloudflare-workers/.dev.vars`）
- 例: `wrangler secret put OPENAI_API_KEY`, `wrangler secret put DISCORD_BOT_TOKEN_RIN`
- Worker では `setTimeout/Interval` を避け、Durable Object alarm を使用する
