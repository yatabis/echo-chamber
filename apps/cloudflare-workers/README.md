# @echo-chamber/cloudflare-workers

Cloudflare Worker / Durable Object の実装本体です。
エントリ、ルーティング、Cloudflare 依存ロジック、テストをこの workspace に集約しています。

## 役割

- Worker エントリ (`src/index.ts`)
- ルーティング:
  - `GET /`
  - `GET /instances`
  - `GET /dashboard` / `GET /dashboard/*`
  - `ALL /:instanceId`
  - `ALL /:instanceId/*`（Durable Object へフォワード）
- Durable Object `Echo` 実装 (`src/echo`)
- Cloudflare KV / Workers AI / OpenAI 連携 (`src/config`, `src/echo`, `src/embedding`, `src/utils`)
- Echo の runtime bindings 解決 (`src/config/echo-runtime-bindings.ts`)
- Cloudflare 依存テスト (`src/**/*.test.ts`, `test/**`)
- Wrangler 設定と bindings 管理
- Dashboard ビルド成果物の静的配信 (`public/dashboard`)

## 依存

- `@echo-chamber/core`

## 主要ファイル

- `wrangler.jsonc`
- `worker-configuration.d.ts`
- `src/index.ts`
- `src/echo/index.tsx`
- `src/config/echo-runtime-bindings.ts`
- `vitest.config.ts`
- `test/`
- `public/`

## コマンド

- `pnpm --filter @echo-chamber/cloudflare-workers dev`
- `pnpm --filter @echo-chamber/cloudflare-workers start`
- `pnpm --filter @echo-chamber/cloudflare-workers cf-typegen`
- `pnpm --filter @echo-chamber/cloudflare-workers deploy`
- `pnpm --filter @echo-chamber/cloudflare-workers test:run`
- `pnpm --filter @echo-chamber/cloudflare-workers test:coverage`
- `pnpm --filter @echo-chamber/cloudflare-workers typecheck`

## メモ

- ローカル環境変数は `apps/cloudflare-workers/.dev.vars` を利用します。
- ルートの `pnpm dev` / `pnpm start` / `pnpm deploy` はこの workspace のコマンドを呼び出します。
- ルートの `pnpm test:run` / `pnpm test:coverage` もこの workspace のテストを実行します。
- `wrangler.jsonc` 変更時は `pnpm cf-typegen` を実行してください。
- Echo の persona 定義は `@echo-chamber/core/echo/instance-definitions` にあり、この workspace では runtime bindings だけを解決します。
- DO を動かすには `ECHO_KV` に `chat_channel_discord_*` / `thinking_channel_discord_*` を投入してください（ローカルは `wrangler kv key put --local`）。
