# @echo-chamber/cloudflare-workers

Cloudflare Worker / Durable Object の実装本体です。
エントリ、ルーティング、Cloudflare 依存ロジック、テストをこの workspace に集約しています。

## 役割

- Worker エントリ (`src/index.ts`)
- Cloudflare Access JWT の Worker-side 検証 (`src/auth/cloudflare-access.ts`)
- ルーティング:
  - `GET /`
  - `GET /instances`
  - `GET /dashboard` / `GET /dashboard/*`
  - `ALL /:instanceId`
  - `ALL /:instanceId/*`（Durable Object へフォワード）
- Durable Object `Echo` 実装 (`src/echo`)
- Echo event の DO SQLite archive と dashboard session log API
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
- `src/auth/cloudflare-access.ts`
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
- Workers AI binding はローカル模擬されず、ローカル開発でも remote binding として利用量が発生し得ます。
- `pnpm dev` / `pnpm start` はローカル開発セッションだけ Worker 名 `echo-chamber-local-dev` を使用し、Wrangler の preview token で保護します。`pnpm deploy` は引き続き `wrangler.jsonc` の Worker 名 `echo-chamber` を使用します。
- ルートの `pnpm test:run` / `pnpm test:coverage` もこの workspace のテストを実行します。
- `wrangler.jsonc` 変更時は `pnpm cf-typegen` を実行してください。
- `ENVIRONMENT=local` 以外では Cloudflare Access JWT を必須とし、production / preview の hostname ごとに別の AUD を検証します。Access 関連 binding の欠落や未知の hostname は `403 Forbidden` になります。
- Access team domain と production / preview の hostname / AUD は `wrangler.jsonc` の `ACCESS_*` vars で管理します。これらは Secret ではありません。
- Echo の persona 定義は `@echo-chamber/core/echo/instance-definitions` にあり、この workspace では runtime bindings だけを解決します。
- DO を動かすには `ECHO_KV` に `thinking_channel_discord_*` を投入してください（ローカルは `wrangler kv key put --local`）。chat channels は `src/config/echo-runtime-bindings.ts` の固定定義を使います。
- Echo event は instance ごとの Durable Object SQLite に 90 日分保持します。R2 binding は使いません。
