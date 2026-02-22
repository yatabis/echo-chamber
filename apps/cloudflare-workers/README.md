# @echo-chamber/app-cloudflare-workers

Cloudflare Worker のアプリ層です。  
この workspace は配線と配信を担当し、実ロジックは主に `@echo-chamber/cloudflare-workers` にあります。

## 役割

- Worker エントリ (`src/index.ts`)
- ルーティング:
  - `GET /`
  - `GET /dashboard` / `GET /dashboard/*`
  - `ALL /:instanceId/*`（Durable Object へフォワード）
- Wrangler 設定と bindings 管理
- Dashboard ビルド成果物の静的配信 (`public/dashboard`)

## 依存

- `@echo-chamber/cloudflare-workers`
- `@echo-chamber/core`

## 主要ファイル

- `wrangler.jsonc`
- `worker-configuration.d.ts`
- `src/index.ts`
- `public/`

## コマンド

- `pnpm --filter @echo-chamber/app-cloudflare-workers dev`
- `pnpm --filter @echo-chamber/app-cloudflare-workers start`
- `pnpm --filter @echo-chamber/app-cloudflare-workers cf-typegen`
- `pnpm --filter @echo-chamber/app-cloudflare-workers deploy`
- `pnpm --filter @echo-chamber/app-cloudflare-workers typecheck`

## メモ

- ルートの `pnpm dev` / `pnpm start` / `pnpm deploy` はこの workspace のコマンドを呼び出します。
- `wrangler.jsonc` 変更時は `pnpm cf-typegen` を実行してください。
