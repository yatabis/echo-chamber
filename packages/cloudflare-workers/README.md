# @echo-chamber/cloudflare-workers

Cloudflare Workers / Durable Objects 依存の実装を提供する package です。
`@echo-chamber/core` の型・ロジックを利用し、Cloudflare 実行環境に接続します。

## 役割

- Durable Object `Echo` 実装（`src/echo`）
- Cloudflare KV / Workers AI / OpenAI 連携
- メモリ・ノート・思考エンジンの Cloudflare 側実装
- Worker アプリ層（`apps/cloudflare-workers`）へ export 提供

## 依存

- `@echo-chamber/core`

## 公開 API

- `Echo`（`src/index.ts`）

## コマンド

- `pnpm --filter @echo-chamber/cloudflare-workers test`
- `pnpm --filter @echo-chamber/cloudflare-workers test:run`
- `pnpm --filter @echo-chamber/cloudflare-workers test:coverage`
- `pnpm --filter @echo-chamber/cloudflare-workers typecheck`

## テスト

- Workers 環境依存テスト: `packages/cloudflare-workers/src/**/*.test.ts`
- 共通モック/セットアップ: `packages/cloudflare-workers/test`
