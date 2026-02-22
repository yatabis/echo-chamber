# @echo-chamber/core

Cloudflare 非依存のコアロジックと共有型を提供する package です。
他 workspace はこの package を中心に型と純粋ロジックを参照します。

## 役割

- Echo の共通型と usage 集計ロジック
- Dashboard DTO 型
- Discord API ラッパ
- 日時/ベクトル/エラーなどのユーティリティ
- システムプロンプト定義（`rin` / `marie`）

## 依存ルール

- Cloudflare 固有型・Cloudflare API には依存しない
- `@echo-chamber/app-cloudflare-workers` への逆依存は禁止

## 主要エクスポート

- Echo 関連の型・定数・usage ロジック（`src/echo/*`）
- 共有型（`src/types/*`）
- ユーティリティ（`src/utils/*`）
- Dashboard DTO（`src/dashboard/types.ts`）
- `systemPromptRin` / `systemPromptMarie`

## コマンド

- `pnpm --filter @echo-chamber/core test`
- `pnpm --filter @echo-chamber/core test:run`
- `pnpm --filter @echo-chamber/core typecheck`

## テスト配置

- 純粋ロジックテスト: `packages/core/src/**/*.test.ts`
- 共通ヘルパー: `packages/core/test/helpers`
