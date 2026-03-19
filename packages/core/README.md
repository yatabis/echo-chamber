# @echo-chamber/core

Cloudflare 非依存のコアロジックと共有型を提供する package です。
他 workspace はこの package を中心に型と純粋ロジックを参照します。

## 役割

- Echo の共通型と usage 集計ロジック
- agent / runtime 分離のための port interface
- 日時/ベクトル/エラーなどのユーティリティ
- システムプロンプト定義（`rin` / `marie`）

## 依存ルール

- Cloudflare 固有型・Cloudflare API には依存しない
- `@echo-chamber/cloudflare-workers` への逆依存は禁止

## 主要エクスポート

- Echo 関連の型・定数・usage ロジック（`@echo-chamber/core/echo/*`）
- port interface（`@echo-chamber/core/ports/*`）
- 共有型（`@echo-chamber/core/types/*`）
- ユーティリティ（`@echo-chamber/core/utils/*`）
- システムプロンプト（`@echo-chamber/core/llm/prompts/*`）

## 公開方針

- ルート import（`@echo-chamber/core`）は使わない
- 必要な記号は必ず subpath import で参照する

## コマンド

- `pnpm --filter @echo-chamber/core test`
- `pnpm --filter @echo-chamber/core test:run`
- `pnpm --filter @echo-chamber/core typecheck`

## テスト配置

- 純粋ロジックテスト: `packages/core/src/**/*.test.ts`
- 共通ヘルパー: `packages/core/test/helpers`
