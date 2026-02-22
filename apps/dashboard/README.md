# @echo-chamber/dashboard

React + Vite ベースの Dashboard アプリです。
Cloudflare Worker 経由で配信されることを前提に、`/{instanceId}/json` と
`/{instanceId}/usage` から状態を取得します。

## 役割

- Dashboard UI (`src/App.tsx`)
- instance 選択と状態表示（`rin` / `marie`）
- `@echo-chamber/core` の DTO 型を利用した表示

## データソース

- `GET /{instanceId}/json`: status / memories / notes / usage
- `GET /{instanceId}/usage`: usage 履歴

## コマンド

- `pnpm --filter @echo-chamber/dashboard dev`
- `pnpm --filter @echo-chamber/dashboard build`
- `pnpm --filter @echo-chamber/dashboard preview`
- `pnpm --filter @echo-chamber/dashboard typecheck`

## ビルド出力

- 出力先: `apps/cloudflare-workers/public/dashboard`
- 設定: `apps/dashboard/vite.config.ts`

## メモ

- 単体開発は `dev`、Worker 同梱用は `build` を使います。
