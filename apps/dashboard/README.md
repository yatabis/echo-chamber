# @echo-chamber/dashboard

React + Vite ベースの Dashboard アプリです。
Cloudflare Worker 経由で配信されることを前提に、TanStack Router で
`/dashboard`（一覧）と `/dashboard/:instanceId`（詳細）を構成します。

## 役割

- Dashboard UI (`src/App.tsx`)
- 一覧→詳細の画面遷移（`rin` / `marie`）
- ノート検索、メモリページング、usage 7/30日切替
- `@echo-chamber/core` の DTO 型を利用した表示

## データソース

- `GET /instances`: 一覧表示用のサマリー
- `GET /{instanceId}`: 詳細表示用 `EchoStatus`

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
