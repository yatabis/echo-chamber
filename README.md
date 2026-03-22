# E.C.H.O Chamber

Cloudflare Workers / Durable Objects 上で動作する、Discord 連携型の思考エージェントです。
OpenAI Responses API と Embedding を利用し、複数インスタンスを 1 つの Worker で運用します。

## モノレポ構成

```text
apps/
  cloudflare-workers/        # Worker エントリ・DO実装・wrangler 設定・Cloudflare依存テスト・静的配信
  dashboard/                 # React + Vite ダッシュボード
packages/
  core/                      # Echo agent のドメイン / application 層（tool spec, prompt, session, ports）
  contracts/                 # Worker / Dashboard 間の API contract（DTO + zod schema）
  openai-adapter/            # ModelPort の OpenAI Responses 実装
  discord-adapter/           # Chat / Notification / ThoughtLog の Discord 実装
  cloudflare-runtime/        # Memory / Note など Cloudflare runtime 実装
```

## 依存ルール

- `packages/core` は Cloudflare 固有型や provider SDK に依存しない
- `packages/contracts` は API 境界の型と schema を持ち、UI 実装や runtime 実装を持たない
- adapter package は `packages/core` に依存する
- `packages/cloudflare-runtime` は `packages/core` に依存し、Cloudflare 固有実装を閉じ込める
- `apps/cloudflare-workers` は composition root として adapter / core を束ねる
- `apps/dashboard` は agent core ではなく API contract に依存する形へ寄せる
- workspace package は root barrel ではなく subpath import で参照する
- 禁止: `packages/core -> adapter/apps` の逆依存

## 前提条件

- Node.js `22.16.0`（`package.json` の Volta 設定準拠）
- `pnpm`
- Cloudflare アカウント
- Discord Bot（Echo インスタンス用 + ログ通知用）
- OpenAI API キー

## セットアップ

```bash
pnpm install
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler login
pnpm dev
```

補足:

- `pnpm dev` は `apps/cloudflare-workers` を対象に `wrangler types && wrangler dev` を実行します。
- dashboard の単体開発は `pnpm --filter @echo-chamber/dashboard dev` を使用します。

## 環境変数と Secret

### 必須キー

| キー名                    | 用途                               |
| ------------------------- | ---------------------------------- |
| `OPENAI_API_KEY`          | OpenAI API 認証                    |
| `DISCORD_BOT_TOKEN`       | ログ通知用 Discord Bot Token       |
| `DISCORD_BOT_TOKEN_RIN`   | `rin` インスタンス用 Bot Token     |
| `DISCORD_BOT_TOKEN_MARIE` | `marie` インスタンス用 Bot Token   |
| `LOG_CHANNEL_ID`          | ログ通知先チャンネル ID            |
| `ENVIRONMENT`             | 実行環境判定（`local` / それ以外） |

ローカル開発時は、`apps/cloudflare-workers/.dev.vars` に上記キーを設定します。

### Secret 設定例

```bash
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put OPENAI_API_KEY
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN_RIN
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN_MARIE
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put LOG_CHANNEL_ID
```

`ENVIRONMENT=local` のときのみ `POST /{instanceId}/run` が有効です。

## KV 初期化（thinking channel ID）

`ECHO_KV` に以下キーを登録してください。

- `thinking_channel_discord_rin`
- `thinking_channel_discord_marie`

chat 用チャンネル定義は `apps/cloudflare-workers/src/config/echo-runtime-bindings.ts` に固定で持ちます。
ここでは `thinkingChannelId` だけを KV に設定します。

設定例:

```bash
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler kv key put --binding ECHO_KV --local thinking_channel_discord_rin "<THINKING_CHANNEL_ID>"
```

`pnpm dev` で使うローカルKVに投入するため、ローカル開発時は `--local` を付けてください。

## 実行・開発コマンド

| コマンド                                            | 用途                                                               |
| --------------------------------------------------- | ------------------------------------------------------------------ |
| `pnpm dev`                                          | Worker ローカル起動（型生成付き）                                  |
| `pnpm start`                                        | Worker ローカル起動                                                |
| `pnpm cf-typegen`                                   | Worker 型定義生成                                                  |
| `pnpm deploy`                                       | Cloudflare へデプロイ                                              |
| `pnpm --filter @echo-chamber/dashboard dev`         | Dashboard 単体開発                                                 |
| `pnpm dashboard:build`                              | Dashboard ビルド（Worker assets に出力）                           |
| `pnpm test:run`                                     | `core` / `contracts` / adapter / runtime / worker のテスト実行     |
| `pnpm test:coverage`                                | `core` / `contracts` / adapter / runtime / worker の coverage 集約 |
| `pnpm lint:check` / `pnpm typecheck` / `pnpm check` | 品質チェック                                                       |

## HTTP エンドポイント

### Worker ルート

| Method | Path              | 説明                              |
| ------ | ----------------- | --------------------------------- |
| `GET`  | `/`               | ヘルスチェック                    |
| `GET`  | `/instances`      | インスタンス一覧（name/state）    |
| `GET`  | `/dashboard`      | Dashboard SPA 本体                |
| `GET`  | `/dashboard/*`    | Dashboard 静的配信 + SPA fallback |
| `ALL`  | `/{instanceId}`   | 対象 Durable Object にフォワード  |
| `ALL`  | `/{instanceId}/*` | 対象 Durable Object にフォワード  |

### Echo Durable Object (`/{instanceId}` 配下)

| Method | Path                    | 説明                                                   |
| ------ | ----------------------- | ------------------------------------------------------ |
| `GET`  | `/{instanceId}/`        | ステータス/メモリ/ノート/usage の JSON（`EchoStatus`） |
| `GET`  | `/{instanceId}/summary` | 一覧用サマリー JSON                                    |
| `POST` | `/{instanceId}/wake`    | 強制 wake                                              |
| `POST` | `/{instanceId}/sleep`   | 強制 sleep                                             |
| `POST` | `/{instanceId}/run`     | 手動実行（`ENVIRONMENT=local` のみ）                   |

## テスト方針（概要）

- agent ドメイン / 純粋ロジック: `packages/core/src/**/*.test.ts`
- API contract / schema: `packages/contracts/src/**/*.test.ts`
- provider adapter: `packages/openai-adapter/src/**/*.test.ts`, `packages/discord-adapter/src/**/*.test.ts`
- Cloudflare runtime: `packages/cloudflare-runtime/src/**/*.test.ts`
- Worker / Durable Object / route: `apps/cloudflare-workers/src/**/*.test.ts`
- Dashboard は現状、専用 test script ではなく build / typecheck と contract parser で整合を保つ
- `pnpm test:coverage` は monorepo 内の package / worker coverage を順に実行する
- `pnpm test:coverage` は `@cloudflare/vitest-pool-workers` の都合で sandbox 外の実行を前提にする

## 運用メモ

- Worker 設定ファイル: `apps/cloudflare-workers/wrangler.jsonc`
- Worker 型定義: `apps/cloudflare-workers/worker-configuration.d.ts`
- Dashboard build 出力先: `apps/cloudflare-workers/public/dashboard`
- dashboard 系 API contract は `packages/contracts/src/dashboard/schemas.ts` を正とする
