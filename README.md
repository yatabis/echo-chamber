# E.C.H.O Chamber

Cloudflare Workers / Durable Objects 上で動作する、Discord 連携型の思考エージェントです。
OpenAI Responses API と Embedding を利用し、複数インスタンスを 1 つの Worker で運用します。

## モノレポ構成

```text
apps/
  cloudflare-workers/        # Worker エントリ・DO実装・wrangler 設定・Cloudflare依存テスト・静的配信
  dashboard/                 # React + Vite ダッシュボード
packages/
  core/                      # Cloudflare 非依存ロジック・共有型・Discord API ラッパ
```

## 依存ルール

- `packages/core` は Cloudflare 固有型に依存しない
- `apps/cloudflare-workers` は Cloudflare 実装を持ち、`packages/core` に依存する
- `apps/dashboard` は `packages/core` の型を利用する
- 禁止: `packages/core -> apps/cloudflare-workers` の逆依存

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

## KV 初期化（チャンネル ID）

`ECHO_KV` に以下キーを登録してください。

- `chat_channel_discord_rin`
- `thinking_channel_discord_rin`
- `chat_channel_discord_marie`
- `thinking_channel_discord_marie`

設定例:

```bash
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler kv key put --binding ECHO_KV --local chat_channel_discord_rin "<CHAT_CHANNEL_ID>"
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler kv key put --binding ECHO_KV --local thinking_channel_discord_rin "<THINKING_CHANNEL_ID>"
```

`pnpm dev` で使うローカルKVに投入するため、ローカル開発時は `--local` を付けてください。

## 実行・開発コマンド

| コマンド                                            | 用途                                         |
| --------------------------------------------------- | -------------------------------------------- |
| `pnpm dev`                                          | Worker ローカル起動（型生成付き）            |
| `pnpm start`                                        | Worker ローカル起動                          |
| `pnpm cf-typegen`                                   | Worker 型定義生成                            |
| `pnpm deploy`                                       | Cloudflare へデプロイ                        |
| `pnpm --filter @echo-chamber/dashboard dev`         | Dashboard 単体開発                           |
| `pnpm dashboard:build`                              | Dashboard ビルド（Worker assets に出力）     |
| `pnpm test:run`                                     | `core` + `app-cloudflare-workers` テスト実行 |
| `pnpm test:coverage`                                | Cloudflare 側テストのカバレッジ生成          |
| `pnpm lint:check` / `pnpm typecheck` / `pnpm check` | 品質チェック                                 |

## HTTP エンドポイント

### Worker ルート

| Method | Path              | 説明                              |
| ------ | ----------------- | --------------------------------- |
| `GET`  | `/`               | ヘルスチェック                    |
| `GET`  | `/dashboard`      | `/dashboard/` へリダイレクト      |
| `GET`  | `/dashboard/*`    | Dashboard 静的配信 + SPA fallback |
| `ALL`  | `/{instanceId}/*` | 対象 Durable Object にフォワード  |

### Echo Durable Object (`/{instanceId}` 配下)

| Method | Path                  | 説明                                   |
| ------ | --------------------- | -------------------------------------- |
| `GET`  | `/{instanceId}/`      | ステータス UI                          |
| `GET`  | `/{instanceId}/json`  | ステータス/メモリ/ノート/usage の JSON |
| `POST` | `/{instanceId}/wake`  | 強制 wake                              |
| `POST` | `/{instanceId}/sleep` | 強制 sleep                             |
| `POST` | `/{instanceId}/run`   | 手動実行（`ENVIRONMENT=local` のみ）   |
| `GET`  | `/{instanceId}/usage` | usage 履歴 JSON                        |

## テスト方針（概要）

- Cloudflare 依存ロジックのテスト: `apps/cloudflare-workers/src/**/*.test.ts`
- 純粋ロジックのテスト: `packages/core/src/**/*.test.ts`
- 共通モックは依存境界に合わせて `packages/core/test` と `apps/cloudflare-workers/test` に分離

## 運用メモ

- Worker 設定ファイル: `apps/cloudflare-workers/wrangler.jsonc`
- Worker 型定義: `apps/cloudflare-workers/worker-configuration.d.ts`
- Dashboard build 出力先: `apps/cloudflare-workers/public/dashboard`
