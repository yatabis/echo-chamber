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

| キー名                           | 用途                                          |
| -------------------------------- | --------------------------------------------- |
| `OPENAI_API_KEY`                 | OpenAI API 認証                               |
| `DISCORD_BOT_TOKEN`              | ログ通知用 Discord Bot Token                  |
| `DISCORD_BOT_TOKEN_RIN`          | `rin` インスタンス用 Bot Token                |
| `DISCORD_BOT_TOKEN_MARIE`        | `marie` インスタンス用 Bot Token              |
| `LOG_CHANNEL_ID`                 | ログ通知先チャンネル ID                       |
| `ENVIRONMENT`                    | 実行環境判定（`local` / それ以外）            |
| `ACCESS_TEAM_DOMAIN`             | Cloudflare Access team domain の HTTPS origin |
| `ACCESS_PRODUCTION_HOSTNAME`     | production Worker の hostname                 |
| `ACCESS_PRODUCTION_AUD`          | production Access application の AUD          |
| `ACCESS_PREVIEW_HOSTNAME_SUFFIX` | preview Worker hostname の suffix             |
| `ACCESS_PREVIEW_AUD`             | preview Access application の AUD             |

ローカル開発時は、`apps/cloudflare-workers/.dev.vars` に Secret と `ENVIRONMENT=local` を設定します。`ACCESS_*` は non-local 環境で必須ですが、local bypass では参照しません。

### インスタンスごとの LLM / token limit 設定

通常の LLM / token limit は `packages/core/src/echo/instance-definitions.ts` の各 instance 定義で管理します。
API key などの secret と、一時的な上書きだけを環境変数で指定します。

Rapid-MLX や LM Studio などの OpenAI 互換 Chat Completions server を一時的に使う場合は、対象 instance の prefix を付けてローカルの `apps/cloudflare-workers/.dev.vars` に追加します。

```dotenv
MARIE_MAIN_LLM_PROVIDER=openai-compatible
MARIE_MAIN_LLM_MODEL=qwen3.6-27b
MARIE_MAIN_LLM_BASE_URL=http://localhost:8000/v1

MARIE_DAILY_HARD_TOKEN_LIMIT=250000
MARIE_DAILY_SOFT_TOKEN_LIMIT=150000
MARIE_HARD_TOKEN_LIMIT_BUFFER_FACTOR=1.5
```

prefix は `RIN_` / `MARIE_` を使います。prefix なしの `MAIN_LLM_*` や `DAILY_*_TOKEN_LIMIT` は、instance 定義に該当項目が無い場合の global fallback です。
`openai-compatible` は server 製品に依存しない Chat Completions 接続です。LM Studio と Rapid-MLX のどちらを使う場合もこの値を指定します。
`*_MAIN_LLM_MODEL` は接続先 runtime でロードしたモデルの identifier に合わせてください。
OpenAI Responses API の reasoning effort は `*_MAIN_LLM_REASONING_EFFORT` または `MAIN_LLM_REASONING_EFFORT` で一時上書きできます。値は `none` / `minimal` / `low` / `medium` / `high` / `xhigh` です。
OpenAI 互換 Chat Completions server では `*_MAIN_LLM_BASE_URL` が必須です。`*_MAIN_LLM_API_KEY` を省略すると OpenAI client 用の `not-needed` を使用するため、server 側で認証を有効にした場合だけ明示します。
Chat Completions API 利用時は、prompt template が user message を必須とするモデル向けに `developer` message を `user` role として渡します。
local runtime には `max_tokens: 32768`、`temperature: 0.7`、`top_p: 0.8`、`presence_penalty: 1.5`、`top_k: 20`、`chat_template_kwargs: { enable_thinking: false }` を固定で指定します。

標準の LM Studio / Rapid-MLX には独自 cache field を送りません。E.C.H.O. 向け session cache protocol を実装した専用 runtime を使う場合だけ、次を追加します。

```dotenv
MARIE_MAIN_LLM_RUNTIME_PROFILE=echo-session-cache-v1
```

この version 付き profile は、instance ごとの session ID と、最初の完了済み exchange を境に `pinned` から `rolling` へ切り替わる slot を独自 `cache` body で送ります。会話履歴そのものは通常の Chat Completions と同じく `messages` で再送します。

### Secret 設定例

```bash
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put OPENAI_API_KEY
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN_RIN
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put DISCORD_BOT_TOKEN_MARIE
pnpm --filter @echo-chamber/cloudflare-workers exec wrangler secret put LOG_CHANNEL_ID
```

`ENVIRONMENT=local` のときのみ `POST /{instanceId}/run` が有効です。

## Cloudflare Access 認証

`ENVIRONMENT=local` 以外の Worker request は、`Cf-Access-Jwt-Assertion` の署名、issuer、AUD、時刻 claim を Worker 内でも検証します。production と preview は別の AUD を使い、request hostname に一致しない token は `403 Forbidden` で拒否します。Access binding の欠落や未知の hostname も fail closed です。

Cloudflare Zero Trust 側では、production / preview の両 Access application を次の状態に保ちます。

- Identity provider は Cloudflare account のみを使用する
- Cloudflare account の MFA を有効にする
- Instant authentication と Binding Cookie を有効にする

ローカル開発では `ENVIRONMENT=local` に限って Access JWT 検証を省略します。`ENVIRONMENT` を未設定にして bypass することはできません。

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

| Method | Path                            | 説明                                                   |
| ------ | ------------------------------- | ------------------------------------------------------ |
| `GET`  | `/{instanceId}/`                | ステータス/メモリ/ノート/usage の JSON（`EchoStatus`） |
| `GET`  | `/{instanceId}/summary`         | 一覧用サマリー JSON                                    |
| `GET`  | `/{instanceId}/session-logs`    | Dashboard 用 session log JSON                          |
| `GET`  | `/{instanceId}/action-analysis` | Dashboard 用 action analysis JSON                      |
| `POST` | `/{instanceId}/wake`            | 強制 wake                                              |
| `POST` | `/{instanceId}/sleep`           | 強制 sleep                                             |
| `POST` | `/{instanceId}/run`             | 手動実行（`ENVIRONMENT=local` のみ）                   |

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
- Echo event は instance ごとの Durable Object SQLite に 90 日分保持し、日次 sleep 中に保持期限切れの event を削除する
