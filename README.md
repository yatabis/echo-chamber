# E.C.H.O Chamber

Cloudflare Workers と Durable Object 上で動作する、Discord 連携型の思考エージェントです。OpenAI Responses API と Embedding を利用し、複数インスタンスを 1 つの Worker で運用できます。

## 主な機能

- 定期思考実行（Durable Object Alarm）
- Discord メッセージの読み取り・送信・リアクション
- Embedding ベースのメモリ保存 / 検索
- ノート管理（作成・一覧・検索・更新・削除）
- インスタンスごとのステータス UI
- 日別トークン使用量の蓄積と可視化

## アーキテクチャ

- ルーティング: `src/index.ts`
  - `GET /` はヘルスチェックを返却
  - `/:instanceId/*` を `Echo` Durable Object にフォワード
- コア: `src/echo/index.tsx`
  - インスタンス設定の遅延初期化
  - state / alarm / usage / notes の管理
  - ステータスページと JSON API を提供
- インスタンス設定: `src/config/echo-registry.ts`
  - Echo インスタンスの Discord Token、KV キー、プロンプトを定義
- 永続化
  - Durable Object Storage: state, notes, usage
  - KV (`ECHO_KV`): インスタンス別 Discord チャンネル ID
- 外部連携
  - OpenAI: `src/llm/openai/*`
  - Discord: `src/discord/*`

## 前提条件

- Node.js `22.16.0`（`package.json` の Volta 設定準拠）
- `pnpm`
- Cloudflare アカウント
- Discord Bot（Echo インスタンス用、加えてログ通知用）
- OpenAI API キー

## セットアップ

```bash
pnpm install
pnpm exec wrangler login
pnpm dev
```

補足:

- `pnpm dev` は `wrangler types && wrangler dev` を実行します。
- ローカルで Worker を動かす前に、次章の Secret / KV 設定を完了してください。

## 環境変数と Secret

### 必須キー

| キー名                     | 用途                               | 例                   |
| -------------------------- | ---------------------------------- | -------------------- |
| `OPENAI_API_KEY`           | OpenAI API 認証                    | `sk-...`             |
| `DISCORD_BOT_TOKEN`        | ログ通知用 Discord Bot Token       | `xxxxxxxx`           |
| `DISCORD_BOT_TOKEN_{name}` | Echo インスタンス用 Bot Token      | `xxxxxxxx`           |
| `LOG_CHANNEL_ID`           | ログ通知先チャンネル ID            | `123456789012345678` |
| `ENVIRONMENT`              | 実行環境判定（`local` / それ以外） | `local`              |

### Secret 設定コマンド例

```bash
pnpm exec wrangler secret put OPENAI_API_KEY
pnpm exec wrangler secret put DISCORD_BOT_TOKEN
pnpm exec wrangler secret put DISCORD_BOT_TOKEN_{name}
pnpm exec wrangler secret put LOG_CHANNEL_ID
```

`ENVIRONMENT` について:

- `ENVIRONMENT=local` のときのみ `POST /{instanceId}/run` が有効です。
- ローカル開発時は `.dev.vars` で `ENVIRONMENT=local` を指定してください。

## KV 初期化（インスタンス別チャンネル ID）

`ECHO_KV` には以下 キーを事前登録してください。

- `chat_channel_discord_{name}`
- `thinking_channel_discord_{name}`

設定コマンド例:

```bash
pnpm exec wrangler kv key put --binding ECHO_KV chat_channel_discord_{name} "<CHAT_CHANNEL_ID>"
pnpm exec wrangler kv key put --binding ECHO_KV thinking_channel_discord_{name} "<THINKING_CHANNEL_ID>"
```

## 実行・開発コマンド

| コマンド             | 用途                           |
| -------------------- | ------------------------------ |
| `pnpm dev`           | 型生成 + Wrangler ローカル起動 |
| `pnpm start`         | Wrangler ローカル起動          |
| `pnpm cf-typegen`    | Worker 型定義生成              |
| `pnpm test`          | Vitest 実行                    |
| `pnpm test:coverage` | カバレッジ付きテスト実行       |
| `pnpm lint`          | ESLint（自動修正あり）         |
| `pnpm typecheck`     | TypeScript 型チェック          |
| `pnpm deploy`        | Cloudflare へデプロイ          |

## HTTP エンドポイント（運用インターフェース）

| Method | Path                  | 説明                                            |
| ------ | --------------------- | ----------------------------------------------- |
| `GET`  | `/`                   | ヘルスチェック（`E.C.H.O Chamber is running.`） |
| `GET`  | `/{instanceId}/`      | ステータス UI                                   |
| `GET`  | `/{instanceId}/json`  | ステータス・メモリ・ノート・usage の JSON       |
| `POST` | `/{instanceId}/wake`  | 強制 wake（state を `Idling` に）               |
| `POST` | `/{instanceId}/sleep` | 強制 sleep（alarm 停止）                        |
| `POST` | `/{instanceId}/run`   | 手動実行（`ENVIRONMENT=local` のみ）            |
| `GET`  | `/{instanceId}/usage` | usage 履歴 JSON                                 |

## 運用メモ・注意点

- ログは `createLogger` で Discord に通知されます。
  - `ENVIRONMENT=local`: `debug` 以上通知
  - それ以外: `info` 以上通知
- Alarm 間隔は 1 分です（`ALARM_CONFIG.INTERVAL_MINUTES = 1`）。
- 毎日 03:00 に `Idling` なら sleep へ遷移し、07:00 に `Sleeping` なら wake します。

## トラブルシューティング

- Secret 未設定
  - 症状: OpenAI / Discord 呼び出しで認証エラー
  - 対応: `wrangler secret put` を再実行
- KV キー未設定
  - 症状: `Chat channel ID not found...` / `Thinking channel ID not found...`
  - 対応: キーを `ECHO_KV` に登録
- Discord 権限不足
  - 症状: メッセージ取得・送信・リアクションで失敗
  - 対応: Bot のチャンネル権限（Read/Send/Add Reactions）を確認

## この README が明確化する公開インターフェース

- HTTP エンドポイント群
- 必要 Secret キー
- 必要 KV キー

コード上の公開 API / 型の追加変更はありません。
