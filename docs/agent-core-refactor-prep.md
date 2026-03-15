# Agent Core Refactor Prep

## 目的

今回のリファクタリングの主目的は、Echo の本来のコアである agent のドメインロジックを `core` に集約し、OpenAI / Discord / Cloudflare Durable Objects などの具体実装を adapter 側へ分離することにある。

現状は `apps/cloudflare-workers` に以下が集中している。

- agent loop
- tool 定義と handler
- 起動時コンテキスト注入
- 実行ポリシー
- 永続化実装
- OpenAI Responses API 呼び出し

その一方で、`packages/core` には純粋ドメインではない Discord 実装や Dashboard 向け DTO が含まれている。

## 今回の狙い

- `packages/core` を Echo agent の純粋ドメイン / アプリケーション層にする
- LLM provider, chat provider, persistence, scheduler を port 越しに差し替え可能にする
- Cloudflare Worker を composition root に縮める
- prompt / tool spec / context injection の正規定義を `core` に寄せる

## 非目標

- 最初の段階で機能追加を行わない
- 最初の段階で永続化方式を全面変更しない
- 最初の段階で Dashboard の UI 改修を主目的にしない

## 目指す責務分離

### `packages/core`

保持するもの:

- agent loop の意味論
- tool の正規定義
- prompt builder
- context injection のルール
- 実行ポリシー
- port interface
- Echo のドメイン型

保持しないもの:

- OpenAI SDK 呼び出し
- Discord REST 呼び出し
- Cloudflare DO / KV / SQLite / Alarm
- Dashboard DTO

### 想定 package

- `packages/core`
  - pure domain + application
- `packages/contracts`
  - Worker / Dashboard 間 DTO
- `packages/openai-adapter`
  - OpenAI Responses API 実装
- `packages/discord-adapter`
  - Discord chat / notification / thought-log 実装
- `packages/cloudflare-runtime`
  - DO / KV / SQLite / scheduler / logger 実装

命名は実装前に再確認する。重要なのは package 名ではなく依存方向である。

## 実装前に確定すべきこと

### 1. `core` の境界

以下を `core` の責務として明文化する。

- agent session の開始条件
- 初期コンテキストの構築
- LLM に渡すメッセージ列の構築方針
- tool 実行と終了条件
- run budget / sleep-wake policy の判断

### 2. port 一覧

最低限必要な port を確定する。

- `ModelPort`
- `ChatPort`
- `NotificationPort`
- `ThoughtLogPort`
- `MemoryPort`
- `NotePort`
- `ContextPort`
- `ClockPort`
- `SchedulerPort`
- `LoggerPort`

`ToolContext` が Cloudflare 実装型や具体クラスを直接持たない形にする。

### 3. tool の正規定義形式

tool は `core` を唯一の正規定義源にする。

含めるべきもの:

- name
- description
- input schema
- output schema
- domain handler

OpenAI function tool への変換は adapter 側で行う。

### 4. prompt 管理方針

prompt は手書きの長文固定値だけでなく、以下の構成で扱う。

- persona / behavior / tool guidance の静的素材
- tool カタログから生成されるツール説明
- 起動時コンテキスト注入ブロック
- 現在時刻や memory snapshot のような runtime block

prompt 内のツール説明は実装から自動生成できる形を目指す。手書きの tool 一覧は drift を生みやすい。

### 5. context injection のモデル

「以前のコンテキスト」を何として扱うかを先に定義する。

候補:

- latest memory のみ
- 明示的な session context snapshot
- memory + note + recent notification summary の合成

少なくとも以下は分けて考える。

- 長期記憶
- セッション継続用の短期コンテキスト
- 外界の最新通知

### 6. 永続化の抽象境界

永続化は `core` から見て repository / store として扱う。

現状の保存先:

- Note: DO storage
- Usage / state: DO storage
- Memory: DO SQLite
- Chat channel binding: KV

実装前に「どこを抽象化し、どこを移行対象外にするか」を決める。

### 7. API contract の切り出し

Dashboard DTO は `core` から切り離す。

候補:

- `packages/contracts`
- `apps/dashboard` ローカル DTO

少なくとも agent core と dashboard representation は別物として扱う。

### 8. 移行の刻み方

一気に移すのではなく、以下の順番で移行する。

1. port と core 側の canonical types を導入
2. tool 定義を `core` に移す
3. prompt builder を `core` に移す
4. agent loop を `core` に移す
5. OpenAI adapter を分離
6. Discord adapter を分離
7. Cloudflare runtime adapter を分離
8. Dashboard contract を分離

## 既存コードの移設候補

### `packages/core` へ寄せる候補

- `apps/cloudflare-workers/src/echo/thinking-engine/index.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/*`
- `apps/cloudflare-workers/src/echo/index.tsx` 内の run policy 部分
- `packages/core/src/llm/prompts/*` は整理して残す

### adapter 側へ出す候補

- `apps/cloudflare-workers/src/llm/openai/client.ts`
- `packages/core/src/discord/*`
- `packages/core/src/utils/thinking-stream.ts`
- `apps/cloudflare-workers/src/echo/memory-system/index.ts`
- `apps/cloudflare-workers/src/echo/note-system/index.ts`
- `apps/cloudflare-workers/src/utils/logger.ts`

### `core` から外す候補

- `packages/core/src/dashboard/*`

## 実装前の確認項目

- `core` の公開 API を最小化できているか
- `core -> adapter` の逆依存が発生しないか
- OpenAI 固有型が `core` に漏れないか
- Discord メッセージ形式が `core` の domain model を汚染しないか
- Cloudflare の `Env`, `DurableObjectStorage`, `SqlStorage` が `core` に入ってこないか
- tool 説明と実装の二重管理を解消できるか
- 既存の state / usage / note / memory データ互換を壊さないか

## 最初の実装タスク

最初の PR では以下だけを行う。

- 新しい package 境界の雛形作成
- port interface の導入
- tool の canonical definition を `core` に移す
- 現行 Worker 実装がその port を満たすように adapter を薄く挟む

この段階では大規模な機能変更やストレージ移行は行わない。

## 備考

- `marie` の prompt 未設定
- `EmotionEngine` は未実装
- `context/tasks/knowledge` 削除処理は意図の確認が必要
- prompt と実装 tool の名称差分があるため、最初に drift を解消する

これらはリファクタリング中に巻き取るか、明示的に保留事項として分ける。
