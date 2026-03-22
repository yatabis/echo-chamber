# Cloudflare Workers LLM Redesign Plan

## 目的

`apps/cloudflare-workers/src/llm` に残っているリファクタ途中の責務混在を解消し、次の境界に揃える。

- `packages/core`: provider 非依存の agent tool 定義と実行契約
- `packages/openai-adapter`: OpenAI Responses API adapter
- `apps/cloudflare-workers`: Cloudflare / Discord / runtime の composition root
- `embedding`: tool / model とは別軸の provider 選択

今回の作業は一括で進めず、レビュー前提で段階的に進める。

## 実施状況

- Step 0: 完了
- Step 1: 完了
- Step 2: 完了
- Step 3: 完了
- Step 4: このメモ更新時点で docs / README 整理まで完了

以下の step 詳細は、計画時点の変更範囲メモとして残している。旧パスを含む記述は実施履歴として扱う。

## 現在の問題

- `src/llm/openai/functions/*` は OpenAI 固有ではなく、runtime tool 実装である
- `src/llm` 配下に tool 実装と embedding 実装が混在している
- `Tool` class が OpenAI 向け `definition` を持つが、現行フローでは未使用で責務が重複している
- tool specification の正本は `packages/core` にある一方、実行ツールの列挙と実装は worker 側に散っている
- `tool-context` は LLM 層ではなく composition root の責務である
- embedding の抽象が `packages/cloudflare-runtime` にあり、所有境界がやや不自然である

## 目標状態

### ディレクトリの考え方

- `packages/core/src/agent/runtime-tools/*`
  - provider 非依存の executable tool 実装
- `apps/cloudflare-workers/src/echo/tool-context.ts`
  - Discord / MemorySystem / NoteSystem / Logger を `ToolExecutionContext` に束ねる
- `apps/cloudflare-workers/src/embedding/*`
  - embedding provider 実装と factory

### 役割分担

- `core` は tool spec と tool 実行 contract の正本を持つ
- `openai-adapter` は `ModelToolContract -> OpenAI function tool` 変換だけを担う
- worker は runtime port を組み立てて core tool を bind する
- embedding は `llm` から切り離し、独立した provider 選択として扱う

## 実行ステップ

### Step 0: 計画メモ作成

目的:

- 段階計画を固定し、レビュー単位を明確にする

変更範囲:

- `docs/cloudflare-workers-llm-redesign-plan.md`

完了条件:

- このメモが作成されていること

### Step 1: `tool-context` を composition root 側へ移動

目的:

- `tool-context` を `llm/openai/functions` から切り離し、責務名と配置を正す

変更範囲:

- `apps/cloudflare-workers/src/llm/openai/functions/tool-context.ts`
- `apps/cloudflare-workers/src/echo/index.tsx`
- 関連 test / mock

変更内容:

- `tool-context.ts` を `apps/cloudflare-workers/src/echo/tool-context.ts` へ移動
- import path を更新
- `llm/openai/functions` 依存から `ToolContext` 型依存を外す準備をする

完了条件:

- 参照先がすべて新パスに揃う
- 既存テストが通る

レビュー観点:

- `tool-context` の責務と置き場所が妥当か
- `echo` が composition root として読めるようになっているか

### Step 2: runtime tool 実装を `packages/core` に移す

目的:

- `openai/functions/*` の誤った配置を解消し、provider 非依存 tool 実装を `core` に集約する

変更範囲:

- `packages/core/src/agent/runtime-tools/*`
- `apps/cloudflare-workers/src/llm/openai/functions/*`
- `apps/cloudflare-workers/src/echo/index.tsx`
- 関連 test

変更内容:

- `Tool` class を `core` 側へ移設または `AgentSessionTool` ベースに再定義
- `chat` / `memory` / `note` / `thinking` の runtime tool 実装を `core` に移す
- `RUNTIME_TOOLS` の正本を `core` に寄せる
- worker 側は `ToolExecutionContext` を使って tool を bind するだけにする
- 未使用の `definition` を削除する

完了条件:

- `apps/cloudflare-workers/src/llm/openai/functions` が不要になる
- `openai-adapter` と tool 実装の責務重複が解消される
- `core` と worker の依存方向が崩れていない

レビュー観点:

- provider 非依存の executable tool を `core` に置く設計が妥当か
- `AgentSessionTool` との整合が取れているか

### Step 3: embedding を `llm` から分離

目的:

- embedding を tool / model と切り離し、独立した provider selection として整理する

変更範囲:

- `apps/cloudflare-workers/src/llm/embedding-factory.ts`
- `apps/cloudflare-workers/src/llm/openai/embedding.ts`
- `apps/cloudflare-workers/src/llm/workersai/embedding.ts`
- `apps/cloudflare-workers/src/echo/index.tsx`
- 関連 test

変更内容:

- `apps/cloudflare-workers/src/embedding/create-embedding-service.ts` を新設
- provider 実装を `src/embedding/providers/*` に移動
- `echo/index.tsx` の import を更新
- `src/llm` ディレクトリを削除可能な状態にする

完了条件:

- embedding 実装が `llm` から完全に消える
- `echo/index.tsx` から見た責務が明確になる

レビュー観点:

- embedding の置き場所と命名が妥当か
- 現時点で package 分離せず app 内に残す判断が妥当か

### Step 4: 抽象名の整理と後始末

目的:

- 名前・export・README を最終構成に合わせる

変更範囲:

- README / docs
- 使われなくなった file / export
- 必要なら `EmbeddingService` 抽象の再配置

変更内容:

- 未使用 export / test helper / import を削除
- README の `src/llm` 記述を更新
- 次の追加 refactor が必要なら別メモに切り出す

完了条件:

- 構成説明と実ファイル配置が一致している
- 不要ファイルが残っていない

レビュー観点:

- 今回の refactor のスコープが適切に閉じているか
- 次フェーズへ持ち越す論点が明確か

## 各ステップの品質ゲート

コード変更を含む Step では毎回以下を実行する。

- `pnpm lint:check`
- `pnpm typecheck`
- `pnpm format:check`
- 関連する test

Step 完了条件:

- error / warning が 0 件
- 変更意図とファイル配置が一致
- レビュー報告後、そのステップで一旦停止

## 今回の進め方

1. 各ステップ開始前に対象と編集方針を共有する
2. 1 ステップ分だけ実装する
3. 品質チェックを実行する
4. 変更内容とチェック結果を報告して停止する
5. レビュー承認後に次ステップへ進む
