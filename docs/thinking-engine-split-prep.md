# Thinking Engine Split Prep

## 目的

`apps/cloudflare-workers/src/echo/thinking-engine/index.ts` の責務を分解し、agent の思考 orchestration を `packages/core` に移しつつ、Cloudflare / Discord / OpenAI / Workers AI への具体依存は `apps/cloudflare-workers` に残す。

ここで重要なのは、`ThinkingEngine` という名前の class を単純に移動することではない。現在の実装には engine 本体と composition root が同居しているため、先に責務境界を確定する必要がある。

## 現状整理

現行の `apps/cloudflare-workers/src/echo/thinking-engine/index.ts` は次の責務を持っている。

- `MemorySystem` / `NoteSystem` の生成
- embedding service の生成
- `ToolExecutionContext` の組み立て
- `DiscordThoughtLog` の生成
- `OpenAIResponsesModel` の生成
- prompt と startup input の構築
- `runAgentSession()` の実行
- tool 群の列挙
- DO storage から usage を読む

このうち、後半は engine 本体に近いが、前半と最後の usage 読み出しは Worker composition / DO runtime の責務である。

## 基本方針

### `packages/core` に置くもの

- 思考開始時の orchestration
- startup input の組み立て手順
- `runAgentSession()` の呼び出し順
- thought log の開始/終了メッセージ送信
- tool 実行を含む session のライフサイクル
- usage 差分を返す contract

### `apps/cloudflare-workers` に残すもの

- `Env` からの adapter/runtime 生成
- `MemorySystem` / `NoteSystem` / `EmbeddingService` の生成
- `ToolExecutionContext` の adapter 化
- `ThoughtLogPort` / `ModelPort` の具体実装生成
- DO storage からの usage 読み出しと永続化
- instance config / channel id / bindings の解決

## 非目標

- この段階で `Echo` 全体を分解しきること
- `tool-context` を `core` に移すこと
- embedding 実装の置き場所を今すぐ変えること
- `openai/functions/*` の handler まで同時に全部移設すること

## 責務の分割案

### 1. core 側: `ThinkingEngine`

候補パス:

- `packages/core/src/agent/thinking-engine.ts`

この engine は、次の依存だけを受け取る。

- `ModelPort`
- `ThoughtLogPort`
- executable tools の配列
- `LoggerPort`
- prompt 構築に必要な input
- startup 時に注入する latest memory snapshot
- current usage total

この engine は次を知らない。

- `Env`
- `DurableObjectStorage`
- `SqlStorage`
- Discord token / channel id
- OpenAI API key
- Cloudflare bindings

### 2. worker 側: `thinking-engine` factory / composition

候補パス:

- `apps/cloudflare-workers/src/echo/thinking-engine-factory.ts`
- もしくは `apps/cloudflare-workers/src/echo/thinking-engine/index.ts` を factory に縮小

ここで行うこと:

- `MemorySystem` / `NoteSystem` の生成
- embedding service の生成
- `OpenAIResponsesModel` の生成
- `DiscordThoughtLog` の生成
- `ToolExecutionContext` の組み立て
- current usage total の取得
- core `ThinkingEngine` への依存注入

## 依存 contract の考え方

### core engine に渡すべき input

- `systemPrompt`
- `currentDatetime`
- `latestMemory`
- `currentUsageTotal`

`latestMemory` は engine 内で `MemoryPort` から取りに行くのではなく、composition 側で取得して data として渡す方がよい。これは engine を pure orchestration に保つためである。

### core engine に渡すべき service

- `model: ModelPort`
- `thoughtLog: ThoughtLogPort`
- `logger: LoggerPort`
- `tools: AgentExecutableTool[]`

`AgentExecutableTool` は `runAgentSession()` が必要とする `name / contract / execute` を満たすだけの小さな型にする。

## startup sequence の扱い

現行は engine 内で次をやっている。

1. prompt input を作る
2. `checkNotificationsFunction` の tool call input を足す
3. 同 tool の tool result input を足す
4. その後に `runAgentSession()` を始める

この sequence 自体は engine の思考開始手順なので、`core` に残す価値が高い。

ただし、特定の `checkNotificationsFunction` 実装に engine が依存する形は避けるべきである。扱い方は次のどちらか。

- engine が `startupTool` を明示的に受け取り、その tool を先行実行する
- composition 側が startup input を完成させて engine に渡す

現時点では後者より前者の方が責務がはっきりしている。startup sequence は engine の振る舞いであり、ただの data 準備ではないためである。

## current usage の扱い

現行の engine は DO storage から今日の usage を読み、終了ログに合計 token を出している。

これは `core` の責務ではない。`core` engine は

- session usage delta を返す
- 終了ログで表示したい current total usage は値として受け取る

に留めるべきである。

つまり、usage の読取元は worker 側、usage の増分は engine 側という分離にする。

## test の分け方

### core 側で必要な test

- startup sequence が期待順で組み立てられること
- thought log の start / complete が送信されること
- `runAgentSession()` に tool 群と initial input が正しく渡ること
- session usage delta がそのまま返ること
- startup tool / thought log の失敗時の扱い

### worker 側で残す test

- `createToolExecutionContext()` の adapter 組み立て
- `ThinkingEngine` factory が runtime 実装を正しく束ねること
- embedding provider の選択
- DO storage から current usage を読む helper

今ある `apps/cloudflare-workers/src/llm/openai/functions/*.test.ts` は、thinking-engine 分割とは別軸のテストなので、この段階では原則そのまま維持する。

## 段階的な実装順

### Step 1

core 側に新しい `ThinkingEngine` contract を追加する。

- engine constructor input の型
- executable tool の型
- current usage / latest memory を受ける input 型

この段階では worker 側の既存 `ThinkingEngine` は消さない。

### Step 2

現行 `ThinkingEngine` の pure orchestration を core に移す。

- `buildInitialInput()`
- thought log 開始/終了
- `runAgentSession()` 呼び出し

### Step 3

worker 側の `ThinkingEngine` を factory / wrapper に縮める。

- runtime 実装生成
- core engine への依存注入
- DO storage から current usage を取得

### Step 4

core / worker 双方の test を再配置する。

- orchestration test は core
- composition test は worker

## 実装時の注意

- `ThinkingEngine` の移設と `tool handler` の移設を同じ diff でやらない
- `tool-context` は worker 側に残す
- `runAgentSession()` の抽象化を増やしすぎない
- current usage の read/write 責務を engine に戻さない
- startup sequence の contract は、後で provider を変えても崩れない形にする

## レビューで確認したいこと

- core engine が受け取る依存の粒度は適切か
- startup sequence を engine 責務として持たせるのが妥当か
- current usage total を value injection にする方針で問題ないか
- worker 側に残す composition の範囲が広すぎないか
