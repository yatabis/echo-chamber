# Agent Core Refactor Checkpoints

## レビュー用タスクリスト

- [x] Checkpoint 1: adapter package scaffolds
  - 状態: レビュー済み / コミット済み
  - コミット: `bf6a328`
  - レビュー対象: package 境界の雛形
- [x] Checkpoint 2: core port interfaces
  - 状態: レビュー済み / コミット済み
  - コミット: `1c97d1b`
  - レビュー対象: port の粒度と概念分離
- [x] Checkpoint 3: canonical tool definitions in core
  - 状態: レビュー済み / コミット済み
  - コミット: `7651bb2`
  - レビュー対象: tool spec を `core` の正規定義源として置く設計
- [ ] Checkpoint 4: tool runtime context を port に寄せる
  - 状態: 実装完了 / レビュー待ち / 未コミット
  - レビュー対象: handler が concrete 実装ではなく port を使う構造
- [ ] Checkpoint 5: prompt builder を core に寄せる
  - 状態: 未着手
  - レビュー対象: prompt と tool spec の単一ソース化
- [ ] Checkpoint 6: agent loop の意味論を core に寄せる
  - 状態: 未着手
  - レビュー対象: loop / finish 条件 / usage 管理の core 移設
- [ ] Checkpoint 7: OpenAI adapter 分離
  - 状態: 未着手
  - レビュー対象: `ModelPort` の OpenAI 実装と変換層
- [ ] Checkpoint 8: Discord adapter 分離
  - 状態: 未着手
  - レビュー対象: chat / notification / thought-log の分離
- [ ] Checkpoint 9: Cloudflare runtime 分離
  - 状態: 未着手
  - レビュー対象: DO / KV / SQLite / scheduler を runtime package に隔離
- [ ] Checkpoint 10: contracts と dashboard 依存整理
  - 状態: 未着手
  - レビュー対象: dashboard DTO を agent core から切り離す構造
- [ ] 横断タスク: barrel export 廃止
  - 状態: 未着手
  - レビュー対象: subpath export への移行完了とバレル削除

## 目的

このメモは、大規模リファクタリングの進捗を「ユーザーレビュー単位」で追跡するためのチェックポイント表である。

- どこまで完了しているか
- どこがレビュー済みか
- 今どの段階の差分を持っているか
- この後どの順番で進めるか

を、コンパクション後でも再開できる形で外部化する。

## 完了済みチェックポイント

### Checkpoint 1: adapter package scaffolds

状態:

- 実装完了
- ユーザーレビュー済み
- コミット済み

コミット:

- `bf6a328` `add adapter package scaffolds`

内容:

- `packages/contracts`
- `packages/openai-adapter`
- `packages/discord-adapter`
- `packages/cloudflare-runtime`

を workspace に追加した。

この段階で行ったこと:

- package 雛形の作成
- `package.json`, `tsconfig.json`, `README.md`, `src/index.ts` の追加
- ルート `typecheck` 対象の更新
- ルート README の構造説明更新

レビュー観点:

- package 境界の方向性
- 依存方向の妥当性
- 今後の移設先として自然か

### Checkpoint 2: core port interfaces

状態:

- 実装完了
- ユーザーレビュー済み
- コミット済み

コミット:

- `1c97d1b` `add core port interfaces`

内容:

- `packages/core/src/ports/` を追加した
- runtime 実装が満たすべき抽象境界を `core` に導入した

追加した port:

- `ChatPort`
- `NotificationPort`
- `ThoughtLogPort`
- `MemoryPort`
- `NotePort`
- `ContextPort`
- `ModelPort`
- `ClockPort`
- `SchedulerPort`
- `LoggerPort`

レビュー中の修正反映:

- `Memory` と `Note` の定義位置を分離
- `Chat` と `Notification` を分離
- `ThoughtLog` を `Chat` と独立させた

レビュー観点:

- port の粒度がドメイン概念に沿っているか
- runtime 実装都合の型が `core` に漏れていないか

## 現在のチェックポイント

### Checkpoint 4: tool runtime context を port に寄せる

状態:

- 実装完了
- ユーザーレビュー待ち
- 未コミット

内容:

- `ToolContext` を `core` の `ToolExecutionContext` 基準に変更
- Worker 側で chat / notification / memory / note / logger の port adapter を組み立てるよう変更
- tool handler から `instanceConfig`, `MemorySystem`, `NoteSystem`, `DurableObjectStorage` への直接依存を外した

新設した主なファイル:

- `packages/core/src/agent/tool-context.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/tool-context.ts`

変更した主なファイル:

- `apps/cloudflare-workers/src/echo/thinking-engine/index.ts`
- `apps/cloudflare-workers/src/echo/memory-system/index.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/chat.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/index.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/memory.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/note.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/chat.test.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/memory.test.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/note.test.ts`
- `apps/cloudflare-workers/test/mocks/tool.ts`
- `packages/core/package.json`

この段階で達成したこと:

- handler が `ctx.chat`, `ctx.notifications`, `ctx.memory`, `ctx.notes`, `ctx.logger` の port 経由で動くようになった
- `ThinkingEngine` は concrete 実装を生成しても、tool へは port だけを渡す形になった
- OpenAI function definition の型は OpenAI 固有型ではなくローカルな shape として扱うように整理した

この段階ではまだやっていないこと:

- prompt builder の `core` 移設
- agent loop の意味論の `core` 移設
- OpenAI / Discord / Cloudflare runtime package への本格的な実装移設

品質チェック:

- `pnpm format`
- `pnpm lint:check`
- `pnpm typecheck`

は通過済み。

レビュー観点:

- `ToolExecutionContext` の粒度が妥当か
- `ThinkingEngine` が composition root として自然な形になっているか
- `handler -> port -> concrete implementation` の依存方向が妥当か

## 次のチェックポイント候補と横断タスク

### Checkpoint 4: tool runtime context を port に寄せる

目的:

- `ToolContext` から具体実装依存を外す
- `MemorySystem`, `NoteSystem`, `Logger`, `DurableObjectStorage` などを直接持たないようにする

やること:

- Worker 側 `ToolContext` の見直し
- `core` の port を使う context interface の導入
- handler から Cloudflare 具体型を追い出す準備

完了条件:

- tool handler が `core` port 経由で必要な機能に触れる構造になる

### Checkpoint 5: prompt builder を core に寄せる

目的:

- prompt と tool 実装の drift を止める

やること:

- static prompt 素材の整理
- tool catalog を使ったツール説明生成
- runtime context block の差し込みポイント整理

完了条件:

- prompt の中で手書きの tool 一覧を持たない形に寄せられる

### Checkpoint 6: agent loop の意味論を core に寄せる

目的:

- `ThinkingEngine` の中核ロジックを `core` に移す

やること:

- 初期メッセージ構築の抽象化
- turn loop / finish 条件 / usage 累積の責務分解
- OpenAI 依存を抜いた agent session 形式の導入

完了条件:

- loop の意味論が `core`
- OpenAI SDK 呼び出しは adapter

という構造になる

### Checkpoint 7: OpenAI adapter 分離

目的:

- `OpenAIClient` を `packages/openai-adapter` に移す

やること:

- `ModelPort` の OpenAI 実装を作る
- `core` tool spec -> OpenAI function tool 変換
- OpenAI 固有 usage / response 型の吸収

完了条件:

- `core` から OpenAI 型 import が消える

### Checkpoint 8: Discord adapter 分離

目的:

- Discord 実装を `core` から外す

やること:

- `packages/core/src/discord/*` を adapter 側へ移す
- `ThinkingStream` 相当を `ThoughtLogPort` 実装へ置き換える
- chat / notification / thought-log の実装を分ける

完了条件:

- `core` に Discord REST 実装が残らない

### Checkpoint 9: Cloudflare runtime 分離

目的:

- Cloudflare 固有の永続化とスケジューリングを runtime package に隔離する

やること:

- `MemorySystem`, `NoteSystem` を `packages/cloudflare-runtime` に移す
- state / usage / context store の Cloudflare 実装を切り出す
- alarm / KV / SQLite / Durable Object storage 依存を worker entry 以外から追い出す

完了条件:

- `core` と adapter が Cloudflare 型を import しない
- Worker は composition root と entry に近い責務だけを持つ

### Checkpoint 10: contracts と dashboard 依存整理

目的:

- dashboard representation を agent core から分離する

やること:

- Dashboard DTO を `packages/contracts` へ移す
- `apps/dashboard` の import を `core` から `contracts` に寄せる
- Worker 側の API response 組み立てを contracts 基準で整理する

完了条件:

- `packages/core` から dashboard 専用 DTO と util が消える

### 横断タスク: barrel export 廃止

目的:

- `core` の公開面を subpath export に限定し、依存の見通しを良くする

やること:

- 既存 import を root barrel から subpath import に移す
- `src/index.ts` と `src/ports/index.ts` への依存を段階的に解消する
- 不要になった barrel export を削除する

位置づけ:

- これは独立 checkpoint というより、複数 checkpoint にまたがって進める横断的な整理である
- まとめて最後にやるのではなく、移設した箇所から順次 subpath import に寄せる

完了条件:

- `@echo-chamber/core` のルート import に依存する箇所がなくなる

## 現時点の方針

優先順位:

1. Checkpoint 4 のレビュー完了
2. Checkpoint 5 で prompt と tool catalog の接続を整理する
3. Checkpoint 6 で agent loop の意味論を core に寄せる
4. その後に adapter 分離へ進む

## 再開時の注意

- 直近の未コミット差分は Checkpoint 4 のみ
- 直近のレビュー対象は tool runtime context の port 化
- barrel export 廃止は横断タスクとして扱うが、まだ未着手
