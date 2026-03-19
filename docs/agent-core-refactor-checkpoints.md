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
- [x] Checkpoint 4: tool runtime context を port に寄せる
  - 状態: レビュー済み / コミット済み
  - コミット: `af934b0`
  - レビュー対象: handler が concrete 実装ではなく port を使う構造
- [x] Checkpoint 5: prompt builder を core に寄せる
  - 状態: レビュー済み / コミット済み
  - コミット: `84a3026`
  - レビュー対象: prompt と tool spec の単一ソース化
- [x] Checkpoint 6: agent loop の意味論を core に寄せる
  - 状態: レビュー済み / コミット済み
  - コミット: `302241f`
  - レビュー対象: loop / finish 条件 / usage 管理の core 移設
- [x] Checkpoint 7: OpenAI adapter 分離
  - 状態: レビュー済み / コミット済み
  - コミット: `c7baa96`
  - レビュー対象: `ModelPort` の OpenAI 実装と変換層
- [x] Checkpoint 8: Discord adapter 分離
  - 状態: レビュー済み / コミット済み
  - コミット: `4ea2bb4`
  - レビュー対象: chat / notification / thought-log の分離
- [ ] Checkpoint 9: Cloudflare runtime 分離
  - 状態: 実装完了 / レビュー待ち / 未コミット
  - レビュー対象: Memory / Note / logger 実装を runtime package に隔離
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

### Checkpoint 7: OpenAI adapter 分離

状態:

- 実装完了
- ユーザーレビュー済み
- コミット済み

コミット:

- `c7baa96` `extract openai adapter package`

内容:

- OpenAI Responses API 実装を `packages/openai-adapter` へ移し、`OpenAIResponsesModel` を `ModelPort` adapter として切り出した
- worker 側の `ThinkingEngine` は package 側 adapter を使うだけにし、OpenAI 実装本体は workspace package に閉じ込めた
- `core` の `convertUsage` を `ModelUsage` 基準へ変更し、`core` から OpenAI 型 import を除去した
- OpenAI adapter のテストを package 側へ移し、ルート `test:run` に組み込んだ

レビュー観点:

- OpenAI 実装が package 単位で独立した責務になっているか
- `core` から OpenAI 型依存が実際に消えているか
- worker 側に残した shim と package 境界の取り方が次段階の分離に耐えるか

### Checkpoint 8: Discord adapter 分離

状態:

- 実装完了
- ユーザーレビュー済み
- コミット済み

コミット:

- `4ea2bb4` `extract discord adapter package`

内容:

- Discord REST 実装と port adapter を `packages/discord-adapter` へ移した
- worker 側の `tool-context` は `createDiscordChatPort` / `createDiscordNotificationPort` を組み立てるだけにし、chat / notification の実装詳細を adapter に閉じ込めた
- `ThinkingStream` 相当は `DiscordThoughtLog` として adapter に移し、worker 側は `ThoughtLogPort` 実装を使うだけにした
- `core` に残っていた Discord 実装ファイルと test helper を削除した
- `discord-adapter` 側に `chat-port` / `notification-utils` / `discord-thought-log` のテストを追加した

レビュー観点:

- Discord 実装が `core` から実際に消えているか
- chat / notification / thought-log の概念分離が adapter でも保たれているか
- worker 側に残した shim と test setup が次段階の runtime 分離に耐えるか

## 現在のチェックポイント

### Checkpoint 9: Cloudflare runtime 分離

状態:

- 実装完了
- ユーザーレビュー待ち
- 未コミット

内容:

- `MemorySystem` と `NoteSystem` を `packages/cloudflare-runtime` に移し、SQLite / Durable Object storage に触る実装本体を worker package の外へ出した
- `EmbeddingService` 抽象を runtime package に追加し、worker 側の OpenAI / Workers AI 実装はその interface を満たす形にした
- logger の中核実装を `CloudflareRuntimeLogger` として runtime package に移し、worker 側は Discord 送信 callback を差し込む thin wrapper だけにした
- worker 側には `src/runtime/*` の shim を置き、entry / thinking engine / tool-context から runtime package source を参照する形に整理した
- Memory / Note のテストを runtime package 側に移し、ルート `test:run` に組み込んだ

新設した主なファイル:

- `packages/cloudflare-runtime/src/embedding-service.ts`
- `packages/cloudflare-runtime/src/memory-system.ts`
- `packages/cloudflare-runtime/src/memory-system.test.ts`
- `packages/cloudflare-runtime/src/note-system.ts`
- `packages/cloudflare-runtime/src/note-system.test.ts`
- `packages/cloudflare-runtime/src/runtime-logger.ts`
- `packages/cloudflare-runtime/vitest.config.ts`
- `apps/cloudflare-workers/src/runtime/embedding-service.ts`
- `apps/cloudflare-workers/src/runtime/logger.ts`
- `apps/cloudflare-workers/src/runtime/memory-system.ts`
- `apps/cloudflare-workers/src/runtime/note-system.ts`

変更した主なファイル:

- `apps/cloudflare-workers/src/echo/index.tsx`
- `apps/cloudflare-workers/src/echo/thinking-engine/index.ts`
- `apps/cloudflare-workers/src/llm/embedding-factory.ts`
- `apps/cloudflare-workers/src/llm/openai/embedding.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/tool-context.ts`
- `apps/cloudflare-workers/src/llm/workersai/embedding.ts`
- `apps/cloudflare-workers/src/utils/logger.ts`
- `package.json`
- `packages/core/package.json`
- `packages/cloudflare-runtime/package.json`
- `packages/cloudflare-runtime/tsconfig.json`
- `packages/cloudflare-runtime/README.md`

この段階で達成したこと:

- Memory / Note / logger の Cloudflare runtime 実装が worker package から切り離された
- worker 側の `ThinkingEngine` と `Echo` は persistence 実装の詳細ではなく runtime service を組み立てる側に近づいた
- worker 側には package source を参照する薄い shim だけを残し、entry 側の変更を最小化した
- runtime package 単位のテストが追加され、SQLite / Durable Object storage ベースのロジックを worker test から分離できた

この段階ではまだやっていないこと:

- state / usage / context store の runtime package への移設
- alarm / KV / scheduler 依存の worker entry からの切り離し
- worker 側の shim を不要にする import 解決の最終整理

品質チェック:

- `pnpm format`
- `pnpm lint:check`
- `pnpm typecheck`
- `pnpm test:run`

は通過済み。

レビュー観点:

- `MemorySystem` / `NoteSystem` / logger の実装責務が runtime package に適切に集約されているか
- worker 側が composition root に寄る方向になっているか
- 次段階で state / usage / context / scheduler を runtime package に移す余地が見えているか

## 次のチェックポイント候補と横断タスク

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

1. Checkpoint 5 のレビュー完了
2. Checkpoint 6 で agent loop の意味論を core に寄せる
3. その後に adapter 分離へ進む

## 再開時の注意

- 直近の未コミット差分は Checkpoint 5 のみ
- 直近のレビュー対象は prompt builder の core 移設
- barrel export 廃止は横断タスクとして扱うが、まだ未着手
