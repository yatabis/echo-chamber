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
- [x] Checkpoint 9: Cloudflare runtime 分離
  - 状態: レビュー済み / コミット済み
  - コミット: `2b85f0f`
  - レビュー対象: Memory / Note / logger 実装を runtime package に隔離
- [x] Checkpoint 10: contracts と dashboard 依存整理
  - 状態: レビュー済み / コミット済み
  - コミット: `e92826c`
  - レビュー対象: dashboard DTO を agent core から切り離す構造
- [ ] 横断タスク: barrel export 廃止
  - 状態: 継続中
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

### 横断タスク: barrel export 廃止（root entrypoint removal）

状態:

- 実装完了
- ユーザーレビュー待ち
- 未コミット

内容:

- 直前の `worker shim cleanup` は `b18dfbc` (`reduce worker adapter shims`) でコミット済み
- `core root import reduction` は `4f6eeff` (`replace core root imports with subpath imports`) でコミット済み
- `packages/core/src/ports/index.ts` を削除し、`ports/context` と `ports/runtime` も subpath export で直接公開する形に変えた
- 各 package の `src/index.ts` と `"."` export を削除し、subpath のみを正式な公開面にした
- `packages/core/README.md` を subpath import 前提の公開方針に更新した
- worker 側に残していた Discord 互換 shim を削除し、`discord-adapter` の subpath を直接 import する形に変えた
- worker test setup も package subpath を直接 mock するように更新した

新設した主なファイル:

- なし

削除した主なファイル:

- `apps/cloudflare-workers/src/discord/client.ts`
- `packages/cloudflare-runtime/src/index.ts`
- `packages/contracts/src/index.ts`
- `packages/core/src/index.ts`
- `packages/core/src/ports/index.ts`
- `packages/discord-adapter/src/index.ts`
- `packages/openai-adapter/src/index.ts`

変更した主なファイル:

- `apps/cloudflare-workers/src/echo/index.tsx`
- `apps/cloudflare-workers/src/echo/thinking-engine/index.ts`
- `apps/cloudflare-workers/src/llm/openai/functions/tool-context.ts`
- `apps/cloudflare-workers/src/utils/logger.ts`
- `apps/cloudflare-workers/test/setup.ts`
- `apps/cloudflare-workers/vitest.config.ts`
- `apps/cloudflare-workers/package.json`
- `packages/core/src/index.ts`
- `packages/core/package.json`
- `packages/core/README.md`
- `packages/cloudflare-runtime/package.json`
- `packages/contracts/package.json`
- `packages/discord-adapter/package.json`
- `packages/openai-adapter/package.json`

この段階で達成したこと:

- package root entrypoint への依存を前提にしない構成に寄せられた
- `core/src/index.ts` と `core/src/ports/index.ts` に加えて、各 package の空 `src/index.ts` も不要化できた
- worker 側の Discord adapter 参照も相対 shim ではなく package subpath に統一できた

この段階ではまだやっていないこと:

- request / response schema の本格導入

品質チェック:

- `pnpm format`
- `pnpm lint:check`
- `pnpm typecheck`
- `pnpm test:run`

は通過済み。

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

1. 横断タスク: worker shim cleanup のレビュー完了
2. `@echo-chamber/core` ルート import を subpath import へ段階的に置き換える
3. barrel export 廃止を横断タスクとして継続する

## 再開時の注意

- 直近のコミット済み checkpoint は barrel export 廃止の一部である worker shim cleanup（`b18dfbc`）
- 直近のコミット済み checkpoint には `core root import reduction`（`4f6eeff`）も含まれる
- 直近の作業対象は `core` バレルファイルの実体削除
- 次のレビュー対象は root entrypoint を完全に消しても運用上問題ないかどうか
