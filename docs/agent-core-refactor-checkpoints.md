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
- [ ] Checkpoint 7: OpenAI adapter 分離
  - 状態: 実装完了 / レビュー待ち / 未コミット
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

### Checkpoint 7: OpenAI adapter 分離

状態:

- 実装完了
- ユーザーレビュー待ち
- 未コミット

内容:

- OpenAI Responses API 実装を `packages/openai-adapter` へ移し、`OpenAIResponsesModel` を `ModelPort` adapter として切り出した
- worker 側の `ThinkingEngine` は package 側 adapter を使うだけにし、OpenAI 実装本体は workspace package に閉じ込めた
- `core` の `convertUsage` を `ModelUsage` 基準へ変更し、`core` から OpenAI 型 import を除去した
- OpenAI adapter のテストを package 側へ移し、ルート `test:run` に組み込んだ

新設した主なファイル:

- `packages/openai-adapter/src/openai-responses-model.ts`
- `packages/openai-adapter/src/openai-responses-model.test.ts`

変更した主なファイル:

- `apps/cloudflare-workers/src/echo/thinking-engine/index.ts`
- `apps/cloudflare-workers/src/llm/openai/client.ts`
- `apps/cloudflare-workers/src/echo/index.tsx`
- `apps/cloudflare-workers/package.json`
- `package.json`
- `packages/core/package.json`
- `packages/core/src/echo/usage.ts`
- `packages/core/src/echo/usage.test.ts`
- `packages/core/src/ports/model.ts`

この段階で達成したこと:

- OpenAI 実装本体が worker workspace の外に出て、package 単位で差し替え可能な位置に置かれた
- OpenAI 固有の request / response / log formatting は adapter package に集約された
- `core` の usage 変換は provider-neutral になり、OpenAI 型への依存が消えた
- worker 側には package source を参照する薄い shim だけを残し、entry 側の変更を最小化した

この段階ではまだやっていないこと:

- Discord adapter への分離
- Cloudflare runtime package への本格的な実装移設
- worker 側の shim を不要にする import 解決の最終整理

品質チェック:

- `pnpm format`
- `pnpm lint:check`
- `pnpm typecheck`
- `pnpm test:run`

は通過済み。

レビュー観点:

- OpenAI 実装が package 単位で独立した責務になっているか
- `core` から OpenAI 型依存が実際に消えているか
- worker 側に残した shim と package 境界の取り方が次段階の分離に耐えるか

## 次のチェックポイント候補と横断タスク

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

1. Checkpoint 5 のレビュー完了
2. Checkpoint 6 で agent loop の意味論を core に寄せる
3. その後に adapter 分離へ進む

## 再開時の注意

- 直近の未コミット差分は Checkpoint 5 のみ
- 直近のレビュー対象は prompt builder の core 移設
- barrel export 廃止は横断タスクとして扱うが、まだ未着手
