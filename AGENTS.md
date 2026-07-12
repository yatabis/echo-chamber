# Repository Guidelines

## Working Agreement

- ユーザー向けの最終出力は日本語で行う
- JSDoc は原則として省略せず、基本的にきっちり記述する
- ユーザーは音声入力を用いるため typo が多い前提で読み取り、表記ゆれや誤記があっても文脈から意図を汲んで対応する
- unresolved な error / warning を残したまま完了しない
- code change 後は `pnpm lint:check`、`pnpm typecheck`、`pnpm format:check`、関連する test を実行し、問題があれば修正して再実行する
- 振る舞いを変える変更では、可能な限り test を先に追加または更新してから実装する
- マージ前の最低条件は `pnpm check` と `pnpm test:run` が通ること
- ユーザーの指示は重要な入力だが、常に正しいとは仮定しない。直近の指示を機械的に実行せず、これまでの合意、リポジトリの状態、実装上の整合性、失われる差分、技術的に正しい方針を照らし合わせて、自分の判断で行動を決める
- ユーザーの指示が矛盾している、危険である、前提を取り違えている、またはユーザー自身の意図と違う可能性がある場合は、実行前に立ち止まって確認する。必要なら「その操作はおそらく意図と違う」「こちらの方が正しい」と明確に指摘する
- ユーザーから「戻して」「消して」「やめて」「全部捨てて」のような取り消し・破棄系の指示が出た場合、単なる作業指示として即実行しない。まず、直前の実装・説明・判断に問題があった可能性、ユーザーの指示や理解が混乱している可能性、既存の合意と矛盾している可能性を検討し、何を戻すべきかを確認してから動く
- staged / unstaged / worktree / index を確認する場合は、`git status --short`、`git diff --name-status`、`git diff --cached --name-status` などの read-only コマンドで対象を切り分ける。特に `AM` / `MM` のような mixed state では、staged と unstaged を同じものとして扱わない
- 破壊的または不可逆に近い Git 操作は、ユーザーの言葉を字義通りに処理せず、目的と失われる差分を説明して確認する。ユーザーの最新指示よりも、これまでの合意・レビュー状態・作業の正しさを優先して検証する

### Git Index Ownership

- staged diff は、ユーザーが内容をレビュー済みと判断した境界として扱う
- Agent は `git add`、`git restore --staged`、`git reset`、`git apply --cached`、interactive staging など、stage / unstage / index を変更する操作を一切行わない。タスク遂行上必要に見えても、stage 操作はユーザーに委ねる
- Agent が関与していない staged diff の追加・削除・変更は、ユーザーがレビューして stage した、または不明点を確認するため unstage した可能性があるものとして扱う
- 身に覚えのない staged 状態を発見しても、勝手に戻す、補完する、再度 stage する、worktree の状態へ揃えるなどの操作を行わない
- staged 状態について判断が必要な場合や、ユーザーの意図と異なる可能性がある場合は、index に触れる前に必ずユーザーへ確認する

## Architectural Invariants

- `packages/core` は Cloudflare 固有型や provider SDK に依存しない
- `packages/contracts` は API 境界の型 / schema を持ち、runtime 実装を持たない
- adapter / runtime package は `packages/core` に依存してよいが、`packages/core` から `apps/*` や adapter へ逆依存させない
- `apps/cloudflare-workers` は composition root として `core` / `contracts` / adapter / runtime を束ねる
- workspace package は root barrel ではなく subpath import で参照する

## Runtime Constraints

- Secret は Wrangler secrets を使い、ローカル開発時のみ `apps/cloudflare-workers/.dev.vars` を使う
- Worker / Durable Object では `setTimeout` / `setInterval` を避け、alarm を使う
- Dashboard / Durable Object / Echo event / tool を追加・変更するときは、`docs/cloudflare-runtime-budget.md` の request / storage budget を確認し、API request、DO request、rows read、rows written、外部 API call の増分を説明できる状態にする
- Dashboard の read path では raw event や embedding BLOB の unbounded scan を避け、pre-aggregated read model または bounded latest rows を使う
- 動作確認や本番調査でも Cloudflare Free plan の制約を意識し、production endpoint の反復実行で rows read / written を消費しない。必要な確認は local / test / analytics / bounded diagnostic endpoint を優先する
- 既存データが削除されなくても、表示・検索・集計の対象から外れる変更、固定上限で以前見えていたデータが見えなくなる変更、過去指標の連続性が失われる変更は破壊的変更として扱い、実装前にユーザーへ明示する
