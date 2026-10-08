# E.C.H.O. の能力と設計の棚卸し

保守性・拡張性を改善するため、維持したい能力、現在の実現方法、検証の根拠を対応づける。個体の独立性と経験の継続を中心に置き、思考ループの役割分担や実行順序も見直せるようにする。

この文書はレビュー用の草案である。維持対象と優先順位は候補として示す。既存の規範仕様は引き続き有効であり、ここに記した検討点だけで実装の振る舞いを変更しない。新しいアーキテクチャや全面再実装の採否は、この棚卸しと比較検証を踏まえて判断する。

## 対象と根拠の読み方

確認した実装の基準は Git HEAD `7e8e48cc7ed65ff08390fc4872cf5a06b1f1e70e`。Hosted は Cloudflare Workers / Durable Objects の入口、Local は `apps/local-runtime` と Native 推論の接続を指す。

検証の根拠は次のように区別する。

- **テスト**: 既存テストが確認する契約。モデル応答、外部サービス、保存先を fixture / mock に置き換える範囲がある。外部サービスへの実配送や実モデルの判断品質まで保証しない。
- **過去の実モデル記録**: 指定したモデル・環境・課題で得られた結果。現在の実行結果や、通常アプリケーションの完成を意味しない。
- **未確認**: 読み取った仕様・実装・証跡から成立を確認できないもの。テストの存在や通過から補わない。

Hosted の実装が存在することと、本番で継続運用できていることも別の証拠として扱う。

## 維持したい能力の候補

### 個体と思考の継続

| 能力                                                       | 現行仕様                                                                                                                               | 主な実装                                                                                                                                                                                                                              | 検証の根拠と限界                                                                                                                                                                                                                                              |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 個体ごとに identity・記憶・感情・推論状態を分離する        | [instance 境界](./echo-registry-boundary.md)、[Native の所有権](../native/echo-inference/docs/architecture.md)                         | [instance 定義](../packages/core/src/echo/instance-definitions.ts)、[Local lifecycle](../apps/local-runtime/src/local-native-inference-runtime.ts)                                                                                    | [定義テスト](../packages/core/src/echo/instance-definitions.test.ts)、[Local テスト](../apps/local-runtime/src/local-native-inference-runtime.test.ts)。全保存先・送信先を通した個体間分離の運用検証は別途必要                                                |
| 呼びかけを観測し、自発的にも思考を開始・終了・再開する     | [起動・スケジュール](./echo-processing-flows.md)、[Main の行動指示](../packages/core/src/llm/prompts/rin.ts)                           | [Hosted Echo](../apps/cloudflare-workers/src/echo/index.tsx)、[終了 tool](../packages/core/src/agent/runtime-tools/finish.ts)                                                                                                         | [起動判定・alarm テスト](../apps/cloudflare-workers/src/echo/index.test.ts)、[session テスト](../packages/core/src/agent/session.test.ts)。Local の常駐 scheduler は未接続                                                                                    |
| 経験を記憶として保存し、必要な経験を想起する               | [Cognitive 仕様](./cognitive-module-architecture.md)、[Memory の予算と上限](./cloudflare-runtime-budget.md)                            | [MemorySystem](../packages/cloudflare-runtime/src/memory-system.ts)、[Cognitive domain](../apps/cloudflare-workers/src/echo/cognitive-module-domain.ts)、[明示的 Memory tool](../packages/core/src/agent/runtime-tools/memory.ts)     | [検索・保存テスト](../packages/cloudflare-runtime/src/memory-system.test.ts)、[phase commit テスト](../apps/cloudflare-workers/src/echo/cognitive-module-domain.test.ts)。Local の実検索 backend は未接続。検索成功と、必要な経験を選べる品質は分けて評価する |
| 感情状態を保持・更新し、判断に参照できるようにする         | [Emotion の出力と handoff](./cognitive-module-architecture.md)                                                                         | [module 生成](../packages/core/src/agent/model-cognitive-module.ts)、[handoff](../packages/core/src/agent/cognitive-module-handoff.ts)、[Cognitive domain](../apps/cloudflare-workers/src/echo/cognitive-module-domain.ts)            | [orchestrator テスト](../packages/core/src/agent/cognitive-module-orchestrator.test.ts)、[domain テスト](../apps/cloudflare-workers/src/echo/cognitive-module-domain.test.ts)。入力への伝達・保存を確認する。感情が行動に与える効果は未確認                   |
| 状況・記憶・感情を使って行動を選び、tool で環境へ作用する  | [思考・tool フロー](./echo-processing-flows.md)、[Main prompt](../packages/core/src/llm/prompts/rin.ts)                                | [ThinkingEngine](../packages/core/src/agent/thinking-engine.ts)、[session](../packages/core/src/agent/session.ts)、[tool catalogue](../packages/core/src/agent/runtime-tools/catalog.ts)                                              | [ThinkingEngine テスト](../packages/core/src/agent/thinking-engine.test.ts)、[session テスト](../packages/core/src/agent/session.test.ts)。既存 workflow 評価は Cognitive 経路と異なり、現行ループ全体の判断品質の受け入れは未完了                            |
| 失敗時も確定状態と未確定状態を区別し、再試行で整合性を保つ | [Cognitive の失敗処理](./cognitive-module-architecture.md)、[Native transaction](../native/echo-inference/docs/architecture.md)        | [orchestrator](../packages/core/src/agent/cognitive-module-orchestrator.ts)、[domain](../apps/cloudflare-workers/src/echo/cognitive-module-domain.ts)、[Local lifecycle](../apps/local-runtime/src/local-native-inference-runtime.ts) | 上記の各テストに、失敗した module だけの再試行、domain version conflict、checkpoint 失敗の保持がある。全外部送信の exactly-once は保証していない                                                                                                              |
| 終了・再起動をまたいで、保存した経験から次の判断へつなぐ   | [Native 統合の受け入れ条件](./native-runtime-integration-readiness.md)、[snapshot 契約](../native/echo-inference/docs/architecture.md) | [Local lifecycle](../apps/local-runtime/src/local-native-inference-runtime.ts)、[Native adapter](../packages/native-inference-adapter/src/native-inference-model.ts)                                                                  | Main snapshot の別プロセス復元に過去の実モデル記録がある。domain fixture を使う [Cognitive 実モデルテスト](../apps/local-runtime/src/real-cognitive-workflow.test.ts)もある。永続 Memory の検索から再起動後の Main の利用までを通した受け入れは未確認         |

### 外部接続と運用

| 能力                                                                   | 現行仕様                                                                                                     | 主な実装                                                                                                                                                                                                                                                         | 検証の根拠と限界                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 複数チャンネルの通知・会話・画像を読み、メッセージや reaction を送る   | [tool と画像入力のフロー](./echo-processing-flows.md)、[ChatPort](../packages/core/src/ports/chat.ts)        | [Chat tools](../packages/core/src/agent/runtime-tools/chat.ts)、[Discord adapter](../packages/discord-adapter/src/chat-port.ts)、[vision 入力](../packages/core/src/agent/session.ts)                                                                            | [Chat tool テスト](../packages/core/src/agent/runtime-tools/chat.test.ts)、[Discord テスト](../packages/discord-adapter/src/chat-port.test.ts)、[session テスト](../packages/core/src/agent/session.test.ts)。実配送・モデル別の画像理解は今回の確認対象外       |
| Note を作成・検索・取得・更新・削除する                                | [tool フロー](./echo-processing-flows.md)、[NotePort](../packages/core/src/ports/note.ts)                    | [Note tools](../packages/core/src/agent/runtime-tools/note.ts)、[NoteSystem](../packages/cloudflare-runtime/src/note-system.ts)                                                                                                                                  | [tool テスト](../packages/core/src/agent/runtime-tools/note.test.ts)、[保存テスト](../packages/cloudflare-runtime/src/note-system.test.ts)。Local の保存先と起動設定は未接続                                                                                     |
| Web ページと Zenn 記事を取得する                                       | [Web の制限と予算](./cloudflare-runtime-budget.md)、[WebPort](../packages/core/src/ports/web-page-reader.ts) | [Web tool](../packages/core/src/agent/runtime-tools/web.ts)、[Hosted reader](../apps/cloudflare-workers/src/web/cloudflare-web-page-reader.ts)、[Zenn adapter](../apps/cloudflare-workers/src/zenn/create-zenn-port.ts)                                          | [reader テスト](../apps/cloudflare-workers/src/web/cloudflare-web-page-reader.test.ts)、[Zenn テスト](../apps/cloudflare-workers/src/zenn/create-zenn-port.test.ts)。Local の egress 制約は Cloudflare と同じとは扱えない                                        |
| モデルや実行環境を接続し、指定した生成・継続契約を守る                 | [ModelPort](../packages/core/src/ports/model.ts)、[Native 統合](./native-runtime-integration-readiness.md)   | [Responses adapter](../packages/openai-adapter/src/openai-responses-model.ts)、[Chat Completions adapter](../packages/openai-adapter/src/openai-chat-completions-model.ts)、[Native adapter](../packages/native-inference-adapter/src/native-inference-model.ts) | 各 adapter のテストと Native の過去の実モデル記録。provider 間の出力品質、対応モデル、画像・schema 対応を同等とは扱わない                                                                                                                                        |
| usage を記録し、実行量を制御する                                       | [Main と Cognitive の usage・request 予算](./cloudflare-runtime-budget.md)                                   | [usage 計算](../packages/core/src/echo/usage.ts)、[Hosted Echo](../apps/cloudflare-workers/src/echo/index.tsx)、[request budget](../apps/cloudflare-workers/src/echo/external-request-budget.ts)                                                                 | [usage テスト](../packages/core/src/echo/usage.test.ts)、[失敗 session の usage 保存テスト](../apps/cloudflare-workers/src/echo/index.test.ts)、[budget テスト](../apps/cloudflare-workers/src/echo/external-request-budget.test.ts)。実課金明細との照合は対象外 |
| 思考・行動・失敗を保存し、通知・集計・障害調査に使う                   | [event 仕様](./echo-event-logging.md)                                                                        | [EchoEventPort](../packages/core/src/ports/echo-event.ts)、[event 配送](../apps/cloudflare-workers/src/utils/echo-event.ts)、[archive](../apps/cloudflare-workers/src/echo/event-archive.ts)                                                                     | [配送テスト](../apps/cloudflare-workers/src/utils/echo-event.test.ts)、[archive テスト](../apps/cloudflare-workers/src/echo/event-archive.test.ts)。通知先への実到達と過去集計の移行は別途確認する                                                               |
| Dashboard で個体・記憶・感情・Note・usage・session・行動分析を観察する | [HTTP・Dashboard フロー](./echo-processing-flows.md)、[読み取り予算](./cloudflare-runtime-budget.md)         | [UI](../apps/dashboard/src/App.tsx)、[DTO/schema](../packages/contracts/src/dashboard/schemas.ts)、[Hosted read API](../apps/cloudflare-workers/src/echo/index.tsx)                                                                                              | [schema テスト](../packages/contracts/src/dashboard/schemas.test.ts)、[Hosted route テスト](../apps/cloudflare-workers/src/echo/index.test.ts)。Local 用 API・静的配信は未接続。UI のブラウザ操作検証は今回未実施                                                |
| 設定・secret・アクセス権を分離して運用する                             | [設定と Access](../README.md)、[instance 境界](./echo-registry-boundary.md)                                  | [binding 解決](../apps/cloudflare-workers/src/config/echo-runtime-bindings.ts)、[Access 検証](../apps/cloudflare-workers/src/auth/cloudflare-access.ts)                                                                                                          | [binding テスト](../apps/cloudflare-workers/src/config/echo-runtime-bindings.test.ts)、[Access テスト](../apps/cloudflare-workers/src/auth/cloudflare-access.test.ts)。Local の認証・公開範囲は後続の設計対象                                                    |

現行 tool は [catalogue](../packages/core/src/agent/runtime-tools/catalog.ts)の17件で確認できる。`think_deeply` は現在、成功結果を返す handler であり、追加推論や別 model の呼び出しは行わない。名称から推論能力の追加を推定しない。

## 見直せる設計判断

以下は現行方式の記録と検討点である。新しい方式を採用する提案や、妥当性の結論ではない。振る舞いを変える場合は、前節の能力と次節の比較ケースを使って評価する。

| 設計判断                             | 現在の方式                                                                                                                                             | 見直す際に確かめること                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| module の責務と実行タイミング        | Main の各 turn 前に Memory・Emotion を同じ context から並列実行し、終了時にも実行する                                                                  | この分担・頻度が想起、判断、記銘の品質と遅延・コストに適しているか。順序や module を変える範囲を明示する                            |
| 共有する履歴と専用指示               | Cognitive は共有 context 全体を受け取り、Main persona と tool catalogue は受け取らない。Main 出力は Cognitive context 上で `think` exchange に変換する | それぞれの判断に必要な情報が届くか。短縮・選別によって情報を失わないか。prompt と履歴の変換も比較対象にする                         |
| Main への結果の渡し方                | 確定済みの想起・感情を system-owned tool exchange として渡す。Native は入力専用の `origin` を扱う                                                      | 結果の由来・確定状態を保ち、通常の未解決 tool call と混同しないか。provider との継続契約も確認する                                  |
| 記憶の表現・検索・保持               | semantic / episode、感情付き保存、embedding・vector 候補・rerank。Hosted は最大500件、vector 候補20件、最終5件                                         | 過去の必要な経験が検索できるか。上限・順位・保持方式の変更で検索対象や既存データの意味を失わないか                                  |
| session 間に戻す domain 状態         | 前 `post_main` の Memory 1件と Emotion を Cognitive の初期 context に戻す                                                                              | 何を継続情報とするのが適切か。終了時の要約と、保存済み Memory 全体の検索を混同しないか                                              |
| 感情の表現と作用                     | valence・arousal・labels を生成し、handoff と Memory 保存に使う                                                                                        | 表現が必要な状態を捉えているか。値を渡すことと、判断へ有効に作用することを分けて検証する                                            |
| 失敗・再試行・継続判断               | 一時失敗を同一入力から1回再試行し、phase が確定できなければ Main を先へ進めない                                                                        | 不正な状態で行動しない性質と、利用可能性の両方を保てるか。既に起きた外部作用や usage の扱いを定義する                               |
| 起動・終了・資源配分                 | Hosted alarm、未読・token limit・next wake による判定、Main 最大10 turn、共有 request hard gate                                                        | 無入力時の自発的思考と呼びかけへの反応を維持するか。予算内の完走・停止・復帰を確認する                                              |
| domain と推論状態の保存境界          | Cognitive phase の domain commit と、Main の session 境界 snapshot は別々。Memory・Emotion の生成状態は次回へ持ち越さない                              | 再起動時にどの確定状態から再開するか。domain と KV/GDN の保存失敗を区別し、利用できる状態を明確にする                               |
| Native 内部の lifecycle とキャッシュ | 補助 lane の `initial` / `reset` 指定、session 内入力キャッシュ、解放完了待ち                                                                          | [保留中の判断](./native-runtime-integration-readiness.md)にある状態指定と解放待機の必要性。失敗の報告先・寿命・順序を含めて判断する |

Cloudflare 固有型を Core へ持ち込まないこと、contracts に runtime 実装を置かないこと、bounded read を使うことは現在の [AGENTS.md](../AGENTS.md)の制約である。provider・storage の交換性を考える際にも遵守する。

## 現状の接続と未確認の範囲

Hosted は起動判定、ThinkingEngine、tool、Cognitive domain 保存、event archive、Dashboard API を束ねている。Local は3 module の Core 接続と Native process / state lifecycle が実装され、domain・tool・retry policy を呼び出し側から注入する。[Native 統合仕様](./native-runtime-integration-readiness.md)では、Local の永続保存・検索・通常起動、常駐 scheduler、外部送信、Dashboard 接続を残る要件としている。

手元には初回実会話と SQLite domain 保存・別プロセス復元の補助記録もある。ただし通常起動への接続、実検索 backend、復元後の次の会話での Memory 利用は受け入れ済みと扱えない。保存物を読み戻す検証と、読み戻した経験で次の判断ができる検証は分ける。

既存の [workflow 評価](../packages/model-evaluation/README.md)は `runAgentSession` を直接呼び、Cognitive coordinator を接続しない。tool と保存先も fixture である。シナリオや scorer は再利用候補だが、通過を現行ループ全体の受け入れに読み替えない。

過去の実モデル検証は次のローカル補助記録を参照した。Git 管理外のため、共有先で参照できる証跡にする場合は、対象コード・モデル・コマンド・結果・fixture の範囲を別途収録する。

- `.artifacts/model-evaluation/2026-09-22-native-cognitive-final/report.md`: Qwen3.6-35B-A3B-MLX-4bit での Cognitive、入力キャッシュ、KV/GDN、schema、Main snapshot 検証。domain・課題・tool は fixture の範囲がある。
- `.artifacts/model-evaluation/2026-09-22-local-conversation/report.md`: 実 prompt による初回会話と SQLite / snapshot の別プロセス復元。記憶検索 backend と再起動後の次の会話は未確認と記されている。試行コードが現在の通常入口に存在する証拠にはしない。

## 比較の基準にする動作の候補

以下は検証計画の草案であり、今回の作業で受け入れ済みとは扱わない。出力文の完全一致を要求する代わりに、観測・参照・行動・保存の trace と結果を確認する。実モデルの判断品質は複数条件・試行で比較し、一度の成功から安定性を推定しない。

| ケース                             | 維持・比較したい結果                                                                                | 現在の根拠と次に必要な確認                                                                                            |
| ---------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 呼びかけへの反応と自発的な起動     | 無入力でも条件に応じて起動でき、呼びかけを観測できる。送信の要否・終了・次回起動を選べる            | Hosted の判定テストは存在する。自発的な判断品質と Local scheduler は別途確認する                                      |
| 新しい経験の保存と再起動後の想起   | 初回に得た情報を保存し、別プロセスの次の session で実検索し、その結果を Main が判断に使う           | snapshot と domain 読み戻しに過去記録がある。実検索・通常入口・次回 Main 利用をつなぐ確認が必要                       |
| 過去の予定を現在の変更で更新する   | 古い予定が Memory にあっても、現在の取消・変更を使い、保存状態と報告が矛盾しない                    | 既存 workflow fixture を実 ThinkingEngine / Cognitive 経路へ接続する。後続 session も確認する                         |
| 同時に複数個体を動かす             | 一方の入力・保存・cancel が、他方の domain・推論状態・送信先を変えない                              | Local lifecycle と Native の lane 分離の根拠がある。domain と外部接続を含む検証が必要                                 |
| module・tool・保存が途中で失敗する | 成功済み処理を不用意に再実行せず、未確定状態を区別し、usage と失敗原因を残して復帰できる            | Core / Hosted / Local の失敗テストを利用する。外部送信後の応答喪失、domain と snapshot の片方だけの失敗を追加評価する |
| 運用上限に達し、再開する           | request・turn・storage の制約内で停止し、観察情報を残し、再開時にも保存・検索・過去集計の意味を保つ | budget、usage、archive のテストを利用する。移行時のデータ範囲と集計の連続性は別途比較する                             |

Note、画像、Web/Zenn、Dashboard、Access の能力は各テスト・API contract も併用して比較する。中心ループの検証だけで、これらの機能維持まで証明したとは扱わない。

## 最初のレビューで決めること

1. 維持したい能力の候補に抜けや過剰な約束がないか。自発的思考、経験の継続、感情の作用をどのように評価したいか。
2. 思考ループの設計判断のうち、先に改善したいものと、比較のため当面維持するものはどれか。
3. 比較基準を先に接続する場所をどう選ぶか。現行機能の比較には Hosted、Local の経験継続の受け入れには保存・検索・通常入口の接続が必要になる。

レビュー後、必要な比較検証を定め、その結果と変更範囲から構造の候補を選ぶ。大きいファイルや module 数だけで置き換え順を決めない。構造整理とループの振る舞いの変更は、効果を個別に確認できる単位に分ける。

## この棚卸しで実施した確認

現行文書、主な入口・port・保存・tool・API 実装、関連テストの契約を読み取った。CI で指定された pnpm 10.12.2 と Node 24.21.0 で `pnpm test:run` を実行し、825件が成功、実モデルなどを使う opt-in の10件がスキップされた。Hosted の250件はログ出力先の権限エラーを解消して再実行し、成功した。残りの package にも runner の error / warning はなかった。

初回確認では実行環境の pnpm が既存の設定を扱えず、テスト開始前に停止した。CI と同じ pnpm 10.12.2 で依存関係を復旧し、ルート `package.json` の `packageManager` に指定した。品質フックも、PATH 上の pnpm の自動切り替えへ依存せず、`npm exec` で指定版を起動するよう修正した。フックは非対話で動作し、npm cache と Wrangler ログの既定出力先には一時ディレクトリを使う。lint・型・書式・テストと判定条件は維持する。当時は自動追加された `allowBuilds` を取り除き、workspace 設定・lockfile・Git index の変更は残さなかった。

通常テストは実モデル用環境変数と live gate を無効にして実行した。Rust / Metal、実モデル、外部 API、本番 endpoint、UI のブラウザ操作は初回確認では再検証していない。仕様の現状確認と通常テストの成功を、未完成の Local 接続や判断品質の受け入れへ広げない。

2026-10-08 に pnpm 12.6.0 へ移行した。`packageManager` と CI の指定を揃え、ビルド許可の5件を `pnpm-workspace.yaml` の `allowBuilds` へ移した。Node 24.21.0 は Volta / CI で供給し、`devEngines.runtime` の `onFail: error` で不一致を検出する。制限付き環境で Node の自動取得に伴う領域外への書き込みを必要としない。lockfile は pnpm 本体の固定情報を追加し、アプリの依存関係部分は移行前と完全に一致する。新規ディレクトリへの `pnpm install --frozen-lockfile`、lint・型・書式、通常テスト825件（opt-in の10件はスキップ）、決定的評価14件、Dashboard ビルド、Stop 品質フックが成功した。Stop フック経由で Native の書式と状態テストも成功した。Dashboard ビルドには既存の Zod のコメントを Rollup が注釈と解釈する警告が2件残る。実モデル / Metal や外部サービスの再検証、GitHub 上での CI 実行は行っていない。

pnpm 移行後はログインシェルの有無にかかわらず、PATH 上の pnpm が警告なく12.6.0を返すことを確認した。品質フックから `npm exec` による起動と npm cache の指定を取り除き、`CI=true` と一時ディレクトリへの `WRANGLER_LOG_PATH` 指定だけを残した。ログ出力先の指定がない元のフックでは、チェックの終了コードが0でも Wrangler のログ書き込みに `EPERM` が出たため、この指定は維持する。

変更した Markdown の Prettier 書式、相対リンクの参照先、差分の空白エラーを確認した。runtime のコードは変更していない。
