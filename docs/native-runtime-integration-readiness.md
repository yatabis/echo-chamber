# Native runtime 統合の現状と実装順序

確認日: 2026-09-12。実装の確認対象は `676b3397994fa68993a0aeff9c05df7cb33786c3`。以下は、この時点の検証結果と、現行Cognitive ModuleをNativeへ接続するための残作業である。

## 現在の到達点

Native推論基盤へのmain取り込みと、実推論で見つかった継続ツール引数の型保持修正は完了した。通常の品質チェックと、継続推論・復旧・連続batch・20分負荷などの実モデル検証を通過している。一方、実モデルのworkflow gateは未合格で、現在のCognitive Moduleを使ったE.C.H.O.全体のNative実行も未接続である。

[PR #1](https://github.com/yatabis/echo-chamber/pull/1)の範囲は、Rust/MLX推論エンジン、TypeScript adapter、ローカルプロセスと推論stateの管理、およびその検証まで。次の実装は、実際の`ThinkingEngine`とCognitive coordinatorをNativeへ接続する結合テストから始める。ローカルdomain保存・運用入口・本番のNative切り替えは、その接続に続く作業である。

このPRのマージだけでCloudflare Workersの本番推論経路は切り替わらない。推論stateの保存と、Memory/Emotion等のdomain保存は別の境界として扱う。

## 完了した変更と確認した版

| 版                                                                                                                                    | 内容                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `b8729cb7f83c64e29e4613e2ce3f1ca545f184f7`                                                                                            | 取り込んだmain                                                                |
| [`d75b1f8c44a41522945b385f6c1a77bdad5350ab`](https://github.com/yatabis/echo-chamber/commit/d75b1f8c44a41522945b385f6c1a77bdad5350ab) | Nativeへmainをマージ。lockfile、lint/format、型、現在のcore契約への追随を完了 |
| [`676b3397994fa68993a0aeff9c05df7cb33786c3`](https://github.com/yatabis/echo-chamber/commit/676b3397994fa68993a0aeff9c05df7cb33786c3) | 継続推論で確定済みtool schemaを保持する修正と回帰テスト3件                    |

最初の仮マージ調査で確認したinstall・lint・format・既存テストの失敗は、main取り込み時に解消した。以降の検証結果は、その修復を反映した版のものである。

型保持の不具合は、継続要求の`tools: []`をそのまま出力解析に使い、例えば`{"step":2}`を`{"step":"2"}`として返すものだった。[Rust server](../native/echo-inference/crates/echo-inference/src/local_server.rs)でinstanceごとの確定済みcatalogを保持し、推論stateのcommit成功時だけ更新する。失敗・キャンセル時には旧定義を保持し、ツールなしの新sessionでは消去する。数値計算kernel、weights、KV/GDN保存形式、wire request形式はこの修正で変更していない。

## 検証結果

### 型保持修正後の通常チェック

| 検証                                                               | 結果                                        |
| ------------------------------------------------------------------ | ------------------------------------------- |
| `pnpm check`、`pnpm lint:check`、`pnpm format:check`               | 合格。全対象のtypecheckを含む               |
| `pnpm test:run`                                                    | 767件成功、通常設定では実モデルgate 8件skip |
| `pnpm eval:check`                                                  | 31件成功                                    |
| Rust release build、`cargo fmt --all -- --check`                   | 合格                                        |
| Rust workspace全体のall-targets/all-features Clippy、warnings deny | 合格                                        |
| Rust workspace全体のall-features tests                             | 85件成功、0件skip                           |

`676b339`の[GitHub CI](https://github.com/yatabis/echo-chamber/actions/runs/34686558004)も成功した。CIのNode側の成功と、別途実機で行ったRust/Metal検証は区別する。通常テストでskipする実モデルgateは個別に有効化して実行したが、以下のworkflow gateは未合格である。

### 実モデル検証

同一のローカル`Qwen3.6-35B-A3B-MLX-4bit`（4 shards）、MLX 0.32.0と対応するmlx-c、Metal、Mac16,11・64 GiB・macOS 26.6.2を使用。GPU負荷は直列に実行し、モデルのdownload、外部モデルAPI、本番Cloudflare APIは使用していない。

| ケース                                                      | 結果                                               | 検証した版 |
| ----------------------------------------------------------- | -------------------------------------------------- | ---------- |
| 実モデル・production sampling、出力上限・強制EOS            | 合格                                               | `676b339`  |
| snapshot復旧、local runtimeの起動・終了・別プロセスでの再開 | 合格                                               | `676b339`  |
| continuous batch、late join、cancel/rollback                | 合格                                               | `676b339`  |
| Main/Memory/Emotionの合成lane負荷                           | 合格。現行Cognitive経路の統合検証とは別            | `676b339`  |
| 8回の継続推論・全履歴replay比較                             | 整数引数を維持し、最終生成結果も一致               | `676b339`  |
| Rapid-MLXとの短文比較・長期session比較                      | 測定7回・3回で比較対象の最終出力が一致             | `676b339`  |
| 標準20分の連続負荷                                          | 合格。実測1,201.188秒、711回、88,324 tokens        | `676b339`  |
| 思考workflow                                                | **未合格**。3件中2件が全行動条件に合格             | `676b339`  |
| greedy実モデルprobe、16k・6 stateのsoak、stateful速度検証   | 合格                                               | `d75b1f8`  |
| chunked prefill、2k/8k/16k/32k context                      | 既存testの必須条件に合格。下記の一致範囲に制限あり | `d75b1f8`  |

型保持修正後は、型解析・継続・session切替・復旧・キャンセルに関係する検証を再実行した。`d75b1f8`で通った数値計算・長文context検証をすべて修正後に再実行したとは扱わない。

20分負荷の生成速度中央値は90.47 tokens/秒、最初と最後の5分の中央値は90.03 / 93.51 tokens/秒だった。active GPUメモリの始終差は0 byte、allocator cacheの増分は3,674,112 bytes。この条件で速度低下と継続的なactive memory増加を検出しなかったという結果であり、全運用条件の保証ではない。

Rapid-MLXの長期session比較は論理履歴を揃えたが、adapterによる実入力は139 tokens異なり、checkpoint方針も異なる。同一token列・同一cache条件での速度比較としては扱わない。

### 未合格のworkflowと検証の限界

- `queued_priority_after_session_boundary`と`transient_note_update_failure`は、全行動条件に合格した。
- `state_revision_across_cold_start`には、初期予定の未保存、中止を「完了」と答える誤り、未登録`update_emotion`呼び出し、最初のsessionの`max_turns`終了が残った。型保持修正で全workflowが合格したわけではない。
- workflow gateの`adaptivePrefillObservedAtOrAboveEightKiTokens`もfalse。観測した各workflowの最大入力は6,245 / 6,335 / 6,248 tokensで、8,192以上の該当requestがなかった。このworkflow内での検証範囲の不足であり、長文prefill自体は別のcontext/chunked検証で扱った。
- chunked prefillの合格範囲は、初回生成のtoken/output一致、入力実行回数、stateサイズ・token計数など、既存testの必須条件。GDNを持ち越すnew sessionでは24 token目で生成が分岐し、stateのbit一致もない。既存testが観測のみとする項目であり、完全なbit一致は保証しない。

現在の[Main prompt](../packages/core/src/llm/prompts/rin.ts)は、各turnでMemory/Emotion由来のsystem-owned exchangeを受け取る前提である。一方、[workflow harness](../packages/model-evaluation/src/qwen36-eat-readiness/runtime-workflow-harness.ts)は`runAgentSession`を直接呼び、Cognitive coordinatorを接続せず、評価専用の`session_record`付き終了契約を使う。現在のMain契約と評価経路の差が残存失敗へ与える寄与は未検証であり、原因をモデル単体や数値engineに断定しない。

## 現行Cognitive Moduleとの接続障害

以下は初回の仮マージ調査で再現し、`676b339`でも該当コードの契約が未変更であることを確認した事項。実モデルでの行動評価とは別の、呼び出し境界の問題である。

### Mainの2回目の入力をNativeが拒否する

現在の[session](../packages/core/src/agent/session.ts)は、Mainのtool実行結果とCognitive coordinatorの追加入力を結合し、前の`responseToken`を付けて次の生成を要求する。[Cognitive factory](../apps/cloudflare-workers/src/echo/cognitive-modules.ts)の追加入力は、確定した`search_memory`と`update_emotion`のtool call/resultである。

[Native adapter](../packages/native-inference-adapter/src/native-inference-model.ts)の`validateContinuationInput`は、直前のpending callと順序・件数・IDが一致するtool resultだけを受け入れる。pending callがない場合は空の再試行だけを許可する。[Rust chat encoder](../native/echo-inference/crates/echo-inference/src/chat.rs)もcontinuationをtool-result suffixに制限している。

初回診断では、実際の`ThinkingEngine`とHosted coordinatorに、合成domain/module応答、Native adapter/client、メモリ内transportを接続し、次を観測した。GPU生成や本番domain保存は行っていない。

```text
Cognitive request: 4回
確定したphase: pre_mainが2回
Native生成command: 1回
2回目のMain入力: tool_result, tool_call, tool_result, tool_call, tool_result
失敗: native continuation accepts only results for the pending tool calls
```

必要なのは、MainのKV/GDNとsessionの連続性を維持したまま、正規のpending tool resultsとruntimeが確定したexchangeを順序どおり追加できる契約である。TypeScriptの検査だけを緩めず、wire schema、Rust encoding、公式Qwen templateに対するsuffixのtoken一致、EOS境界を一緒に検証する。wireの意味を変える場合はprotocol versionの更新も扱う。

### Cognitive requestの指定がNativeへ届かない

[ModelCognitiveModuleRunner](../packages/core/src/agent/model-cognitive-module.ts)は`responseFormat`、`maxOutputTokens`、`signal`を`ModelPort.generate()`へ渡すが、Nativeの`executeRequest` / `prepareCommand`はこれらをwire実行へ反映していない。

| 初回診断の入力                        | 観測                                 |
| ------------------------------------- | ------------------------------------ |
| JSON Schema `cognitive_memory_recall` | wireへschema/nameが渡らない          |
| `maxOutputTokens: 7`、モデル側上限128 | wireの`max_new_tokens`は128          |
| 呼び出し前にabort済みの`signal`       | 生成commandが1件送られ、通常解決する |

coreは返答後にJSON/schemaを検証するため、不正JSONがそのままdomainへcommitされるという指摘ではない。Nativeの生成・結果検証、requestごとの上限、開始前と実行中のabortを接続し、未対応の指定は明示的に拒否する必要がある。cancelと完了の競合、rollback、失敗時のretryとusageも検証する。

### Auxiliary laneの寿命とphase入力を接続する

core runnerは各phaseで共有context全体を渡し、`previousResponseToken`を付けない。既存stateがあるNativeでは`new_session`となり、既定でGDNを保持してKVを初期化する。一方、[local runtime](../apps/local-runtime/src/local-native-inference-runtime.ts)はMemory/Emotionのephemeral laneをプロセス起動時に一度だけ開く。

単にmodelを差し替えると、共有context全体の再入力と過去GDNの持ち越しが重なる。これはコードから確認した状態遷移であり、生成品質への悪影響を測定した結果ではない。phaseごとに初期化するか、確定stateからdeltaを入力するかを決め、同一phaseの再試行・成功済みsiblingの保持・次activationの開始状態を定義する。Mainのdurable laneとの独立性は維持する。

## 次の実装と受け入れ条件

### 1. 実際のCognitive経路を通す結合テスト

`ThinkingEngine`、`ParallelCognitiveModuleOrchestrator`、`NativeInferenceModel`を組み合わせる。モデル生成結果・domain保存先・transportのみを決定的なfixtureへ差し替え、実際の継続入力検査を通す。次の経路と条件を先にテストへ固定する。

1. Memory/Emotionの`pre_main`を実行し、両結果をdomainへcommitする。
2. Mainの初回生成が通常のtool callを返す。
3. tool resultを解決し、次の`pre_main`を実行・commitする。
4. pending tool resultsと確定済みsystem-owned exchangeを所定順序で渡し、Mainの2回目の生成を実行する。
5. Mainが`finish_thinking`を返し、`post_main`の保存まで完了する。

未解決・重複・順序違いのMain tool resultは引き続き拒否し、任意のassistant tool callを無条件に許可しない。system-ownedの`update_emotion`をMainが実行できるtoolとして登録しない。moduleまたはdomain commitが失敗した場合はMainの次の生成へ進まず、成功済みusageと失敗原因を保持する。Main専用promptとmoduleの共有contextも分離する。

### 2. 呼び出し契約と評価経路を整合させる

| 順序              | 対象と完了条件                                                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 継続入力          | adapter・protocol・Rust rendererを修正し、上記の結合テスト、pending call照合、token prefix、cancel/rollbackを通す。毎turnのnew_sessionや全履歴再投入で代用しない                 |
| Module request    | schema・出力上限・abortを実行へ反映し、不正な出力やcommit失敗でMainを進めない                                                                                                    |
| Auxiliary state   | phase再試行と次activationが、それぞれ定義した状態から開始し、共有contextを重複蓄積しない                                                                                         |
| Cognitive factory | Workerのprompt・handoffのprovider非依存部分をmodel/domain注入可能にする。HostedのSDK・env設定・retry policyをcoreへ持ち込まない                                                  |
| Workflow評価      | 実際のThinkingEngine/Cognitive経路に接続し、予定変更・緊急通知・一時失敗の3件を再検証する。中止を完了と答える失敗の期待値を緩めず、8k以上の観測条件は明示的な長文fixtureで満たす |

この結合テストの成功と、実モデルのworkflow全行動条件の成功は別々に確認する。Hostedの既存テストとNativeの継続・rollback検証も維持する。

### 3. 1インスタンスのlocal sessionと再起動

`apps/local-runtime`へ実行入口とdomain保存を接続する。Memory/Emotionの保存にはversion・idempotency・一括commit、Memory検索にはlocal SQLとembedding/rerankingの境界が必要で、保存だけを実装して検索同等性を満たしたとは扱わない。Note等は既存の小さいstorage interfaceを再利用できるか確認する。

専用state rootとテキストの合成入力で、起動、Cognitive、Main、tool実行、`post_main`、domain保存、Main checkpoint、正常終了、別プロセス再起動、次sessionへの引き継ぎまで検証する。保存の重複、phase失敗、checkpoint失敗、終了中のcancelも含める。常駐scheduler、外部送信、Dashboard移行、本番切り替えは別の運用判断とする。

## 運用境界と再現資料

- 推論stateとlaneの仕様、MLX環境・build・probeコマンドは[Native README](../native/echo-inference/README.md)、評価設定は[model-evaluation README](../packages/model-evaluation/README.md)を参照する。
- 現行Cognitiveの保存・失敗契約は[Cognitive Module architecture](./cognitive-module-architecture.md)を使う。既存`current.json`は自動移行せず拒否する[Native保存契約](../native/echo-inference/README.md#durable-state)を維持し、移行時は対象stateを明示して判断する。
- 2026-09-12の調査では、main由来の既存事項としてDashboard buildのzod注釈warning 2件と、`@cloudflare/vitest-plugin > miniflare > sharp@0.35.2`のHigh `GHSA-rgj7-g3m4-5g8c`を確認した。今回の型保持修正で解消したとは扱わない。
- この文書更新によるCloudflare request、DO request、rows read/written、外部API callの実行時増分は0。後続のHosted factory抽出も[Cloudflare runtime budget](./cloudflare-runtime-budget.md)と照合する。

実行ログ・JSON・provenanceは、ローカルのGit管理外`.artifacts/model-evaluation/2026-09-12-native-live/`に保存した。初回接続診断は`.artifacts/model-evaluation/2026-09-12-native-integration/`、この更新前の調査文書は`.artifacts/model-evaluation/2026-09-12-native-docs/`に保持する。これらはリポジトリへ同梱されないため、レビューで必要な到達点・未合格項目・次の受け入れ条件は本文に記載した。
