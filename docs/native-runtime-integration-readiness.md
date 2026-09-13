# Native runtime と Cognitive Module の統合

Native backend で E.C.H.O. の思考 session を実行するための接続範囲と、未実装の要件を定義する。Cognitive Module の責務・保存契約は[Cognitive Module 設計](./cognitive-module-architecture.md)、KV/GDN・プロトコル・snapshot の仕様は[Native architecture](../native/echo-inference/docs/architecture.md)を参照する。

## 接続範囲

Native 推論基盤は、モデルの常駐実行、TypeScript adapter、KV/GDN の継続、キャンセル時の rollback、Main state の保存・復旧を提供する。確定済みの Cognitive 結果を Main の継続入力へ渡す経路も実装されている。

Memory/Emotion 自身の Native 生成から domain 保存までを含む、アプリケーション全体の実行は未接続である。推論 state の snapshot はモデル内部の状態を保存するもので、Memory・Emotion・Note 等の domain 保存は別途必要になる。

| 境界                                             | 実装状況                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Main への Cognitive handoff                      | Core formatter、Native adapter、protocol v11、Rust renderer が対応                 |
| ThinkingEngine と Cognitive coordinator の処理順 | モデル応答・domain 保存先・transport を fixture にした結合テストで検証             |
| Cognitive request の指定                         | `responseFormat`、`maxOutputTokens`、`signal` の Native 実行への反映が未実装       |
| Memory/Emotion の推論 state                      | 独立した ephemeral lane は実装済み。phase・再試行・次 activation との対応が未接続  |
| ローカルアプリケーション                         | provider を注入できる composition、domain 保存、実行入口、全体の実モデル評価が必要 |

## Main の継続入力

[session](../packages/core/src/agent/session.ts)は Main の tool 実行結果と Cognitive coordinator の追加入力を結合し、前の `responseToken` を付けて次の生成を要求する。[handoff formatter](../packages/core/src/agent/cognitive-module-handoff.ts)は、domain commit 済みの `search_memory` と `update_emotion` を tool call/result の組にする。

Native は次の順序で継続入力を受理する。

1. Main が生成した未解決 call に対する tool result を、件数・ID・順序まで照合する。
2. その後に、`origin: "runtime"` を持つ確定済み call/result の組を受け入れる。
3. 確定済みの EOS 境界に Qwen template の suffix を追加し、既存 KV/GDN から推論を続ける。

Main に未解決 call がなければ、空の再試行または確定済み runtime exchange を受理する。新しい通常 message は `new_session` を必要とする。adapter と Rust renderer は、由来のない call、未解決・不一致の組、suffix 内の重複 ID を拒否する。

`origin` は runtime が付ける入力専用属性で、モデル出力から引き継がない。これは確定済みの履歴を識別するためのもので、`update_emotion` を Main の実行可能 tool として登録するものではない。Native adapter と engine は protocol version 11 で揃える。

## 検証方法と確認範囲

| 検証                                                                                             | 確認する契約                                                                                                                                           | 実モデル・fixture の範囲                                                                                          |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [Cognitive 結合テスト](../packages/native-inference-adapter/src/cognitive-continuation.test.ts)  | `pre_main → Main → pre_main → Main finish → post_main` の順序、tool/文章出力後の継続、domain commit 前の Main 停止、失敗時の usage 保持、prompt の分離 | ThinkingEngine・coordinator・runner・Native adapter/client は実装を使用。モデル応答・保存先・transport は fixture |
| [公式 template の比較](../native/echo-inference/oracles/qwen35_cognitive_continuation_parity.py) | 既存 prefix と追加 suffix が、公式 Qwen template の全体入力と byte/token 単位で一致する                                                                | tool/文章での完了と Cognitive exchange の有無を組み合わせた fixture。GPU 生成は行わない                           |
| [実モデル probe](../packages/native-inference-adapter/src/real-model-probe.ts)                   | 2 session・各2生成で、tool call の解析、結果を含む回答、resident prefix の再利用、state 長の更新を確認する                                             | Main は実モデル。`cognitive` モードの Memory/Emotion 結果と domain 状態は fixture                                 |

実モデル probe は各 session の最初を含むすべての Main turn で handoff を渡す。初回の Memory recall は空で、継続時には観測した lookup 結果を含める。`plain` と `cognitive` の両入力について、greedy と production sampling を検証対象とする。

通常の Node テスト、Rust テスト、実モデル probe は実行条件が異なる。モデルと Metal を必要とする検証は明示的に実行する。環境設定とコマンドは[Native README](../native/echo-inference/README.md#build)を参照する。

## 残る実装要件

### 1. Cognitive request の指定を実行へ反映する

[ModelCognitiveModuleRunner](../packages/core/src/agent/model-cognitive-module.ts)は `responseFormat`、`maxOutputTokens`、`signal` を `ModelPort.generate()` へ渡す。[Native adapter](../packages/native-inference-adapter/src/native-inference-model.ts)はこれらを wire 実行へ反映しておらず、生成上限には model 作成時の設定を使う。

Core は返答後に JSON/schema を検証する。Native 側にも生成・結果検証、request ごとの上限、開始前と実行中の abort を接続し、未対応の指定は明示的に拒否する必要がある。完了条件には、cancel と完了の競合、rollback、再試行、使用済み usage の保持を含める。

### 2. Auxiliary state と phase の寿命を対応させる

Core runner は各 phase で共有 context 全体を渡し、`previousResponseToken` を付けない。既存 state がある Native では `new_session` となり、既定で GDN を保持して KV を初期化する。一方、[local runtime](../apps/local-runtime/src/local-native-inference-runtime.ts)は Memory/Emotion の ephemeral lane をプロセス内で保持する。

この組み合わせでは共有 context の再入力と過去 GDN の持ち越しが重なる。phase ごとに初期化するか、確定 state から差分を入力するかを定義する必要がある。再試行、成功済み sibling、次 activation の開始状態をそれぞれ定め、Main の durable lane との独立性を維持する。

### 3. Provider と domain を注入できる composition を作る

[Cognitive factory](../apps/cloudflare-workers/src/echo/cognitive-modules.ts)の prompt 組み立てと module 設定を、model/domain の実装を注入できる形にする。共通の handoff formatter は Core が所有し、Hosted の SDK・環境設定・retry policy は Worker 側が所有する。

### 4. 実際の Cognitive 経路で workflow を評価する

[既存 workflow harness](../packages/model-evaluation/src/qwen36-eat-readiness/runtime-workflow-harness.ts)は `runAgentSession` を直接呼び、Cognitive coordinator を接続せず、評価専用の `session_record` 付き終了契約を使う。各 turn の Cognitive exchange を前提とする[Main prompt](../packages/core/src/llm/prompts/rin.ts)とは入力・終了契約が異なる。

実際の ThinkingEngine/Cognitive 経路へ評価を接続し、予定変更、緊急通知、一時的な tool 失敗を再検証する。既存の予定変更ケースには、初期予定の未保存、中止を「完了」と答える応答、未登録の `update_emotion` 呼び出し、`max_turns` 終了が観測されている。モデル単体と評価経路の寄与は未確定で、全 workflow の受け入れは未完了である。

長文 prefill の評価では、新規入力が8,192 tokens 以上となる fixture を用意する。chunked prefill は BF16 の演算順序が変わるため、単一実行との state の bit 一致を前提にしない。比較条件は[long-input prefill](../native/echo-inference/README.md#long-input-prefill)に従う。

### 5. ローカル保存・起動・再起動を接続する

Memory/Emotion の domain 保存には version、idempotency、一括 commit が必要になる。Memory 検索には local SQL と embedding/reranking の境界を接続し、Note 等には既存の storage interface を利用する。

1インスタンスの受け入れ検証では、専用 state root を使って次の一連の動作を確認する。

1. 起動し、Cognitive、Main、tool 実行、`post_main`、domain 保存を完了する。
2. Main checkpoint を保存して正常終了する。
3. 別プロセスで再起動し、保存済みの推論 state と domain 状態から次の session を開始する。
4. 保存の重複、phase 失敗、checkpoint 失敗、終了中の cancel でも整合性を保つ。

常駐 scheduler、外部送信、Dashboard 接続、本番 backend の切り替えは、このローカル実行を受け入れた後の運用要件とする。

## 実装・検証時の参照先

- [Native architecture](../native/echo-inference/docs/architecture.md): lane の所有権、状態遷移、保存・復旧の契約。
- [Native README](../native/echo-inference/README.md): MLX 環境、build、probe コマンド。
- [model-evaluation README](../packages/model-evaluation/README.md): 評価シナリオ、実行条件、結果の保存方針。
- [Cloudflare runtime budget](./cloudflare-runtime-budget.md): Hosted composition を変更する際の request・storage 予算。`origin` の伝達自体は既存入力への属性追加で、API/DO request、rows read/written、外部 API call を追加しない。
