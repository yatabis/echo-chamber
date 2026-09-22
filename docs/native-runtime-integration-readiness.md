# Native runtime と Cognitive Module の統合

Native backend で E.C.H.O. の思考 session を実行するための接続範囲と、未実装の要件を定義する。Cognitive Module の責務・保存契約は[Cognitive Module 設計](./cognitive-module-architecture.md)、KV/GDN・プロトコル・snapshot の仕様は[Native architecture](../native/echo-inference/docs/architecture.md)を参照する。

## 接続範囲

Native 推論基盤は、モデルの常駐実行、TypeScript adapter、KV/GDN の継続、キャンセル時の rollback、Main state の保存・復旧を提供する。確定済みの Cognitive 結果を Main の継続入力へ渡す経路も実装されている。

LocalNativeInferenceRuntime.think() は、Native の Main・Memory・Emotion を既存の ThinkingEngine/Cognitive coordinator に接続する。保存先と tool は呼び出し側が注入する。ローカルの永続保存実装とアプリケーション起動設定は未接続であり、推論 state の snapshot とは別に Memory・Emotion・Note 等の domain 保存が必要になる。

| 境界                                             | 実装状況                                                                                              |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Main への Cognitive handoff                      | Core formatter、Native adapter、protocol v15、Rust renderer が対応                                    |
| ThinkingEngine と Cognitive coordinator の処理順 | モデル応答・domain 保存先・transport を fixture にした結合テストで検証                                |
| Cognitive request の指定                         | `responseFormat` の生成時のスキーマ制約・最終出力の契約検証、request ごとの出力上限、abort を実装済み |
| Memory/Emotion の推論 state                      | 前回の生成状態は引き継がず、同一 session の入力 checkpoint を条件付きで再利用                         |
| ローカルアプリケーション                         | Native composition と Core の実行入口は実装済み。永続保存・起動設定・実運用 workflow の評価が必要     |

## Main の継続入力

[session](../packages/core/src/agent/session.ts)は Main の tool 実行結果と Cognitive coordinator の追加入力を結合し、前の `responseToken` を付けて次の生成を要求する。[handoff formatter](../packages/core/src/agent/cognitive-module-handoff.ts)は、domain commit 済みの `search_memory` と `update_emotion` を tool call/result の組にする。

Native は次の順序で継続入力を受理する。

1. Main が生成した未解決 call に対する tool result を、件数・ID・順序まで照合する。
2. その後に、`origin: "runtime"` を持つ確定済み call/result の組を受け入れる。
3. 確定済みの EOS 境界に Qwen template の suffix を追加し、既存 KV/GDN から推論を続ける。

Main に未解決 call がなければ、空の再試行または確定済み runtime exchange を受理する。新しい通常 message は `new_session` を必要とする。adapter と Rust renderer は、由来のない call、未解決・不一致の組、suffix 内の重複 ID を拒否する。

`origin` は runtime が付ける入力専用属性で、モデル出力から引き継がない。これは確定済みの履歴を識別するためのもので、`update_emotion` を Main の実行可能 tool として登録するものではない。Native adapter と engine は protocol version 15 で揃える。

## Cognitive request の実行契約

`responseFormat` は prompt と別の実行パラメーターとしてエンジンへ渡す。Rust 側が推論開始前にスキーマを文法へコンパイルし、各 token の選択前に不正な継続候補を除外する。EOS は JSON 値が完成した時点でのみ選択できる。Core の prompt を中間層が追記・変更することはない。

対応範囲は [Native README](../native/echo-inference/README.md#request-controls) に定義した JSON Schema draft-7 の部分集合とする。未対応の条件やコンパイラー警告は推論開始前に拒否し、近似や prompt 指示へフォールバックしない。文法の状態は生成 request ごとに独立し、通常生成・バッチ生成・キャンセル後の再試行で同じ契約を守る。継続 request でも出力形式を個別に指定できる。生成後の schema 検証はエンジン・通信経路の不変条件を確認するために残す。

`maxOutputTokens` は request ごとに指定できる。正の安全な整数を要求し、model 作成時の上限を超える値は拒否する。省略時はその上限を使い、model の上限も resident owner の上限以内に収める。

`signal` が開始前に abort 済みなら生成を送らず、実行中なら generate の後に cancel を送る。adapter は terminal event まで lane を占有する。`cancelled` なら直前の KV/GDN と継続情報を維持し、`completed` が競合に勝った場合は確定済みの応答を受理する。cancel の送信失敗で engine の状態が不明になった client は、以降の要求を拒否する。

長さ制限による未完了と、エンジン・通信経路の不変条件に反する最終出力は、エンジンが EOS で閉じて確定した state を保持したまま使用量付きのエラーを返す。キャンセルでは、完了した prefill chunk と生成済み token の使用量を protocol v15 の `cancelled.usage` で返す。Cognitive coordinator は `ModelGenerationError` の使用量を失敗・再試行後も集計する。domain commit の可否と、モデル内部で state が確定したかどうかは別の境界である。

## 検証方法と確認範囲

| 検証                                                                                             | 確認する契約                                                                                                                                           | 実モデル・fixture の範囲                                                                                          |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [Cognitive 結合テスト](../packages/native-inference-adapter/src/cognitive-continuation.test.ts)  | `pre_main → Main → pre_main → Main finish → post_main` の順序、tool/文章出力後の継続、domain commit 前の Main 停止、失敗時の usage 保持、prompt の分離 | ThinkingEngine・coordinator・runner・Native adapter/client は実装を使用。モデル応答・保存先・transport は fixture |
| [公式 template の比較](../native/echo-inference/oracles/qwen35_cognitive_continuation_parity.py) | 既存 prefix と追加 suffix が、公式 Qwen template の全体入力と byte/token 単位で一致する                                                                | tool/文章での完了と Cognitive exchange の有無を組み合わせた fixture。GPU 生成は行わない                           |
| [実モデル probe](../packages/native-inference-adapter/src/real-model-probe.ts)                   | 2 session・各2生成で、tool call の解析、結果を含む回答、resident prefix の再利用、state 長の更新を確認する                                             | Main は実モデル。`cognitive` モードの Memory/Emotion 結果と domain 状態は fixture                                 |
| [state integrity テスト](../native/echo-inference/crates/echo-inference/src/state_integrity.rs)  | 途中キャンセル・observer 失敗後の全 KV/GDN テンソル保持、再試行、6並列から1件離脱した際の他 lane の独立性                                              | 実モデルを使う token-level runtime テスト。比較用テンソルは独立したファイルに保存する                             |

実モデル probe は各 session の最初を含むすべての Main turn で handoff を渡す。初回の Memory recall は空で、継続時には観測した lookup 結果を含める。`plain` と `cognitive` の両入力について、greedy と production sampling を検証対象とする。

state integrity テストはテンソルの値・shape・dtype を比較する。TypeScript probe が報告する state 長や再利用 token 数の一致だけでは、GDN テンソル全体の一致を確認したことにはならない。

[実モデル request-control テスト](../packages/native-inference-adapter/src/real-request-controls.test.ts)は、Memory検索・Memory保存・Emotion の各 schema を greedy/production sampling で生成し、出力上限、途中 abort、使用量と再試行を確認する。prompt が非 JSON の文章を要求してもスキーマが強制されること、異なるスキーマでの実バッチ生成、Unicode と tool 記法を含む文字列も確認する。module prompt の判断品質や phase の寿命は検証しない。

通常の Node テスト、Rust テスト、実モデル probe は実行条件が異なる。モデルと Metal を必要とする検証は明示的に実行する。環境設定とコマンドは[Native README](../native/echo-inference/README.md#build)を参照する。

## Memory/Emotion の状態と composition

Memory/Emotion の計算の基準は、Core が各呼び出しへ渡す専用指示と共有 context 全体とする。前回の生成後の GDN を残して同じ履歴を再入力する意味を追加しないため、local runtime は両 adapter を `statePolicy: independent` で構築する。Main の既存の継続仕様は変更しない。

初回は `initial`、既存 state がある呼び出しは `reset` を使う。どちらも前回の生成後の KV/GDN は引き継がず、成功時だけ lane の current state を置き換える。独立呼び出しは `previousResponseToken` と durable 保存を拒否する。

`runThinkingSession()` は Memory/Emotion に同じ session ID を渡す。Native は各 lane に1個、生成ヘッダー直前の入力トークン列と KV/GDN を保持する。同じ session 内で保存済みトークン列全体が今回の入力の先頭と一致するときだけ再利用し、追加部分を計算して checkpoint を更新する。Core の入力内容やロールは変更しない。生成は別の作業用状態から進め、前回の生の生成結果は入力キャッシュへ混ぜない。

Memory の想起から記銘への指示変更は一致しないため全入力を再計算する。Emotion は同じ指示と伸長した共有 context なら再利用できる。生成の途中 abort 後も、完了済みの入力 checkpoint は再試行に使える。入力 checkpoint の完成前に中断した場合は以前の checkpoint を保持する。session の成功・失敗どちらでも `clear_input_cache` で解放する。session 間の継続は domain 保存と初期 context が担当する。

Core は同じ phase の再試行に同じ入力を渡し、成功済みの sibling を保持する。この再試行契約と、Native の入力キャッシュの寿命は別々に管理する。

[共通 factory](../packages/core/src/agent/model-cognitive-module.ts) は Core の prompt・schema・handoff と、注入された model/domain/retry policy を束ねる。Hosted の SDK・環境設定・期限・retry 判定は Worker 側に残る。[local runtime](../apps/local-runtime/src/local-native-inference-runtime.ts) の `think()` は Native model を注入し、session の排他と Main checkpoint に既存の `runThinkingSession()` を使う。呼び出し側は domain、tool、Main prompt、Cognitive の retry policy・期限・出力上限を指定する。

[実モデル Cognitive テスト](../apps/local-runtime/src/real-cognitive-workflow.test.ts) は、Core の実際の Memory/Emotion prompt と schema で、greedy/production sampling の各2 sessionを実行する。初回は tool で確認した結果を使って終了し、次回は前 session の確定結果を参照する。Main の各生成前の `pre_main` と終了後の `post_main`、2回目の session の Memory の途中 abort と再試行、成功済み Emotion の保持、全使用量の照合、前 session の確定結果の再入力を確認する。3 module の生成・Native transport・Core orchestration は実装を使用し、Main の課題と tool は検証用、domain の検索・保存はメモリ上の fixture とする。永続化・再起動や、新しい課題を session をまたいで遂行する運用品質の受け入れを代替しない。

Rust の state integrity テストは、単独・バッチの `reset` 後の全 KV/GDN が同じ入力を空から実行した対照と一致すること、およびキャンセル・observer 失敗で直前の全テンソルが保たれることを確認する。入力キャッシュのテストは、同じ chunk 境界での空からの計算との一致、生成による入力 checkpoint の汚染がないこと、prefill/生成の中断、batch 内の分離、session・指示変更時の不使用と解放も確認する。

[長文入力キャッシュの実モデルテスト](../apps/local-runtime/src/real-input-cache.test.ts)は、伸長する context と同じ最終入力の cold 実行を比較し、再利用 token 数と新規計算 token 数が全入力に一致することを確認する。プリフィル時間は診断値として記録し、マシン依存の速度倍率を合否条件にはしない。

## 保留中の設計判断

Native Cognitive 統合を進めるため、以下は現行実装を維持する。ただし、設計上の妥当性について合意済みとは扱わない。ローカルの起動・session 完了・異常時復旧を接続する際に再評価する。

### 独立推論の状態遷移指定

Memory/Emotion 用 adapter は、確定済み state の有無に応じて `initial` と `reset` を送り分ける。両者の独立推論と入力キャッシュの再利用条件は同じであり、呼び出し側の存在認識と Native 内部の state を照合することは、推論の正しさに必須ではない。

検討点は、この区別と毎回の `state_transition` 指定を Memory/Emotion に要求する必要性である。独立推論の方針を Native の state 登録時に設定し、初回登録・置き換えを Native 内部で扱う形も候補とする。Main の継続仕様は、この見直しとは分けて扱う。

### キャッシュ解放と session 完了の結合

現在は `clear_input_cache` の完了通知を待ち、解放に失敗すると、推論本体が成功していても `think()` がエラーを返す。確定済みの domain 更新や実行済み tool は巻き戻さず、Main checkpoint の保存は引き続き試みる。

session ID による再利用の分離と、解放要求を後続の推論が追い越さない処理順序は、完了通知の待機とは別に成立する。検討点は、解放完了を session の成功条件に含める必要性と、解放失敗をどこへ報告するかである。待機を外す場合も、処理順序、残存キャッシュの寿命、通信・Native process 障害の通知を定義する必要がある。

## 残る実装要件

### 1. 実運用の Cognitive workflow を評価する

[既存 workflow harness](../packages/model-evaluation/src/qwen36-eat-readiness/runtime-workflow-harness.ts)は `runAgentSession` を直接呼び、Cognitive coordinator を接続せず、評価専用の `session_record` 付き終了契約を使う。各 turn の Cognitive exchange を前提とする[Main prompt](../packages/core/src/llm/prompts/rin.ts)とは入力・終了契約が異なる。

実際の ThinkingEngine/Cognitive 経路へ評価を接続し、予定変更、緊急通知、一時的な tool 失敗を再検証する。既存の予定変更ケースには、初期予定の未保存、中止を「完了」と答える応答、未登録の `update_emotion` 呼び出し、`max_turns` 終了が観測されている。モデル単体と評価経路の寄与は未確定で、全 workflow の受け入れは未完了である。

長文 prefill の評価では、新規入力が8,192 tokens 以上となる fixture を用意する。chunked prefill は BF16 の演算順序が変わるため、単一実行との state の bit 一致を前提にしない。比較条件は[long-input prefill](../native/echo-inference/README.md#long-input-prefill)に従う。

### 2. ローカル保存・起動・再起動を接続する

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
- [Cloudflare runtime budget](./cloudflare-runtime-budget.md): Hosted composition を変更する際の request・storage 予算。`origin` の伝達と Cognitive 構築の共通化は、API/DO request、rows read/written、外部 API call を追加しない。
