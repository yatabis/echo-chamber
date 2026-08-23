# Cognitive Module 設計

## 用語

- **思考 session**: E.C.H.O. が起動してから `finish_thinking` または turn 上限で終了するまでの1回の思考。
- **Main model turn**: 思考 session 内で Main model を1回呼び出す単位。
- **Main**: 思考、runtime tool の選択、行動、終了判断を担う model。
- **Memory Module**: Main に代わって記憶の想起と記銘を担う model。
- **Emotion Module**: E.C.H.O. の現在の感情状態を更新する model。
- **Cognitive Module**: Memory Module、Emotion Module、および両者を Main と接続する runtime 処理の総称。
- **system-owned tool exchange**: Main が選んだ tool call ではなく、Cognitive Module の結果を通常の tool call / result と同じ形で Main の入力へ追加する履歴。

## 責務とインターフェース

Memory Module と Emotion Module には、その時点までに確定した Main の入力、出力、tool call / result、Memory の検索結果、Emotion の状態を渡す。同じタイミングで実行する両 Module は、同一の確定済み入力から並列に動作する。

Memory Module の出力は実行タイミングによって異なる。

| タイミング              | 出力                | runtime の処理                                     |
| ----------------------- | ------------------- | -------------------------------------------------- |
| 各 Main model turn の前 | `{ query }`         | `search_memory` を実行する                         |
| 思考 session の終了時   | `{ content, type }` | `store_memory` と同じ Memory 保存処理を1回実行する |

`query` と `content` は1〜500文字、`type` は `semantic` または `episode` とする。

Emotion Module は毎回、次の現在状態を返す。

```json
{
  "valence": 0.0,
  "arousal": 0.0,
  "labels": []
}
```

`valence` は -1.0〜1.0、`arousal` は 0.0〜1.0 とする。`labels` は最大5件、各ラベルは最大12文字とする。

## 実行の流れ

### 各 Main model turn の前

1. Memory Module と Emotion Module を同じ context から実行する。
2. runtime が Memory Module の `{ query }` で Memory を検索する。
3. runtime が検索結果と Emotion を確定する。
4. Main の入力へ次の system-owned tool exchange を追加する。
5. Main model turn を実行する。

```text
search_memory({ query })
→ { success: true, results: [...] }

update_emotion({ valence, arousal, labels })
→ { success: true }
```

Main は検索された Memory と現在の Emotion を、この tool exchange から観測する。system-owned tool exchange であることは内部の call ID と Cognitive Module のログで識別する。

### 思考 session の終了時

Memory Module と Emotion Module を最後に1回ずつ実行する。runtime は Memory Module の `{ content, type }` に同じタイミングの Emotion を関連付けて Memory を保存し、Emotion の現在状態も更新する。この処理は終了境界につき1回だけ commit する。

## Main からの Memory 利用

`search_memory` と `store_memory` は Main の runtime tool に残す。Main は必要と判断した場合に明示的に使用できる。Main が `store_memory` を使用した場合も、runtime が確定済みの現在の Emotion を関連付ける。

## 失敗時の動作

一時的な接続失敗、タイムアウト、rate limit、provider の server error では、失敗した Module または Memory 操作だけを同じ入力から1回再試行する。

再試行しても成功しない場合や一時的ではないエラーの場合は、そのタイミングの結果を commit せず、Main を先へ進めない。外部 request 数は thinking session 全体の hard gate を共有する。

## 設定とログ

Memory / Emotion が共有する model と reasoning effort は、`EchoInstanceDefinition.cognitiveModules` に instance ごとに指定する。

Cognitive Module の model event は Main と同じ payload policy で記録し、`cognitiveModule: memory | emotion` を追加する。Cognitive Module だけに適用する本文の redaction は行わない。
