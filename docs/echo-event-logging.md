# Echo Event Logging

> Status: 議論のベース。現行の `EchoEventPort` 実装を説明しつつ、未実装の archive / Discord sink / dashboard については設計方針として整理する。

## 背景

Echo のログは、もともと Discord が「通知」と「保存」を兼ねる形になっていた。

Discord は、今エージェントが何をしているかを素早く確認する用途には向いている。一方で、後から検索・集計・分析する保存先としては弱い。ログ量が増えるとチャンネルが氾濫し、人間が見るべき情報も埋もれやすくなる。

整理したい要求は大きく 2 つある。

- Discord を静かにし、人間が今見る価値のある通知に絞る
- Discord に出さない詳細ログも、後から観察・分析できる形で残す

## ログの用途

ログは 1 種類ではなく、少なくとも次の用途に分かれる。

### クイック監視

人間が Discord や dashboard で短時間に確認する用途。

見たいものは、エージェントが詰まっていないか、ツールを適切に使えているか、重要な warning / error が出ていないかである。

### 観察

Cloudflare Workers Observability などで、直近の挙動を追う用途。

実行された session、tool call、memory search、run decision、usage 記録などを時系列で確認できる必要がある。

### 分析

日次・週次で集計し、改善に使う用途。

例:

- 1 日にどの tool が何回使われたか
- 1 session あたり平均何回 tool call があるか
- tool failure rate
- session duration
- token usage
- soft / hard limit による skip 回数
- memory search の query、hit count、score 分布
- rerank 前後で順位が大きく変わった memory

### 再現・障害調査

外部 API 失敗、token limit 判定、alarm 実行、state 遷移などを後から追い、なぜその挙動になったかを説明できるようにする用途。

## 現行の役割分担

### `ThoughtLogPort` / `LoggerPort`

廃止済みの旧 port。

以前は `ThoughtLogPort` が Discord の thought channel へ思考ログを流し、`LoggerPort` が console と Discord log channel へ運用ログを流していた。

現行実装ではどちらも runtime path から外し、残す価値がある情報だけを semantic event として定義する。

- model adapter が返した assistant message / reasoning は `model.output.emitted`
- tool call は `tool.called` / `tool.completed` / `tool.failed`
- usage は `usage.recorded`
- memory search / rerank / re-embedding / embedding usage は `memory.*`
- alarm、state、run decision は `system.*`

`log.emitted` のような汎用 event は作らない。message 文字列をそのまま移すのではなく、後から集計・監査しやすい payload に分解する。

### `EchoEventPort`

Echo の運用・分析イベントを構造化して扱う。

発火元は「何が起きたか」だけを渡す。`category` や `streams` は event type から中央で派生する。配送先は event に持たせず、sink policy 側で決める。

現在の Worker 実装では、`kind: "echo_event"` の JSON として Cloudflare Workers の console log に出す。severity に応じて `console.debug` / `console.info` / `console.warn` / `console.error` を使い分ける。

## 設計原則

### Event は配送先ではなく事実を表す

`EchoEvent` は「どこへ送るか」ではなく、「何が起きたか」を表す。

発火元が Discord、dashboard、archive などの配送先を知り始めると、表示・通知の都合が business logic 側へ混ざる。配送判断は sink policy に寄せる。

### 発火元は最小情報だけを渡す

発火元が渡すのは次の最小 event input にする。

```ts
type EchoEventInput = {
  type: EchoEventType;
  severity: EchoEventSeverity;
  summary: string;
  payload?: Record<string, unknown>;
};
```

`createEchoEvent()` が `type` から `category` と `streams` を補う。

### `streams` は読み方の index

`streams` は配送先ではない。dashboard や archive 上で、その event をどの文脈で読むかを示す index である。

- `thought`: エージェントの思考・行動の流れとして読む
- `system`: 実行基盤、制御、エラー、状態遷移として読む
- `analysis`: 後から集計・評価・改善に使う

### `channels` / `visibility` は event に持たせない

過去の検討では `visibility` や `channels` を event に持たせる案もあった。しかし、それを発火元が指定すると配送責務が event 生成側に戻る。

そのため、現行方針では `EchoEvent` から `visibility` を外し、配送先は sink policy が決める。

## Event type と stream

現在の event type と派生値は次のとおり。

| type                                       | category  | streams                         |
| ------------------------------------------ | --------- | ------------------------------- |
| `session.started`                          | `session` | `thought`, `system`, `analysis` |
| `session.completed`                        | `session` | `thought`, `system`, `analysis` |
| `session.failed`                           | `session` | `thought`, `system`, `analysis` |
| `model.turn.started`                       | `model`   | `analysis`                      |
| `model.turn.completed`                     | `model`   | `analysis`                      |
| `model.output.emitted`                     | `model`   | `thought`, `analysis`           |
| `model.exchange.recorded`                  | `model`   | `analysis`                      |
| `model.provider.warning`                   | `model`   | `system`, `analysis`            |
| `tool.called`                              | `tool`    | `thought`, `analysis`           |
| `tool.completed`                           | `tool`    | `system`, `analysis`            |
| `tool.failed`                              | `tool`    | `thought`, `system`, `analysis` |
| `memory.evicted`                           | `memory`  | `system`, `analysis`            |
| `memory.embedding.generated`               | `memory`  | `system`, `analysis`            |
| `memory.search.started`                    | `memory`  | `system`, `analysis`            |
| `memory.search.completed`                  | `memory`  | `system`, `analysis`            |
| `memory.search.failed`                     | `memory`  | `system`, `analysis`            |
| `memory.reembedding.skipped`               | `memory`  | `system`, `analysis`            |
| `memory.reembedding.started`               | `memory`  | `system`, `analysis`            |
| `memory.reembedding.item_failed`           | `memory`  | `system`, `analysis`            |
| `memory.reembedding.completed`             | `memory`  | `system`, `analysis`            |
| `memory.rerank.failed`                     | `memory`  | `system`, `analysis`            |
| `memory.rerank.fallback`                   | `memory`  | `system`, `analysis`            |
| `system.schedule.alarm_triggered`          | `system`  | `system`, `analysis`            |
| `system.schedule.alarm_completed`          | `system`  | `system`, `analysis`            |
| `system.schedule.alarm_scheduled`          | `system`  | `system`, `analysis`            |
| `system.schedule.next_wake_at_updated`     | `system`  | `system`, `analysis`            |
| `system.schedule.next_wake_at_cleared`     | `system`  | `system`, `analysis`            |
| `system.schedule.next_wake_at_invalidated` | `system`  | `system`, `analysis`            |
| `system.echo_state.changed`                | `system`  | `system`, `analysis`            |
| `system.echo_state.change_rejected`        | `system`  | `system`, `analysis`            |
| `system.echo_state.change_failed`          | `system`  | `system`, `analysis`            |
| `system.run.failed`                        | `system`  | `system`, `analysis`            |
| `system.run.precondition_failed`           | `system`  | `system`, `analysis`            |
| `system.run_decision.evaluated`            | `system`  | `system`, `analysis`            |
| `usage.recorded`                           | `usage`   | `system`, `analysis`            |

`model.output.emitted` は assistant message / reasoning のみを扱い、tool call は含めない。tool handler 内部の詳細エラーは、model に返す tool output からは除き、`tool.failed` payload の `diagnostics` に載せる。API payload の裏取りが必要な場合は `model.exchange.recorded` を `debug` severity / `analysis` stream として出す。

## Sink policy の考え方

### Console / Observability

現在実装されている sink。

Cloudflare Workers Observability で拾いやすいように、次のような構造化 JSON を console に出す。

```json
{
  "timestamp": "2026-05-31T00:00:00.000Z",
  "kind": "echo_event",
  "source": "cloudflare-workers",
  "instanceId": "rin",
  "sessionId": "session-id",
  "type": "tool.called",
  "category": "tool",
  "streams": ["thought", "analysis"],
  "severity": "info",
  "summary": "search_memory called",
  "payload": {}
}
```

この sink は短期観察向けであり、長期保存の正本ではない。

### Discord

未実装の将来 sink。

Discord は「今、人間が見る価値がある通知」だけに絞る。`analysis` 単独の event を Discord に出すことは原則避ける。

Discord へ送る候補:

- `session.failed`
- `tool.failed`
- warning / error
- token hard / soft limit による重要な skip
- memory search が 0 件、低スコア、異常に遅い場合
- session summary

Discord の配送先は event に直接持たせず、sink 側で `streams` を見て thought / system のチャンネルへ分ける。

### Archive

未実装の将来 sink。

長期保存と分析の正本にする。D1 と R2 を併用する案が有力である。

- D1: event type、instance id、session id、tool name、created at、duration、success、token count などの index
- R2: raw payload、LLM input / output、thought text、memory search candidate 全文などの大きいデータ

D1 だけに全文を入れると肥大化しやすい。R2 だけだと検索・集計が難しい。D1 は index、R2 は raw archive と分ける。

### Dashboard

Dashboard は channel ではなく consumer である。

Dashboard は archive または observability から event を読み、`streams` / `type` / `category` を使って thought view、system view、analysis view を構成する。

## 現時点の未決事項

- Discord に出す event の最小セット
- `session.started` / `session.completed` を常に Discord に出すか、summary だけにするか
- archive sink の保存先を D1 + R2 にするか、まずは R2 JSONL から始めるか
- raw payload の redaction 方針
- payload が大きい event の分割・圧縮・sampling 方針
- Cloudflare Observability の retention を前提に、どこまで console sink に頼るか
- `model.turn.*` を `analysis` 専用のままにするか、system stream にも出すか

## 実装上の境界

`packages/core` は `EchoEventPort` と event type / category / stream の定義だけを持つ。Cloudflare、Discord、D1、R2 などの具体 sink には依存しない。

`apps/cloudflare-workers` は composition root として、当面は `ConsoleEchoEventPort` を差し込む。将来 Discord sink や archive sink を追加する場合も、core から具体 runtime へ逆依存させない。

`emitEchoEvent()` は best-effort とし、event 配送の失敗で agent 本体を落とさない。
