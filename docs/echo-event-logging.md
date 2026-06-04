# Echo Event Logging

> Status: 現行設計メモ。`EchoEventPort` を中心に、console / Discord / DO SQLite archive / dashboard の役割分担を整理する。

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

現在の Worker 実装では、console、Discord、DO SQLite archive に配送する。console では `kind: "echo_event"` の JSON として Cloudflare Workers の console log に出す。severity に応じて `console.debug` / `console.info` / `console.warn` / `console.error` を使い分ける。

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

実装済みの短期通知 sink。

Discord は「今、人間が見る価値がある通知」だけに絞る。`analysis` 単独の event を Discord に出すことは原則避ける。

現行 policy では次を Discord に送る。

- `category === "session"` の event
- `severity === "error"` の event
- `severity === "warn"` かつ `system` stream を持つ event

これにより `session.started` / `session.completed` は毎回 Discord に出る。`model.turn.completed` の `no_tool_calls` は `analysis` 専用なので Discord には出ない。

Discord の配送先は event に直接持たせない。現行 Worker では instance ごとの `thinkingChannelId` へ送る。

### Archive

実装済みの dashboard / 後追い確認用 sink。

保存先は instance ごとの DO SQLite である。各 Durable Object が自分の Echo event を所有し、90 日分の raw event を保持する。

R2 / KV / D1 への日次退避は行わない。過去ログを外部分析基盤へ載せたくなった場合は、その時点で DO SQLite からの export 経路を別途設計する。

`echo_events` は次の情報だけを持つ。

- `id`
- `created_at_ms`
- `archive_day`
- `session_id`
- `type`
- `category`
- `severity`
- `streams_json`
- `summary`
- `payload_json`

`instance_id` は持たない。DO 自体が instance 単位だからである。

人間が活動日として読む operational day の境界は JST 07:00 である。一方、event archive の保存区切りは日次 sleep 開始時刻に合わせて JST 03:00 とする。これは activity の意味境界ではなく、日次 sleep 中に保持期限 cleanup を行うための archive day である。

日次 sleep window 中に event retention cleanup を実行し、90 日を超えた `archive_day` の event を DO SQLite から削除する。

```sql
DELETE FROM echo_events
WHERE archive_day < :cutoffDay;
```

`cutoffDay` は cleanup 実行時刻から 90 日を引いた時刻を、JST 03:00 境界の archive day に変換して決める。現在の archive day と保持期間内の archive day は削除しない。

cleanup は wake alarm より重要度が低い。日次 sleep では、まず wake alarm を設定してから cleanup を試みる。cleanup が失敗した場合は `system.schedule.alarm_completed` に `eventRetentionCleanup.status = "failed"` として記録し、severity は `warn` にする。ただし wake alarm は維持する。

### Dashboard

Dashboard は channel ではなく consumer である。

現行 dashboard は `/:instanceId/session-logs` から現在の archive day の session log だけを読む。過去ログは基本的に dashboard では扱わない。必要になった場合は DO SQLite に保持されている raw event から別の export / 分析基盤を作る。

Dashboard は Cloudflare Observability の代替ではない。raw event stream をそのまま読む用途は Observability に寄せる。

Dashboard API の `/:instanceId/session-logs` は raw event を返さない。Worker 側で保存済み event から session log view model を組み立てて `sessionLogs` として返す。Dashboard は session log を新しい順に並べ、各 session log を折りたためる activity として描画する。

#### Dashboard session log の責務

Dashboard session log は event stream の見た目を変えたものではない。Echo の 1 session を人間が読み返すための read model である。

そのため、変換の起点は「event type ごとの表示可否」ではなく「session 中に Echo が何を考え、何を実行し、どう終わったか」である。raw event は材料であり、1 raw event が必ず 1 activity になるとは限らない。

Dashboard session log は次の問いに答える。

- いつ session が始まり、いつ終わったか
- Echo がどんな自然言語出力をしたか
- どの tool を、どんな入力で、どんな結果として使ったか
- session 中に重要な異常や欠落があったか
- 最終的に session が正常終了、警告終了、失敗のどれだったか

Dashboard session log は次の問いには答えない。

- alarm / schedule / run decision が内部的にどう評価されたか
- raw API request / response payload がどうだったか
- archive / Discord / console sink に何が配送されたか
- session 外の状態遷移や運用イベントがどう発生したか

これらは Cloudflare Observability、DO SQLite に保持された raw event、または将来の分析基盤で見る。

#### 変換方針

変換は次の順で行う。

1. 現在の archive day の raw event を読む
2. `sessionId` を持つ event だけを session ごとに束ねる
3. session 内の raw event から session log activity を組み立てる
4. tool event は `callId` 単位で畳み込む
5. activity が 1 件もない session は dashboard response に出さない
6. session log は最新 activity の時刻で新しい順に並べる
7. session log 内の activity は時系列順に並べる

`sessionId` を持たない event は session log の材料ではない。これは dashboard response で後から落とす filter ではなく、read model の入力境界である。

#### 採用する event

Session log に採用する raw event は allowlist とする。unknown event や「warn/error だから」という理由だけで activity 化しない。

主に次を採用する。

- `session.*`: 思考 session の開始・終了・失敗
- `model.output.emitted`: Echo の自然言語出力
- `model.provider.warning`: session 中の provider 警告
- `tool.called` / `tool.completed` / `tool.failed`: tool 実行の開始と結果。ただし同じ `callId` の開始・完了は dashboard 上では 1 つの行動として扱う
- `memory.search.completed` / `memory.search.failed`: session 中の関連 memory 参照
- `model.turn.completed` の `no_tool_calls`: 確実に気づけるよう activity log に出す
- session log として意味を持つ warning / error: 活動ログ上の issue として表示する

Tool activity は `tool.completed` / `tool.failed` を主行とする。対応する `tool.called` があれば input を補助情報として吸収する。対応する completion / failure がない `tool.called` だけ、未完了 tool call として activity 化する。これにより同じ tool call が dashboard 上で重複して見えない。

#### 採用しない event

次は session log の材料にしない。

- `model.exchange.recorded`: raw API payload であり、session の読み物ではない
- `model.turn.started`: 通常は activity として読む情報がない
- `system.run_decision.*`: session 開始前の運用判断であり、session log ではなく observability の対象
- `system.echo_state.*`: instance 状態遷移であり、session log ではなく observability の対象
- `system.schedule.*`: alarm / sleep / wake の運用イベントであり、session log ではなく observability の対象
- `usage.recorded`: session 結果の集計値は `session.completed` から読めるため、別 activity にしない
- `memory.embedding.*` / `memory.reembedding.*` / `memory.evicted` / `memory.rerank.*`: memory subsystem の運用イベントであり、session log ではなく observability の対象
- `sessionId` を持たない event: session log の入力境界外

#### 実装方針

実装は `event type -> activity creator` の汎用 registry にしない。registry は event stream projection に見えやすく、event type を足せば dashboard に出る構造になりやすい。

代わりに、session log builder として実装する。

- `buildSessionLogs`: archive day の raw event を session 単位へ束ねる
- `buildSessionLog`: 1 session 分の activity log を作る
- `createSessionLifecycleActivity`: session の開始・終了・失敗を扱う
- `createModelActivity`: natural language output / provider warning / no tool calls を扱う
- `createToolActivity`: `callId` 単位で tool call を畳み込む
- `createMemoryActivity`: session 中の memory search 結果を扱う

この構造なら、review 時に「これは session log の構成要素か」という問いで判断できる。event type を増やす場合も、まず採用理由を docs に追加してから実装する。

## 現時点の未決事項

- Discord 通知を thought / system など複数チャンネルへ分けるか
- raw payload の redaction 方針
- payload が大きい event の分割・圧縮・sampling 方針
- Cloudflare Observability の retention を前提に、どこまで console sink に頼るか
- `model.turn.*` を `analysis` 専用のままにするか、system stream にも出すか
- 90 日分の DO SQLite raw event を外部分析基盤へ export する経路を作るか

## 実装上の境界

`packages/core` は `EchoEventPort` と event type / category / stream の定義だけを持つ。Cloudflare、Discord、D1、R2、DO SQLite などの具体 sink には依存しない。

`apps/cloudflare-workers` は composition root として、console / Discord / DO SQLite archive sink を差し込む。core から具体 runtime へ逆依存させない。

`emitEchoEvent()` は best-effort とし、event 配送の失敗で agent 本体を落とさない。
