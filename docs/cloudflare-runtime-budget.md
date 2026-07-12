# Cloudflare Runtime Budget

> Status: 現行運用メモ。Dashboard と Echo runtime の API request、Durable Object request、storage rows read / written の見積もりを整理する。

## 背景

Echo は Cloudflare Workers + Durable Objects 上で動く。Free plan では、HTTP request 数だけでなく Durable Object の request 数と SQLite-backed storage の rows read / written が制約になる。

特に dashboard は read-only に見えるが、開くだけで複数の Durable Object request と storage read を発生させる。今後の機能追加では、画面や API の便利さだけでなく、操作ごとの request / storage cost を設計入力として扱う。

## Cloudflare 側の制約

Cloudflare Durable Objects pricing docs による現在の主な Free plan 制約は次のとおり。

Source: <https://developers.cloudflare.com/durable-objects/platform/pricing/>

| 項目                      | Free plan limit | 備考                                                                                      |
| ------------------------- | --------------- | ----------------------------------------------------------------------------------------- |
| Durable Object requests   | 100,000 / day   | HTTP request、RPC session、WebSocket message、alarm invocation を含む                     |
| Duration                  | 13,000 GB-s/day | active execution 中が対象                                                                 |
| SQLite rows read          | 5,000,000 / day | SQLite-backed DO storage の read 制約                                                     |
| SQLite rows written       | 100,000 / day   | insert / update / delete が対象                                                           |
| SQL stored data           | 5 GB total      | SQLite-backed DO storage                                                                  |
| Daily reset               | 00:00 UTC       | JST では 09:00                                                                            |
| `setAlarm()`              | 1 row written   | alarm 設定も write として数える                                                           |
| key-value storage methods | rows 課金対象   | `get()` / `put()` / `delete()` / `list()` も hidden SQLite table に対する操作として数える |
| delete                    | rows written    | 削除も write として数える                                                                 |

## 設計原則

- Dashboard は raw data scan ではなく、pre-aggregated read model または bounded latest rows だけを読む。
- `SELECT *` は表示に必要な bounded rows に限定する。集計用途では BLOB や巨大 JSON payload を読まない。
- 期間集計は raw event を 1 / 7 / 30 日分走査しない。日次集計などの小さい read model から組み立てる。
- 新しい dashboard endpoint を追加するときは、この文書に request / storage budget を追記する。
- 新しい background event を追加するときは、1 alarm / 1 session あたりの追加 rows written と日次見積もりを確認する。
- production 調査で endpoint を反復実行しない。負荷調査は local / staging / Cloudflare analytics を優先する。

## 基準値

現在の固定設定:

| 項目               | 値                        |
| ------------------ | ------------------------- |
| Echo instances     | `rin`, `marie` の 2 件    |
| chat channels      | instance あたり 3 channel |
| regular alarm      | 1 分間隔                  |
| daily sleep window | JST 03:00-07:00           |

alarm invocation 数の概算:

| ケース                  | 1 instance | 2 instances |
| ----------------------- | ---------- | ----------- |
| sleep なしの最大        | 1,440 /day | 2,880 /day  |
| daily sleep window あり | 1,200 /day | 2,400 /day  |

## Dashboard の操作別 budget

以下の storage read / write は実装上のクエリ形状から見た設計見積もりであり、Cloudflare が実際に数える rows read は SQLite の実行計画や内部 metadata にも左右される。設計レビューでは、この表より大きくなる変更を危険信号として扱う。

### Dashboard shell / static assets

| 項目                 | 内容                                       |
| -------------------- | ------------------------------------------ |
| Browser API requests | `GET /dashboard`, `GET /dashboard/*`       |
| DO requests          | 0                                          |
| Storage reads        | 0                                          |
| Storage writes       | 0                                          |
| 外部 API             | 0                                          |
| 備考                 | `ASSETS.fetch()` で静的 dashboard を返す。 |

### Instance list

Dashboard 一覧画面を開く、または一覧で Refresh する操作。

| 項目                 | 内容                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| Browser API requests | `GET /instances`                                                                  |
| Internal DO requests | `GET /:id/summary` x `ECHO_INSTANCE_IDS.length`。現在は 2 件                      |
| Storage reads        | instance ごとに state、alarm、next wake、usage、note summary、memory summary      |
| Storage writes       | warm object なら 0。cold initialization では `id` / `name` 保存が発生する場合あり |
| 外部 API             | 0                                                                                 |

`/:id/summary` は raw event を読まない。note は最大 200 件、memory は summary query のみを使う。

### Instance detail

詳細画面を開く、または詳細で Refresh する操作。タブ切り替えだけでは追加 API request は発生しない。

| 項目                 | 内容                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| Browser API requests | `GET /:id`, `GET /:id/session-logs`, `GET /:id/action-analysis` の 3 本           |
| DO requests          | 3                                                                                 |
| Storage writes       | warm object なら 0。cold initialization では `id` / `name` 保存が発生する場合あり |
| 外部 API             | 0                                                                                 |

endpoint ごとの storage read:

| Endpoint                   | Storage read shape                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `GET /:id`                 | state、alarm、next wake、context、usage、notes 最大 200 件、embedding BLOB なし memory 最大 500 件      |
| `GET /:id/session-logs`    | 現在 archive day の `echo_events` を `LIMIT 200` で読む                                                 |
| `GET /:id/action-analysis` | 日次 action-analysis stats と tool stats だけを最大 30 archive day 分読む。raw `echo_events` は読まない |

`GET /:id/action-analysis` は raw `echo_events` を読まない。1 / 7 / 30 day の period summary は、最大 30 日分の daily stats と tool stats から組み立てる。

### Manual wake / sleep

Dashboard からではなく API として存在する操作。

| 操作         | API request       | DO requests | Storage reads              | Storage writes                                                                 |
| ------------ | ----------------- | ----------- | -------------------------- | ------------------------------------------------------------------------------ |
| manual wake  | `POST /:id/wake`  | 1           | state                      | `setAlarm()`、state、`alarm_scheduled` event、state change event               |
| manual sleep | `POST /:id/sleep` | 1           | state                      | state、alarm delete、state change event。delete は rows written として扱われる |
| local run    | `POST /:id/run`   | 1           | `alarm()` の run path 相当 | local environment のみ。production では not found                              |

## Echo alarm の budget

### Skip path

alarm が発火したが、思考 session を実行しない場合。

| 項目           | 見積もり                                                                   |
| -------------- | -------------------------------------------------------------------------- |
| DO requests    | 1 alarm invocation                                                         |
| Storage reads  | `id`、`name`、state、usage、next wake などの小さい key read                |
| Storage writes | event rows 約 4 件 + `setAlarm()` 1 件                                     |
| 外部 API       | Discord unread check: 3 channels x 2 calls = 約 6 calls / alarm / instance |

代表的な event rows:

- `system.schedule.alarm_triggered`
- `system.run_decision.evaluated`
- `system.schedule.alarm_scheduled`
- `system.schedule.alarm_completed`

通常稼働時の概算:

| ケース                  | DO requests/day | Storage writes/day | Discord API calls/day |
| ----------------------- | --------------- | ------------------ | --------------------- |
| 1 instance, sleep あり  | 約 1,200        | 約 6,000           | 約 7,200              |
| 2 instances, sleep あり | 約 2,400        | 約 12,000          | 約 14,400             |
| 2 instances, sleep なし | 約 2,880        | 約 14,400          | 約 17,280             |

### Run path

alarm が思考 session を実行する場合。skip path の固定コストに、session / model turn / tool call の変動コストが加わる。

変数:

| 記号 | 意味                   |
| ---- | ---------------------- |
| `T`  | model turn 数。最大 10 |
| `C`  | tool call 数           |
| `M`  | memory search 回数     |
| `S`  | store memory 回数      |

追加 request / storage の概算:

| 項目           | 追加コスト                                                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Model API      | `T` calls                                                                                                                           |
| Discord API    | startup `check_notifications` でさらに約 6 calls。`read_chat_messages` / `send_chat_message` / reaction tool は tool 使用分だけ追加 |
| Storage reads  | context load、usage read、memory search source 最大 500 rows、notes tool 使用時は最大 200 notes                                     |
| Storage writes | state Running / Idling、session events、model turn events、tool events、usage、context、next wake、memory store など                |

event rows written の目安:

```text
skip path events
+ state change events
+ session.started
+ session.completed or session.failed
+ usage.recorded
+ 2 * T  (model.turn.started / model.turn.completed)
+ 2 * C  (tool.called / tool.completed or tool.failed)
+ memory / embedding / rerank related events
```

`memory.search` は検索候補として embedding BLOB を含む memory rows を読む。現行の上限は memory 保持上限と同じ 500 rows で、同一 request 内では cache する。Dashboard 表示では embedding BLOB を読まない。

current embedding model と異なる memory row は、異なるベクトル空間や次元が混ざることを避けるため検索対象にしない。代わりに、日次 sleep maintenance で stale memory の再 embedding を最大 500 件実行し、model 変更後も既存 memory が検索対象へ戻る機会を維持する。

## Action analysis read model

集計はテーブルである必要はない。必要条件は、dashboard 表示時に raw event を期間分 scan しないことである。

許容される形:

- SQL daily stats table
- archive day ごとの JSON stats row
- Durable Object storage key に保存した bounded stats

推奨は SQL daily stats table である。理由は、event 保存時に `UPDATE count = count + ?` で差分集計でき、1 / 7 / 30 day の read が小さい rows に閉じるため。

禁止する形:

- `GET /:id/action-analysis` で `echo_events` を 7 / 30 日分 scan する
- dashboard 表示用に raw `payload_json` や embedding BLOB を期間分読む
- production 確認で action-analysis endpoint を反復実行して quota を消費する

## Feature review checklist

Cloudflare runtime、dashboard、Echo event、tool を変更するときは、実装前または PR 前に次を確認する。

- その変更は dashboard 初回表示、Refresh、タブ切り替え、alarm、run session のどこで実行されるか。
- Browser API request、internal DO request、storage read、storage write、外部 API call の数を説明できるか。
- 日次見積もりを `instances x alarms/day x per-alarm cost` で見たとき、Free plan の rows read / written に収まるか。
- 新しい dashboard query が raw table scan、unbounded `list()`、unbounded `SELECT *`、BLOB read を含んでいないか。
- 新しい event type が alarm ごとに発火するなら、rows written/day への影響を見積もったか。
- 集計が必要な UI は、raw event ではなく pre-aggregated read model を読む設計になっているか。
- production で確認する必要がある場合、反復 endpoint 実行ではなく analytics / logs / bounded diagnostic endpoint を使えるか。
- この文書の該当表を更新したか。
