# Cloudflare Runtime Budget

> Status: 現行運用メモ。Dashboard と Echo runtime の API request、Durable Object request、storage rows read / written の見積もりを整理する。

## 背景

Echo は Cloudflare Workers + Durable Objects 上で動く。Free plan では、HTTP request 数だけでなく Durable Object の request 数と SQLite-backed storage の rows read / written が制約になる。

特に dashboard は read-only に見えるが、開くだけで複数の Durable Object request と storage read を発生させる。今後の機能追加では、画面や API の便利さだけでなく、操作ごとの request / storage cost を設計入力として扱う。

## Cloudflare 側の制約

Cloudflare Durable Objects pricing と Workers limits による現在の主な Free plan 制約は次のとおり。

Source: <https://developers.cloudflare.com/durable-objects/platform/pricing/>
Source: <https://developers.cloudflare.com/workers/platform/limits/#subrequests>

| 項目                      | Free plan limit    | 備考                                                                                      |
| ------------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| Durable Object requests   | 100,000 / day      | HTTP request、RPC session、WebSocket message、alarm invocation を含む                     |
| Duration                  | 13,000 GB-s/day    | active execution 中が対象                                                                 |
| SQLite rows read          | 5,000,000 / day    | SQLite-backed DO storage の read 制約                                                     |
| SQLite rows written       | 100,000 / day      | insert / update / delete が対象                                                           |
| SQL stored data           | 5 GB total         | SQLite-backed DO storage                                                                  |
| Daily reset               | 00:00 UTC          | JST では 09:00                                                                            |
| `setAlarm()`              | 1 row written      | alarm 設定も write として数える                                                           |
| key-value storage methods | rows 課金対象      | `get()` / `put()` / `delete()` / `list()` も hidden SQLite table に対する操作として数える |
| delete                    | rows written       | 削除も write として数える                                                                 |
| Subrequests               | 50 / invocation    | Free plan の通常枠。internal service 向けは次行の別枠                                     |
| Internal-service requests | 1,000 / invocation | Free plan。R2、KV、D1 など internal service 向け subrequest の別枠                        |

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

production / preview の全 Worker request は、route 処理より前に Cloudflare Access JWT の署名、issuer、AUD、時刻 claim を検証する。この検証による DO request、storage read / write の増分は 0。署名検証用 JWKS は Worker isolate 内で再利用するため、通常の request では外部 API call も 0 だが、isolate 内の初回検証時または未知の key id への rotation 時には Access team domain の JWKS endpoint へ 1 call 発生し得る。JWT 検証の CPU cost は各 request に加わる。

### Dashboard shell / static assets

| 項目                 | 内容                                                                   |
| -------------------- | ---------------------------------------------------------------------- |
| Browser API requests | `GET /dashboard`, `GET /dashboard/*`                                   |
| DO requests          | 0                                                                      |
| Storage reads        | 0                                                                      |
| Storage writes       | 0                                                                      |
| 外部 API             | warm isolate では 0。初回 JWT 検証時は Access JWKS 1 call の可能性あり |
| 備考                 | `ASSETS.fetch()` で静的 dashboard を返す。                             |

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

`/:id/summary` は Durable Object instance 内で 30 秒だけ in-memory cache する。cache hit 時も DO request は発生するが、storage read は発生しない。state / usage / cognitive domain / next wake / alarm などの更新時は cache を破棄する。

### Instance detail

詳細画面を開く、または詳細で Refresh する操作。タブ切り替えだけでは追加 API request は発生しない。

| 項目                 | 内容                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| Browser API requests | `GET /:id`, `GET /:id/session-logs`, `GET /:id/action-analysis` の 3 本           |
| DO requests          | 3                                                                                 |
| Storage writes       | warm object なら 0。cold initialization では `id` / `name` 保存が発生する場合あり |
| 外部 API             | 0                                                                                 |

endpoint ごとの storage read:

| Endpoint                   | Storage read shape                                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `GET /:id`                 | state、alarm、next wake、cognitive domain、usage、notes 最大 200 件、embedding BLOB なし memory 最大 500 件 |
| `GET /:id/session-logs`    | 現在 archive day の session 付き `echo_events` を partial index 経由で `LIMIT 200` まで読む                 |
| `GET /:id/action-analysis` | 日次 action-analysis stats と tool stats だけを最大 30 archive day 分読む。raw `echo_events` は読まない     |

`GET /:id/action-analysis` は raw `echo_events` を読まない。1 / 7 / 30 day の period summary は、最大 30 日分の daily stats と tool stats から組み立てる。

`GET /:id/session-logs` は `session_id IS NOT NULL` を `LIMIT 200` より前に適用する。1 分間隔の alarm が生成する session なし system event は取得上限を消費しない。`idx_echo_events_archive_day_session_created` partial index により、session なし event を遡って scan せず、最大 200 session event の bounded read を維持する。既存 DO では、この index を初めて作るときだけ保持中の session event に対する index 構築が発生する。定常時の API / DO request、event table row write、外部 API call は増えないが、session event row には partial index の保存・更新コストが加わる。

Dashboard detail の GET DTO は Durable Object instance 内で短時間だけ in-memory cache する。`GET /:id` と `GET /:id/summary`、`GET /:id/session-logs` は 30 秒、`GET /:id/action-analysis` は 60 秒を上限にする。これは Cloudflare edge cache ではないため DO request 数は減らないが、同じ DO instance が生きている間の連続 refresh では storage read を避けられる。

### Manual wake / sleep

Dashboard からではなく API として存在する操作。

| 操作         | API request       | DO requests | Storage reads              | Storage writes                                                                  |
| ------------ | ----------------- | ----------- | -------------------------- | ------------------------------------------------------------------------------- |
| manual wake  | `POST /:id/wake`  | 1           | state                      | `setAlarm()`、state、`alarm_scheduled` event、state change event                |
| manual sleep | `POST /:id/sleep` | 1           | state                      | state、alarm delete、state change event。delete は rows written として扱われる  |
| local run    | `POST /:id/run`   | 1           | state 検証後の思考 session | local environment のみ。alarm 用の未読、token limit、next wake 判定は適用しない |

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

追加 request / storage の概算:

| 項目           | 追加コスト                                                                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model API      | `T` calls                                                                                                                                          |
| Discord API    | startup `check_notifications` でさらに約 6 calls。`read_chat_messages` / `send_chat_message` / reaction tool は tool 使用分だけ追加                |
| Storage reads  | usage / schedule / cognitive domain、Memory recall source 最大500 rows、`search_memory`使用時は追加の最大500 rows、notes tool使用時は最大200 notes |
| Storage writes | state Running / Idling、session / model / tool event、usage、next wake、cognitive domain / Memory commit、`store_memory`使用分等                   |

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

### Cognitive Module

Cognitive phaseのmodel callとstorage操作は、各thinking sessionのrun budgetへ含める。Mainが`store_memory` / `search_memory`を選んだ場合は、そのtool callのcostも同じbudgetへ加算する。

変数`T`をMain model turn数、`P = T + 1`をcognitive phase数とする。各Main model turnの前に`pre_main`を1回、session終了時に`post_main`を1回実行し、各phaseで2 moduleを呼ぶ。

一時エラーでは失敗したmoduleだけを1回再試行する。Cognitive Module用のOpenAI SDK retryは0とし、1 application attemptを1 HTTP attemptとして数える。

| 項目                       | 上限 / 挙動                                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Worker / DO request        | 追加0。同じthinking sessionのDO request内で待機する                                                                                |
| cognitive model requests   | retry無しで`2 * (T + 1)`。全module requestが1回ずつretryした場合は`4 * (T + 1)`。`T = 10`なら22〜44件                              |
| external request hard gate | alarm / manual run全体で40件。Main、Cognitive、Discordの状態取得・tool、Web、Zenn、OpenAI embeddingが共有する                      |
| model request timeout      | 30秒                                                                                                                               |
| Memory recall              | `T`回。各回で最大500 rowsを読み、query embedding 1回、候補があればrerank 1回を行い、最大5件をMainへ渡す                            |
| Memory update              | `post_main`で1件。Memory Moduleの`content` / `type`と同phaseのEmotionを1 SQLite transactionで保存し、次session用の状態にも保持する |
| Event archive writes       | phase / model / commit event分。Cognitive model eventにはMainと同じpayload policyを適用する                                        |
| main model calls / input   | Mainのcall数`T`は変わらないが、各turn前の`search_memory` / `update_emotion` exchangeによりMain requestのinput tokenが増える        |

任意toolを1回使う場合、`search_memory`はquery embedding 1回、候補があればrerank 1回、embedding BLOBを含む最大500 rowsのreadを追加し得る。`store_memory`はembedding 1回とMemory rowのinsert / eviction、関連event writeを追加し得る。OpenAI embeddingを使用する構成ではembedding requestも40件のexternal request hard gateを共有する。既定構成のembeddingとrerankはWorkers AI bindingを使うため、この外部request gateには含めない。

外部requestは送信直前に1件ずつ数える。40件を使い切った場合は追加requestを送らず、現在のsessionを失敗させる。session lifecycleと失敗通知のDiscord eventはこのapplication gateに含めず、別の通知用gateで最大10件に制限する。これにより、Cloudflare上限50件の範囲で通知枠を保ち、通常処理の上限到達が失敗ログやusage保存を妨げないようにする。

既定のWorkers AI embedding構成でも、run判定と起動時通知確認でDiscord APIを約12件使う。`T = 10`かつretry・任意toolなしの場合はMain 10件、Cognitive 22件と合わせて約44件になるため、現行hard gateのまま10 turn完走は保証しない。`T = 8`なら同条件で約38件となる。OpenAI embedding構成では、さらに各recallと終了時保存のembedding requestが加わる。

この40件は、Free planのexternal subrequest上限50件に対して10件の余裕を持たせるためのrun単位の上限である。

`pre_main`では検索結果とEmotionを確定し、`post_main`では1件のMemoryとEmotionを1回のtransactionで保存する。片方のmoduleまたはMemory操作が失敗した場合、そのphaseのstate更新は行わない。

次sessionのMemory / Emotion初期状態は既存のcognitive domain stateから復元する。Memory tableの追加scan、DO request、外部requestは発生しない。

`memory.search` は検索候補として embedding BLOB を含む memory rows を読む。現行の上限は memory 保持上限と同じ 500 rows で、同一 request 内では cache する。Dashboard 表示では embedding BLOB を読まない。

current embedding model と異なる memory row は、異なるベクトル空間や次元が混ざることを避けるため検索対象にしない。日次 sleep maintenance はstale候補を最大500件読む。OpenAI embedding構成で共有external request budgetへ達した場合は、残りを反復せず後続の日次runへ繰り越す。既定のWorkers AI embeddingは内部serviceの上限に従う。

### Public Web page reader

`read_web_page` は model が明示的に tool を呼んだときだけ、同じ思考 session の run path 内で外部 HTTP(S) page を取得する。課金 quota や日次配分ではなく、偶発的な反復・並列取得を単純に抑える session-local counter を使う。

| 項目                | 1 tool call                                            | 1 thinking session                                      |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| Tool call 上限      | -                                                      | 最大 4 call                                             |
| 外部 Web fetch      | 初回 1 + redirect 最大 3 = 最大 4                      | 全 call が redirect 上限へ達した場合、理論上最大 16     |
| Worker / DO request | 実行中の DO request 内で処理するため追加 0             | 追加 0                                                  |
| Storage reads       | 0                                                      | 0                                                       |
| Storage writes      | 既存の `tool.called` と completed / failed の約 2 rows | 最大約 8 event rows。次 model turn 等の既存変動分は別途 |
| Response body       | stream 実測 1,048,576 byte 以下                        | call 間で共有する byte quota は設けない                 |
| 抽出 Markdown       | 最大 64,000 UTF-16 code units                          | session 合計文字 quota は設けない                       |
| Model 可視本文      | 既定 8,000、指定可能最大 12,000 UTF-16 code units      | call ごとの上限だけを適用                               |
| Retry               | 自動 retry なし                                        | retryable 表示はするが reader 自身は再試行しない        |

4 call は並列に実行され得る。各 call の入力 body は 1 MiB で停止するが、stream chunk、結合した byte array、decode 後の文字列、HTML 抽出中の作業文字列が一時的に共存するため、memory の瞬間値は body 上限そのものより大きい。HTMLRewriter は platform API を使い、新しい第三者 runtime dependency は追加しない。CPU、memory、bundle size は local fixture の最大サイズケース、`wrangler deploy --dry-run` の upload size、必要時の preview Worker analytics で確認する。本番 endpoint を負荷測定のために反復実行しない。

2026-08-10 の実装後 dry-run では、Worker 全体の upload size は 1,533.57 KiB、gzip 後 272.33 KiB だった。変更前の同条件 baseline は保存されていないため、これは Web reader の増分ではなく実装後の総量として扱う。

application は初回 URL と各 redirect について scheme、literal / special-use host、既知の internal suffix、credential、credential-like query key、非 default port、HTTPS downgrade を拒否する。hostname の DNS 解決先を application code で検証・pin はしていないため、解決後の egress 制約は Cloudflare platform boundary に依存する。この区別を、local runtime へ同じ adapter を移植できるという意味には解釈しない。

Web tool 自体の監査 event は URL、title、本文、link URL を保存せず、安全な件数・状態 metadata だけを保存する。Main / Cognitive model exchangeには共通payload policyを適用する。Cognitive boundaryにmodel-visible Web resultが含まれる場合、その本文は`model.exchange.recorded`へ保存され得る。Web監査eventのmetadata化はmodel exchangeの保存内容を変更しない。

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
