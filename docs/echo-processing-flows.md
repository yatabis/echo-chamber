# E.C.H.O. Hosted 処理フロー

この文書は、Hosted runtime が入力を受け取り、Main、Memory Module、Emotion Module、runtime tool、永続化、Dashboard へ処理をつなぐ流れを示します。

図形は一般的なフローチャート記法に統一しています。

- 角丸: 開始・終了
- 長方形: 処理
- ひし形: 判定
- 円柱: 永続化
- 二重枠: サブプロセスまたはモデル実行
- 破線矢印: 補助・参照・反復経路

## 全体像

```mermaid
flowchart TD
  subgraph hosted[Hosted runtime]
    http([HTTP]) --> router[Worker Router + Access]
    router -->|instance API / control| echo[[Echo Durable Object]]
    router -->|Dashboard / assets| views
    alarm([Durable Object alarm]) --> echo
    discord[/Discord messages/] -. 未読をpoll .-> echo
    gate{起動条件を満たす?}
    thinking[[ThinkingEngine]]
    cognitive[[Memory / Emotion phase]]
    session[[Main agent session / tool loop]]
    ports[Runtime tools + ports]
    effects[/外部作用/]
    state[(State / memory / notes / usage)]
    views[Echo events / Dashboard]
    idle([次回起動まで待機])

    echo -->|alarm / run| gate
    gate -->|No| idle
    gate -->|Yes| thinking --> cognitive --> session
    session -. 各turn境界 .-> cognitive
    session --> ports --> effects
    session --> state
    cognitive --> state
    state --> views
  end
```

主な実装: `apps/cloudflare-workers/src/echo/index.tsx`、`packages/core/src/agent/thinking-engine.ts`、`packages/core/src/agent/session.ts`

## 起動・スケジュール

```mermaid
flowchart TD
  start([Durable Object alarm]) --> budget[ExternalRequestBudget を作成]
  budget --> init[ensureInitialized]
  init --> sleepStart{日次sleep開始時刻<br/>かつIdling?}
  sleepStart -->|Yes| maintenance[wake alarm確定 / re-embedding / cleanup]
  maintenance --> sleeping([Sleeping])
  sleepStart -->|No| sleepEnd{日次sleep終了時刻<br/>かつSleeping?}
  sleepEnd -->|Yes| wake[wake]
  sleepEnd -->|No| state{実行可能なstate?}
  wake --> state

  state -->|No| alarm[(state / 次回 alarm 更新)]
  state -->|Yes| unread[/未読メッセージ確認/]
  unread --> hasUnread{未読あり?}
  hasUnread -->|Yes| thinking[[ThinkingEngine.think]]
  hasUnread -->|No| limits[(usage / next_wake_at 読み込み)]
  limits --> gate{token limit / next wake条件を満たす?}

  gate -->|No| alarm
  gate -->|Yes| thinking
  thinking --> success{ThinkingEngine 成功?}
  success -->|Yes| persist[(usage / next wake 保存)]
  success -->|No| failed[session.failed event]
  persist --> persisted{保存成功?}
  persisted -->|Yes| alarm
  persisted -->|No| runFailed[system.run.failed event]
  failed --> billed{課金済みusageあり?}
  billed -->|Yes| failedUsage[(usage保存を試行)]
  billed -->|No| runFailed
  failedUsage --> runFailed
  runFailed --> alarm
  alarm --> finish([次回 alarmまで待機])
```

主な実装: `apps/cloudflare-workers/src/echo/index.tsx`

## 思考セッション

```mermaid
flowchart TD
  start([mainSystemPrompt / sharedContext]) --> activation[beginActivation]
  activation --> pre[[Memory + Emotion を並列生成]]
  pre --> preGenerated{生成成功?}
  preGenerated -->|No| fail([fail closed])
  preGenerated -->|Yes| preCommit{pre_main commit 成功?}
  preCommit -->|No| fail
  preCommit -->|Yes| handoff[system-owned handoff]

  handoff --> main[[Main model turn]]
  main --> tools[tool call 実行]
  tools --> boundary{思考を終了する?}
  boundary -->|No| next[次 turn の pre_main]
  next --> pre

  boundary -->|Yes| post[[post_main を並列生成]]
  post --> postGenerated{生成成功?}
  postGenerated -->|No| fail
  postGenerated -->|Yes| postCommit{終了時 commit 成功?}
  postCommit -->|No| fail
  postCommit -->|Yes| result([ThinkingEngineResult])
```

主な実装: `packages/core/src/agent/thinking-engine.ts`、`packages/core/src/agent/cognitive-module-orchestrator.ts`、`packages/core/src/agent/session.ts`

## LLMリクエストとコンテキスト

現行Hosted runtimeで通常の思考セッション中に呼び出すLLMは、Main、Memory Module、Emotion Moduleの3系統です。リンとマリーはMainのsystem promptとmodel設定が異なりますが、コンテキストの組み立て方は共通です。

Embeddingとrerankは会話履歴を持つLLM requestではなく、Memory処理の中でtextや候補を個別に渡します。

### 最初のMain turnまで

```mermaid
flowchart TD
  start([思考session開始]) --> builder[初期promptを組み立てる<br/>buildAgentPromptMessages]
  identity[Instance固有のsystem prompt] --> builder
  contracts[実行可能なtool contracts] --> builder
  clock[開始時の現在日時] --> builder

  builder --> mainPrompt[Main専用system prompt<br/>protocol上はdeveloper role]
  builder --> runtime[共有runtime context]
  runtime --> base[初期共有context]
  startup[check_notificationsのcallとresult] --> base

  storedState[(前session終了時の<br/>Memory / Emotion<br/>存在する場合)] --> cognitiveContext[Cognitive用共有context]
  base --> cognitiveContext

  cognitiveContext --> memoryRequest[[Memory pre_main request<br/>専用prompt + Recall schema]]
  cognitiveContext --> emotionRequest[[Emotion pre_main request<br/>専用prompt + Emotion schema]]

  memoryRequest --> commit[検索とEmotionをphase commit]
  emotionRequest --> commit
  commit --> handoff[search_memoryとupdate_emotionの<br/>system-owned tool exchange]

  mainPrompt --> mainRequest[[Main turn 1 request<br/>tools: 全tool contracts]]
  base --> mainRequest
  handoff --> mainRequest
```

Mainへは前session終了時のMemoryとEmotionを直接渡しません。保持されている場合はMemory ModuleとEmotion Moduleがそれらを初期状態として読み、最初の`pre_main`で新たに検索・更新した結果だけをsystem-owned tool exchangeとしてMainへ渡します。初期状態のMemoryは、前の`post_main`でMemory Moduleが生成して保存した1件です。

### 各LLM requestの内容

| Request             | `input`の構成                                                                                        | 会話の継続方法                                                          | `tools`                   | 出力契約                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------- | -------------------------------------------------- |
| Main turn 1         | Main専用prompt → 現在日時 → startup `check_notifications` call/result → 最初のCognitive handoff      | 初回なので過去のprovider応答なし                                        | 実行可能な全tool contract | 自然言語、tool call                                |
| Main turn 2以降     | 直前turnのtool result、必要ならDiscord画像、そのturn用のCognitive handoff                            | Responses APIまたはChat Completions adapterが前turnまでのMain履歴を接続 | 実行可能な全tool contract | 自然言語、tool call                                |
| Memory `pre_main`   | Recall専用prompt → その時点のCognitive共有context全体。初回は前session終了時のMemory / Emotionを含む | 毎回、共有contextの完全なsnapshotを新規requestとして渡す                | なし                      | `{ query }`のstrict JSON Schema                    |
| Emotion `pre_main`  | Emotion専用prompt → Memoryと同一のCognitive共有context snapshot                                      | 毎回、共有contextの完全なsnapshotを新規requestとして渡す                | なし                      | `{ valence, arousal, labels }`のstrict JSON Schema |
| Memory `post_main`  | Store専用prompt → 終了turnまでを含むCognitive共有context全体                                         | 独立した新規request                                                     | なし                      | `{ content, type }`のstrict JSON Schema            |
| Emotion `post_main` | Emotion専用prompt → Memoryと同一の終了時snapshot                                                     | 独立した新規request                                                     | なし                      | `{ valence, arousal, labels }`のstrict JSON Schema |

MemoryとEmotionは同じphaseで同一のimmutable snapshotを読みます。一方のmodel出力をもう一方へ渡すことはなく、両方の検証成功後にruntimeが結果をまとめてcommitします。

### Cognitive共有contextの増え方

```mermaid
flowchart TD
  start([思考session開始]) --> runtime[現在日時]
  runtime --> startup[check_notifications call / result]
  startup --> restored[前session終了時のMemory / Emotionが<br/>存在すれば追加]
  restored --> pre1[[最初のMemory / Emotion request]]
  pre1 --> handoff1[確定済みhandoffを履歴へ追加]
  handoff1 --> main1[Main turn 1の出力を追加]
  main1 --> result1[実tool resultと画像入力を追加]
  result1 --> continue{Main sessionを継続する?}
  continue -->|Yes| preNext[[次のMemory / Emotion request]]
  preNext --> handoffNext[新しい確定済みhandoffを追加]
  handoffNext --> mainNext[次のMain出力とtool resultを追加]
  mainNext --> continue
  continue -->|No| post[[終了時Memory / Emotion request]]
  post --> finish([Memory保存とEmotion更新をcommit])
```

Cognitive共有contextに入るもの:

- 思考session開始時の現在日時
- startup `check_notifications`の擬似tool callとsanitise済みresult
- 前sessionの`post_main`で確定したMemory 1件とEmotion。保持されている場合だけ追加し、Cognitive Moduleだけが読む
- それ以前の`pre_main`で確定し、Mainへ渡したsystem-owned tool exchange
- 各Main turnのassistant出力、tool call、sanitise済みtool result
- `read_chat_messages`が返した画像のうち、上限内でvision inputへ変換したもの

Cognitive共有contextに入らないもの:

- Main固有のidentity / persona system prompt
- Main用tool catalog
- Memory / Emotionそれぞれの専用prompt
- Memory / Emotionの生のmodel出力。validationとdomain commit後のhandoffだけを残す

### Main履歴のprovider別接続

```mermaid
flowchart TD
  request[CoreのMain ModelRequest] --> provider{Main provider}
  provider -->|OpenAI Responses| responses[今回分のinputを送信]
  previous[(直前のresponse ID)] --> responses
  responses --> linked[previous_response_idで<br/>provider側の履歴へ接続]

  provider -->|OpenAI-compatible Chat Completions| chat[今回分をadapterのmessagesへ追加]
  accumulated[(session内の累積messages)] --> chat
  chat --> full[累積messages全体を毎回送信]
```

- OpenAI Responses APIでは、turn 1だけがMain初期input全体を持ちます。turn 2以降は今回分の増分と`previous_response_id`を送り、provider側に保存された直前までの履歴へ接続します。
- OpenAI-compatible Chat Completionsでは、adapterが同じ思考session中のinputとassistant応答を`messages`へ累積し、毎turnその全体を送ります。互換chat templateを優先するため、coreの`developer` messageは`user` roleへ変換されます。
- `echo-session-cache-v1`はChat Completions requestへcache slot情報を加えるruntime最適化であり、上記のコンテキスト内容を選別する仕組みではありません。
- Memory / Emotionは現状OpenAI Responses API固定ですが、`previous_response_id`では接続しません。各phaseで専用promptとCognitive共有context全体を改めて送ります。

主な実装: `packages/core/src/agent/prompt-builder.ts`、`packages/core/src/agent/thinking-engine.ts`、`packages/core/src/agent/cognitive-module-orchestrator.ts`、`packages/core/src/agent/model-cognitive-module.ts`、`packages/core/src/agent/session.ts`、`apps/cloudflare-workers/src/echo/cognitive-modules.ts`、`packages/openai-adapter/src/`

## Tool・外部作用

```mermaid
flowchart TD
  calls([Main tool_calls]) --> lookup[catalog lookup]
  lookup --> registered{登録済み tool?}
  registered -->|No| unknown[unknown tool result]
  registered -->|Yes| limit{read_web_pageの<br/>session上限を超える?}
  limit -->|Yes| limited[制限エラー result]
  limit -->|No| execute[input検証 / handler実行]

  execute --> kind{tool の種類}
  kind -->|Chat| chat[/read / send / reaction / notify/]
  kind -->|Memory| memory[(search / store)]
  kind -->|Note| note[(CRUD)]
  kind -->|Web / Zenn| web[/bounded fetch/]
  kind -->|Thinking| control[think_deeply / finish_thinking<br/>next_wake_at]

  chat --> external{外部 request budget 内?}
  web --> external
  memory --> embedding{OpenAI embeddingを使う?}
  embedding -->|Yes| external
  embedding -->|No| sanitize
  external -->|No| limited
  external -->|Yes| sanitize[model向け結果を整形 / sanitise]
  note --> sanitize
  control --> sanitize
  unknown --> sanitize
  limited --> sanitize
  sanitize --> next([次 Main turn input])
```

主な実装: `packages/core/src/agent/session.ts`、`packages/core/src/agent/runtime-tools/`

## 永続化・ログ

```mermaid
flowchart TD
  session([Thinking session]) --> runtime[(Echo runtime state)]
  session --> usage[(Usage record)]
  session --> notes[(Note storage)]
  session --> memory[(MemorySystem<br/>明示的runtime tool)]
  session --> cognitive[CognitiveModuleDomainStore]
  cognitive --> cognitiveState[(Cognitive domain state)]
  cognitive --> memory

  runtime --> events[EchoEventPort]
  usage --> events
  notes --> events
  memory --> events
  cognitive --> events

  events --> console[/Console sink/]
  events --> archive[(SQLite event archive)]
  events --> discord[/Discord event 通知/]

  archive --> activities[Session activities]
  archive --> analysis[Action analysis]
  maintenance([日次 maintenance]) --> archive
  maintenance --> memory
```

主な実装: `packages/cloudflare-runtime/src/memory-system.ts`、`apps/cloudflare-workers/src/utils/echo-event.ts`

## HTTP・Dashboard

```mermaid
flowchart TD
  browser([Browser request]) --> access{Access検証を通過?}
  access -->|No| forbidden([403 Forbidden])
  access -->|Yes| route{Worker route}

  route -->|static| assets[Dashboard assets]
  route -->|/instances| instances[instance summaries]
  route -->|instance API| detail[status / logs / analysis]

  detail --> cache[(DO read cache)]
  cache --> bounded[(bounded read model)]
  instances --> contracts[[Dashboard contracts]]
  detail --> contracts
  bounded --> contracts
  contracts --> ui[Dashboard UI]
  assets --> ui
  ui --> finish([Fleet / Cognitive state / usage / sessions / actions])
```

主な実装: `apps/cloudflare-workers/src/index.ts`、`packages/contracts/src/dashboard/`、`apps/dashboard/src/App.tsx`
