# Echo Registry Boundary

## 概要

`echo-registry` 周りの責務分離は完了している。現在は「instance catalogue」「Cloudflare runtime binding 解決」「それを利用する consumer」が明示的に分かれている。

## 現在の境界

### `core`

`core` は instance catalogue として、identity / persona に加えて instance ごとの Main LLM、Cognitive Module、token limit の非secret default を持つ。secretとCloudflare bindingはworker、provider SDK clientはadapter / runtime packageが所有する。

- `EchoInstanceId`
- `EmbeddingConfig`
- instance definition catalogue

実装:

- [packages/core/src/types/echo-config.ts](../packages/core/src/types/echo-config.ts)
- [packages/core/src/echo/instance-definitions.ts](../packages/core/src/echo/instance-definitions.ts)

`EchoInstanceDefinition` の責務は次のとおり。

- `id`
- `name`
- `systemPrompt`
- `mainLlm`
- `cognitiveModules`
- `tokenLimits`

### worker

worker は Cloudflare 依存の runtime binding を解決する。

- Discord bot token の取得
- 固定 chat channel 定義の保持
- thinking channel id の取得
- thinking channel 用 KV key の管理
- Cognitive Module の provider / API、API key、instance env override、global fallback の解決
- `Env` / `KVNamespace` を読む resolver

実装:

- [apps/cloudflare-workers/src/config/echo-runtime-bindings.ts](../apps/cloudflare-workers/src/config/echo-runtime-bindings.ts)
- [apps/cloudflare-workers/src/config/cognitive-module-config.ts](../apps/cloudflare-workers/src/config/cognitive-module-config.ts)

`EchoRuntimeBindings` の責務は次のとおり。

- `discordBotToken`
- `chatChannels`
- `thinkingChannelId`
- `embeddingConfig`

Cognitive Moduleの非secretなmodel / reasoning defaultは`EchoInstanceDefinition.cognitiveModules`に置き、Memory / Emotionがinstance内で共有する。Worker composition rootのresolverは、instance固有env、definition、global env、Worker fallbackの順に値を解決する。

### consumer

`Echo` は definition と runtime bindings を別 field で保持する。

- `ThinkingEngine` には `systemPrompt`
- `main-llm-config` には `mainLlm` の default、`token-limit-config` には `tokenLimits` の default
- `cognitive-module-config` には `cognitiveModules` の default と instance identity
- `DiscordEchoEventPort` には token と `thinkingChannelId`
- `tool-context` には chat 用の binding だけを渡す

chat channels は worker 側の固定定義として管理し、`thinkingChannelId` だけを KV から解決する。

実装:

- [apps/cloudflare-workers/src/echo/index.tsx](../apps/cloudflare-workers/src/echo/index.tsx)
- [apps/cloudflare-workers/src/echo/tool-context.ts](../apps/cloudflare-workers/src/echo/tool-context.ts)
- [apps/cloudflare-workers/src/config/main-llm-config.ts](../apps/cloudflare-workers/src/config/main-llm-config.ts)
- [apps/cloudflare-workers/src/config/token-limit-config.ts](../apps/cloudflare-workers/src/config/token-limit-config.ts)

## 削除したもの

refactor 前に存在した「全部入り DTO」は削除済みである。

- `EchoInstanceConfig`
- `apps/cloudflare-workers/src/config/echo-registry.ts`

これにより、domain config と worker runtime 依存の境界が型の上でも分離された。

## 補足

`embeddingConfig` は引き続き worker 側に置いている。現在の `EmbeddingConfig` は `workersai` を含み、Cloudflare 固有の provider 選択を反映しているためである。将来的に provider 非依存の policy に持ち上げられるなら、その時点で `core` へ寄せる余地がある。
