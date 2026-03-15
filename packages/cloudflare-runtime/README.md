# @echo-chamber/cloudflare-runtime

Cloudflare Workers / Durable Objects / KV / SQLite / Alarm を使った runtime 実装を置く package です。

## 役割

- Cloudflare bindings の吸収
- Durable Object storage / SQLite 実装
- scheduler / logger / repository 実装
- Worker からの composition

## 依存ルール

- `@echo-chamber/core` に依存する
- OpenAI / Discord の adapter 実装には直接依存しない

## メモ

- 初期段階では package 境界の雛形のみを提供します。
