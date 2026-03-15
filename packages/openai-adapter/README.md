# @echo-chamber/openai-adapter

`@echo-chamber/core` が定義する agent / tool / model port を OpenAI SDK に接続する adapter package です。

## 役割

- OpenAI Responses API 実装
- OpenAI function tool 形式への変換
- OpenAI 固有レスポンス型の吸収

## 依存ルール

- `@echo-chamber/core` に依存する
- Cloudflare runtime や Discord 実装には依存しない

## メモ

- 初期段階では package 境界の雛形のみを提供します。
