# @echo-chamber/discord-adapter

Discord REST API への接続をまとめる adapter package です。

## 役割

- Discord REST 呼び出し
- チャット通知取得
- メッセージ送信とリアクション追加

## 依存ルール

- `@echo-chamber/core` に依存する
- Cloudflare runtime には依存しない

## メモ

- 初期段階では package 境界の雛形のみを提供します。
