# @echo-chamber/discord-adapter

`@echo-chamber/core` が要求する chat / notification / thought-log port を Discord に接続する adapter package です。

## 役割

- Discord REST 呼び出し
- チャット通知取得
- メッセージ送信とリアクション追加
- thought log 送信

## 依存ルール

- `@echo-chamber/core` に依存する
- Cloudflare runtime には依存しない

## メモ

- 初期段階では package 境界の雛形のみを提供します。
