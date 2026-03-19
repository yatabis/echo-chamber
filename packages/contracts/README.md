# @echo-chamber/contracts

Worker と Dashboard のあいだで共有する API contract を置く package です。

## 役割

- HTTP レスポンス DTO
- request / response schema
- UI 表現のためではなく API 境界のための型

## 依存ルール

- `@echo-chamber/core` の agent domain には依存しない
- runtime 実装には依存しない

## メモ

- `dashboard` 向け DTO と表示用 util をこの package に置きます。
