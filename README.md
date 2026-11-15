# E.C.H.O. Chamber

E.C.H.O. Chamber を構成する repo をまとめるための repo です。ここには submodule と、この README だけを置きます。
コードと文書は、それぞれの repo が持ちます。

## 構成

| path | repo | 役割 |
| --- | --- | --- |
| `runtime/` | [echo-chamber-lab/echo-chamber-runtime](https://github.com/echo-chamber-lab/echo-chamber-runtime) | Echo の本体。思考セッション、Discord、常駐のループ |
| `infer/` | [echo-chamber-lab/echo-chamber-infer](https://github.com/echo-chamber-lab/echo-chamber-infer) | 推論サーバー（OpenAI 互換 API） |

どちらも private repo です。

## submodule の参照先

- 参照先のコミットは、その組み合わせで動くことを確かめたときだけ更新します。
- 開発は、各 repo を単独で clone して行います。この repo の中では開発しません。
- 起動スクリプトや launchd の設定は、この repo には置きません。

現在の参照先は、この repo を作り直した時点（2026-11-15）の各 repo の `main` の最新です。

## 旧実装

TypeScript と Cloudflare Workers による旧実装は、tag [`legacy-final`](https://github.com/yatabis/echo-chamber/tree/legacy-final) に残しています。
新しい本体は、旧実装の移植ではなく、一から作り直したものです。
