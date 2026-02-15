# Project Structure - E.C.H.O. Chamber

## ルートディレクトリ構成

```
echo-chamber/
├── src/                    # メインソースコード
├── test/                   # テスト設定とヘルパー
├── public/                 # 静的アセット（Assets binding）
├── html/                   # ビルドされたHTML（開発用）
├── coverage/               # テストカバレッジレポート
├── .wrangler/             # Wrangler 一時ファイル・キャッシュ
├── .kiro/                 # Kiro Spec-Driven Development
│   └── steering/          # ステアリングドキュメント
├── .claude/               # Claude Code 設定
│   ├── commands/          # カスタムスラッシュコマンド
│   └── settings.json      # Claude設定
├── package.json           # 依存関係とスクリプト定義
├── tsconfig.json          # TypeScript設定
├── wrangler.jsonc         # Cloudflare Workers設定
├── vitest.config.ts       # Vitest テスト設定
├── eslint.config.js       # ESLint設定
├── CLAUDE.md              # プロジェクト仕様書
└── AGENTS.md              # エージェント設計書
```

## メインソースディレクトリ (`src/`)

### トップレベル構成

```
src/
├── index.ts               # Workers エントリーポイント
├── echo/                  # Echo Durable Object 関連
├── discord/               # Discord API 統合
├── llm/                   # 言語モデル統合
├── utils/                 # 共通ユーティリティ
└── types/                 # 型定義
```

### Echo Durable Object (`src/echo/`)

```
echo/
├── index.tsx              # Echo Durable Object メインクラス
├── constants.ts           # Echo固有の定数定義
├── types.ts               # Echo関連の型定義
├── usage.ts               # Usage tracking ロジック
├── usage.test.ts          # Usage機能のテスト
├── thinking-engine/       # 思考エンジンモジュール
├── emotion-engine/        # 感情エンジンモジュール
├── memory-system/         # 記憶システムモジュール
└── view/                  # UI コンポーネント
    ├── components/        # 再利用可能コンポーネント
    │   ├── Card.tsx       # カード UI コンポーネント
    │   ├── Layout.tsx     # レイアウトコンポーネント
    │   ├── TaskList.tsx   # タスク一覧表示
    │   ├── KnowledgeList.tsx  # 知識ベース表示
    │   └── UsageChart.tsx # 使用量グラフ表示
    └── pages/             # ページレベルコンポーネント
        └── StatusPage.tsx # ステータスページ
```

### Discord統合 (`src/discord/`)

```
discord/
├── index.ts               # Discord API ラッパー
├── index.test.ts          # Discord機能テスト
└── api.ts                 # 低レベルAPI通信
```

### LLM統合 (`src/llm/`)

```
llm/
├── openai/                # OpenAI API 統合
│   ├── client.ts          # OpenAI クライアントとUsage管理
│   ├── client.test.ts     # クライアントテスト
│   ├── embedding.ts       # テキスト埋め込み生成
│   └── functions/         # OpenAI Function Calling
│       ├── index.ts       # 関数定義の集約
│       ├── index.test.ts  # 統合テスト
│       ├── chat.ts        # チャット機能（Discord連携）
│       ├── chat.test.ts
│       ├── memory.ts      # 記憶管理（semantic/episodic）
│       ├── memory.test.ts
│       ├── think.ts       # 深層思考機能
│       ├── think.test.ts
│       ├── finish.ts      # 思考完了通知
│       └── finish.test.ts
└── prompts/               # プロンプトテンプレート
    ├── rin.ts             # Rin用システムプロンプト
    └── marie.ts           # Marie用システムプロンプト
```

### 共通ユーティリティ (`src/utils/`)

```
utils/
├── datetime.ts            # 日時処理ユーティリティ
├── datetime.test.ts       # 日時処理テスト
├── error.ts               # エラーハンドリング
├── logger.ts              # ログ機能
├── vector.ts              # ベクトル計算（コサイン類似度など）
├── vector.test.ts         # ベクトル計算テスト
└── thinking-stream.ts     # 思考ストリーム処理
```

### 設定 (`src/config/`)

```
config/
└── echo-registry.ts       # Echoインスタンス設定・レジストリ
```

### 型定義 (`src/types/`)

```
types/
├── logger.ts              # ログ関連の型定義
└── echo-config.ts         # Echoインスタンス設定の型定義
```

## テスト構成 (`test/`)

```
test/
├── setup.ts               # Vitest セットアップ
├── env.d.ts               # テスト環境型定義
├── tsconfig.json          # テスト用TypeScript設定
├── helpers/               # テストヘルパー
│   └── discord.ts         # Discord API モック
└── mocks/                 # モックオブジェクト
    ├── tool.ts            # OpenAI Tool モック
    └── thinking-stream.ts # ThinkingStream モック
```

## コードファイル命名規則

### ファイル名パターン

- **Implementation**: `index.ts`, `module.ts` - 実装ファイル
- **Tests**: `*.test.ts` - テストファイル（co-location pattern）
- **Types**: `types.ts`, `*.types.ts` - 型定義ファイル
- **Constants**: `constants.ts` - 定数定義
- **Components**: `ComponentName.tsx` - PascalCase for React components

### ディレクトリ名規則

- **kebab-case**: `thinking-engine/`, `memory-system/` - ハイフン区切り
- **camelCase**: 使用しない
- **snake_case**: 使用しない

## Import組織パターン

### Import順序

1. **External libraries**: `cloudflare:workers`, `hono`, `openai` etc.
2. **Internal modules**: 相対パスでの内部モジュール
3. **Types**: `import type` による型のみのインポート

### Import例

```typescript
// 外部ライブラリ
import { DurableObject } from 'cloudflare:workers';
import { Hono } from 'hono';

// 内部モジュール
import { getUnreadMessageCount } from '../discord';
import { formatDatetime } from '../utils/datetime';
import { createLogger } from '../utils/logger';

// 型のみ
import type { EchoState, Knowledge, Task } from './types';
import type { Logger } from '../utils/logger';
```

## 主要なアーキテクチャパターン

### 1. Co-location Testing Pattern

実装ファイルと同じディレクトリにテストファイルを配置：

```
src/echo/usage.ts
src/echo/usage.test.ts
```

### 2. Durable Object Pattern

```typescript
export class Echo extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // 初期化
  }

  async fetch(request: Request): Promise<Response> {
    // HTTP エンドポイント処理
  }

  async alarm(): Promise<void> {
    // 定期実行処理
  }
}
```

### 3. Hono Router Pattern

```typescript
this.router = new Hono()
  .basePath('/:id')
  .get('/', async (c) => {
    /* GET handler */
  })
  .post('/action', async (c) => {
    /* POST handler */
  });
```

### 4. Function Calling Pattern

```typescript
export const functionDefinition = {
  name: 'function_name',
  description: 'Function description',
  parameters: {
    type: 'object',
    properties: {
      /* parameters */
    },
    required: [
      /* required fields */
    ],
  },
} as const;

export async function functionImplementation(args: Args): Promise<Result> {
  // 実装
}
```

### 5. Usage Tracking Pattern

```typescript
// OpenAI API 呼び出し
const usage = await openaiClient.call(params);

// 使用量を累積
await echoInstance.updateUsage(convertUsage(usage));
```

## Component階層

### Layout Component Structure

```
StatusPage                 # ページレベルコンポーネント
├── Layout                 # レイアウトラッパー
│   ├── Card               # 個別セクション
│   │   ├── TaskList       # タスク表示
│   │   ├── KnowledgeList  # 知識表示
│   │   └── UsageChart     # 使用量グラフ
│   └── Card
└── ...
```

### Styling Convention

```typescript
// インライン CSS-in-JS パターン
return (
  <div style={{
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem'
  }}>
    {content}
  </div>
);
```

## 設定ファイル構成

### TypeScript設定

- **Root**: `tsconfig.json` - メイン設定
- **Test**: `test/tsconfig.json` - テスト専用設定

### Quality Tools設定

- **ESLint**: `eslint.config.js` - Flat config形式
- **Prettier**: `package.json` 内でのスクリプト設定
- **Vitest**: `vitest.config.ts` - テスト設定

### Cloudflare設定

- **Wrangler**: `wrangler.jsonc` - Workers設定とバインディング
- **Types**: `worker-configuration.d.ts` - 自動生成型定義

## 開発ワークフロー原則

### 1. ファイル作成の優先順位

1. **編集を優先**: 既存ファイルの編集を新規作成より優先
2. **Co-location**: テストは実装ファイルと同じディレクトリに配置
3. **型安全性**: TypeScriptの厳格な設定を活用

### 2. モジュール境界

- **Echo**: Durable Object関連のコアロジック
- **Discord**: 外部サービス統合
- **LLM**: AI機能とプロンプト管理
- **Utils**: 汎用ユーティリティ

### 3. 依存関係の方向

```
Echo (core) ← Discord, LLM
Utils → すべてのモジュール
Types → すべてのモジュール
```

---

**最終更新**: 2026-02-16
**バージョン**: 1.1
**ステータス**: Always Included - 全ファイル組織において参照必須
