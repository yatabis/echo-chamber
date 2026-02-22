# Claude Code Spec-Driven Development

Kiro-style Spec Driven Development implementation using claude code slash commands, hooks and agents.

## Project Context

### Paths

- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`
- Commands: `.claude/commands/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications

- Check `.kiro/specs/` for active specifications
- Use `/kiro:spec-status [feature-name]` to check progress

## Development Guidelines

- Think in English, but generate responses in Japanese (思考は英語、回答の生成は日本語で行うように)

## Workflow

### Phase 0: Steering (Optional)

`/kiro:steering` - Create/update steering documents
`/kiro:steering-custom` - Create custom steering for specialized contexts

**Note**: Optional for new features or small additions. Can proceed directly to spec-init.

### Phase 1: Specification Creation

1. `/kiro:spec-init [detailed description]` - Initialize spec with detailed project description
2. `/kiro:spec-requirements [feature]` - Generate requirements document
3. `/kiro:spec-design [feature]` - Interactive: "requirements.mdをレビューしましたか？ [y/N]"
4. `/kiro:spec-tasks [feature]` - Interactive: Confirms both requirements and design review

### Phase 2: Progress Tracking

`/kiro:spec-status [feature]` - Check current progress and phases

## Development Rules

1. **Consider steering**: Run `/kiro:steering` before major development (optional for new features)
2. **Follow 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Approval required**: Each phase requires human review (interactive prompt or manual)
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Steering Configuration

### Current Steering Files

Managed by `/kiro:steering` command. Updates here reflect command changes.

### Active Steering Files

- `product.md`: Always included - Product context and business objectives
- `tech.md`: Always included - Technology stack and architectural decisions
- `structure.md`: Always included - File organization and code patterns

### Custom Steering Files

<!-- Added by /kiro:steering-custom command -->
<!-- Format:
- `filename.md`: Mode - Pattern(s) - Description
  Mode: Always|Conditional|Manual
  Pattern: File patterns for Conditional mode
-->

### Inclusion Modes

- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., `"*.test.js"`)
- **Manual**: Reference with `@filename.md` syntax

## Development Commands

- `pnpm dev` - Start Worker dev server with type generation (`apps/cloudflare-workers`)
- `pnpm start` - Start Worker dev server without type generation
- `pnpm cf-typegen` - Generate TypeScript types from `apps/cloudflare-workers/wrangler.jsonc`
- `pnpm deploy` - Deploy Worker from root script
- `pnpm dashboard:build` - Build dashboard into Worker assets
- `pnpm --filter @echo-chamber/dashboard dev` - Run dashboard dev server only

## Quality Assurance Commands

**CRITICAL**: Always run these commands after code changes to ensure zero errors before task completion.

- `pnpm lint:check` - ESLint with zero warnings tolerance (strict mode)
- `pnpm format:check` - Prettier format checking
- `pnpm typecheck` - TypeScript type checking
- `pnpm lint` - ESLint with auto-fix (use for fixing, not checking)
- `pnpm format` - Prettier with auto-format (use for fixing, not checking)

### Quality Check Workflow

```bash
# After any code changes, run these in sequence:
pnpm lint:check
pnpm typecheck
pnpm format:check
```

**Never complete tasks with unresolved errors from these commands.**

## Automated Testing Strategy

This project implements **t-wada式TDD (Test-Driven Development)** using the latest Cloudflare Workers testing environment.

### Testing Commands

**CRITICAL**: Always run tests after code changes to ensure functionality is preserved.

- `pnpm test:run` - Run all workspace test suites once (`core` + `app-cloudflare-workers`)
- `pnpm test:coverage` - Generate coverage report (`@echo-chamber/app-cloudflare-workers`)
- `pnpm --filter @echo-chamber/core test:run` - Run core package tests only
- `pnpm --filter @echo-chamber/app-cloudflare-workers test:run` - Run Cloudflare app tests only

### Claude Code Testing Constraints

**IMPORTANT**: Claude Code should avoid interactive commands due to execution limitations.

**✅ Claude Code Should Use:**

- `pnpm test:run` - Single execution with clear pass/fail results
- `pnpm test:coverage` - Coverage measurement and reporting
- `pnpm lint:check` - Code quality verification
- `pnpm typecheck` - Type safety confirmation

**❌ Claude Code Should NOT Use:**

- `pnpm tdd` - Interactive UI requires human interaction
- `pnpm test:watch` - Watch mode not suitable for automated execution
- `pnpm test:ui` - Browser-based UI cannot be operated by Claude Code

**Recommended Claude Code Workflow:**

1. Write test code
2. Execute `pnpm test:run` immediately for verification
3. Fix any failing tests
4. Re-run `pnpm test:run` to confirm green state
5. Proceed to next test iteration

This approach ensures continuous quality verification while respecting Claude Code's execution constraints.

### Testing Architecture

**Technology Stack (2025 Best Practices):**

- **@cloudflare/vitest-pool-workers** - Executes tests in `workerd` runtime environment
- **Vitest** - Fast, modern test runner with TypeScript support
- **@vitest/ui** - Interactive testing interface for TDD workflows

**Key Benefits:**

- Tests run in actual Cloudflare Workers runtime (`workerd`), not Node.js emulation
- Automatic isolation of Durable Objects and KV storage between tests
- Real-time feedback for Red-Green-Refactor TDD cycles

### Test Structure

**Co-location Pattern:**

- Core logic tests are co-located in `packages/core/src/**/*.test.ts`
- Cloudflare-dependent tests are co-located in `apps/cloudflare-workers/src/**/*.test.ts`
- Shared test helpers/mocks are split by boundary:
  - `packages/core/test/**`
  - `apps/cloudflare-workers/test/**`

**Discovery & Execution:**

- Use patterns:
  - `packages/core/src/**/*.test.ts`
  - `apps/cloudflare-workers/src/**/*.test.ts`
- Run all tests: `pnpm test:run`
- Run per package:
  - `pnpm --filter @echo-chamber/core test:run`
  - `pnpm --filter @echo-chamber/app-cloudflare-workers test:run`

### Testing Patterns

**Durable Objects Testing:**

- **Integration Tests**: Use `SELF.fetch()` to test HTTP endpoints and full request flows
- **Unit Tests**: Use `runInDurableObject()` helper for direct instance testing

**External Dependencies:**

- **Discord API**: Fully mocked with realistic response patterns
- **OpenAI API**: Comprehensive mock including usage tracking
- **Environment Variables**: Isolated test environment setup

### TDD Workflow

**CRITICAL**: このプロジェクトでは厳格なt-wada式TDDプロセスを遵守する。

#### TodoWriteツールによる強制管理

仕様変更・新機能開発時は**必ず**以下のタスクテンプレートをTodoWriteで設定：

```
- 🔴 Red: 失敗するテストを書く
- ✅ テスト実行して失敗確認 (pnpm test:run)
- 🟢 Green: 最小限実装でテストを通す
- ✅ テスト実行して成功確認 (pnpm test:run)
- 🔵 Refactor: 必要に応じてリファクタリング
```

#### 必須プロセス

1. **🔴 Red**: 失敗するテストを書く
   - 実装前に**必ずテスト実行**して赤を確認
   - 一度に一つの変更のみ

2. **🟢 Green**: 最小限の実装でテストを通す
   - テスト実行して緑を確認
   - 過度な実装は禁止（最小限で止める）

3. **🔵 Refactor**: コード品質向上（必要時のみ）
   - テストを壊さない範囲で改善
   - 品質チェック実行必須

#### 絶対禁止事項

❌ **テストと実装の同時変更**
❌ **Redフェーズのスキップ**
❌ **テスト実行せずに次ステップへ進む**
❌ **一度に複数の変更**

### Characterization Tests

For existing legacy code, we use **Characterization Tests** to:

- Document current behavior before refactoring
- Prevent regressions during code improvements
- Enable safe architectural changes

**Next Steps:**

- Create Characterization Tests for core Echo Durable Object logic
- Expand integration test coverage for Discord/OpenAI interactions
- Establish CI/CD pipeline with automated testing

## Architecture Overview

This project is a monorepo Cloudflare Workers application built with TypeScript and Hono.

### Workspace Components

- **Worker App** (`apps/cloudflare-workers/`) - Entry point, Wrangler config, static assets
- **Dashboard App** (`apps/dashboard/`) - React + Vite frontend, built into Worker assets
- **Core Package** (`packages/core/`) - Cloudflare-independent logic and shared types
- **Cloudflare Runtime** (`apps/cloudflare-workers/src/`) - Cloudflare-dependent runtime implementation

### Runtime Components

- **Worker Entrypoint** (`apps/cloudflare-workers/src/index.ts`) - Routes `/`, `/dashboard/*`, and `/:instanceId/*`
- **Echo Durable Object** (`apps/cloudflare-workers/src/echo/index.tsx`) - Main agent lifecycle and APIs
- **Memory / Thinking / Emotion Engines** (`apps/cloudflare-workers/src/echo/*`)
- **OpenAI Client & Tool System** (`apps/cloudflare-workers/src/llm/openai/*`)
- **Discord API & Shared DTO/Utils** (`packages/core/src/discord/*`, `packages/core/src/dashboard/*`, `packages/core/src/utils/*`)
- **Instance Registry** (`apps/cloudflare-workers/src/config/echo-registry.ts`)

### Cloudflare Resources

- **Durable Object**: `Echo` bound as `ECHO`
- **KV Namespace**: `ECHO_KV`
- **Workers AI**: `AI`
- **Static Assets Binding**: `ASSETS`
- **Config File**: `apps/cloudflare-workers/wrangler.jsonc`

### Key Patterns

- `apps/cloudflare-workers` contains both wiring and Cloudflare runtime implementation
- `apps/cloudflare-workers` depends on `packages/core`, never vice versa
- Dashboard build output is emitted to `apps/cloudflare-workers/public/dashboard`
- Durable Object alarms drive periodic execution
- OpenAI usage is tracked and accumulated per day

### Entry Points

- `GET /` - Health check
- `GET /dashboard` - Redirect to `/dashboard/`
- `GET /dashboard/*` - Static dashboard assets + SPA fallback
- `ALL /:instanceId/*` - Forward to corresponding Echo Durable Object

**Echo Durable Object endpoints** (`/:instanceId/*`):

- `GET /:instanceId/` - Status page
- `GET /:instanceId/json` - Status/memory/note/usage JSON
- `POST /:instanceId/wake` - Force wake
- `POST /:instanceId/sleep` - Force sleep
- `POST /:instanceId/run` - Manual run (`ENVIRONMENT=local` only)
- `GET /:instanceId/usage` - Usage statistics

Always run `pnpm cf-typegen` when changing `apps/cloudflare-workers/wrangler.jsonc`.

## OpenAI Usage Management

This project tracks OpenAI usage to control token consumption across recursive tool-calling flows.

### Architecture

**OpenAIClient** (`apps/cloudflare-workers/src/llm/openai/client.ts`)

- `call()` returns cumulative `ResponseUsage` across recursive calls
- Logs a warning when response usage is missing

**Echo Durable Object** (`apps/cloudflare-workers/src/echo/index.tsx`)

- Accumulates daily usage in Durable Object storage
- Applies dynamic token limits
- Stores usage by date key (`YYYY-MM-DD`)
