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

- `pnpm dev` - Start development server with type generation (combines `wrangler types && wrangler dev`)
- `pnpm cf-typegen` - Generate TypeScript types from Wrangler configuration
- `pnpm test:run path/to/specific.test.ts` - Run a specific test file

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

- `pnpm test:run` - Run tests once and exit
- `pnpm test:coverage` - Generate test coverage report

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

**Co-location Pattern:** Tests are co-located with source code using the `.test.ts` suffix. Each module's tests are placed alongside its implementation files within the `src/` directory.

**Discovery & Execution:**

- Use `Glob` tool with pattern `src/**/*.test.ts` to find all test files
- Run directory tests: `pnpm test:run src/path/to/module/`
- Run specific file: `pnpm test:run src/path/to/file.test.ts`
- Run all tests: `pnpm test:run`

**Key Benefits of Co-location:**

- Tests are immediately discoverable next to their implementation
- Module boundaries are clear and testable units are obvious
- Refactoring moves tests and code together naturally

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

This is a Cloudflare Workers application built with Hono framework and TypeScript. The architecture consists of:

### Core Components

- **Main Worker** (`src/index.ts`) - Hono application serving as the entry point with basic routing
- **Echo Durable Object** (`src/echo/index.tsx`) - Implements the core Echo functionality as a Durable Object with RPC methods
- **Memory System** (`src/echo/memory-system/`) - Semantic and episodic memory with embedding-based retrieval
- **Thinking Engine** (`src/echo/thinking-engine/`) - Deep thinking and reflection module
- **Emotion Engine** (`src/echo/emotion-engine/`) - Emotional state management
- **OpenAI Client** (`src/llm/openai/client.ts`) - GPT-5 Responses API client with usage tracking and recursive function call handling
- **Discord Integration** (`src/discord/`) - Discord API wrapper for chat functionality
- **Tool System** (`src/llm/openai/functions/`) - OpenAI function calling tools: chat (Discord messaging), memory (semantic/episodic storage), think (deep thinking), finish (thinking completion)
- **Embedding Client** (`src/llm/openai/embedding.ts`) - Text embedding generation for semantic search
- **Prompt Templates** (`src/llm/prompts/`) - Character-specific system prompts (rin.ts, marie.ts)
- **Instance Registry** (`src/config/echo-registry.ts`) - Echo instance configuration and registration

### Cloudflare Resources

- **Durable Object**: `Echo` class bound as `ECHO` - provides persistent state and RPC capabilities
- **KV Namespace**: `ECHO_KV` - key-value storage for the application
- **SQLite**: Configured for the Echo Durable Object via migrations

### Key Patterns

- The main worker routes requests to Durable Object instances using `c.env.ECHO.idFromName()`
- Durable Objects expose RPC methods that can be called directly from the worker
- Type definitions are auto-generated by Wrangler and stored in `worker-configuration.d.ts`
- **Usage Management**: OpenAI API calls are tracked with dynamic hourly limits and daily caps
- **Function Calling**: Extensive use of OpenAI's function calling for Discord interactions, task management, and context storage
- **Alarm-based Scheduling**: Durable Objects use alarms for periodic execution (1-minute intervals)

### Entry Points

- Root path `/` returns a simple status message
- All paths under `/rin/*` are routed to an Echo Durable Object instance named "Rin"

**Debug Endpoints** (local only):

- `GET /rin/` - Status page (HTML)
- `GET /rin/json` - Status data (JSON format)
- `POST /rin/wake` - Wake up Echo instance
- `POST /rin/sleep` - Put Echo to sleep
- `POST /rin/run` - Force execution cycle
- `POST /rin/reset` - Clear memory and tasks
- `GET /rin/usage` - Token usage statistics

Always run `wrangler types` when making changes to `wrangler.jsonc` to keep TypeScript definitions up to date.

## OpenAI Usage Management

This application includes comprehensive OpenAI API usage tracking and management to prevent unexpected token consumption.

### Architecture

**OpenAIClient** (`src/llm/openai/client.ts`)

- `call()` method returns cumulative `ResponseUsage` across all recursive function calls
- Automatically logs warning when API response lacks usage information

**Echo Durable Object** (`src/echo/index.ts`)

- Accumulates daily usage statistics in Durable Object storage
- Implements dynamic token limits based on time-proportional allocation
- Stores usage data by date: `{ "2025-07-28": ResponseUsage, ... }`
