# Repository Guidelines

## Project Structure & Modules

- `src/`: TypeScript sources. Entrypoint `src/index.ts` (Cloudflare Worker). Key areas: `discord/`, `echo/` (Durable Object with memory-system, thinking-engine, emotion-engine, view components using `hono/jsx`), `llm/` (OpenAI client, embedding, prompts, function calling tools), `config/`, `utils/`, and `types/`.
- `test/`: Test setup and helpers (`helpers/`, `mocks/`). Test files are co-located with source in `src/**/*.test.ts`.
- `public/`: Static assets (optional via Wrangler assets binding).
- Config: `wrangler.jsonc` (Durable Object `Echo`, KV `ECHO_KV`), `tsconfig.json`, `eslint.config.js`, `vitest.config.ts`.

## Build, Test, and Development

- Install deps: `pnpm install`
- Local dev (Workers): `pnpm dev` or `pnpm start` — runs Wrangler dev with typegen.
- Typegen only: `pnpm cf-typegen`
- Test (watch/UI): `pnpm test`, `pnpm test:watch`, `pnpm test:ui`
- Coverage: `pnpm test:coverage` — reports to `coverage/`.
- Lint/Format/Types: `pnpm lint`, `pnpm format`, `pnpm typecheck`, or all: `pnpm check`
- Deploy (Cloudflare): `pnpm deploy`

## Coding Style & Naming

- Language: TypeScript (strict). JSX via `hono/jsx`.
- Formatting: Prettier (2-space indent, 80 cols, semicolons, single quotes). Run `pnpm format`.
- Linting: ESLint with TypeScript rules. Enforced: explicit return types, type-only imports, import ordering, no unused vars (prefix unused with `_`). Run `pnpm lint`.
- Names: files `kebab-case.ts`; React-style components `PascalCase.tsx`; tests `*.test.ts` (co-located with source).

## Testing Guidelines

- Framework: Vitest with `@cloudflare/vitest-pool-workers` (Workers/miniflare pool). Globals enabled.
- Location: Tests co-located with source in `src/**/*.test.ts`. Helpers in `test/helpers/`, mocks in `test/mocks/`.
- Coverage: Istanbul provider includes `src/**/*.ts` (excludes logger/discord API). Aim to keep or raise existing coverage.
- Run locally: `pnpm test` (watch) or `pnpm test:coverage` for reports.

## Commits & Pull Requests

- Commits: concise, present tense; English or Japanese acceptable. Prefer scope in subject (e.g., `echo: …`, `llm: …`). Link issues in body when relevant.
- PRs: include purpose, summary of changes, screenshots/JSON samples for API/view changes, reproduction steps, and linked issues. Ensure `pnpm check` and tests pass.

## Security & Config

- Secrets: use Wrangler secrets in cloud; for local dev use `.dev.vars` only. Examples: `wrangler secret put OPENAI_API_KEY`, `wrangler secret put DISCORD_BOT_TOKEN`.
- Avoid `setTimeout/Interval` in Workers; use Durable Object alarms per ESLint rules.
