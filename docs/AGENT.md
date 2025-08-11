# ScoreGenius Project

ScoreGenius is a full-stack web application with TypeScript frontend and Node.js backend.
The core functionality lives in the `src/` folder, with separate client (`client/`)
and server (`server/`) components.

## Build & Commands

- Typecheck and lint everything: `npm run check`
- Fix linting/formatting: `npm run check:fix`
- Run tests: `npm test -- --run --no-color`
- Run single test: `npm test -- --run src/file.test.ts`
- Start development server: `npm run dev`
- Build for production: `npm run build`
- Preview production build: `npm run preview`

### Development Environment

- Frontend dev server: http://localhost:5173
- Backend dev server: http://localhost:8080
- Database: Supabase (managed PostgreSQL)
- Redis cache: not used

## Code Style

- TypeScript: Strict mode with exactOptionalPropertyTypes, noUncheckedIndexedAccess
- Tabs for indentation (2 spaces for YAML/JSON/MD)
- Single quotes, no semicolons, trailing commas
- Use JSDoc docstrings for documenting TypeScript definitions, not `//` comments
- 100 character line limit
- Imports: Use consistent-type-imports
- Use descriptive variable/function names
- In CamelCase names, use "URL" (not "Url"), "API" (not "Api"), "ID" (not "Id")
- Prefer functional programming patterns
- Use TypeScript interfaces for public APIs
- NEVER use `@ts-expect-error` or `@ts-ignore` to suppress type errors

## Testing

- Vitest for unit testing
- Testing Library for component tests
- Playwright for E2E tests
- When writing tests, do it one test case at a time
- Use `expect(VALUE).toXyz(...)` instead of storing in variables
- Omit "should" from test names (e.g., `it("validates input")` not `it("should validate input")`)
- Test files: `*.test.ts` or `*.spec.ts`
- Mock external dependencies appropriately

## Architecture

- Frontend: React with TypeScript
- Backend: Express.js (ESM JavaScript)
- Database: Supabase (PostgreSQL via Supabase JS client)
- State management: React Query + React Context
- Styling: Tailwind CSS
- Build tool: Vite
- Package manager: npm

## Security

- Use appropriate data types that limit exposure of sensitive information
- Never commit secrets or API keys to repository
- Use environment variables for sensitive data
- Validate all user inputs on both client and server
- Use HTTPS in production
- Regular dependency updates
- Follow principle of least privilege

## Git Workflow

- ALWAYS run `npm run check` before committing
- Fix linting errors with `npm run check:fix`
- Run `npm run build` to verify typecheck passes
- NEVER use `git push --force` on the main branch
- Use `git push --force-with-lease` for feature branches if needed
- Always verify current branch before force operations

## Configuration

When adding new configuration options, update all relevant places:

1. Environment variables in `.env.example`
2. Configuration schemas in `src/config/`
3. Documentation in README.md

All configuration keys use consistent naming and MUST be documented.
