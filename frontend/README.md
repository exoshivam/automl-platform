# AutoML Platform: frontend

React 19 + TypeScript (strict) + Vite, TanStack Router/Query, Zustand (UI state only), Tailwind v4 with shadcn-style Radix components, react-hook-form + Zod. Recharts, react-konva and Uppy are installed for the data/labeling/training screens.

## Run
```sh
npm install              # .npmrc sets legacy-peer-deps (an optional-peer clash inside @hookform/resolvers)
cp .env.example .env     # optional
npm run dev              # http://127.0.0.1:5180 (strictPort)
```
In this repo the service runner owns the shared 5180 dev server.

| Script | |
|---|---|
| `npm run typecheck` | `tsc --noEmit` (TypeScript 7) |
| `npm test` | Vitest unit tests |
| `npm run e2e` | Playwright against :5180 (`npx playwright install chromium` once) |
| `npm run gen:api` | `../backend/openapi.json` → `src/api/schema.d.ts` (runs openapi-typescript with TS 5 via npx, since TS 7 has no JS compiler API) |

## API and mocks
- `VITE_API_URL` (default `/api/v1`, relative). In dev, Vite proxies `/api` (incl. WebSocket) to FastAPI on `127.0.0.1:8010`, so calls are same-origin: no CORS, first-party refresh cookie.
- Access token in memory only; refresh/logout use the httpOnly cookie with `credentials: 'include'` + `X-CSRF: 1`. Any 401 triggers one refresh, then a retry (`src/lib/api.ts`).
- Errors are RFC 9457 problem+json (`ApiError`, with `fieldErrors()` for 422s and `retryAfter` for 429s).
- **MSW mocks** (dev only) cover only groups the backend hasn't built yet (`NOT_BUILT_YET` in `src/lib/env.ts`); auth, orgs, projects, templates and from-description use the real API. Mocked screens show a **Demo data** badge. Override per browser with `localStorage.setItem('automl.mocks','all'|'none'|'data,training')` (or `VITE_MOCKS`).
  - With `automl.mocks=all` (offline demo): login `demo@automl.dev` / `demo1234`; mock data persists in localStorage.
  - Registering sends a verification "email" listed at `/_dev/mail`; the check-email screen shows a dev link to it.
  - Put `[llm-down]` in a task description to get the 503 `llm_unavailable` fallback.
- Generated types are used for auth, user and workspace shapes (`src/api/types.ts`); project/data types are hand-written from PRD §11 until those OpenAPI groups land.

## Layout
`src/features/{auth,shell,projects}` pages, `src/components/ui` primitives, `src/api` types + query hooks, `src/mocks` MSW handlers, `src/router.tsx` routes.
