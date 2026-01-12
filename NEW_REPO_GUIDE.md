# New Repo Guide (App/API Greenfield)

This repo already contains a clear target boundary in `RESTRUCTURE_MIGRATION_PLAN.md`: **user-facing app** vs **background compute** with **shared contracts**.

If the intent is to start a new repo (rather than migrating this one), use that same boundary as a repo boundary.

---

## Recommended split

### `herenews-app` (greenfield)

- Owns all user-facing code:
  - `frontend/web/` (React)
  - `frontend/api/` (FastAPI BFF: serves the web app + exposes `/api/*`)
- Owns `shared/contracts/`:
  - OpenAPI (`openapi.json`/`openapi.yaml`)
  - DB schema snapshot (if needed by product/dev tooling)
  - Queue message formats (if frontend needs to display job states)

### `herenews-compute` (this repo, trimmed or a new repo)

- Owns background compute only:
  - `backend/workers/`
- Owns the shared Python domain package:
  - `shared/python/` → published as `here` (private index) or consumed via git/submodule.

---

## Day-1 engineering checklist (min friction)

- Make **OpenAPI the contract** (not `API.md`): generate types for the frontend and validate on CI.
- Keep a strict dependency rule:
  - `frontend/api` imports `here.*` (shared package) but **never** imports worker modules.
  - workers import `here.*` but **never** import BFF modules.
- Build thin vertical slices from `product/APP_API_V2_USER_STORIES.md` (Auth status → Inquiry list/detail → contribute/stake).

---

## What to reuse from this repo

- Product spec + flows: `product/MVP1-SPECIFICATION.md`
- UX/system story: `product/USER-STORY-TOPOLOGY-INQUIRY.md`
- Current endpoint inventory (for mapping into OpenAPI): `API.md`
- Architecture boundary and test matrix patterns: `RESTRUCTURE_MIGRATION_PLAN.md`

