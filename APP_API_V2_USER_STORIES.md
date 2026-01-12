# App/API V2 — User Stories (Greenfield Frontend + BFF)

This backlog is intentionally aligned to `RESTRUCTURE_MIGRATION_PLAN.md`:

- **User-facing app** = `frontend/web` (React) + `frontend/api` (FastAPI BFF).
- **Compute** = background workers only.
- **Contracts** live in `shared/contracts/` (OpenAPI, DB schema, queue formats) and drive frontend development.

It also maps to the existing API surface documented in `API.md` (events, surfaces, inquiry, auth, user, entities, pages).

---

## 0) Repo Strategy (Starting a New Repo)

If we start greenfield, keep the *architecture boundary* from the migration plan, but make it *repo boundary*:

### Option A (Recommended): Two repos + versioned shared package

- `herenews-app` (new): `frontend/web` + `frontend/api` + `shared/contracts`
- `herenews-compute` (current or new): `backend/workers` + `shared/python` (the `here` package)
- Dependency between repos is via:
  - HTTP (app talks to compute through DB/queues), and/or
  - a versioned Python package (`here`) published to a private index (mirrors “Publish Shared Package (Optional)” in `RESTRUCTURE_MIGRATION_PLAN.md`).

### Option B: Monorepo (current plan, but executed cleanly)

Keep everything in one repo and implement the target structure directly (`frontend/web`, `frontend/api`, `backend/workers`, `shared/python`, `shared/contracts`).

---

## 1) Contracts-First Deliverables (Unblocks Frontend From Day 1)

**C1. OpenAPI as source of truth**
- As a frontend engineer, I can generate a typed client from OpenAPI so I can build UI without guessing payloads.
- Acceptance criteria:
  - `shared/contracts/openapi.json` (or `openapi.yaml`) exists and is versioned.
  - Generated TypeScript client/types compile in `frontend/web`.

**C2. Error model + auth model**
- As a product/engineering team, we have consistent errors and auth/session semantics so UI states are predictable.
- Acceptance criteria:
  - Standard error schema (e.g., `{ code, message, details? }`) defined and used across endpoints.
  - Auth mechanism documented (cookie session vs bearer token) and represented in OpenAPI security schemes.

---

## 2) Epics + User Stories

Each story includes a suggested API mapping based on `API.md`. The exact routes should be formalized in OpenAPI.

### Epic A — Authentication & Session

**A1. Guest browsing**
- As a user, I can browse public content without logging in so I can evaluate the product before committing.
- API: `GET /auth/status`
- Acceptance criteria:
  - App renders in “Guest” state when unauthenticated.
  - Calls that require auth degrade gracefully with a login prompt.

**A2. Login/logout**
- As a user, I can log in via Google OAuth and log out so my account, credits, and actions persist.
- API: `GET /auth/login`, `GET /auth/logout`, `GET /auth/me`
- Acceptance criteria:
  - Redirect-based login works end-to-end; after login, user identity appears in header.
  - Logout clears session and returns to Guest state.

### Epic B — Home / Discovery (Events feed)

**B1. Events list with filters**
- As a user, I can browse events with key signals (status/tension/coverage) so I can decide what to investigate.
- API: `GET /events`
- Acceptance criteria:
  - Supports pagination (limit/offset) and basic filters (status, min_coherence, etc.).
  - Loading/empty/error states are designed and implemented.

**B2. “Needs you” view (tensions + gaps)**
- As a user, I can see which events have contradictions or missing corroboration so I can help where it matters.
- API: `GET /event/{event_id}/tensions`, `GET /event/{event_id}/epistemic`
- Acceptance criteria:
  - UI highlights tension types (e.g., `single_source_only`, `unresolved_conflict`, `need_primary_source`).
  - “Call to action” links into the relevant event facet/inquiry.

### Epic C — Event Detail (Narrative + Evidence)

**C1. Event detail page**
- As a user, I can open an event and see narrative, key metrics, and evidence so I can understand “what happened” and “how we know”.
- API: `GET /event/{event_id}`
- Acceptance criteria:
  - Renders event + citations, shows relationships (children/parent), and handles missing pieces gracefully.

**C2. Topology visualization**
- As a power user, I can view event topology so I can understand claim structure and contradictions.
- API: `GET /event/{event_id}/topology`
- Acceptance criteria:
  - UI renders topology graph with interactive highlighting for contradictions/update chains.
  - Includes a “fallback” view if graph rendering fails (table/list).

### Epic D — Surfaces (Claim clusters / facets)

**D1. Surfaces browse**
- As a user, I can browse surfaces so I can find coherent claim clusters and drill down.
- API: `GET /surfaces`, `GET /surfaces/stats`
- Acceptance criteria:
  - List view supports `min_claims` and pagination.
  - Stats module renders top anchors and summary metrics.

**D2. Surface detail**
- As a user, I can open a surface to see its claims and internal relations so I can evaluate the evidence cluster.
- API: `GET /surfaces/{surface_id}`
- Acceptance criteria:
  - Claims list supports sorting (recency/source diversity/confidence if available).
  - Relations render as either graph or structured list.

### Epic E — Inquiry (MVP1)

This epic mirrors `product/MVP1-SPECIFICATION.md` but frames it as a greenfield build.

**E1. Inquiry listing**
- As a user, I can browse inquiries by status and sort by bounty/uncertainty/activity so I can find where to contribute.
- API: `GET /inquiry`
- Acceptance criteria:
  - Carousels or sections for Open/Resolved/Top Bounties/Contested.
  - Search by title/keywords; URL query is shareable.

**E2. Inquiry detail**
- As a user, I can view an inquiry’s current best answer, confidence, tasks, and contributions so I can decide what to do next.
- API: `GET /inquiry/{inquiry_id}`, `GET /inquiry/{inquiry_id}/trace`
- Acceptance criteria:
  - Clearly shows belief state + confidence/entropy + “why” (trace).
  - Includes evidence gaps/tasks panel.

**E3. Contribute evidence**
- As a user, I can submit a contribution with text and optional source URL so I can add evidence to an inquiry.
- API: `POST /inquiry/{inquiry_id}/contribute`
- Acceptance criteria:
  - Validations (length, URL format) and clear error messages.
  - Contribution appears in feed after submit.

**E4. Stake bounty**
- As a user, I can stake credits to increase an inquiry bounty so I can incentivize resolution.
- API: `POST /inquiry/{inquiry_id}/stake`, `GET /user/credits`
- Acceptance criteria:
  - Credits balance updates immediately after staking.
  - Prevents staking more than available balance.

### Epic F — Entities, Claims, Pages (Explainers & Provenance)

**F1. Entity page**
- As a user, I can view an entity’s description, related claims, and related events so I can follow the graph of meaning.
- API: `GET /entity/{entity_id}`
- Acceptance criteria:
  - Shows enrichment fields when present (Wikidata label/description/image/geo).

**F2. Claim page**
- As a user, I can open a claim and see its source and linked entities so I can audit provenance.
- API: `GET /claim/{claim_id}`
- Acceptance criteria:
  - Clearly displays the source (URL, publisher) and extracted entities.

**F3. Page browser**
- As a user, I can browse and open ingested pages so I can inspect extracted claims in context.
- API: `GET /pages`, `GET /page/{page_id}`
- Acceptance criteria:
  - List supports status filters (e.g., `semantic_complete`) and pagination.

### Epic G — User / Credits / History

**G1. Profile + credits**
- As a user, I can view my profile and credits balance so I understand my ability to participate.
- API: `GET /user/profile`, `GET /user/credits`
- Acceptance criteria:
  - Header shows current credit balance when authenticated.

**G2. My contributions/stakes**
- As a user, I can see my contributions and stakes so I can track impact and rewards over time.
- API: `GET /user/contributions`, `GET /user/stakes`, `GET /user/transactions`
- Acceptance criteria:
  - Basic filters by time/status and a simple summary (“total staked”, “total impact”).

---

## 3) Thin Vertical Slices (Suggested Build Order)

This sequence keeps the “BFF + OpenAPI + web UI” loop tight.

1. **Auth status + guest browsing** (A1) + basic shell UI
2. **Inquiry list/detail** (E1, E2) using existing endpoints
3. **Contribute + stake** (E3, E4) with real credits checks
4. **Events list + event detail** (B1, C1) to connect inquiry ↔ topology
5. **Topology view** (C2) + surfaces browse/detail (D1, D2)
6. **Entities/claims/pages** (F1–F3) for provenance depth
7. **User history** (G1–G2) for retention loop

