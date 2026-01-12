# Repository Restructure Migration Plan

**Target:** Separate user-facing app (web + API) from compute layer (workers) with shared domain logic
**Principle:** frontend/ = serves users directly | backend/ = background computation only
**Current Status:** Well-architected dependencies, but flat structure with sys.path hacks
**Risk Level:** MEDIUM (good architecture, but ~200 file moves)

---

## Architecture Philosophy

### Current Problem
```
backend/
├── main.py          # User-facing API ← WRONG LOCATION
├── api/             # User-facing routes ← WRONG LOCATION
├── workers/         # Background compute ← CORRECT
├── models/          # Shared domain ← UNCLEAR OWNERSHIP
├── repositories/    # Shared data access ← UNCLEAR OWNERSHIP
└── backend/         # Duplicate tree ← DEAD CODE
```

**Issue:** "backend" conflates two concerns:
1. **User-facing HTTP API** (responds to requests, serves frontend)
2. **Background workers** (async computation, no user interaction)

### Target Philosophy

```
frontend/            # "app" - everything that serves users directly
├── web/             # React SPA (current frontend/)
└── api/             # FastAPI server (from backend/main.py + backend/api/)
                     # Responds to HTTP, serves /app routes, proxies requests

backend/             # "compute" - background processing only
└── workers/         # All worker entrypoints and modules
                     # No HTTP server, no user interaction

shared/              # Import-only, no runtime entrypoints
├── python/          # herenews package (domain models/repos/services/reee)
└── contracts/       # OpenAPI spec, DB schema, queue formats
```

**Key Insight:** The API server is part of the "frontend" because:
- It serves the React app (static files)
- It responds to user HTTP requests
- It's the "backend-for-frontend" (BFF pattern)
- Users block waiting for its responses

Workers are true "backend" because:
- No user interaction
- Async/background processing
- Pull jobs from queues
- Write results to database

---

## Target Structure (Aligned with User Model)

```
service_farm/
├── frontend/                  # USER-FACING APP
│   ├── web/                   # React SPA (current frontend/)
│   │   ├── app/               # React components
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── api/                   # FastAPI BFF (from backend/)
│       ├── main.py            # API entrypoint
│       ├── routes/            # API route modules (from backend/api/)
│       │   ├── auth.py
│       │   ├── stories.py
│       │   ├── surfaces.py
│       │   ├── entities.py
│       │   ├── inquiry.py
│       │   └── ...
│       └── middleware/        # CORS, sessions, auth
│
├── backend/                   # NON-USER-FACING COMPUTE
│   └── workers/               # Background processing
│       ├── run_workers.py     # Unified worker manager
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── run.py
│       │   └── worker.py
│       ├── knowledge/
│       │   ├── run.py
│       │   └── worker.py
│       ├── weaver/
│       │   ├── run.py
│       │   └── principled.py
│       ├── canonical/
│       │   ├── run.py
│       │   └── worker.py
│       ├── inquiry/
│       │   └── run.py
│       └── common/
│           └── claim_loader.py
│
├── shared/                    # IMPORT-ONLY (no entrypoints)
│   ├── python/                # Installable package
│   │   ├── setup.py
│   │   └── here/              # "here" namespace (from HereNews)
│   │       ├── __init__.py
│   │       ├── models/        # Domain models (Claim, Surface, Event, etc.)
│   │       ├── repositories/  # Data access (ClaimRepo, SurfaceRepo, etc.)
│   │       ├── services/      # Business logic (Neo4j, JobQueue, etc.)
│   │       ├── reee/          # Epistemic engine
│   │       ├── config/        # Settings, DB connections
│   │       └── utils/         # ID generation, URL utils, etc.
│   └── contracts/             # Schemas & interfaces
│       ├── openapi.json       # Generated from FastAPI
│       ├── schema.sql         # DB schema source of truth
│       └── queue_contracts.md # Redis queue message formats
│
├── docker/                    # Docker configurations
│   ├── frontend.Dockerfile    # Builds web + runs API
│   └── backend.Dockerfile     # Runs workers only
│
├── docker-compose.yml         # Two services: frontend (app) + backend (workers)
│
├── scripts/                   # Maintenance utilities
│   └── maintenance/
│       ├── bootstrap_surfaces.py
│       ├── enqueue_*.py
│       └── ...
│
├── legacy/                    # Archived code
│   ├── backend_backend/       # The duplicate tree
│   └── old_migrations/
│
└── infra/
    └── README.md              # Points to ../infra repo for Postgres/Neo4j/Redis
```

---

## Dependency Flow (Clean Layered Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER REQUESTS                              │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  FRONTEND/API (FastAPI BFF)                                     │
│  - Routes (stories, surfaces, entities, auth, etc.)             │
│  - Middleware (CORS, sessions)                                  │
│  - Static file serving (React SPA)                              │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ imports from ↓
             │
┌────────────▼────────────────────────────────────────────────────┐
│  SHARED/PYTHON (here package)                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Services Layer (business logic)                         │  │
│  │  - Neo4jService, JobQueue, EntityManager                 │  │
│  └────────────┬─────────────────────────────────────────────┘  │
│               │                                                  │
│  ┌────────────▼─────────────────────────────────────────────┐  │
│  │  Repositories Layer (data access)                        │  │
│  │  - ClaimRepository, SurfaceRepository, etc.              │  │
│  └────────────┬─────────────────────────────────────────────┘  │
│               │                                                  │
│  ┌────────────▼─────────────────────────────────────────────┐  │
│  │  Models Layer (domain entities)                          │  │
│  │  - Claim, Surface, Entity, Event, Case                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│               │                                                  │
│  ┌────────────▼─────────────────────────────────────────────┐  │
│  │  REEE (epistemic engine - pure computation)              │  │
│  │  - Kernel, builders, comparators, topology               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ imports from ↑
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  BACKEND/WORKERS (background compute)                           │
│  - Extraction worker (crawl pages)                              │
│  - Knowledge worker (extract entities/claims)                   │
│  - Weaver worker (cluster surfaces → events)                    │
│  - Canonical worker (build stories from events)                 │
│  - Inquiry worker (resolve epistemic inquiries)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ writes to ↓
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  INFRASTRUCTURE (external, not in this repo)                    │
│  - PostgreSQL (pages, users, transactions)                      │
│  - Neo4j (knowledge graph: claims, surfaces, events, entities)  │
│  - Redis (job queues)                                           │
└─────────────────────────────────────────────────────────────────┘
```

**No circular dependencies:**
- frontend/api imports from shared/python ✓
- backend/workers imports from shared/python ✓
- frontend/api does NOT import from backend/workers ✓
- backend/workers does NOT import from frontend/api ✓

---

## Current State Analysis

### Runtime Entrypoints

| Entrypoint | Current Location | Purpose | Target Location |
|------------|------------------|---------|-----------------|
| `main.py` | backend/ | FastAPI server + React serving | **frontend/api/** |
| `run_workers.py` | backend/ | Unified worker manager | **backend/workers/** |
| `run_extraction_worker.py` | backend/ | Extraction worker | **backend/workers/extraction/run.py** |
| `run_knowledge_worker.py` | backend/ | Knowledge worker | **backend/workers/knowledge/run.py** |
| `run_weaver_worker.py` | backend/ | Weaver worker | **backend/workers/weaver/run.py** |
| `run_canonical_worker.py` | backend/ | Canonical worker | **backend/workers/canonical/run.py** |
| `run_inquiry_resolver.py` | backend/ | Inquiry worker | **backend/workers/inquiry/run.py** |

### Shared Modules (Must Move to shared/python/herenews/)

| Module | Current | Lines | Imported By |
|--------|---------|-------|-------------|
| models/ | backend/models/ | ~2000 | API + workers |
| repositories/ | backend/repositories/ | ~3500 | API + workers |
| services/ | backend/services/ | ~4000 | API + workers + repos |
| config/ | backend/config/ | ~200 | Everyone |
| reee/ | backend/reee/ | ~15000 | Workers + some API |
| utils/ | backend/utils/ | ~500 | API + workers |

### Dead Code to Archive

| Path | Size | Status |
|------|------|--------|
| backend/backend/ | ~20MB | Duplicate tree, zero imports |
| backend/test_eu/ | ~15MB | Already deleted in git |
| Various api/*.py | Small | Already deleted (chat.py, etc.) |

### sys.path Hacks (Must Eliminate)

| File | Hack | Reason |
|------|------|--------|
| backend/main.py | `sys.path.insert(0, parent)` | Minimal, OK for now |
| backend/workers/*.py | `sys.path.insert(0, parent.parent)` | Navigate from workers/ to backend/ |
| backend/run_*.py | `sys.path.insert(0, parent)` | Same as main.py |
| backend/enqueue_*.py | `sys.path.insert(0, 'backend/backend')` | **BROKEN**, looks for duplicate |
| backend/reee/tests/*.py | `sys.path.insert(0, parent^3)` | Navigate up from deep test files |

---

## Migration Phases

### Phase 0: Pre-Flight Checks (30 min)
- Verify Docker builds
- Run test suite
- Validate API endpoints
- Test worker processing
- Create git tag for rollback

### Phase 1: Dead Code Cleanup (15 min)
- Archive backend/backend/ → legacy/backend_backend/
- Verify no imports reference it
- Test services still start

### Phase 2: Create Shared Package (1 hour)
- Create shared/python/here/ structure
- Write setup.py for installable package
- Copy (not move) modules: models, repositories, services, config, reee, utils
- Install in dev mode: `pip install -e shared/python`
- Test imports: `from here.models.domain import Claim`
- Update Dockerfile to install shared package

### Phase 3: Migrate Frontend (API) (1.5 hours)
- Create frontend/api/ structure
- Move backend/main.py → frontend/api/main.py
- Move backend/api/ → frontend/api/routes/
- Move backend/middleware/ → frontend/api/middleware/
- Move backend/endpoints*.py → frontend/api/routes/
- Update all imports: `from models.*` → `from here.models.*`
- Remove sys.path hacks
- Update Dockerfile.app → docker/frontend.Dockerfile
- Update docker-compose.yml for new paths
- Test: API starts, health checks pass, frontend loads

### Phase 4: Migrate Backend (Workers) (2 hours)
- Create backend/workers/ submodule structure
- Move run_*.py → backend/workers/*/run.py
- Move workers/*.py → backend/workers/*/worker.py
- Update imports: backend.* → here.*
- Remove ALL sys.path hacks
- Update run_workers.py to use new paths
- Create docker/backend.Dockerfile (workers only)
- Update docker-compose.yml for workers service
- Test: Workers start, process jobs from queue

### Phase 5: Contracts & Docs (30 min)
- Generate OpenAPI schema → shared/contracts/openapi.json
- Optional: Generate TypeScript types from OpenAPI
- Document queue contracts → shared/contracts/queue_contracts.md
- Consolidate DB schema → shared/contracts/schema.sql
- Create infra/README.md (points to ../infra)

### Phase 6: Final Cleanup (30 min)
- Remove old shared modules from backend/
- Move utility scripts → scripts/maintenance/
- Update root README with new structure
- Update docker-compose.yml with final paths
- Create .dockerignore

### Phase 7: Integration Testing (1 hour)
- Build all services from scratch
- Run full test suite
- Test end-to-end flows (submit job → worker processes → query API)
- 24-hour soak test

**Total Time:** ~7-8 hours (split across 2 days recommended)

---

## Stability Test Matrix

### Test 1: Package Installation
```bash
cd shared/python
pip install -e .
python -c "from here.models.domain import Claim; print('✓')"
python -c "from here.services.neo4j_service import Neo4jService; print('✓')"
python -c "from here.reee import Engine; print('✓')"
```

### Test 2: Frontend API
```bash
docker-compose up -d frontend
curl -f http://localhost:7272/health
curl -f http://localhost:7272/api/health
curl -f http://localhost:7272/api/stories?limit=1
curl -f http://localhost:7272/api/surfaces?limit=1
curl -f http://localhost:7272/  # React app loads
```

### Test 3: Backend Workers
```bash
docker-compose up -d backend
docker-compose logs backend | grep "Started extraction"
docker-compose logs backend | grep "Started knowledge"
docker-compose logs backend | grep "Started weaver"

# Submit test job
docker exec herenews-frontend python -c "
import asyncio
from here.services.job_queue import JobQueue

async def test():
    queue = await JobQueue.create()
    await queue.push('queue:extraction:high', {'page_id': 'test', 'url': 'https://test.com'})
    print('✓ Job queued')

asyncio.run(test())
"

# Verify worker picked it up
timeout 30 bash -c 'until docker-compose logs backend | grep -q "Processing.*test"; do sleep 1; done'
echo "✓ Worker processed"
```

### Test 4: No sys.path Hacks
```bash
# Should return ZERO results
grep -r "sys\.path\.insert" frontend/ backend/ shared/
```

### Test 5: Import Correctness
```bash
# All shared imports use here.*
grep -r "from models\." frontend/api backend/workers --include="*.py" && echo "❌ FAIL" || echo "✓ PASS"
grep -r "from repositories\." frontend/api backend/workers --include="*.py" && echo "❌ FAIL" || echo "✓ PASS"
grep -r "from services\." frontend/api backend/workers --include="*.py" && echo "❌ FAIL" || echo "✓ PASS"

# Should use here.*
grep -r "from here\." frontend/api backend/workers --include="*.py" | wc -l  # Should be ~100+
```

### Test 6: Unit Tests
```bash
docker exec herenews-frontend pytest /shared/python/here/reee/tests/ -v
```

### Test 7: End-to-End Flow
```bash
# 1. Submit URL to extraction queue
# 2. Extraction worker crawls it
# 3. Knowledge worker extracts entities/claims
# 4. Weaver clusters into surfaces
# 5. Canonical worker builds stories
# 6. Query API for the story

# Full test:
docker exec herenews-frontend python -c "
import asyncio
from here.services.job_queue import JobQueue

async def e2e_test():
    queue = await JobQueue.create()
    await queue.push('queue:extraction:high', {
        'page_id': 'e2e-test',
        'url': 'https://bbc.com/news/world'
    })
    print('✓ E2E test job submitted')

asyncio.run(e2e_test())
"

# Monitor logs for full pipeline
docker-compose logs -f | grep "e2e-test"
```

---

## Docker Configuration

### docker/frontend.Dockerfile
```dockerfile
# Multi-stage: Build React web app, then run FastAPI server

# Stage 1: Build React frontend
FROM node:18-alpine AS web-builder
WORKDIR /frontend/web
COPY frontend/web/package*.json ./
RUN npm install
COPY frontend/web/ ./
RUN npm run build

# Stage 2: Python API server with built frontend
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install shared package
COPY shared/python /shared/python
RUN pip install -e /shared/python

# Copy API code
COPY frontend/api /app/frontend/api

# Copy built React app
COPY --from=web-builder /output /static

# Serve API
WORKDIR /app/frontend/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7272"]
```

### docker/backend.Dockerfile
```dockerfile
# Workers only (no web build, no API server)

FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install shared package
COPY shared/python /shared/python
RUN pip install -e /shared/python

# Install Playwright (for extraction worker)
RUN playwright install chromium --with-deps

# Copy worker code only
COPY backend/workers /app/backend/workers

# Run unified worker manager
CMD ["python", "-m", "backend.workers.run_workers", "--no-api", "--workers", "2"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  # Frontend: React app + FastAPI server (serves users)
  frontend:
    build:
      context: .
      dockerfile: docker/frontend.Dockerfile
    container_name: herenews-frontend
    network_mode: host
    env_file: .env
    volumes:
      - ./frontend/api:/app/frontend/api
      - ./shared/python:/shared/python
    command: uvicorn main:app --host 0.0.0.0 --port 7272 --reload
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7272/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Backend: Workers (background processing)
  backend:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    container_name: herenews-backend
    network_mode: host
    env_file: .env
    volumes:
      - ./backend/workers:/app/backend/workers
      - ./shared/python:/shared/python
    command: python -m backend.workers.run_workers --no-api --workers 2
    restart: unless-stopped
    depends_on:
      - frontend
```

---

## Rollback Procedures

### Emergency Rollback (Complete Failure)
```bash
docker-compose down
git reset --hard pre-restructure-$(date +%Y%m%d)
docker-compose build --no-cache
docker-compose up -d
curl http://localhost:7272/health
```

### Partial Rollback (Keep Shared Package, Revert Moves)
```bash
# Keep shared/python, revert file moves
git revert <bad-commit-range>
docker-compose build
docker-compose up -d
```

### Recovery Checklist
- [ ] Services start
- [ ] Health endpoints return 200
- [ ] Workers process jobs
- [ ] Frontend loads
- [ ] Document failure reason

---

## Success Criteria

### Technical
- [ ] Zero `sys.path` manipulations
- [ ] All imports use `from here.*`
- [ ] Docker builds < 5 min
- [ ] Services start < 30 sec
- [ ] All tests pass
- [ ] No circular dependencies

### Structural
- [ ] frontend/ contains ONLY user-facing code (web + API)
- [ ] backend/ contains ONLY background workers
- [ ] shared/ contains ONLY importable modules (no entrypoints)
- [ ] Clear dependency flow: frontend → shared ← backend

### Operational
- [ ] Developer onboarding: `pip install -e shared/python && docker-compose up`
- [ ] Can modify shared/ and see changes in both frontend and backend
- [ ] Debugging: Clear logs per service (frontend vs backend)
- [ ] Documentation matches actual structure

---

## Next Steps After Migration

1. **CI/CD Updates**
   - Build docker/frontend.Dockerfile and docker/backend.Dockerfile separately
   - Run tests in shared/python/herenews/
   - Deploy frontend and backend independently

2. **Publish Shared Package (Optional)**
   - Push "here" package to private PyPI
   - Version it properly (semver)
   - Frontend and backend can depend on specific versions

3. **Developer Docs**
   - Update CONTRIBUTING.md
   - Add "Where to put new code" decision tree
   - Document package development workflow

4. **Monitoring**
   - Alert on sys.path usage (should be zero)
   - Track import errors as critical metric
   - Monitor service startup times

---

## Appendix: Complete File Mapping

### Frontend (user-facing)
**FROM backend/** → **TO frontend/api/**
- main.py
- api/ → routes/
- middleware/
- endpoints.py → routes/artifacts.py
- endpoints_rogue.py → routes/rogue.py

### Backend (compute)
**FROM backend/** → **TO backend/workers/**
- run_workers.py → run_workers.py
- run_extraction_worker.py → extraction/run.py
- workers/extraction_worker.py → extraction/worker.py
- run_knowledge_worker.py → knowledge/run.py
- workers/knowledge_worker.py → knowledge/worker.py
- run_weaver_worker.py → weaver/run.py
- workers/principled_weaver.py → weaver/principled.py
- run_canonical_worker.py → canonical/run.py
- workers/canonical_worker.py → canonical/worker.py
- run_inquiry_resolver.py → inquiry/run.py
- workers/claim_loader.py → common/claim_loader.py

### Shared (import-only)
**FROM backend/** → **TO shared/python/here/**
- models/ → models/
- repositories/ → repositories/
- services/ → services/
- config/ → config/
- reee/ → reee/
- utils/ → utils/

### Scripts (maintenance)
**FROM backend/** → **TO scripts/maintenance/**
- analyze_event_structure.py
- bootstrap_surfaces.py
- cleanup_neo4j_pages.py
- enqueue_*.py
- queue_*.py
- reprocess_*.py
- reset_*.py
- regenerate_*.py

### Legacy (archived)
**FROM backend/** → **TO legacy/**
- backend/ → backend_backend/
- (deleted files already in git history)

---

**END OF MIGRATION PLAN**

*Last Updated: 2026-01-07*
*Status: Ready for Review*
