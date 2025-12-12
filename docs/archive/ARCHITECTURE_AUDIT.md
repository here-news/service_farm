# 🔍 Complete Architecture Audit

## Executive Summary

**Status**: System is in **transitional state** with significant inconsistencies between Story (OLD) and Event (NEW) terminology.

**Critical Issues**:
1. ❌ **API route conflict**: Two different routers mounted at `/api/events`
2. ❌ **Terminology mismatch**: Neo4j uses "Story", backend uses "Event", frontend uses "Story"
3. ❌ **Frontend mismatch**: Frontend is Lit Element (not React), uses old Story API
4. ⚠️ **Container confusion**: No 7272 port, frontend container serves legacy HTML
5. ⚠️ **Obsolete code**: backend/api/story.py and services/neo4j_client.py still referenced

---

## Current Architecture Flow

```
User Browser
    ↓
[Port 8080] frontend container (legacy HTML tools)
    OR
[Port 8000] /app → static/ (React/Lit built assets - NOT BUILT YET)
    ↓
[Port 8000] main.py (FastAPI)
    ↓
    ├─→ /api/auth → backend/api/auth.py → middleware/ → repositories/user_repository
    ├─→ /api/comments → backend/api/comments.py → repositories/comment_repository
    ├─→ /api/chat → backend/api/chat.py → repositories/chat_session_repository
    ├─→ /api/stories → backend/api/story.py → services/neo4j_client (OBSOLETE!)
    ├─→ /api/events → endpoints_events.py → repositories/event_repository ✅
    ├─→ /api/events → backend/api/events.py → event submissions ❌ CONFLICT!
    ├─→ /api/map → backend/api/map.py
    └─→ /api/coherence → backend/api/coherence.py
    ↓
Data Layer:
    ├─→ PostgreSQL (users, comments, chat_sessions, pages, embeddings)
    ├─→ Neo4j (Events/Stories, Claims, Entities, relationships)
    └─→ Redis (job queues)
    ↓
Workers (background):
    ├─→ extraction_worker.py (3x) - Extract pages
    ├─→ knowledge_worker.py (2x) - Extract entities/claims
    └─→ event_worker_neo4j.py (1x) - Form events
```

---

## 1. Container Configuration

### docker-compose.yml

| Service | Container | Port | Purpose | Status |
|---------|-----------|------|---------|--------|
| **api** | herenews-api | 8000 | FastAPI backend | ✅ **Should rename to "app"** |
| **frontend** | herenews-frontend | 8080 | Legacy HTML tools | ⚠️ **Obsolete, remove** |
| postgres | herenews-postgres | 5432 | PostgreSQL + pgvector | ✅ Good |
| neo4j | herenews-neo4j | 7474, 7687 | Neo4j graph DB | ✅ Good |
| redis | herenews-redis | 6379 | Job queues | ✅ Good |
| worker-extraction-{1,2,3} | herenews-worker-extraction-* | - | Page extraction | ✅ Good |
| worker-knowledge-{1,2} | herenews-worker-knowledge-* | - | Entity/claim extraction | ✅ Good |
| worker-event | herenews-worker-event | - | Event formation | ✅ Good |

**Issues**:
- ❌ **Port 7272 doesn't exist** (user mentioned it, but not configured)
- ❌ **frontend container is obsolete** - serves legacy HTML, not React app
- ⚠️ **api container** should be renamed to "app" (serves both frontend + API)

**Recommendations**:
1. Remove frontend container (frontend/ now in same repo)
2. Rename "api" service to "app"
3. Configure main port as 7272 or keep 8000 (user preference)
4. api/app container should serve frontend/static/ at /app route

---

## 2. API Routes Audit

### ✅ Working Routes (Unified Backend)

| Endpoint | Router | Purpose | Status |
|----------|--------|---------|--------|
| `/api/auth/login` | backend/api/auth.py | Google OAuth | ✅ Good |
| `/api/auth/callback` | backend/api/auth.py | OAuth callback | ✅ Good |
| `/api/auth/status` | backend/api/auth.py | Check auth | ✅ Good |
| `/api/comments/*` | backend/api/comments.py | Comment CRUD | ✅ Good |
| `/api/chat/unlock` | backend/api/chat.py | Unlock chat | ✅ Good |
| `/api/chat/message` | backend/api/chat.py | Send message | ✅ Good |
| `/api/preview` | backend/api/preview.py | URL preview | ✅ Good |
| `/api/map/entities` | backend/api/map.py | Hot entities | ✅ Good |
| `/api/map/locations` | backend/api/map.py | Hot locations | ✅ Good |
| `/api/coherence/*` | backend/api/coherence.py | Coherence score | ✅ Good |
| `/api/extraction/*` | backend/api/extraction.py | Manual extraction | ✅ Good |
| `/api/event/*` | backend/api/event_page.py | Event pages | ✅ Good |

### ❌ Problematic Routes

| Endpoint | Router | Issue | Fix |
|----------|--------|-------|-----|
| `/api/stories` | backend/api/story.py | OLD terminology, uses neo4j_client | **Remove or update** |
| `/api/events` | endpoints_events.py | Event listing (CORRECT) | ✅ Keep |
| `/api/events` | backend/api/events.py | Event submissions (CONFLICT!) | **Rename to `/api/submissions`** |
| `/api/v2/artifacts` | endpoints.py | Artifact submission | Merge with submissions? |

**Critical Issue**:
```python
# main.py has TWO routers at /api/events!
app.include_router(backend_events_router, tags=["Events - Backend"])  # endpoints_events.py
app.include_router(app_events.router, tags=["Events - Community"])    # backend/api/events.py
```

**One will override the other!** This is a bug.

---

## 3. Frontend Audit

### Current Structure

```
frontend/                           # Webapp frontend (Lit Element)
├── src/
│   ├── main.ts                     # Root: Uses "Story" terminology
│   ├── components/
│   │   ├── story-detail.ts         # Story detail view
│   │   ├── story-chat-sidebar.ts   # Chat UI
│   │   ├── comment-thread.ts       # Comments UI
│   │   └── news-card.ts            # Story card
│   └── utils/
│       └── storyUrl.ts             # Story URL helper
├── package.json                    # Lists React (unused!)
└── vite.config.ts                  # Build to ../static/

frontend-legacy/                    # Legacy HTML tools
├── event.html                      # Event viewer (standalone)
├── timeline.html                   # Timeline viz
├── map.html                        # Map interface
└── index.html                      # Dashboard
```

**Issues**:
- ❌ **Frontend uses Lit Element**, NOT React (despite package.json)
- ❌ **All frontend components use "Story"** terminology
- ❌ **Frontend calls `/api/stories`** (obsolete endpoint)
- ⚠️ **Not built yet** - no static/ directory exists
- ⚠️ **Package.json lies** - lists React dependencies but code uses Lit

**API Calls Frontend Makes**:
```typescript
// In main.ts and components:
fetch('/api/stories')                    // ❌ OLD endpoint
fetch('/api/stories/{story_id}')         // ❌ OLD endpoint
fetch('/api/comments/story/{story_id}')  // ⚠️ Should be event_id
fetch('/api/chat/unlock')                // ✅ Good
fetch('/api/auth/status')                // ✅ Good
```

---

## 4. Terminology Consistency Audit

### Current State (INCONSISTENT!)

| Layer | Terminology | ID Format | Status |
|-------|-------------|-----------|--------|
| **Neo4j** | `Story` nodes | `story_id` | ❌ OLD |
| **Backend Models** | `Event` class | `ev_xxxxxxxx` | ✅ NEW |
| **Backend Repos** | `EventRepository` | `ev_xxxxxxxx` | ✅ NEW |
| **Backend API (old)** | `story.py` router | `story_id` | ❌ OLD |
| **Backend API (new)** | `events.py` router | `event_id` | ✅ NEW |
| **Frontend** | `Story` interface | `story_id` | ❌ OLD |
| **Database (PostgreSQL)** | `events` table | `event_id` (str) | ⚠️ MIXED |

### What Needs to Change

1. **Neo4j**: Rename `Story` nodes → `Event` nodes
2. **Frontend**: Rename all `Story` → `Event` throughout
3. **API**: Remove or migrate `/api/stories` → `/api/events`
4. **Database**: Already uses `events` table (good!)

---

## 5. Data Model Flow (CORRECT UNDERSTANDING)

### Domain Models (backend/models/domain/)
```
Event (ev_xxx)        → Root entity, represents a news event
  ├─ Page (pg_xxx)    → Article/webpage about the event
  │   └─ Claim (cl_xxx) → Factual statement from page
  │       └─ Entity (en_xxx) → Person/Org/Location mentioned
  │
  ├─ User (UUID)      → User account (Google OAuth)
  ├─ Comment (cm_xxx) → User comment on event
  └─ ChatSession (cs_xxx) → AI chat about event
```

### Repositories → Data Storage

| Repository | PostgreSQL | Neo4j | Purpose |
|------------|------------|-------|---------|
| EventRepository | events table | Event nodes | Event metadata |
| PageRepository | pages, embeddings | Page nodes | Article content |
| ClaimRepository | - | Claim nodes | Factual claims |
| EntityRepository | - | Person/Org/Location | Named entities |
| UserRepository | users | - | User accounts |
| CommentRepository | comments | - | User comments |
| ChatSessionRepository | chat_sessions, messages | - | AI chat |

**Neo4j as Source of Truth**: Knowledge graph (Events, Pages, Claims, Entities, relationships)
**PostgreSQL**: Content storage (page text), vectors (embeddings), user data

---

## 6. What User Wants (Corrected Understanding)

### Target Architecture

```
User → http://localhost:7272
    ↓
[Container: herenews-app] (renamed from "api")
    ↓
    ├─→ /app → Homepage (React/Lit frontend)
    │   ├─ Event feed (list of live events, not stories!)
    │   ├─ Google OAuth login
    │   ├─ Comment threads per event
    │   ├─ Map page (hot entities + hot locations)
    │   └─ Graph visualization
    │
    └─→ /api/* → Backend API
        ├─ /api/events → List events (✅ keep)
        ├─ /api/submissions → Submit new event (rename from /api/events)
        ├─ /api/auth/* → Authentication
        ├─ /api/comments/* → Comments
        ├─ /api/chat/* → AI chat
        ├─ /api/map/* → Map data (entities, locations)
        └─ /api/coherence/* → Event scoring
    ↓
Repositories → Models → PostgreSQL + Neo4j
    ↓
Workers (separate containers, shared models)
```

---

## 7. Critical Issues Summary

### 🔴 **CRITICAL** (Must Fix Immediately)

1. **API Route Conflict**: `/api/events` mounted twice
   - Fix: Rename backend/api/events.py → backend/api/submissions.py
   - Change prefix to `/api/submissions`

2. **Obsolete Code Still Active**: backend/api/story.py uses neo4j_client
   - Fix: Remove story.py or migrate to use EventRepository

3. **Neo4j Schema Mismatch**: Uses "Story" nodes, not "Event"
   - Fix: Migration script to rename Story → Event in Neo4j

### 🟡 **HIGH PRIORITY** (Should Fix Soon)

4. **Frontend Uses Old API**: Calls `/api/stories`
   - Fix: Update frontend to call `/api/events`

5. **Frontend Terminology**: All components use "Story"
   - Fix: Rename Story → Event throughout frontend

6. **Container Naming**: "api" container misleading
   - Fix: Rename to "app" (serves both frontend + API)

7. **Frontend Not Built**: No static/ directory
   - Fix: Run `cd frontend && npm run build`

### 🟢 **LOW PRIORITY** (Nice to Have)

8. **Frontend Container Obsolete**: Serves legacy HTML
   - Fix: Remove from docker-compose.yml

9. **Port Confusion**: User mentioned 7272, currently 8000
   - Fix: Decide on standard port (7272 or 8000)

10. **Package.json Mismatch**: Lists React, uses Lit
    - Fix: Clean up dependencies or migrate to React

---

## 8. Recommended Cleanup Steps

### Phase 1: Fix API Conflicts (URGENT)

```bash
# 1. Rename event submissions router
mv backend/api/events.py backend/api/submissions.py

# 2. Update router prefix
sed -i 's|prefix="/api/events"|prefix="/api/submissions"|g' backend/api/submissions.py

# 3. Update main.py import
sed -i 's|events as app_events|submissions as app_submissions|g' main.py
sed -i 's|app_events.router|app_submissions.router|g' main.py

# 4. Remove or migrate story.py
# Option A: Remove entirely
rm backend/api/story.py
sed -i '/story.router/d' main.py

# Option B: Migrate to use EventRepository instead of neo4j_client
# (requires code rewrite)
```

### Phase 2: Update Frontend

```bash
# 1. Update frontend to use /api/events (not /api/stories)
find frontend/src -type f -name "*.ts" -exec sed -i 's|/api/stories|/api/events|g' {} +

# 2. Rename Story → Event throughout frontend
find frontend/src -type f -name "*.ts" -exec sed -i 's/Story/Event/g' {} +
find frontend/src -type f -name "*.ts" -exec sed -i 's/story/event/g' {} +

# 3. Rename files
mv frontend/src/components/story-detail.ts frontend/src/components/event-detail.ts
mv frontend/src/components/story-chat-sidebar.ts frontend/src/components/event-chat-sidebar.ts
# ... etc

# 4. Build frontend
cd frontend && npm run build
# Output: ../static/
```

### Phase 3: Neo4j Migration

```cypher
// Rename Story nodes to Event nodes
MATCH (s:Story)
SET s:Event
REMOVE s:Story
```

### Phase 4: Container Cleanup

```yaml
# docker-compose.yml

# Remove frontend service (obsolete)
# Rename api → app
services:
  app:  # renamed from "api"
    build:
      context: .  # Build from root, not backend/
      dockerfile: Dockerfile
    container_name: herenews-app
    ports:
      - "7272:8000"  # External 7272, internal 8000
    volumes:
      - ./backend:/app/backend
      - ./frontend:/app/frontend
      - ./static:/app/static
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. Correct Data Flow (Post-Cleanup)

```
User Browser
    ↓
http://localhost:7272/app
    ↓
[herenews-app container]
main.py serves:
    ├─ /app → static/index.html (built React/Lit frontend)
    ├─ /app/assets → static/assets/
    └─ /api/* → Backend API
        ├─ /api/events → List events (endpoints_events.py)
        ├─ /api/events/{id} → Event detail
        ├─ /api/submissions → Submit event URL (backend/api/submissions.py)
        ├─ /api/comments/event/{id} → Comments
        ├─ /api/auth/* → Google OAuth
        ├─ /api/chat/* → AI chat
        └─ /api/map/* → Map data
    ↓
Repositories:
    ├─ EventRepository → PostgreSQL + Neo4j (Event nodes)
    ├─ CommentRepository → PostgreSQL (comments table)
    ├─ ChatSessionRepository → PostgreSQL (chat_sessions)
    └─ UserRepository → PostgreSQL (users table)
    ↓
Data Storage:
    ├─ Neo4j: Event nodes, Page nodes, Claim nodes, Entity nodes, relationships
    └─ PostgreSQL: users, comments, chat_sessions, pages (content), embeddings
    ↓
Workers (background):
    ├─ extraction_worker.py → Process URL → Create Page
    ├─ knowledge_worker.py → Extract Claims + Entities → Link to Wikidata
    └─ event_worker.py → Form Events from Pages
```

---

## 10. Next Actions (Priority Order)

1. ✅ **Fix API route conflict** (Phase 1)
2. ✅ **Remove obsolete backend/api/story.py**
3. ✅ **Rename container "api" → "app"**
4. ⚠️ **Update frontend API calls** (/api/stories → /api/events)
5. ⚠️ **Migrate Neo4j Story → Event nodes**
6. ⚠️ **Build frontend** (npm run build)
7. ⚠️ **Test end-to-end flow**

---

**Summary**: System needs cleanup to align Story (OLD) → Event (NEW) terminology consistently across Neo4j, backend, and frontend. Main blocker is API route conflict at `/api/events`.
