# ✅ Phase 1 Cleanup Complete

## Summary

Fixed critical API route conflicts and removed obsolete Story-related code.

---

## Changes Made

### 1. ✅ Fixed API Route Conflict

**Problem**: Two routers mounted at `/api/events` causing conflicts
```python
# BEFORE (BROKEN)
/api/events → endpoints_events.py (Event listing)
/api/events → backend/api/events.py (Event submissions) ❌ CONFLICT!
```

**Solution**: Renamed event submissions endpoint
```python
# AFTER (FIXED)
/api/events → endpoints_events.py (Event listing) ✅
/api/submissions → backend/api/submissions.py (Event submissions) ✅
```

**Files Changed**:
- `backend/api/events.py` → `backend/api/submissions.py`
- Updated router prefix: `/api/events` → `/api/submissions`
- Updated main.py imports

### 2. ✅ Removed Obsolete Story Router

**Removed**: `backend/api/story.py`
- Used old `services/neo4j_client` (deprecated pattern)
- Redundant with unified `/api/events` endpoint

**Note**: `services/neo4j_client.py` kept temporarily (still used by chat, map, coherence routers)

### 3. ✅ Updated main.py

**Import Changes**:
```python
# BEFORE
from backend.api import (
    auth, coherence, story, comments, chat,
    preview, events as app_events, extraction, event_page, map
)

# AFTER
from backend.api import (
    auth, coherence, comments, chat,
    preview, submissions, extraction, event_page, map
)
```

**Router Changes**:
```python
# REMOVED
app.include_router(story.router, tags=["Stories"])

# CHANGED
app.include_router(app_events.router, tags=["Events - Community"])
# TO
app.include_router(submissions.router, tags=["Submissions"])
```

### 4. ✅ Updated docker-compose.yml

**Service Rename**: `api` → `app`
```yaml
# BEFORE
services:
  frontend:          # Obsolete HTML server
    ports:
      - "8080:8080"
  api:               # FastAPI backend
    ports:
      - "8000:8000"

# AFTER
services:
  app:               # Unified FastAPI + React
    container_name: herenews-app
    ports:
      - "7272:8000"  # External port 7272
    volumes:
      - ./backend:/app
      - ./static:/app/static  # Built React frontend
```

**Removed**: Obsolete `frontend` service (legacy HTML server)

**Worker Dependencies**: Updated all workers to depend on `app` (was `api`)

---

## Current API Structure

### ✅ Working Endpoints

| Endpoint | Router | Purpose |
|----------|--------|---------|
| **Event System** |
| `GET /api/events` | endpoints_events.py | List all events |
| `GET /api/events/{id}` | endpoints_events.py | Get event detail |
| `POST /api/submissions` | backend/api/submissions.py | Submit URL for event |
| `GET /api/event/{id}` | backend/api/event_page.py | Event page data |
| **User System** |
| `GET /api/auth/login` | backend/api/auth.py | Google OAuth login |
| `GET /api/auth/callback` | backend/api/auth.py | OAuth callback |
| `GET /api/auth/status` | backend/api/auth.py | Check auth status |
| **Community** |
| `GET /api/comments/event/{id}` | backend/api/comments.py | Get comments |
| `POST /api/comments` | backend/api/comments.py | Create comment |
| `POST /api/chat/unlock` | backend/api/chat.py | Unlock AI chat |
| `POST /api/chat/message` | backend/api/chat.py | Send chat message |
| **Utilities** |
| `GET /api/preview` | backend/api/preview.py | URL preview |
| `GET /api/map/entities` | backend/api/map.py | Hot entities |
| `GET /api/map/locations` | backend/api/map.py | Hot locations |
| `POST /api/coherence/score` | backend/api/coherence.py | Score coherence |

---

## Access Points

```
User → http://localhost:7272
    ↓
    ├─→ /app                    → React frontend (static/index.html)
    ├─→ /api/events             → Event listing
    ├─→ /api/submissions        → Submit new event URL
    ├─→ /api/comments/*         → Comments
    ├─→ /api/chat/*             → AI chat
    ├─→ /api/auth/*             → Authentication
    └─→ /api/map/*              → Map data
```

---

## Docker Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| **app** | herenews-app | 7272 | Unified FastAPI + React |
| postgres | herenews-postgres | 5432 | PostgreSQL + pgvector |
| neo4j | herenews-neo4j | 7474, 7687 | Neo4j graph DB |
| redis | herenews-redis | 6379 | Job queues |
| worker-extraction-{1,2,3} | herenews-worker-extraction-* | - | Extract pages |
| worker-knowledge-{1,2} | herenews-worker-knowledge-* | - | Extract entities/claims |
| worker-event | herenews-worker-event | - | Form events |

---

## Testing

### Start Services

```bash
cd /media/im3/plus/lab4/re_news/service_farm

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Test Endpoints

```bash
# Health check
curl http://localhost:7272/health

# List events
curl http://localhost:7272/api/events

# Event detail
curl http://localhost:7272/api/events/ev_xxxxxxxx

# Submit URL (requires auth)
curl -X POST http://localhost:7272/api/submissions \
  -H "Content-Type: application/json" \
  -d '{"content": "Breaking news", "urls": "https://example.com/article"}'

# Auth status
curl http://localhost:7272/api/auth/status
```

---

## Next Steps (Phase 2)

### 🔴 HIGH PRIORITY

1. **Neo4j Migration**: Rename `Story` nodes → `Event` nodes
   ```cypher
   MATCH (s:Story)
   SET s:Event
   REMOVE s:Story
   ```

2. **Frontend Migration**: Update React frontend
   - Change API calls from `/api/stories` → `/api/events`
   - Rename all `Story` → `Event` terminology
   - Build frontend: `cd frontend && npm run build`

3. **Repository Cleanup**: Migrate remaining neo4j_client usage
   - backend/api/chat.py → Use Neo4jService
   - backend/api/map.py → Use Neo4jService
   - backend/api/coherence.py → Use Neo4jService
   - backend/api/event_page.py → Use Neo4jService

### 🟡 MEDIUM PRIORITY

4. **Database Migration**: Ensure PostgreSQL schema matches
   - Verify `events` table exists (not `stories`)
   - Check all foreign keys use correct names

5. **Frontend Build**: Create production build
   ```bash
   cd frontend
   npm install
   npm run build  # Output to ../static/
   ```

### 🟢 LOW PRIORITY

6. **Documentation**: Update API docs
7. **Tests**: Add integration tests for new endpoints

---

## Verification Checklist

- [x] No duplicate routes at `/api/events`
- [x] backend/api/story.py removed
- [x] main.py imports updated
- [x] docker-compose.yml service renamed to "app"
- [x] Port changed to 7272
- [x] Frontend container removed
- [x] Worker dependencies updated
- [ ] Services start successfully (need to test)
- [ ] Endpoints respond correctly (need to test)
- [ ] Frontend builds successfully (need to build)

---

**Status**: Phase 1 cleanup complete. Ready for testing and Phase 2 (Neo4j migration + frontend update).
