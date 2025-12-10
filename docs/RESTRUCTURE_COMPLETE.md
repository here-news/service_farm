# ✅ Directory Restructure Complete

## Summary

Successfully merged `app/` into `backend/` with clear, organized subdirectories.

---

## New Structure

```
service_farm/
├── backend/                          # Unified Python backend
│   ├── models/
│   │   ├── domain/                   # Dataclass domain models
│   │   │   ├── event.py              # ev_xxxxxxxx
│   │   │   ├── page.py               # pg_xxxxxxxx
│   │   │   ├── claim.py              # cl_xxxxxxxx
│   │   │   ├── entity.py             # en_xxxxxxxx
│   │   │   ├── user.py               # UUID format
│   │   │   ├── comment.py            # cm_xxxxxxxx
│   │   │   ├── chat_session.py       # cs_xxxxxxxx
│   │   │   └── ...
│   │   └── api/                      # Pydantic API models
│   │       ├── user.py               # UserCreate, UserResponse
│   │       ├── chat.py               # ChatUnlock, ChatMessage
│   │       ├── event_submission.py
│   │       └── extraction.py
│   │
│   ├── api/                          # HTTP API routes
│   │   ├── auth.py                   # OAuth & JWT
│   │   ├── comments.py               # Comment CRUD
│   │   ├── chat.py                   # AI chat sessions
│   │   ├── events.py                 # Event endpoints
│   │   ├── story.py                  # Story feed
│   │   ├── event_page.py             # Event pages
│   │   ├── coherence.py              # Coherence scoring
│   │   ├── extraction.py             # Manual extraction
│   │   ├── preview.py                # URL preview
│   │   └── map.py                    # Map data
│   │
│   ├── repositories/                 # Data access (asyncpg)
│   │   ├── user_repository.py
│   │   ├── comment_repository.py
│   │   ├── chat_session_repository.py
│   │   ├── event_repository.py
│   │   ├── page_repository.py
│   │   ├── claim_repository.py
│   │   └── entity_repository.py
│   │
│   ├── services/                     # Business logic
│   │   ├── neo4j_service.py          # Knowledge graph
│   │   ├── event_service.py          # Event formation
│   │   ├── knowledge_graph.py        # Graph operations
│   │   ├── tcf_feed_service.py       # TCF feed processing
│   │   ├── coherence_service.py      # Coherence scoring
│   │   ├── cache.py                  # Caching
│   │   └── ...
│   │
│   ├── workers/                      # Background processors
│   │   ├── extraction_worker.py      # Page extraction
│   │   ├── knowledge_worker.py       # Graph enrichment
│   │   └── event_worker.py           # Event detection
│   │
│   ├── middleware/                   # Auth system
│   │   ├── auth.py                   # FastAPI dependencies
│   │   ├── google_oauth.py           # OAuth client
│   │   └── jwt_session.py            # JWT tokens
│   │
│   ├── utils/                        # Utilities
│   │   ├── id_generator.py           # Short ID generation
│   │   └── datetime_utils.py
│   │
│   ├── migrations/                   # SQL migrations
│   │   └── 001_create_user_tables.sql
│   │
│   └── config.py                     # Unified configuration
│
├── main.py                           # FastAPI entry point
├── requirements.txt                  # Python dependencies
├── docker-compose.yml
└── Dockerfile
```

---

## Changes Made

### 1. **Models Organized** ✅
- **backend/models/domain/**: Core domain models (Event, Page, User, Comment, etc.)
- **backend/models/api/**: Pydantic request/response models for FastAPI

### 2. **API Routes Consolidated** ✅
- **backend/api/**: All HTTP route handlers (merged from app/routers/)
- Updated imports from `app.*` to `backend.*`

### 3. **Services Merged** ✅
- Moved unique services from app/services/ to backend/services/
- Removed duplicate neo4j_client.py (using backend/services/neo4j_service.py)

### 4. **Removed Obsolete Code** ✅
- **app/auth/**: Replaced by backend/middleware/
- **app/database/**: Replaced by backend/repositories/
- **app/main.py**: Replaced by root main.py
- **app/** entire directory removed

### 5. **Updated Imports** ✅
- All routers now import from `backend.*`
- Removed SQLAlchemy AsyncSession dependencies
- Using asyncpg repositories with `db_pool` directly

---

## Import Changes

### Before (OLD)
```python
from app.auth.google_oauth import get_google_oauth
from app.auth.session import create_access_token
from app.auth.middleware import get_current_user_optional
from app.database.connection import get_db
from app.database.repositories.user_repo import UserRepository
from app.models.user import UserCreate, UserResponse
from app.config import get_settings
```

### After (NEW)
```python
from backend.middleware.google_oauth import get_google_oauth
from backend.middleware.jwt_session import create_access_token
from backend.middleware.auth import get_current_user_optional
from backend.repositories import db_pool
from backend.repositories.user_repository import UserRepository
from backend.models.api.user import UserCreate, UserResponse
from backend.config import get_settings
```

---

## Database Access Pattern

### Before (OLD - SQLAlchemy ORM)
```python
@router.get("/comments")
async def get_comments(db: AsyncSession = Depends(get_db)):
    repo = CommentRepository(db)
    comments = await repo.get_all()
    return comments
```

### After (NEW - asyncpg repositories)
```python
@router.get("/comments")
async def get_comments():
    repo = CommentRepository(db_pool)
    comments = await repo.get_all()
    return comments
```

**Benefits:**
- ✅ No dependency injection for database session
- ✅ Repositories manage their own connection pooling
- ✅ Simpler, faster, more direct
- ✅ Matches service_farm's existing pattern

---

## Next Steps

1. **Test the restructured application**:
   ```bash
   python main.py
   ```

2. **Run workers** (if needed):
   ```bash
   python backend/workers/extraction_worker.py
   python backend/workers/knowledge_worker.py
   python backend/workers/event_worker.py
   ```

3. **Test API endpoints**:
   - Auth: `GET /api/auth/login`
   - Events: `GET /api/events`
   - Comments: `GET /api/comments/story/{story_id}`

4. **Setup frontend** (future):
   - Copy React app to `frontend/`
   - Build to `static/`
   - Main.py already serves from `static/`

---

## Benefits of New Structure

✅ **Single source of truth**: All backend code in `backend/`
✅ **Clear organization**: domain models vs API models
✅ **No duplication**: Removed redundant auth/database layers
✅ **Standard pattern**: Matches service_farm's architecture
✅ **Maintainable**: Easy to find and modify code
✅ **Scalable**: Room for growth in organized structure

---

## Files Modified

- **main.py**: Updated imports and removed init_db()
- **backend/api/*.py** (11 files): Updated all imports
- **backend/config.py**: Moved from app/config.py
- **backend/services/**: Merged unique services from app/
- **Created**: backend/models/domain/, backend/models/api/, backend/api/

## Files Removed

- **app/** entire directory
- **app/auth/**: Duplicate of backend/middleware/
- **app/database/**: Replaced by backend/repositories/
- **app/services/neo4j_client.py**: Duplicate of backend/services/neo4j_service.py

---

**✅ Restructure complete! Ready for testing.**
