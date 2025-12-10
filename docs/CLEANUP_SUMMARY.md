# ✅ App Directory Cleanup Complete

## What Was Removed

### ❌ Duplicates
```
app/auth/                    # Removed - Use backend/middleware/
  - google_oauth.py          →  backend/middleware/google_oauth.py
  - session.py               →  backend/middleware/jwt_session.py
  - middleware.py            →  backend/middleware/auth.py
```

### ❌ Deprecated
```
app/main.py                  # Removed - Use root main.py
```

### ❌ Legacy ORM
```
app/database/                # Removed - Use backend/repositories/
  - models.py                # SQLAlchemy ORM
  - connection.py            # SQLAlchemy connection
  - repositories/            # Old repo pattern
    - user_repo.py           →  backend/repositories/user_repository.py
    - comment_repo.py        →  backend/repositories/comment_repository.py
    - chat_session_repo.py   →  backend/repositories/chat_session_repository.py
    - event_submission_repo.py
```

## What Remains (Cleaned App Structure)

```
app/
├── config.py              # Settings (used by routers)
├── models/                # Pydantic API models
│   ├── user.py            # UserCreate, UserResponse, etc.
│   ├── chat.py            # Chat API models
│   ├── event_submission.py
│   └── extraction.py
├── routers/               # API route handlers
│   ├── auth.py            # ⚠️ Needs update to use backend/middleware
│   ├── comments.py        # ⚠️ Needs update to use backend/repositories
│   ├── chat.py            # ⚠️ Needs update to use backend/repositories
│   ├── events.py          # ⚠️ Needs update to use backend/repositories
│   ├── story.py
│   ├── event_page.py
│   ├── coherence.py
│   ├── extraction.py
│   ├── preview.py
│   └── map.py
└── services/              # App-specific services
    ├── neo4j_client.py    # Keep (has embedding search)
    ├── tcf_feed_service.py
    ├── coherence_service.py
    └── cache.py
```

## Next Steps

### 1. Update Router Imports
Routers still import from old `app.database` and `app.auth`. Need to update to:
```python
# OLD
from app.auth.middleware import get_current_user_optional
from app.database.repositories.user_repo import UserRepository
from app.database.connection import get_db

# NEW
from backend.middleware.auth import get_current_user_optional
from backend.repositories.user_repository import UserRepository
# Use backend db_pool directly (no get_db needed)
```

### 2. Frontend Structure
React frontend should go in:
```
service_farm/
├── frontend/              # React app
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/     # API clients
│   │   └── App.tsx
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
└── static/                # Built frontend (npm run build output)
```

## Benefits of Cleanup

✅ **Single source of truth** - backend/ for all data access
✅ **No duplication** - auth, repos, models unified
✅ **Modern pattern** - asyncpg + dataclasses vs SQLAlchemy ORM
✅ **Simpler** - Less code to maintain
✅ **Faster** - asyncpg is faster than SQLAlchemy

## Remaining Work

- [ ] Update app/routers/ imports (auth, comments, chat, events)
- [ ] Setup React frontend in frontend/
- [ ] Build frontend → static/
- [ ] Test complete integration
