# Complete Webapp → Service Farm Merge Plan

## Overview
Merge webapp's community features into service_farm backend to create unified application.

## File Mapping

### 1. Auth System (COPY → backend/middleware/)
```
../webapp/app/auth/google_oauth.py    → backend/middleware/google_oauth.py
../webapp/app/auth/session.py         → backend/middleware/jwt_session.py
../webapp/app/auth/middleware.py      → backend/middleware/auth.py (DONE)
```

### 2. API Routes (COPY → backend/api/)
```
../webapp/app/routers/auth.py         → backend/api/auth.py
../webapp/app/routers/comments.py     → backend/api/comments.py
../webapp/app/routers/chat.py         → backend/api/chat.py
../webapp/app/routers/story.py        → backend/api/events.py (rename Story→Event)
../webapp/app/routers/events.py       → backend/api/event_submissions.py
```

### 3. Models (ALREADY DONE ✅)
```
User, Comment, ChatSession created in backend/models/
```

### 4. Repositories (ALREADY DONE ✅)
```
UserRepository, CommentRepository, ChatSessionRepository in backend/repositories/
```

### 5. Services (COPY → backend/services/)
```
../webapp/app/services/neo4j_client.py    → ALREADY EXISTS (use existing)
../webapp/app/services/tcf_feed_service.py → backend/services/ (if needed)
```

### 6. Config (MERGE)
```
../webapp/app/config.py settings → Add to backend/main.py or create backend/config.py
```

### 7. Frontend (REPLACE)
```
../webapp/frontend/* → service_farm/frontend/ (REPLACE simple HTML with full app)
```

### 8. Dependencies (MERGE)
```
../webapp/requirements.txt → service_farm/requirements.txt (ADD TO)
```

## Execution Order

1. ✅ Models created
2. ✅ Repositories created  
3. ✅ Database schema created
4. ✅ Tests validated
5. ⏳ Move auth system
6. ⏳ Move API routes
7. ⏳ Update main.py
8. ⏳ Merge requirements.txt
9. ⏳ Replace frontend
10. ⏳ Test end-to-end

## Changes Needed During Move

### Terminology Updates
- `story_id` → `event_id`
- `Story` nodes → `Event` nodes in Neo4j queries
- `/api/stories/` → `/api/events/`

### Import Updates
- `app.config` → direct env vars or backend config
- `app.models.user` → `models.user`
- `app.database.repositories` → `repositories`
- SQLAlchemy sessions → asyncpg pool

### Dependency Injection
- `Depends(get_db)` → `Depends(get_db_pool)`
- SQLAlchemy session → asyncpg connection pool
