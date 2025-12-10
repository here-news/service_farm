# Directory Restructuring Options

## Problem
Current structure is confusing:
```
❌ backend/    - Intelligence engine
❌ app/        - API routes (weird name!)
```

Both are Python backend code, but the naming is unclear.

---

## ✅ **OPTION 1: Merge Everything into `backend/`** (RECOMMENDED)

**Why separate them?** They're both Python backend. Merge them!

### Before:
```
backend/
  ├── models/
  ├── repositories/
  ├── workers/
  ├── services/
  └── middleware/

app/
  ├── models/          # Pydantic models
  ├── routers/         # API routes
  ├── services/        # Some services
  └── config.py
```

### After:
```
backend/
  ├── models/
  │   ├── domain/           # Domain models (Event, Page, User)
  │   └── api/              # Pydantic API models (from app/models/)
  │
  ├── repositories/         # Data access
  │
  ├── api/                  # All API routes (from app/routers/)
  │   ├── auth.py
  │   ├── comments.py
  │   ├── chat.py
  │   ├── events.py
  │   ├── artifacts.py
  │   └── ...
  │
  ├── workers/              # Background workers
  │
  ├── services/             # Business logic (merged from both)
  │
  ├── middleware/           # Auth middleware
  │
  └── config.py             # Unified config
```

**Commands:**
```bash
# Merge models
mkdir -p backend/models/domain backend/models/api
mv backend/models/*.py backend/models/domain/
mv app/models/*.py backend/models/api/

# Merge routers → backend/api/
mv app/routers/* backend/api/

# Merge services
mv app/services/* backend/services/

# Remove empty app/
rm -rf app/

# Update imports in all files
# from app.models.user import UserCreate
# → from backend.models.api.user import UserCreate
```

**Pros:**
- ✅ Clear single backend
- ✅ No confusion
- ✅ Standard Python project layout

**Cons:**
- ⚠️ Need to update many imports

---

## OPTION 2: Rename `app/` → `api/`

Keep separation but clearer naming.

### Structure:
```
backend/          # Core logic (workers, services, repos)
api/              # HTTP API layer (routes, Pydantic models)
frontend/         # React UI
```

**Commands:**
```bash
mv app api
# Update imports: from app. → from api.
```

**Pros:**
- ✅ Clearer than "app"
- ✅ Minimal changes

**Cons:**
- ⚠️ Still have two Python directories
- ⚠️ Unclear separation of concerns

---

## OPTION 3: Flatten Everything to Root

No nested directories, everything at top level.

### Structure:
```
service_farm/
├── models/
├── repositories/
├── api/
├── workers/
├── services/
├── middleware/
├── frontend/
└── main.py
```

**Pros:**
- ✅ Very simple

**Cons:**
- ❌ Root gets cluttered
- ❌ Hard to distinguish backend vs workers vs API

---

## 🎯 **RECOMMENDED: Option 1**

Merge everything into `backend/` with clear subdirectories:

```
service_farm/
│
├── backend/                    # All Python backend code
│   ├── models/
│   │   ├── domain/             # Dataclass domain models
│   │   │   ├── event.py
│   │   │   ├── page.py
│   │   │   ├── user.py
│   │   │   └── ...
│   │   └── api/                # Pydantic API models
│   │       ├── user.py         # UserCreate, UserResponse
│   │       ├── chat.py
│   │       └── ...
│   │
│   ├── repositories/           # Data access
│   │   ├── user_repository.py
│   │   ├── event_repository.py
│   │   └── ...
│   │
│   ├── api/                    # HTTP routes
│   │   ├── auth.py
│   │   ├── comments.py
│   │   ├── chat.py
│   │   ├── events.py
│   │   ├── artifacts.py        # From endpoints.py
│   │   └── ...
│   │
│   ├── workers/                # Background processors
│   │   ├── extraction_worker.py
│   │   ├── knowledge_worker.py
│   │   └── event_worker.py
│   │
│   ├── services/               # Business logic
│   │   ├── neo4j_service.py
│   │   ├── event_service.py
│   │   ├── tcf_feed_service.py
│   │   └── ...
│   │
│   ├── middleware/             # Auth
│   │   ├── auth.py
│   │   ├── google_oauth.py
│   │   └── jwt_session.py
│   │
│   ├── utils/                  # Utilities
│   │   └── id_generator.py
│   │
│   └── config.py               # Configuration
│
├── frontend/                   # React UI (from webapp)
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── frontend-legacy/            # HTML tools (preserved)
│   ├── event.html
│   └── timeline.html
│
├── static/                     # Built frontend (from npm build)
│
├── main.py                     # FastAPI entry point
├── requirements.txt
└── docker-compose.yml
```

---

## Migration Script for Option 1

```bash
#!/bin/bash
# Reorganize service_farm structure

cd service_farm

echo "🔧 Reorganizing directory structure..."

# 1. Organize models
echo "  📦 Organizing models..."
mkdir -p backend/models/domain backend/models/api
mv backend/models/*.py backend/models/domain/ 2>/dev/null
mv app/models/*.py backend/models/api/ 2>/dev/null

# 2. Move routers to backend/api
echo "  🛣️  Moving API routes..."
mkdir -p backend/api
mv app/routers/*.py backend/api/ 2>/dev/null

# 3. Merge services
echo "  ⚙️  Merging services..."
mv app/services/* backend/services/ 2>/dev/null

# 4. Move config
echo "  ⚙️  Moving config..."
mv app/config.py backend/config.py 2>/dev/null

# 5. Remove empty app/
echo "  🗑️  Removing empty app/..."
rm -rf app/

# 6. Move frontend
echo "  🎨 Reorganizing frontend..."
mv frontend frontend-legacy 2>/dev/null
cp -r ../webapp/frontend ./frontend 2>/dev/null || echo "  ⚠️  Webapp frontend not found"

echo "✅ Reorganization complete!"
echo ""
echo "⚠️  TODO: Update imports in Python files"
echo "   - app.models → backend.models.api"
echo "   - app.routers → backend.api"
echo "   - app.services → backend.services"
```

---

## Decision Matrix

| Criteria | Option 1 (Merge) | Option 2 (Rename) | Option 3 (Flatten) |
|----------|------------------|-------------------|-------------------|
| Clarity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Standard | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Effort | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Scalable | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Recommendation: Option 1** - Do it once, do it right.

---

**Which option do you prefer?**
