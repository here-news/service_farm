# Breathing Knowledge Base - Demo

Thin vertical slice demonstrating the resource-first architecture with progressive enhancement.

## Architecture

**Resource-First Pattern**: Instant stub creation + async worker enrichment

```
User submits URL
     ↓
  Instant response (< 100ms)
  - New URL: stub + commission workers
  - Existing URL: best shot from DB
     ↓
  Workers enrich in background
  (extraction → semantic → event)
     ↓
  Webapp polls for updates
```

## Components

1. **FastAPI Backend** (port 8001)
   - `POST /api/artifacts/draft?url=...` - Submit URL, get instant best shot
   - `GET /api/artifacts/draft/{id}` - Poll for status updates

2. **Three Workers** (async pipeline)
   - **Extraction Worker**: URL → Content (trafilatura)
   - **Semantic Worker**: Content → Entities + Claims (GPT-4o-mini)
   - **Event Worker**: Claims → Events (matching/clustering)

3. **One-Page Webapp**
   - Input any URL
   - Live status updates (polls every 2s)
   - Visual confidence bar
   - Entity tags, content preview

4. **Isolated Environment**
   - PostgreSQL on port 5433 (isolated from production)
   - Redis for worker queues
   - Fresh database (no production data)

## Quick Start

```bash
# 1. Start demo (automatically loads OPENAI_API_KEY from ../.env)
chmod +x start_demo.sh stop_demo.sh
./start_demo.sh

# 2. Open webapp
# The script will show the file:// path
# Or serve via HTTP:
cd frontend
python3 -m http.server 8002
# Visit http://localhost:8002
```

**Note**: The startup script automatically loads environment variables from `../.env` including the OpenAI API key.

## Demo Scenarios

### Scenario A: Existing URL
Submit a URL already in the database:
- Response: Instant best shot with full data
- No workers commissioned (already extracted)
- Shows entities, events, high confidence

### Scenario B: New URL
Submit a URL not in the database:
- Response: Instant stub (status: "stub", confidence: 0.3)
- Workers commissioned automatically
- Poll to see progressive updates:
  - `stub` → `extracting` → `extracted` → `entities_extracted` → `complete`
- Confidence increases as workers finish

## Status Flow

```
stub (0.3)
  ↓ [Extraction Worker]
extracting (0.3)
  ↓
extracted (0.5)
  ↓ [Semantic Worker]
entities_extracted (0.8)
  ↓ [Event Worker]
complete (1.0)
```

## Database Schema

Simplified for demo (production uses artifact abstraction):

- **pages**: URL, content, status, language
- **entities**: Canonical name, type, confidence
- **claims**: Factual assertions from content
- **events**: Clustered claims
- **page_entities, page_events, event_entities**: Relationships

## File Structure

```
demo/
├── docker-compose.demo.yml    # PostgreSQL + Redis
├── schema.sql                 # Database schema
├── start_demo.sh              # Startup script
├── stop_demo.sh               # Shutdown script
├── backend/
│   ├── main.py                # FastAPI app
│   ├── endpoints.py           # /artifacts/draft
│   ├── workers.py             # 3 workers
│   └── requirements.txt       # Dependencies
├── frontend/
│   └── index.html             # One-page demo UI
└── logs/
    ├── backend.log
    └── workers.log
```

## Monitoring

```bash
# Watch backend logs
tail -f logs/backend.log

# Watch workers
tail -f logs/workers.log

# Check PostgreSQL
docker exec -it demo-postgres psql -U demo_user -d demo_phi_here
# \dt to list tables
# SELECT * FROM pages;

# Check Redis queues
docker exec -it demo-redis redis-cli
# LLEN queue:extraction:high
# LLEN queue:semantic:normal
```

## Testing API Directly

```bash
# Submit URL
curl -X POST "http://localhost:8001/api/artifacts/draft?url=https://example.com/article"

# Poll status
curl "http://localhost:8001/api/artifacts/draft/{artifact_id}"

# Health check
curl "http://localhost:8001/health"
```

## Stop Demo

```bash
./stop_demo.sh
```

Data is preserved in Docker volume. To completely reset:

```bash
docker volume rm demo_demo_pg_data
```

## Next Steps After Demo

1. Validate flow works end-to-end
2. Test with real URLs (news articles)
3. Observe progressive enhancement
4. Measure response times
5. Decide on migration strategy for old data
