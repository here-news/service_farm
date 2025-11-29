# Browser Extension - PostgreSQL Migration

## What Changed

The browser extension has been migrated from **Firestore** to **PostgreSQL** for rogue task polling.

### Before (Firestore)
- Extension polled Firestore `rogue_extraction_tasks` collection
- Required Firebase SDK and API credentials
- Direct database access from browser extension

### After (PostgreSQL)
- Extension polls service-farm REST API endpoints
- No Firebase dependencies
- Cleaner architecture with API-based communication

## New API Endpoints

The service-farm now provides two endpoints for the browser extension:

### 1. Poll for Tasks
```
GET /api/rogue/poll
```

**Response (200):**
```json
{
  "task_id": "5192ccd4528bc9b3084f5f81fa6dde35",
  "url": "https://politico.com/article",
  "created_at": "2025-11-25T14:39:52.886574Z",
  "extraction_task_id": "119a48b3-6fd8-4525-b22a-3b3a1a3c8ebc"
}
```

**Response (404):** No pending tasks

**Behavior:**
- Atomically claims next pending task using `FOR UPDATE SKIP LOCKED`
- Task status changes from `pending` → `processing`
- Returns task details

### 2. Update Task Status
```
PATCH /api/rogue/{task_id}
```

**Form Parameters:**
- `status` (required): "completed" or "failed"
- `metadata` (optional): JSON string with extracted metadata
- `error_message` (optional): Error message if failed

**Response (200):**
```json
{
  "success": true,
  "task_id": "5192ccd4528bc9b3084f5f81fa6dde35",
  "status": "completed"
}
```

## Files Changed

1. **`manifest.json`**
   - Updated version: 1.2.7 → 2.0.0
   - Changed service worker: `background.js` → `background_postgres.js`
   - Removed `type: "module"` (no longer using Firebase ES modules)

2. **`scripts/background_postgres.js`** (NEW)
   - Polls PostgreSQL via REST API
   - No Firebase dependencies
   - Simpler, more maintainable code

3. **`scripts/background.js`** (DEPRECATED)
   - Original Firestore implementation
   - Kept for reference

## Configuration

Update the service-farm URL in `background_postgres.js`:

```javascript
const SERVICE_FARM_URL = 'http://localhost:8080'; // Change for production
```

For production:
```javascript
const SERVICE_FARM_URL = 'https://service-farm-here.run.app';
```

## Installation

1. **Update service-farm** (deploy new API endpoints):
   ```bash
   cd /media/im3/plus/lab4/re_news/service_farm
   git pull
   # Deploy to Cloud Run
   ```

2. **Rebuild browser extension**:
   ```bash
   cd browser_extension
   # Zip for Chrome Web Store
   zip -r here-rogue-extractor-v2.0.0.zip . -x "*.git*" -x "node_modules/*"
   ```

3. **Load in Chrome** (for testing):
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `browser_extension` directory

4. **Verify**:
   - Extension badge should show status
   - Check browser console logs: `🚀 HERE News Rogue URL Extractor - PostgreSQL Mode`
   - Submit a blocked URL and watch extension pick it up

## Testing

### Test rogue task creation:
```bash
# Submit a known paywall URL
curl -X POST http://localhost:8080/submit \
  -d "url=https://www.politico.com/news/2025/11/24/halligan-dismissed-james-comey-cases-00667735" \
  -d "force=false"
```

### Check rogue queue:
```bash
# Poll for next task
curl http://localhost:8080/api/rogue/poll
```

### Monitor extension:
- Open extension popup to see status
- Check Chrome DevTools → Extensions → Service Worker logs
- Look for: `📋 Found task: <task_id>`

## Troubleshooting

### Extension not picking up tasks
1. Check service-farm URL in `background_postgres.js`
2. Verify service-farm is running and accessible
3. Check browser console for CORS errors
4. Verify rogue tasks exist: `docker exec service-farm python3 -c "..."`

### Tasks stuck as "processing"
- Extension crashed during extraction
- Tasks will auto-recover after timeout (future enhancement)
- Manual fix: `UPDATE tasks.rogue_tasks SET status='pending' WHERE status='processing';`

### CORS errors
- Service-farm needs CORS headers for extension origin
- Add to `main.py` if needed:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["*"])
  ```

## Rollback Plan

If issues occur, rollback to Firestore version:

1. **Revert manifest.json:**
   ```json
   "background": {
     "service_worker": "scripts/background.js",
     "type": "module"
   }
   ```

2. **Reload extension** in Chrome

3. **Verify Firestore** still has pending tasks

## Migration Benefits

✅ **No Firebase dependencies** - Simpler, fewer moving parts
✅ **Unified architecture** - All services use PostgreSQL
✅ **Better monitoring** - PostgreSQL query logs
✅ **Atomic task claiming** - `FOR UPDATE SKIP LOCKED` prevents race conditions
✅ **Easier testing** - Can poll API directly with curl

## Next Steps

- [ ] Deploy service-farm with new API endpoints
- [ ] Test extension with pending rogue task (Politico URL)
- [ ] Monitor logs for successful extraction
- [ ] Verify metadata is persisted correctly
- [ ] Deploy to production
- [ ] Publish updated extension (v2.0.0) to Chrome Web Store
