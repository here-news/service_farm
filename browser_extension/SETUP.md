# Browser Extension Setup Guide

Complete setup guide for the HERE News Rogue URL Extractor extension.

## Prerequisites

- Chrome, Edge, or Firefox browser
- Access to Firebase Console
- `here2-474221` GCP project access

## Step 1: Configure Firestore

### 1.1 Deploy Firestore Security Rules

```bash
# Navigate to service_farm directory
cd /media/im3/plus/lab4/re_news/service_farm

# Deploy Firestore rules
firebase deploy --only firestore:rules
```

Or manually in Firebase Console:
1. Go to https://console.firebase.google.com/project/here2-474221/firestore/rules
2. Copy contents from `firestore.rules`
3. Click "Publish"

### 1.2 Create Firestore Index (Optional)

For better query performance:

```bash
# Create composite index on status + created_at
gcloud firestore indexes composite create \
  --collection-group=rogue_extraction_tasks \
  --query-scope=COLLECTION \
  --field-config field-path=status,order=ASCENDING \
  --field-config field-path=created_at,order=ASCENDING \
  --project=here2-474221
```

Or in Firebase Console:
1. Go to https://console.firebase.google.com/project/here2-474221/firestore/indexes
2. Click "Add index"
3. Collection: `rogue_extraction_tasks`
4. Fields:
   - `status` (Ascending)
   - `created_at` (Ascending)
5. Query scope: Collection
6. Click "Create"

## Step 2: Get Firebase Configuration

1. Go to Firebase Console: https://console.firebase.google.com/project/here2-474221/settings/general
2. Scroll to "Your apps"
3. Click "Web app" or create new web app
4. Copy the `firebaseConfig` object

Example:
```javascript
const firebaseConfig = {
  apiKey: "AIza...",
  authDomain: "here2-474221.firebaseapp.com",
  projectId: "here2-474221",
  storageBucket: "here2-474221.appspot.com",
  messagingSenderId: "123...",
  appId: "1:123..."
};
```

## Step 3: Configure Extension

Edit `browser_extension/scripts/background.js`:

```javascript
// Replace this section (around line 9-15)
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",              // Replace with actual API key
  authDomain: "here2-474221.firebaseapp.com",
  projectId: "here2-474221",
  storageBucket: "here2-474221.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID", // Replace with actual sender ID
  appId: "YOUR_APP_ID"                 // Replace with actual app ID
};
```

## Step 4: Install Extension

### Chrome / Edge

1. Open `chrome://extensions/` (or `edge://extensions/`)
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Navigate to `/media/im3/plus/lab4/re_news/service_farm/browser_extension`
5. Click "Select Folder"
6. Extension should appear in toolbar

### Firefox

1. Open `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on..."
3. Navigate to `/media/im3/plus/lab4/re_news/service_farm/browser_extension`
4. Select `manifest.json`
5. Extension loads (note: temporary in Firefox, reloads on browser restart)

For permanent Firefox installation:
1. Package extension: `web-ext build` (requires web-ext tool)
2. Submit to Mozilla Add-ons: https://addons.mozilla.org/developers/
3. Wait for review (~1-7 days)

## Step 5: Verify Installation

### 5.1 Check Extension Status

1. Click extension icon in toolbar
2. Popup should show:
   - "Polling: Active" (green)
   - "Current Task: None"
   - Pending/Completed counts

### 5.2 Check Browser Console

1. Go to `chrome://extensions/` (or `edge://extensions/`)
2. Find "HERE News - Rogue URL Extractor"
3. Click "Inspect views: background page" (or "service worker")
4. Console should show:
   ```
   🚀 HERE News Rogue URL Extractor - Background worker initialized
   🔄 Starting task polling...
   🔍 Polling for pending tasks...
   ✅ No pending tasks
   ```

### 5.3 Create Test Task

Test the extension with a manual Firestore task:

```javascript
// In Firebase Console → Firestore → Data
// Create document in collection: rogue_extraction_tasks

{
  url: "https://www.bbc.com/news/world",
  status: "pending",
  created_at: <Timestamp: now>,
  priority: 0,
  attempts: 0
}
```

Watch extension console - should see:
```
📋 Found task: abcdef12...
🌐 Opening URL: https://www.bbc.com/news/world
✅ Metadata extracted: { title: "...", description: "..." }
✅ Task completed: abcdef12...
```

## Step 6: Deploy CloudRun Integration

The CloudRun integration is already in `main.py`. Deploy latest code:

```bash
cd /media/im3/plus/lab4/re_news/service_farm

# Commit changes
git add .
git commit -m "feat: Add browser extension for rogue URL extraction

- Created browser extension with mobile emulation
- Added Firestore task queue for blocked URLs
- Integrated rogue URL detection in CloudRun
- Check extension cache before attempting extraction
- Queue blocked URLs for manual extraction"

# Deploy to CloudRun
./deploy.sh
```

## Step 7: Test End-to-End

### 7.1 Submit Reuters URL

```bash
# Test with known blocked URL
curl -X POST "https://story-engine-here-179431661561.us-central1.run.app/trigger/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.reuters.com/investigations/trump-20-maga-aligned-influencers-media-emerge-new-mainstream-2025-11-08/"
  }'
```

### 7.2 Watch CloudRun Logs

```bash
gcloud run services logs read story-engine-here \
  --project=here2-474221 \
  --region=us-central1 \
  --limit=50
```

Should see:
```
🌐 Extracting fresh content from https://www.reuters.com/...
🔄 Retrying with screenshot capture
📸 Uploading screenshot to GCS (123456 bytes)
✅ Screenshot saved: gs://here2-474221-extraction-artifacts/domains/reuters.com/.../screenshot.png
🚨 Detected rogue URL (bot blocking): https://www.reuters.com/...
📋 Created rogue extraction task: abcdef123...
📋 Queued for browser extension extraction
```

### 7.3 Watch Extension Extract Metadata

Extension console should show (within 30 seconds):
```
📋 Found task: abcdef123...
🌐 Opening URL: https://www.reuters.com/...
✅ Metadata extracted: { title: "Trump: 20 MAGA-aligned influencers...", ... }
✅ Task completed: abcdef123...
```

### 7.4 Verify Firestore

Check task in Firestore Console:
```
Collection: rogue_extraction_tasks
Document: abcdef123...

Fields:
  url: "https://www.reuters.com/..."
  status: "completed"  ← Should change from "pending"
  metadata: {
    title: "Trump: 20 MAGA-aligned influencers...",
    description: "...",
    thumbnail_url: "...",
    canonical_url: "...",
    ...
  }
  completed_at: <Timestamp>
```

### 7.5 Test Cache Hit

Submit same Reuters URL again:

```bash
curl -X POST "https://story-engine-here-179431661561.us-central1.run.app/trigger/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.reuters.com/investigations/trump-20-maga-aligned-influencers-media-emerge-new-mainstream-2025-11-08/"
  }'
```

CloudRun logs should show:
```
💾 Cache hit for rogue URL: https://www.reuters.com/...
   Title: Trump: 20 MAGA-aligned influencers...
🎯 Using browser extension metadata (rogue URL cache hit)
✅ Rogue extraction completed: 45 words from browser extension
```

**No new extraction!** - Uses cached metadata from Firestore.

## Troubleshooting

### Extension not polling

**Symptom**: Popup shows "Polling: Stopped"

**Fix**:
1. Click extension icon
2. Click "▶️ Start Polling"
3. Check browser console for errors

### Firebase permission denied

**Symptom**: Console shows "Missing or insufficient permissions"

**Fix**:
1. Verify Firestore rules deployed: `firebase deploy --only firestore:rules`
2. Check rules allow read/write: https://console.firebase.google.com/project/here2-474221/firestore/rules
3. Ensure API key is correct in `background.js`

### Tasks not appearing in extension

**Symptom**: CloudRun queues task, extension doesn't pick it up

**Fix**:
1. Check Firestore collection name: `rogue_extraction_tasks` (no typos)
2. Verify task status is `"pending"` (not `"completed"`)
3. Check extension polling: popup should show "Polling: Active"
4. Manually refresh: Click "🔄 Refresh Status"

### Metadata extraction fails

**Symptom**: Extension opens tab but returns empty metadata

**Fix**:
1. Check if site has Open Graph tags (view page source, search for `og:`)
2. Verify mobile User-Agent works (test manually in DevTools)
3. Check content script loads (inspect tab, check console)
4. Try different `bot_type` in background.js (facebook, telegram)

### CloudRun not detecting rogue URLs

**Symptom**: Screenshots captured but no Firestore task created

**Fix**:
1. Check `is_rogue_url()` logic in `rogue_url_handler.py`
2. Verify screenshot_bytes exists: look for "📸 Uploading screenshot to GCS" in logs
3. Check content_text for blocking keywords: "verification required", "captcha", etc.

## Security Considerations

### Production Deployment

Before production use, secure the Firestore access:

**Option 1: Firebase Anonymous Auth**
```javascript
// In background.js
import { getAuth, signInAnonymously } from 'firebase/auth';

const auth = getAuth(app);
await signInAnonymously(auth);
```

Update Firestore rules:
```javascript
match /rogue_extraction_tasks/{taskId} {
  allow read, write: if request.auth != null;
}
```

**Option 2: API Key Validation**
Create Cloud Function to validate requests:
```javascript
// CloudRun checks API key before writing to Firestore
const validApiKey = process.env.EXTENSION_API_KEY;
if (request.headers['x-api-key'] !== validApiKey) {
  return res.status(401).send('Unauthorized');
}
```

**Option 3: Service Account (Recommended)**
1. Create service account for extension
2. Grant Firestore read/write to specific collection
3. Use service account credentials in extension

### Rate Limiting

Prevent abuse by limiting task creation:

```python
# In rogue_url_handler.py
def create_task(...):
    # Check if too many pending tasks from same domain
    domain = urlparse(url).netloc
    pending_count = len(list(self.collection
        .where('status', '==', 'pending')
        .where('url', '>=', f'https://{domain}')
        .where('url', '<', f'https://{domain}\uffff')
        .stream()))

    if pending_count > 10:
        print(f"⚠️ Too many pending tasks for {domain}, skipping")
        return None
```

## Cost Analysis

### Free Tier (Estimated)

- **Firestore reads**: ~8,640/day (polling every 30s) → Free (50k/day limit)
- **Firestore writes**: ~50/day (task creation + updates) → Free (20k/day limit)
- **Firestore storage**: ~1 MB (100 tasks × 10KB) → Free (1 GB limit)

### Paid Tier (If exceeded)

- **Firestore reads**: $0.06 per 100k reads
- **Firestore writes**: $0.18 per 100k writes
- **Firestore storage**: $0.18 per GB/month

**Estimated cost**: < $0.10/month for ~50 rogue URLs/day

## Monitoring

### Extension Metrics

Track extraction success rate:

```javascript
// In background.js, after processTask():
chrome.storage.local.get(['stats'], (result) => {
  const stats = result.stats || { total: 0, success: 0, failed: 0 };
  stats.total++;
  if (metadata) stats.success++;
  else stats.failed++;
  chrome.storage.local.set({ stats });
});
```

Display in popup:
```javascript
// In popup.js
chrome.storage.local.get(['stats'], (result) => {
  const stats = result.stats || { total: 0, success: 0, failed: 0 };
  const successRate = (stats.success / stats.total * 100).toFixed(1);
  document.getElementById('successRate').textContent = `${successRate}%`;
});
```

### CloudRun Metrics

Track rogue URL detection rate:

```bash
# Check rogue task creation count
gcloud logging read 'resource.type="cloud_run_revision"
  AND textPayload=~"Created rogue extraction task"' \
  --project=here2-474221 \
  --limit=100 \
  --format="table(timestamp,textPayload)"
```

## Next Steps

1. **Add extension icons**: Create 16x16, 48x48, 128x128 PNGs
2. **Implement proper auth**: Use Firebase Anonymous Auth or service account
3. **Add retry logic**: Retry failed extractions after timeout
4. **Support manual submission**: Allow users to manually queue URLs via popup
5. **Add analytics**: Track extraction success rate, response times
6. **Package for distribution**: Submit to Chrome Web Store / Firefox Add-ons

## Support

For issues or questions:
1. Check browser console for errors
2. Check Firestore collection in Firebase Console
3. Check CloudRun logs: `gcloud run services logs read story-engine-here`
4. Create issue in repository
