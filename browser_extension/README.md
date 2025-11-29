# HERE News - Rogue URL Extractor Extension

Browser extension that extracts Open Graph metadata from news sites that block automated scrapers (like Reuters).

## How It Works

1. **CloudRun detects blocked URL** (CAPTCHA, HTTP 401/403, bot detection)
2. **Task queued to Firestore** (`rogue_extraction_tasks` collection)
3. **Extension polls Firestore** every 30 seconds for pending tasks
4. **Opens URL in background tab** with mobile emulation (iPhone 14 Pro Max)
5. **Extracts Open Graph metadata** using real browser (bypasses all bot detection)
6. **Uploads metadata to Firestore** for CloudRun to use
7. **Closes tab** and repeats

## Features

- ✅ Mobile User-Agent (reduces ads, matches CloudRun Playwright)
- ✅ Background processing (doesn't interrupt your browsing)
- ✅ Real browser fingerprint (bypasses IP checks, TLS fingerprinting, behavioral analysis)
- ✅ Handles ~2-5% of URLs that are blocked
- ✅ One-time extraction with caching for reuse

## Installation

### Chrome/Edge

1. Open `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `browser_extension` folder
5. Extension will appear in toolbar

### Firefox

1. Open `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select `browser_extension/manifest.json`
4. Extension loads (temporary, until browser restart)

## Configuration

Edit `scripts/background.js` and update Firebase config:

```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "here2-474221.firebaseapp.com",
  projectId: "here2-474221",
  // ... other config
};
```

Get your config from: https://console.firebase.google.com/project/here2-474221/settings/general

## Usage

1. **Install extension** (see above)
2. **Extension auto-starts polling** for tasks
3. **CloudRun automatically queues blocked URLs**
4. **Extension processes them** in background
5. **Check popup** (click extension icon) to see status

### Manual Control

- Click extension icon to open popup
- View current task and stats
- Start/Stop polling
- Refresh status

## Architecture

```
CloudRun (story-engine-here)
  └─> Detects blocked URL
  └─> Creates Firestore task
       └─> rogue_extraction_tasks/{task_id}
            ├─ url
            ├─ status: "pending"
            ├─ created_at
            └─ screenshot_url (optional)

Browser Extension
  └─> Polls Firestore every 30s
  └─> Picks up "pending" task
  └─> Opens URL with mobile emulation
  └─> Extracts metadata (content.js)
  └─> Uploads to Firestore
       └─> rogue_extraction_tasks/{task_id}
            ├─ status: "completed"
            ├─ metadata: {...}
            └─ completed_at

CloudRun
  └─> Reads completed task
  └─> Uses metadata for preview
  └─> Caches in GCS (30 day TTL)
```

## Mobile Emulation

Extension uses iPhone 14 Pro Max profile to match CloudRun Playwright:

- **User-Agent**: Safari on iOS 17.1
- **Viewport**: 430x932 @ 3x density
- **Benefits**:
  - Lighter pages (less ads)
  - Faster loading
  - Consistent with CloudRun extractions

## Security

- Extension requires `debugger` permission (for User-Agent override)
- Only processes URLs from trusted Firestore collection
- No data collection or tracking
- Runs entirely locally on your machine

## Troubleshooting

### Extension not picking up tasks

1. Check Firestore rules allow read/write
2. Verify Firebase config is correct
3. Check browser console for errors (F12 → Console)
4. Ensure polling is active (check popup)

### Metadata extraction fails

1. Check site blocks even mobile browsers
2. Verify content script loads (check page console)
3. Try manually visiting URL to test
4. Check Firestore task has `error_message`

### Extension slows down browser

1. Reduce polling frequency (edit `POLL_INTERVAL_MS`)
2. Limit concurrent tasks (currently 1 at a time)
3. Disable when not needed (stop polling in popup)

## Cost Savings

- **Before**: iFramely API call for every blocked URL ($0.0001-0.001 per call)
- **After**: One-time extraction + 30-day cache
- **Estimated savings**: 60-80% reduction in iFramely costs for blocked sites

## Limitations

- Only works when your machine is running and browser is open
- Still can't bypass sites that require login/subscription
- May trigger CAPTCHA on some sites (requires manual solving)
- Temporary in Firefox (reloads on browser restart)

## Development

### File Structure

```
browser_extension/
├── manifest.json          # Extension config
├── popup.html            # UI for extension popup
├── popup.js              # Popup UI logic
├── scripts/
│   ├── background.js     # Background worker (polls Firestore)
│   └── content.js        # Content script (extracts metadata)
└── icons/               # Extension icons (TODO)
```

### Testing Locally

1. Create test task in Firestore:
```javascript
{
  url: "https://www.reuters.com/world/test-article",
  status: "pending",
  created_at: Timestamp.now()
}
```

2. Watch extension console:
```
chrome://extensions/ → Extension → Background page → Console
```

3. Check extraction:
```
Should see: "📋 Found task: ...", "🌐 Opening URL: ...", "✅ Metadata extracted: ..."
```

## Next Steps

1. Add extension icons (16x16, 48x48, 128x128)
2. Implement Firestore task count queries in popup
3. Add error retry logic
4. Support for manual task submission via popup
5. Analytics/metrics for extraction success rate
