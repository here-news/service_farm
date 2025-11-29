# Browser Extension Troubleshooting Guide

## Common Errors

### 1. "Metadata extraction timeout"

**What it means**: The content script didn't respond within 60 seconds.

**Common causes**:
- **Heavy JavaScript sites** - Page takes >60s to fully load
- **CSP (Content Security Policy)** - Site blocks script injection
- **Readability.js not loading** - Dependency not injected properly
- **Infinite page load** - Site continuously loads resources

**How to diagnose**:
1. Open Chrome DevTools on the extension background page:
   - Go to `chrome://extensions/`
   - Find "HERE News - Rogue URL Extractor (PostgreSQL)"
   - Click "service worker" link
   - Check console logs

2. Check which URL is failing:
   ```
   🌐 Opening URL: https://example.com/article
   📱 Mobile User-Agent set for tab 123
   ❌ Task failed: cf6510e48d2f488e... Error: Metadata extraction timeout
   ```

3. Open the tab manually and check console:
   - The extension opens tabs in background
   - If you see the tab briefly, click it before it closes
   - Open DevTools → Console
   - Look for content script logs:
     ```
     🔍 HERE News - Content script loaded
     🚀 Starting extraction for: https://...
     📄 Page readyState: loading
     ⏳ Waiting for page load...
     ```

**Solutions**:
- **For heavy sites**: Increase `MAX_PROCESSING_TIME_MS` in `background_postgres.js` (currently 60000ms = 60s)
- **For CSP sites**: No workaround - these sites actively block extensions (mark as failed)
- **For slow loading**: Check if site has infinite loading spinners (some paywalls do this)

### 2. "Cannot inject script on this page"

**What it means**: Chrome blocked script injection due to CSP or permissions.

**Common causes**:
- **chrome:// pages** - Extensions can't run on internal Chrome pages
- **chrome-extension:// pages** - Can't inject into other extensions
- **Strict CSP headers** - Site explicitly blocks all inline scripts

**Solution**: Mark these tasks as failed automatically (no workaround).

### 3. "Extracted metadata is empty"

**What it means**: Content script ran but found no title or content.

**Common causes**:
- **CAPTCHA pages** - Bot detection page with no article content
- **Login walls** - "Sign in to continue" pages
- **Error pages** - 404, 403, 500 error pages
- **Blank pages** - Page hasn't rendered yet

**How to diagnose**:
Check content script logs:
```
✅ Metadata extracted in 2.34s: {
  title: null,
  word_count: 0,
  has_content: false
}
```

**Solution**: Mark as failed - these are legitimately inaccessible.

### 4. "Script injection failed"

**What it means**: Readability.js or content.js couldn't be injected.

**Common causes**:
- **Extension files missing** - Readability.js not in scripts/ folder
- **Tab closed too early** - Race condition
- **Chrome permissions** - Extension doesn't have access to that domain

**How to fix**:
1. Check extension files are present:
   ```
   browser_extension/
   ├── scripts/
   │   ├── background_postgres.js ✅
   │   ├── content.js ✅
   │   └── Readability.js ✅
   ```

2. Verify manifest.json has correct permissions:
   ```json
   "host_permissions": [
     "https://*/*",
     "http://*/*"
   ]
   ```

3. Reload extension:
   - Go to `chrome://extensions/`
   - Click reload button

## Debugging Tips

### Enable Verbose Logging

The extension now logs detailed info at each step:

**Background script logs** (`chrome://extensions/` → service worker):
```
🔍 Polling PostgreSQL for pending tasks...
📋 Found task: 5192ccd4528bc... https://politico.com/article
🌐 Opening URL: https://politico.com/article
📱 Mobile User-Agent set for tab 123 (iPhone 14 Pro Max)
✅ Readability.js injected
✅ content.js injected
✅ Metadata extracted: { title: "...", word_count: 500 }
✅ Task completed: 5192ccd4528bc...
🗑️  Removed User-Agent rule 123456
🗑️  Closed tab 123
```

**Content script logs** (DevTools on the tab):
```
🔍 HERE News - Content script loaded
🚀 Starting extraction for: https://politico.com/article
📄 Page readyState: loading
⏳ Waiting for page load...
✅ Page load event fired
📝 Extracted article: 1,234 words
✅ Metadata extracted in 3.45s: { title: "Article Title", word_count: 1234, has_content: true }
✅ Metadata sent to background script
```

### Check Rogue Task Queue

See what tasks are pending:
```bash
curl http://localhost:8080/api/rogue/poll
```

**Response (task available)**:
```json
{
  "task_id": "5192ccd4528bc9b3084f5f81fa6dde35",
  "url": "https://politico.com/article",
  "created_at": "2025-11-25T14:39:52.886574Z",
  "extraction_task_id": "119a48b3-..."
}
```

**Response (no tasks)**:
```json
{"detail":"No pending rogue tasks"}
```

### Check PostgreSQL Directly

View recent rogue tasks:
```bash
docker exec service-farm python3 << 'EOF'
import asyncio
from services.shared import task_store

async def check():
    await task_store.initialize()
    async with task_store.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, url, status, error_message, completed_at
            FROM tasks.rogue_tasks
            ORDER BY updated_at DESC
            LIMIT 5
        """)
        for row in rows:
            print(f"{row['status']:12} | {row['url'][:60]}")
            if row['error_message']:
                print(f"             Error: {row['error_message'][:80]}")
    await task_store.close()

asyncio.run(check())
EOF
```

### Manually Test a URL

1. **Submit URL to create rogue task**:
   ```bash
   # Submit a known paywall URL
   curl -X POST http://localhost:8080/submit \
     -F "url=https://www.politico.com/news/article" \
     -F "force=false"
   ```

2. **Watch extension pick it up**:
   - Open extension service worker console
   - Should see: "📋 Found task: ..."
   - Watch tab open and close

3. **Check if it completed**:
   ```bash
   curl http://localhost:8080/api/rogue/poll
   # Should return 404 if processed
   ```

4. **Check extraction task updated**:
   ```bash
   curl http://localhost:8080/api/task/{task_id}
   # Should show status: "completed" (not "blocked")
   ```

## Known Limitations

### Sites That Will Always Fail:
- **CAPTCHA walls** - reCAPTCHA, Cloudflare challenges
- **Login-required content** - NYTimes subscriber-only articles
- **Strict CSP sites** - Some enterprise sites block all extensions
- **Infinite loaders** - Sites that never finish loading

### Sites That Work Well:
- **Most news sites** - Washington Post, Politico, Reuters
- **Blogs** - Medium, Substack, personal blogs
- **Social media embeds** - Twitter/X cards, Facebook previews
- **Academic papers** - ArXiv, many journal sites

## Performance Notes

### Typical Extraction Times:
- **Fast sites** (static HTML): 2-5 seconds
- **Medium sites** (some JS): 5-15 seconds
- **Heavy sites** (lots of JS): 15-30 seconds
- **Timeout after**: 60 seconds

### Resource Usage:
- **Memory**: ~50MB per tab (Chrome overhead)
- **CPU**: Minimal (just DOM parsing)
- **Network**: Only loads the one page (no additional fetches)

### Polling Frequency:
- **3 seconds** between polls
- **Instant pickup** when task queued
- **~10 URLs/minute** maximum throughput (6s average per URL)

## FAQ

**Q: Why not use the Firestore version?**
A: Migrated to PostgreSQL for unified architecture. Firestore version is deprecated.

**Q: Can I process tasks faster?**
A: Reduce `POLL_INTERVAL_MS` from 3000ms to 1000ms, but this increases API load.

**Q: Can I run multiple extensions?**
A: Yes! Each extension polls independently. Load balancing is automatic via `FOR UPDATE SKIP LOCKED`.

**Q: What happens if extension crashes?**
A: Tasks stuck as "processing" will eventually timeout (future enhancement). Currently, manually reset:
```sql
UPDATE tasks.rogue_tasks SET status='pending' WHERE status='processing';
```

**Q: Can I see which URLs are timing out?**
A: Yes, check service worker console logs. The URL is logged when task is picked up.

**Q: Why does the tab close so fast?**
A: By design - extraction is fast (2-15s typically). Tab is kept open in background only during extraction.

## Need Help?

1. Check extension service worker console (`chrome://extensions/`)
2. Check service-farm logs (`docker logs service-farm --tail 100`)
3. Check PostgreSQL (`docker exec service-farm ...`)
4. Open GitHub issue with:
   - Error message from extension console
   - URL that's failing
   - Whether it times out or fails immediately
   - Browser version and OS
