"""
Gen2 /url endpoint - Submit URL for processing

Instant best shot pattern with iframely metadata
"""
import os
import uuid
import httpx
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter, HTTPException
import asyncpg
from services.job_queue import JobQueue

router = APIRouter()

# Globals (initialized on startup)
db_pool = None
job_queue = None

# Iframely config
IFRAMELY_API_KEY = os.getenv('IFRAMELY_API_KEY', '')
IFRAMELY_URL = "https://iframe.ly/api/iframely"


async def init_services():
    """Initialize database pool and job queue"""
    global db_pool, job_queue

    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=2,
            max_size=10
        )

    if job_queue is None:
        job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
        await job_queue.connect()

    return db_pool, job_queue


def normalize_url(url: str) -> str:
    """
    Normalize URL to canonical form

    Removes:
    - www. prefix
    - Trailing slashes
    - URL fragments (#)
    - Common tracking parameters (utm_*, fbclid, etc.)

    This deduplication happens at instant tier to:
    1. Prevent duplicate iframely calls (saves API quota)
    2. Prevent duplicate worker jobs (saves compute)
    3. Enable instant cache hits on same content
    """
    parsed = urlparse(url)

    # Remove www prefix
    netloc = parsed.netloc.lower()
    if netloc.startswith('www.'):
        netloc = netloc[4:]

    # Remove trailing slash
    path = parsed.path.rstrip('/')

    # Remove tracking parameters (marketing noise)
    # Common tracking params: utm_source, utm_medium, utm_campaign, fbclid, gclid, etc.
    if parsed.query:
        from urllib.parse import parse_qs
        params = parse_qs(parsed.query)

        # Blacklist of tracking parameters to remove
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'mc_cid', 'mc_eid',
            '_ga', '_gl', 'ref', 'source', 'campaign'
        }

        # Keep only non-tracking params
        clean_params = {k: v for k, v in params.items() if k.lower() not in tracking_params}

        # Rebuild query string (sorted for consistency)
        query = '&'.join(f"{k}={v[0]}" for k, v in sorted(clean_params.items()))
    else:
        query = ''

    canonical = urlunparse((
        parsed.scheme or 'https',
        netloc,
        path,
        '',  # params
        query,
        ''   # fragment removed
    ))
    return canonical


async def get_iframely_metadata(url: str) -> dict:
    """
    Get instant metadata from iframely (< 500ms usually)

    Returns rich metadata: title, description, language, image, author, etc.
    Critical for instant tier - gives 99% accurate metadata without extraction

    Returns None on failure (timeout, API error) - graceful degradation
    """
    if not IFRAMELY_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=0.8) as client:  # 800ms timeout
            response = await client.get(
                IFRAMELY_URL,
                params={
                    'url': url,
                    'key': IFRAMELY_API_KEY,  # iframely uses 'key' not 'api_key'
                    'iframe': 0,
                    'omit_script': 1
                }
            )

            if response.status_code == 200:
                data = response.json()

                # Check if iframely returned an error (e.g., status: 403, 404)
                # When successful, iframely doesn't include 'status' field
                # When failed, it returns {status: 403/404, error: "..."}
                if 'error' in data or (data.get('status') and data.get('status') != 200):
                    # iframely failed to fetch (403, paywall, etc.)
                    return None

                meta = data.get('meta', {})

                return {
                    'title': meta.get('title'),
                    'description': meta.get('description'),
                    'language': meta.get('language', 'en'),
                    'author': meta.get('author'),
                    'site': meta.get('site'),
                    'canonical_url': meta.get('canonical') or url,
                    'image': data.get('links', {}).get('thumbnail', [{}])[0].get('href') if data.get('links') else None
                }
            else:
                return None

    except (httpx.TimeoutException, httpx.RequestError, Exception):
        # Graceful degradation - iframely down doesn't break our API
        return None


@router.post("/url")
async def submit_url(url: str):
    """
    Instant best shot for any URL (new or existing)

    **ARCHITECTURE PRINCIPLE**: Resource-First, Not Pipeline
    - Returns immediately (< 500ms) with BEST AVAILABLE state
    - Commissions workers in background for enrichment
    - Progressive enhancement: stub → partial → enriched → complete

    **Contract (open for discussion):**
    - Input: `url` query parameter (e.g., POST /url?url=https://example.com)
    - Output: Full resource state (whatever we have right now)
    - Status codes: 200 (always), 400 (invalid URL)

    **Questions:**
    1. Should we include content_text in response if already extracted?
    2. Should we trigger re-extraction if status='failed'?
    3. Quick domain-based language guess vs wait for extraction?
    """
    pool, queue = await init_services()

    # Validate URL
    if not url or not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    canonical_url = normalize_url(url)

    async with pool.acquire() as conn:
        # Check if URL already exists
        existing = await conn.fetchrow("""
            SELECT
                id, url, canonical_url, title, description,
                author, thumbnail_url, status, language,
                word_count, pub_time, created_at, updated_at
            FROM core.pages
            WHERE canonical_url = $1
        """, canonical_url)

        if existing:
            # SCENARIO A: Existing URL - return best shot from DB
            # Get entity/claim counts
            entity_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT entity_id)
                FROM core.page_entities
                WHERE page_id = $1
            """, existing['id'])

            claim_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM core.claims
                WHERE page_id = $1
            """, existing['id'])

            # Commission background update if needed (status='failed')
            if existing['status'] == 'failed':
                await queue.enqueue('queue:extraction:high', {
                    'page_id': str(existing['id']),
                    'url': url,
                    'retry_count': 0
                })

            return {
                "page_id": str(existing['id']),
                "url": existing['url'],
                "canonical_url": existing['canonical_url'],
                "title": existing['title'],
                "description": existing['description'],
                "author": existing['author'],
                "thumbnail_url": existing['thumbnail_url'],
                "status": existing['status'],
                "language": existing['language'],
                "word_count": existing['word_count'],
                "entity_count": entity_count,
                "claim_count": claim_count,
                "pub_time": existing['pub_time'].isoformat() if existing['pub_time'] else None,
                "created_at": existing['created_at'].isoformat(),
                "updated_at": existing['updated_at'].isoformat() if existing['updated_at'] else None,
                "_commissioned": existing['status'] == 'failed'  # Did we re-enqueue?
            }

        # SCENARIO B: New URL - create stub with iframely instant metadata
        page_id = uuid.uuid4()

        # Get iframely metadata (< 500ms, 99% success rate in Gen1)
        iframely_meta = await get_iframely_metadata(url)

        if iframely_meta:
            # iframely succeeded - use rich metadata
            title = iframely_meta.get('title')
            description = iframely_meta.get('description')
            language = iframely_meta.get('language', 'en')
            author = iframely_meta.get('author')
            thumbnail_url = iframely_meta.get('image')

            # Use iframely's canonical URL if different (deduplication!)
            canonical_from_iframely = iframely_meta.get('canonical_url')
            if canonical_from_iframely and canonical_from_iframely != url:
                canonical_url = normalize_url(canonical_from_iframely)

                # Check again if THIS canonical URL exists
                existing_canonical = await conn.fetchrow("""
                    SELECT id, status FROM core.pages WHERE canonical_url = $1
                """, canonical_url)

                if existing_canonical:
                    # Redirect to existing page (iframely found duplicate!)
                    print(f"✅ iframely deduplication: {url} → {canonical_url}")
                    # Re-fetch existing page data and return
                    return await submit_url(canonical_url)

            # Insert page with iframely metadata
            await conn.execute("""
                INSERT INTO core.pages (
                    id, url, canonical_url, title, description,
                    author, thumbnail_url, language,
                    status, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'preview', NOW())
            """, page_id, url, canonical_url, title, description,
                author, thumbnail_url, language)

            status = 'preview'  # Has metadata, pending extraction

        else:
            # iframely failed - quick domain guess
            domain = urlparse(url).netloc.lower()
            language = 'en'  # Default
            if any(x in domain for x in ['.cn', '.zh', 'chinese']):
                language = 'zh'
            elif any(x in domain for x in ['.fr', 'french']):
                language = 'fr'

            await conn.execute("""
                INSERT INTO core.pages (id, url, canonical_url, status, language, created_at)
                VALUES ($1, $2, $3, 'stub', $4, NOW())
            """, page_id, url, canonical_url, language)

            title = None
            description = None
            author = None
            thumbnail_url = None
            status = 'stub'

        # Commission extraction (async, doesn't block response)
        await queue.enqueue('queue:extraction:high', {
            'page_id': str(page_id),
            'url': url,
            'retry_count': 0
        })

        # Return immediately with best available metadata (< 500ms total)
        return {
            "page_id": str(page_id),
            "url": url,
            "canonical_url": canonical_url,
            "title": title,  # From iframely if available
            "description": description,  # From iframely if available
            "author": author,  # From iframely if available
            "thumbnail_url": thumbnail_url,  # From iframely if available
            "status": status,  # 'preview' if iframely, 'stub' if not
            "language": language,
            "word_count": None,  # Pending extraction
            "entity_count": 0,
            "claim_count": 0,
            "pub_time": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "_commissioned": True,  # Workers commissioned in background
            "_iframely_used": iframely_meta is not None  # Diagnostic
        }


@router.get("/url/{page_id}")
async def get_page_status(page_id: str):
    """
    Get current status of a page

    **Contract (open for discussion):**
    - Returns: { page_id, status, word_count?, entity_count?, ... }
    - Question: How much data to return? Full content? Just metadata?
    - Question: Include related entities/events in response?
    """
    pool, _ = await init_services()

    try:
        page_uuid = uuid.UUID(page_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page_id format")

    async with pool.acquire() as conn:
        page = await conn.fetchrow("""
            SELECT
                id,
                url,
                canonical_url,
                title,
                description,
                author,
                thumbnail_url,
                status,
                language,
                word_count,
                pub_time,
                created_at,
                updated_at
            FROM core.pages
            WHERE id = $1
        """, page_uuid)

        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        # Get entity count
        entity_count = await conn.fetchval("""
            SELECT COUNT(DISTINCT entity_id)
            FROM core.page_entities
            WHERE page_id = $1
        """, page_uuid)

        # Get claim count
        claim_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM core.claims
            WHERE page_id = $1
        """, page_uuid)

        return {
            "page_id": str(page['id']),
            "url": page['url'],
            "canonical_url": page['canonical_url'],
            "title": page['title'],
            "description": page['description'],
            "author": page['author'],
            "thumbnail_url": page['thumbnail_url'],
            "status": page['status'],
            "language": page['language'],
            "word_count": page['word_count'],
            "entity_count": entity_count,
            "claim_count": claim_count,
            "pub_time": page['pub_time'].isoformat() if page['pub_time'] else None,
            "created_at": page['created_at'].isoformat(),
            "updated_at": page['updated_at'].isoformat() if page['updated_at'] else None
        }


@router.get("/queue/stats")
async def get_queue_stats():
    """
    Get queue statistics (diagnostic endpoint)

    Useful for monitoring worker backlog
    """
    _, queue = await init_services()

    return {
        "extraction": await queue.queue_length('queue:extraction:high'),
        "semantic": await queue.queue_length('queue:semantic:high'),
        "event": await queue.queue_length('queue:event:high'),
        "enrichment": await queue.queue_length('queue:enrichment:high')
    }
