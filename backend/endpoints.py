"""
Gen2 /url endpoint - Submit URL for processing

Instant best shot pattern with iframely metadata

Refactored: Uses PageRepository for all data access
"""
import os
import httpx
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter, HTTPException
import asyncpg
from services.job_queue import JobQueue
from repositories import PageRepository
from models.domain.page import Page
from utils.id_generator import generate_page_id

router = APIRouter()

# Globals (initialized on startup)
db_pool = None
job_queue = None
page_repo = None

# Iframely config
IFRAMELY_API_KEY = os.getenv('IFRAMELY_API_KEY', '')
IFRAMELY_URL = "https://iframe.ly/api/iframely"


async def init_services():
    """Initialize database pool, job queue, and repositories"""
    global db_pool, job_queue, page_repo

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

    if page_repo is None:
        page_repo = PageRepository(db_pool)

    return page_repo, job_queue


def normalize_url(url: str) -> str:
    """
    Normalize URL to canonical form

    Removes:
    - www. prefix
    - Trailing slashes (EXCEPT when query string present, to preserve query-driven routes)
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

    # Remove trailing slash ONLY if no query string present
    # (preserves path for query-driven routes like /case-detail/?id=123)
    if parsed.query:
        path = parsed.path  # Keep trailing slash with query strings
    else:
        path = parsed.path.rstrip('/') if parsed.path != '/' else '/'

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


@router.post("/artifacts")
async def submit_artifact(url: str):
    """
    Submit URL and get instant best shot metadata for any content type

    Creates or retrieves an artifact (web page, video, image, PDF, etc.) from a URL.
    Returns the best available metadata immediately, commissioning background workers
    for full extraction and enrichment.

    **ARCHITECTURE PRINCIPLE**: Resource-First, Not Pipeline
    - Returns immediately (< 500ms) with BEST AVAILABLE state
    - Commissions workers in background for enrichment
    - Progressive enhancement: stub → preview → extracted → semantic → complete

    **Endpoint:**
    - POST /api/v2/artifacts?url=https://example.com

    **Returns:**
    - New URL: Instant preview from iframely (title, description, thumbnail, author)
    - Existing URL: Best cached state from database
    - Status codes: 200 (success), 400 (invalid URL)

    **Future-proof:**
    - Currently supports web pages
    - Designed to support videos, images, PDFs, podcasts in future
    """
    page_repo, queue = await init_services()

    # Validate URL
    if not url or not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    canonical_url = normalize_url(url)

    # Check if URL already exists using repository
    existing = await page_repo.get_by_canonical_url(canonical_url)

    if existing:
        # SCENARIO A: Existing URL - return best shot from DB
        # Get entity/claim counts using repository
        entity_count = await page_repo.get_entity_count(existing.id)
        claim_count = await page_repo.get_claim_count(existing.id)

        # Commission background update if needed
        commissioned = False
        if existing.status == 'failed':
            # Extraction failed - retry extraction
            await queue.enqueue('queue:extraction:high', {
                'page_id': str(existing.id),
                'url': url,
                'retry_count': 0
            })
            commissioned = True
        elif existing.status == 'semantic_failed':
            # Semantic failed - retry semantic (extraction was OK)
            await queue.enqueue('queue:semantic:high', {
                'page_id': str(existing.id),
                'url': url,
                'retry_count': 0
            })
            commissioned = True

        return {
            "page_id": str(existing.id),
            "url": existing.url,
            "canonical_url": existing.canonical_url,
            "title": existing.title,
            "description": existing.description,
            "author": existing.author,
            "thumbnail_url": existing.thumbnail_url,
            "status": existing.status,
            "language": existing.language,
            "word_count": existing.word_count,
            "entity_count": entity_count,
            "claim_count": claim_count,
            "pub_time": existing.pub_time.isoformat() if existing.pub_time else None,
            "metadata_confidence": existing.metadata_confidence or 0.0,
            "created_at": existing.created_at.isoformat(),
            "updated_at": existing.updated_at.isoformat() if existing.updated_at else None,
            "_commissioned": commissioned  # Did we re-enqueue?
        }

    # SCENARIO B: New URL - create stub with iframely instant metadata
    page_id = generate_page_id()

    # Get iframely metadata (< 500ms, 99% success rate in Gen1)
    iframely_meta = await get_iframely_metadata(url)

    if iframely_meta:
        # iframely succeeded - use rich metadata
        title = iframely_meta.get('title')
        description = iframely_meta.get('description')
        language = iframely_meta.get('language', 'en')
        author = iframely_meta.get('author')
        thumbnail_url = iframely_meta.get('image')
        site_name = iframely_meta.get('site')  # Publisher name from iframely

        # Use iframely's canonical URL if different (deduplication!)
        canonical_from_iframely = iframely_meta.get('canonical_url')
        if canonical_from_iframely and canonical_from_iframely != url:
            canonical_url = normalize_url(canonical_from_iframely)

            # Check again if THIS canonical URL exists using repository
            existing_canonical = await page_repo.get_by_canonical_url(canonical_url)

            if existing_canonical:
                # Redirect to existing page (iframely found duplicate!)
                print(f"✅ iframely deduplication: {url} → {canonical_url}")
                # Re-fetch existing page data and return
                return await submit_artifact(canonical_url)

        # Extract domain from URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
        except:
            domain = None

        # Create page with iframely metadata using repository
        page = Page(
            id=page_id,
            url=url,
            canonical_url=canonical_url,
            title=title,
            description=description,
            author=author,
            thumbnail_url=thumbnail_url,
            language=language,
            status='preview',
            word_count=0,
            metadata_confidence=0.8,  # iframely is pretty reliable
            content_text='',
            domain=domain,
            site_name=site_name
        )
        await page_repo.create(page)

        # Commission extraction (async, doesn't block response)
        await queue.enqueue('queue:extraction:high', {
            'page_id': str(page_id),
            'url': url,
            'retry_count': 0
        })

        # Return immediately with iframely metadata
        return {
            "page_id": str(page_id),
            "url": url,
            "canonical_url": canonical_url,
            "title": title,
            "description": description,
            "author": author,
            "thumbnail_url": thumbnail_url,
            "status": 'preview',
            "language": language,
            "word_count": None,
            "entity_count": 0,
            "claim_count": 0,
            "pub_time": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "_commissioned": True,
            "_iframely_used": True
        }

    else:
        # iframely failed - quick domain guess
        domain = urlparse(url).netloc.lower()
        site_name = None  # Will be extracted during content extraction
        language = 'en'  # Default
        if any(x in domain for x in ['.cn', '.zh', 'chinese']):
            language = 'zh'
        elif any(x in domain for x in ['.fr', 'french']):
            language = 'fr'

        # Create stub page using repository
        page = Page(
            id=page_id,
            url=url,
            canonical_url=canonical_url,
            language=language,
            status='stub',
            word_count=0,
            metadata_confidence=0.0,
            content_text='',
            domain=domain,
            site_name=site_name
        )
        await page_repo.create(page)

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
async def get_page_status(page_id: str, include_data: bool = True):
    """
    Get current status of a page

    **Contract (open for discussion):**
    - Returns: { page_id, status, word_count?, entity_count?, ... }
    - Question: How much data to return? Full content? Just metadata?
    - Question: Include related entities/events in response?

    **Parameters:**
    - include_data: If true, includes claims and entities in response (default: true)
    """
    page_repo, _ = await init_services()

    # Use repository to get page
    page = await page_repo.get_by_id(page_id)

    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Get entity and claim counts using repository
    entity_count = await page_repo.get_entity_count(page_id)
    claim_count = await page_repo.get_claim_count(page_id)

    # Calculate semantic confidence (0.0-1.0) based on semantic analysis quality
    # This tells the user how reliable the semantic data (claims/entities) is
    word_count = page.word_count or 0
    status = page.status

    if status == 'semantic_complete':
        # Full semantic analysis succeeded
        # Confidence scales with content quality
        if word_count >= 300 and claim_count >= 3 and entity_count >= 3:
            semantic_confidence = 1.0  # High quality article with rich semantic data
        elif word_count >= 150 and claim_count >= 1:
            semantic_confidence = 0.7  # Decent article
        else:
            semantic_confidence = 0.5  # Semantic complete but low quality
    elif status in ['extracted', 'preview'] and word_count >= 100:
        # Extraction succeeded but semantic pending/not started
        semantic_confidence = 0.3  # Has potential
    elif status == 'semantic_failed' and word_count < 100:
        # Insufficient content - paywall/teaser
        semantic_confidence = 0.0
    elif status == 'semantic_failed':
        # Semantic failed for other reasons (no claims, etc.)
        semantic_confidence = 0.1
    else:
        # Stub, preview with low word count, or failed
        semantic_confidence = 0.0

    result = {
        "page_id": str(page.id),
        "url": page.url,
        "canonical_url": page.canonical_url,
        "title": page.title,
        "description": page.description,
        "author": page.author,
        "thumbnail_url": page.thumbnail_url,
        "status": page.status,
        "language": page.language,
        "word_count": page.word_count,
        "entity_count": entity_count,
        "claim_count": claim_count,
        "pub_time": page.pub_time.isoformat() if page.pub_time else None,
        "metadata_confidence": page.metadata_confidence or 0.0,
        "semantic_confidence": semantic_confidence,  # NEW: semantic analysis quality (0.0=no semantic data, 1.0=rich semantic data)
        "created_at": page.created_at.isoformat(),
        "updated_at": page.updated_at.isoformat() if page.updated_at else None
    }

    # Include full claims and entities data if requested and semantic_complete
    if include_data and page.status == 'semantic_complete':
        # Get entities using repository
        entities = await page_repo.get_entities(page_id)

        # Get claims using repository
        claims = await page_repo.get_claims(page_id)

        result['entities'] = entities
        result['claims'] = claims

    return result


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


@router.get("/claims")
async def get_claims(page_id: str):
    """Get all claims for a page"""
    page_repo, _ = await init_services()

    # Use repository to get claims
    claims = await page_repo.get_claims(page_id)
    return {"claims": claims}


@router.get("/entities")
async def get_entities(page_id: str):
    """Get all entities mentioned in a page"""
    page_repo, _ = await init_services()

    # Use repository to get entities
    entities = await page_repo.get_entities(page_id)

    return {
        "entities": entities
    }
