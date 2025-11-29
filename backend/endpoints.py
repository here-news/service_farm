"""
Demo /artifacts/draft endpoint - Instant best shot with background commissioning
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter, HTTPException
import asyncpg
import redis.asyncio as redis

router = APIRouter()

# Redis for worker queues
redis_client = None

# Shared connection pool
db_pool = None

async def get_or_create_pool():
    """Get or create the shared database pool"""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host='demo-postgres',
            port=5432,
            user='demo_user',
            password='demo_pass',
            database='demo_phi_here',
            min_size=2,
            max_size=10
        )
    return db_pool

def normalize_url(url: str) -> str:
    """Normalize URL to canonical form"""
    parsed = urlparse(url)
    # Remove www, trailing slash, fragments
    netloc = parsed.netloc.lower()
    if netloc.startswith('www.'):
        netloc = netloc[4:]

    path = parsed.path.rstrip('/')

    canonical = urlunparse((
        parsed.scheme or 'https',
        netloc,
        path,
        '',  # params
        parsed.query,
        ''   # fragment removed
    ))
    return canonical


async def get_db_pool():
    """Get PostgreSQL connection pool - DEPRECATED, use get_or_create_pool()"""
    return await get_or_create_pool()


def detect_language_from_domain(url: str) -> tuple[str, float]:
    """Quick language detection from domain"""
    domain = urlparse(url).netloc.lower()

    # Common patterns
    if any(x in domain for x in ['.cn', '.zh', 'chinese']):
        return 'zh', 0.8
    if any(x in domain for x in ['.fr', 'french']):
        return 'fr', 0.8
    if any(x in domain for x in ['.es', 'spanish']):
        return 'es', 0.8

    # Default to English
    return 'en', 0.6


@router.post("/artifacts/draft")
async def create_artifact_draft(url: str):
    """
    Instant best shot for any URL (new or existing)

    Returns immediately with current state, commissions workers in background
    """
    canonical_url = normalize_url(url)
    artifact_id = hashlib.md5(canonical_url.encode()).hexdigest()[:16]

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Check if we already have this URL
        existing = await conn.fetchrow("""
            SELECT id, title, language, content_text, status, created_at
            FROM pages
            WHERE canonical_url = $1
        """, canonical_url)

        if existing:
            # SCENARIO A: Existing URL - return best shot from DB

            # Get entities from Neo4j (simplified - would query actual Neo4j)
            entities = await get_entities_for_page(str(existing['id']))

            # Get events
            events = await get_events_for_page(str(existing['id']))

            # Calculate confidence
            confidence = calculate_confidence(existing, entities, events)

            return {
                "artifact_id": str(existing['id']),
                "url": url,
                "canonical_url": canonical_url,
                "status": existing['status'] or "extracted",
                "best_shot": {
                    "title": existing['title'],
                    "language": existing['language'] or 'en',
                    "content_preview": existing['content_text'][:200] if existing['content_text'] else None,
                    "entities": entities,
                    "events": events,
                    "confidence": confidence,
                    "word_count": len(existing['content_text'].split()) if existing['content_text'] else 0
                },
                "created_at": existing['created_at'].isoformat(),
                "_processing": {
                    "commissioned": [],  # Already extracted
                    "message": "Existing content, no workers needed"
                }
            }
        else:
            # SCENARIO B: New URL - create stub, commission workers

            # Quick language detection from domain
            language, lang_conf = detect_language_from_domain(url)

            # Create artifact stub
            page_id = await conn.fetchval("""
                INSERT INTO pages (
                    url, canonical_url, status, language, language_confidence,
                    created_at, updated_at
                )
                VALUES ($1, $2, 'stub', $3, $4, NOW(), NOW())
                RETURNING id
            """, url, canonical_url, language, lang_conf)

            # Commission workers (async, don't wait)
            commissioned = await commission_workers(str(page_id), canonical_url)

            return {
                "artifact_id": str(page_id),
                "url": url,
                "canonical_url": canonical_url,
                "status": "stub",
                "best_shot": {
                    "title": None,
                    "language": language,
                    "content_preview": None,
                    "entities": [],
                    "events": [],
                    "confidence": 0.3,  # Low - just created
                    "word_count": 0
                },
                "created_at": datetime.utcnow().isoformat(),
                "_processing": {
                    "commissioned": commissioned,
                    "message": f"Workers commissioned: {', '.join(commissioned)}",
                    "poll_url": f"/artifacts/draft/{page_id}"
                }
            }


@router.get("/artifacts/draft/{artifact_id}")
async def get_artifact_status(artifact_id: str):
    """
    Poll for artifact status updates

    Used by webapp to check processing progress
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        page = await conn.fetchrow("""
            SELECT id, url, canonical_url, title, language, content_text,
                   status, created_at, updated_at
            FROM pages
            WHERE id = $1
        """, artifact_id)

        if not page:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Get entities and events
        entities = await get_entities_for_page(artifact_id)
        events = await get_events_for_page(artifact_id)

        # Calculate confidence
        confidence = calculate_confidence(page, entities, events)

        return {
            "artifact_id": artifact_id,
            "url": page['url'],
            "canonical_url": page['canonical_url'],
            "status": page['status'],
            "best_shot": {
                "title": page['title'],
                "language": page['language'],
                "content_preview": page['content_text'][:200] if page['content_text'] else None,
                "entities": entities,
                "events": events,
                "confidence": confidence,
                "word_count": len(page['content_text'].split()) if page['content_text'] else 0
            },
            "created_at": page['created_at'].isoformat(),
            "updated_at": page['updated_at'].isoformat()
        }


async def commission_workers(page_id: str, url: str) -> list[str]:
    """
    Enqueue workers to process this URL

    Returns list of commissioned worker names
    """
    global redis_client
    if not redis_client:
        redis_client = await redis.from_url('redis://demo-redis:6379')

    commissioned = []

    # 1. Extraction worker (high priority)
    await redis_client.lpush('queue:extraction:high', f'{{"page_id": "{page_id}", "url": "{url}"}}')
    commissioned.append("extraction_worker")

    # 2. Semantic worker will be auto-triggered after extraction
    # (No need to queue now - extraction worker will queue it)

    # 3. Event worker will be auto-triggered after semantic

    return commissioned


async def get_entities_for_page(page_id: str, conn=None) -> list[dict]:
    """Get entities extracted for this page"""
    if conn:
        # Use provided connection
        entities = await conn.fetch("""
            SELECT e.id, e.canonical_name, e.entity_type, e.confidence
            FROM entities e
            JOIN page_entities pe ON e.id = pe.entity_id
            WHERE pe.page_id = $1
        """, page_id)
    else:
        # Use shared pool
        pool = await get_or_create_pool()
        async with pool.acquire() as conn:
            entities = await conn.fetch("""
                SELECT e.id, e.canonical_name, e.entity_type, e.confidence
                FROM entities e
                JOIN page_entities pe ON e.id = pe.entity_id
                WHERE pe.page_id = $1
            """, page_id)

    return [
        {
            "id": str(e['id']),
            "name": e['canonical_name'],
            "type": e['entity_type'],
            "confidence": e['confidence']
        }
        for e in entities
    ]


async def get_events_for_page(page_id: str, conn=None) -> list[dict]:
    """Get events this page is part of, with hierarchy info"""
    if conn:
        # Use provided connection
        events = await conn.fetch("""
            SELECT e.id, e.title, e.summary, e.event_type, e.location,
                   e.event_start, e.confidence,
                   e.parent_event_id, e.event_scale, e.relationship_type,
                   p.title as parent_title
            FROM events e
            LEFT JOIN events p ON e.parent_event_id = p.id
            JOIN page_events pe ON e.id = pe.event_id
            WHERE pe.page_id = $1
        """, page_id)
    else:
        # Use shared pool
        pool = await get_or_create_pool()
        async with pool.acquire() as conn:
            events = await conn.fetch("""
                SELECT e.id, e.title, e.summary, e.event_type, e.location,
                       e.event_start, e.confidence,
                       e.parent_event_id, e.event_scale, e.relationship_type,
                       p.title as parent_title
                FROM events e
                LEFT JOIN events p ON e.parent_event_id = p.id
                JOIN page_events pe ON e.id = pe.event_id
                WHERE pe.page_id = $1
            """, page_id)

    return [
        {
            "id": str(e['id']),
            "title": e['title'],
            "summary": e['summary'],
            "event_type": e['event_type'],
            "location": e['location'],
            "event_start": e['event_start'].isoformat() if e['event_start'] else None,
            "confidence": e['confidence'],
            "event_scale": e['event_scale'],
            "parent_event_id": str(e['parent_event_id']) if e['parent_event_id'] else None,
            "parent_title": e['parent_title'],
            "relationship_type": e['relationship_type']
        }
        for e in events
    ]


@router.get("/events")
async def list_events():
    """
    List all events with hierarchical structure
    Returns top-level events (macro/meso without parents) with sub-events nested
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Get all events with hierarchy info
        events = await conn.fetch("""
            SELECT
                e.id,
                e.title,
                e.summary,
                e.event_type,
                e.location,
                e.event_start,
                e.confidence,
                e.created_at,
                e.updated_at,
                e.parent_event_id,
                e.event_scale,
                e.relationship_type,
                p.title as parent_title,
                COUNT(DISTINCT pe.page_id) as article_count
            FROM events e
            LEFT JOIN events p ON e.parent_event_id = p.id
            LEFT JOIN page_events pe ON e.id = pe.event_id
            GROUP BY e.id, p.title
            ORDER BY e.updated_at DESC
        """)

        # Build hierarchical structure
        events_by_id = {}
        top_level = []

        for e in events:
            event_dict = {
                "id": str(e['id']),
                "title": e['title'],
                "summary": e['summary'],
                "event_type": e['event_type'],
                "location": e['location'],
                "event_start": e['event_start'].isoformat() if e['event_start'] else None,
                "confidence": e['confidence'],
                "article_count": e['article_count'],
                "created_at": e['created_at'].isoformat(),
                "updated_at": e['updated_at'].isoformat(),
                "updated_at_dt": e['updated_at'],  # Keep datetime for sorting
                "event_scale": e['event_scale'],
                "parent_event_id": str(e['parent_event_id']) if e['parent_event_id'] else None,
                "parent_title": e['parent_title'],
                "relationship_type": e['relationship_type'],
                "sub_events": []
            }
            events_by_id[str(e['id'])] = event_dict

            # Top-level events are those without parents
            if not e['parent_event_id']:
                top_level.append(event_dict)

        # Nest sub-events under their parents
        for event_dict in events_by_id.values():
            if event_dict['parent_event_id'] and event_dict['parent_event_id'] in events_by_id:
                parent = events_by_id[event_dict['parent_event_id']]
                parent['sub_events'].append(event_dict)

        # Sort top_level events by updated_at_dt (newest first)
        top_level.sort(key=lambda x: x['updated_at_dt'], reverse=True)

        # Sort sub_events by updated_at_dt (newest first)
        for event_dict in events_by_id.values():
            if event_dict['sub_events']:
                event_dict['sub_events'].sort(key=lambda x: x['updated_at_dt'], reverse=True)

        # Remove the datetime objects before returning
        for event_dict in events_by_id.values():
            del event_dict['updated_at_dt']

        return {
            "events": top_level,
            "total_events": len(events),
            "top_level_events": len(top_level)
        }


@router.get("/events/{event_id}")
async def get_event_details(event_id: str):
    """
    Get detailed information about an event including full summary and articles
    """
    pool = await get_or_create_pool()

    async with pool.acquire() as conn:
        event = await conn.fetchrow("""
            SELECT e.id, e.title, e.summary, e.event_type, e.location,
                   e.event_start, e.confidence, e.created_at, e.updated_at,
                   e.parent_event_id, e.event_scale, e.relationship_type,
                   p.title as parent_title
            FROM events e
            LEFT JOIN events p ON e.parent_event_id = p.id
            WHERE e.id = $1
        """, event_id)

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get articles in this event
        articles = await conn.fetch("""
            SELECT p.id, p.title, p.url, p.created_at
            FROM pages p
            JOIN page_events pe ON p.id = pe.page_id
            WHERE pe.event_id = $1
            ORDER BY p.created_at DESC
        """, event_id)

        # Get entities involved
        entities = await conn.fetch("""
            SELECT DISTINCT e.id, e.canonical_name, e.entity_type
            FROM entities e
            JOIN event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
        """, event_id)

        # Get sub-events if this is a parent event
        sub_events = await conn.fetch("""
            SELECT id, title, event_scale, relationship_type,
                   COUNT(DISTINCT pe.page_id) as article_count
            FROM events e
            LEFT JOIN page_events pe ON e.id = pe.event_id
            WHERE e.parent_event_id = $1
            GROUP BY e.id, e.title, e.event_scale, e.relationship_type
        """, event_id)

        return {
            "id": str(event['id']),
            "title": event['title'],
            "summary": event['summary'],
            "event_type": event['event_type'],
            "location": event['location'],
            "event_start": event['event_start'].isoformat() if event['event_start'] else None,
            "confidence": event['confidence'],
            "created_at": event['created_at'].isoformat(),
            "updated_at": event['updated_at'].isoformat(),
            "event_scale": event['event_scale'],
            "parent_event_id": str(event['parent_event_id']) if event['parent_event_id'] else None,
            "parent_title": event['parent_title'],
            "relationship_type": event['relationship_type'],
            "sub_events": [
                {
                    "id": str(se['id']),
                    "title": se['title'],
                    "event_scale": se['event_scale'],
                    "relationship_type": se['relationship_type'],
                    "article_count": se['article_count']
                }
                for se in sub_events
            ],
            "articles": [
                {
                    "id": str(a['id']),
                    "title": a['title'],
                    "url": a['url'],
                    "created_at": a['created_at'].isoformat()
                }
                for a in articles
            ],
            "entities": [
                {
                    "id": str(e['id']),
                    "name": e['canonical_name'],
                    "type": e['entity_type']
                }
                for e in entities
            ]
        }


@router.get("/entities/{entity_id}")
async def get_entity_details(entity_id: str):
    """
    Get detailed entity profile with enrichment data
    """
    pool = await get_or_create_pool()

    async with pool.acquire() as conn:
        entity = await conn.fetchrow("""
            SELECT id, canonical_name, entity_type, language, confidence,
                   profile_summary, profile_roles, profile_affiliations,
                   profile_key_facts, profile_locations, mention_count,
                   last_enriched_at, created_at, updated_at
            FROM entities
            WHERE id = $1
        """, entity_id)

        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Get articles mentioning this entity
        articles = await conn.fetch("""
            SELECT p.id, p.title, p.url, p.created_at
            FROM pages p
            JOIN page_entities pe ON p.id = pe.page_id
            WHERE pe.entity_id = $1
            ORDER BY p.created_at DESC
            LIMIT 20
        """, entity_id)

        # Get events involving this entity
        events = await conn.fetch("""
            SELECT e.id, e.title, e.event_type
            FROM events e
            JOIN event_entities ee ON e.id = ee.event_id
            WHERE ee.entity_id = $1
            ORDER BY e.updated_at DESC
        """, entity_id)

        import json as json_lib
        return {
            "id": str(entity['id']),
            "name": entity['canonical_name'],
            "type": entity['entity_type'],
            "language": entity['language'],
            "confidence": entity['confidence'],
            "profile": {
                "summary": entity['profile_summary'],
                "roles": json_lib.loads(entity['profile_roles']) if entity['profile_roles'] else [],
                "affiliations": json_lib.loads(entity['profile_affiliations']) if entity['profile_affiliations'] else [],
                "key_facts": json_lib.loads(entity['profile_key_facts']) if entity['profile_key_facts'] else [],
                "locations": json_lib.loads(entity['profile_locations']) if entity['profile_locations'] else []
            },
            "mention_count": entity['mention_count'],
            "last_enriched_at": entity['last_enriched_at'].isoformat() if entity['last_enriched_at'] else None,
            "created_at": entity['created_at'].isoformat(),
            "updated_at": entity['updated_at'].isoformat(),
            "articles": [
                {
                    "id": str(a['id']),
                    "title": a['title'],
                    "url": a['url'],
                    "created_at": a['created_at'].isoformat()
                }
                for a in articles
            ],
            "events": [
                {
                    "id": str(e['id']),
                    "title": e['title'],
                    "event_type": e['event_type']
                }
                for e in events
            ]
        }


def calculate_confidence(page: dict, entities: list, events: list) -> float:
    """
    Calculate overall confidence score

    Based on:
    - Content extracted (0.3)
    - Entities found (0.3)
    - Events matched (0.2)
    - Language detected (0.2)
    """
    conf = 0.0

    # Has content
    if page.get('content_text'):
        conf += 0.3

    # Has title
    if page.get('title'):
        conf += 0.2

    # Has entities
    if entities:
        conf += 0.3

    # Part of events
    if events:
        conf += 0.2

    return min(1.0, conf)
