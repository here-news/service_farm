"""
Event Tree API Endpoints

GET /api/events - List all events
GET /api/events/{event_id} - Get event detail with tree structure
"""
import os
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException
import asyncpg

router = APIRouter()

# Globals
db_pool = None


async def init_db_pool():
    """Initialize database pool"""
    global db_pool
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
    return db_pool


@router.get("/api/events")
async def list_events(
    status: Optional[str] = None,
    scale: Optional[str] = None,
    limit: int = 50
):
    """List events with filters"""
    pool = await init_db_pool()

    query = """
        SELECT
            e.id, e.title, e.summary, e.event_scale, e.status,
            e.confidence, e.claims_count, e.pages_count,
            e.event_start, e.event_end, e.created_at, e.updated_at,
            e.log_posterior, e.coherence,
            COUNT(DISTINCT rel.related_event_id) FILTER (WHERE rel.relationship_type = 'PART_OF') as child_count
        FROM core.events e
        LEFT JOIN core.event_relationships rel ON e.id = rel.related_event_id AND rel.relationship_type = 'PART_OF'
        WHERE 1=1
    """
    params = []
    param_idx = 1

    if status:
        query += f" AND e.status = ${param_idx}"
        params.append(status)
        param_idx += 1

    if scale:
        query += f" AND e.event_scale = ${param_idx}"
        params.append(scale)
        param_idx += 1

    query += f"""
        GROUP BY e.id, e.title, e.summary, e.event_scale, e.status,
                 e.confidence, e.claims_count, e.pages_count,
                 e.event_start, e.event_end, e.created_at, e.updated_at,
                 e.log_posterior, e.coherence
        ORDER BY e.updated_at DESC
        LIMIT ${param_idx}
    """
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return {
        'events': [dict(row) for row in rows],
        'total': len(rows)
    }


@router.get("/api/events/{event_id}")
async def get_event_tree(event_id: str):
    """
    Get event with full tree structure

    Returns:
    - Event details
    - Child events (via PART_OF relationships)
    - Parent events (via PART_OF relationships)
    - Related events (via other relationship types)
    - Entities
    - Claims
    - Pages
    - Consolidated data (timeline, casualties, etc.)
    """
    pool = await init_db_pool()

    try:
        event_uuid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event ID format")

    async with pool.acquire() as conn:
        # Get event details
        event = await conn.fetchrow("""
            SELECT
                id, title, summary, event_scale, status,
                confidence, claims_count, pages_count,
                event_start, event_end, created_at, updated_at,
                log_prior, log_likelihood, log_posterior, coherence,
                enriched_json
            FROM core.events
            WHERE id = $1
        """, event_uuid)

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        event_dict = dict(event)

        # Get child events (micro events that are PART_OF this event)
        children = await conn.fetch("""
            SELECT
                e.id, e.title, e.event_scale, e.status,
                e.confidence, e.claims_count, e.pages_count,
                e.event_start, e.event_end,
                rel.relationship_type, rel.confidence as rel_confidence
            FROM core.event_relationships rel
            JOIN core.events e ON rel.event_id = e.id
            WHERE rel.related_event_id = $1
                AND rel.relationship_type = 'PART_OF'
            ORDER BY e.event_start NULLS LAST, e.title
        """, event_uuid)

        # Get parent events (meso/macro events this is PART_OF)
        parents = await conn.fetch("""
            SELECT
                e.id, e.title, e.event_scale, e.status,
                e.confidence, e.claims_count, e.pages_count,
                rel.relationship_type, rel.confidence as rel_confidence
            FROM core.event_relationships rel
            JOIN core.events e ON rel.related_event_id = e.id
            WHERE rel.event_id = $1
                AND rel.relationship_type = 'PART_OF'
        """, event_uuid)

        # Get related events (other relationship types)
        related = await conn.fetch("""
            SELECT
                e.id, e.title, e.event_scale, e.status,
                rel.relationship_type, rel.confidence as rel_confidence,
                rel.metadata
            FROM core.event_relationships rel
            JOIN core.events e ON rel.related_event_id = e.id
            WHERE rel.event_id = $1
                AND rel.relationship_type != 'PART_OF'
        """, event_uuid)

        # Get entities with Wikidata info
        entities = await conn.fetch("""
            SELECT
                e.id, e.canonical_name, e.entity_type,
                e.confidence, e.mention_count,
                e.wikidata_qid,
                e.wikidata_properties->>'thumbnail_url' as wikidata_thumbnail,
                e.names_by_language->'en' as aliases
            FROM core.entities e
            JOIN core.event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
            ORDER BY e.mention_count DESC, e.canonical_name
        """, event_uuid)

        # Get claims
        claims = await conn.fetch("""
            SELECT
                c.id, c.text, c.event_time, c.confidence, c.modality,
                ARRAY_AGG(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) as entities
            FROM core.claims c
            JOIN core.pages p ON c.page_id = p.id
            JOIN core.page_events pe ON p.id = pe.page_id
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            LEFT JOIN core.entities e ON ce.entity_id = e.id
            WHERE pe.event_id = $1
            GROUP BY c.id, c.text, c.event_time, c.confidence, c.modality
            ORDER BY c.event_time NULLS LAST, c.text
        """, event_uuid)

        # Get pages
        pages = await conn.fetch("""
            SELECT
                p.id, p.url, p.title, p.description, p.author,
                p.pub_time, p.word_count, p.metadata_confidence,
                p.created_at
            FROM core.pages p
            JOIN core.page_events pe ON p.id = pe.page_id
            WHERE pe.event_id = $1
            ORDER BY p.pub_time DESC NULLS LAST
        """, event_uuid)

    # Extract micro-narratives from enriched_json if available
    micro_narratives = []
    if event_dict.get('enriched_json'):
        import json
        enriched = json.loads(event_dict['enriched_json']) if isinstance(event_dict['enriched_json'], str) else event_dict['enriched_json']
        micro_narratives = enriched.get('micro_narratives', [])

    return {
        'event': event_dict,
        'children': [dict(row) for row in children],
        'parents': [dict(row) for row in parents],
        'related': [dict(row) for row in related],
        'entities': [dict(row) for row in entities],
        'claims': [dict(row) for row in claims],
        'pages': [dict(row) for row in pages],
        'micro_narratives': micro_narratives
    }
