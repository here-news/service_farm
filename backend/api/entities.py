"""
Canonical Entities API
======================

REST endpoints for serving canonical entities.
These are entities enriched with narratives by the CanonicalWorker.

Endpoints:
- GET /api/entities - List entities (paginated, filterable)
- GET /api/entities/{id} - Get single entity with narrative
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel


router = APIRouter(prefix="/api/entities", tags=["Entities"])


class EntitySummary(BaseModel):
    """Entity summary for list view."""
    id: str
    canonical_name: str
    entity_type: Optional[str]
    narrative: Optional[str]
    image_url: Optional[str]
    source_count: int
    surface_count: int


class EntityDetail(BaseModel):
    """Full entity detail with NER and Wikidata enrichment."""
    id: str
    canonical_name: str
    entity_type: Optional[str]
    aliases: List[str]

    # LLM-generated narrative (from canonical worker)
    narrative: Optional[str]
    # Original AI profile summary (from semantic worker)
    profile_summary: Optional[str]

    # Wikidata enrichment
    wikidata_qid: Optional[str]
    wikidata_label: Optional[str]
    wikidata_description: Optional[str]
    image_url: Optional[str]

    # Counts
    source_count: int
    surface_count: int
    claim_count: int
    mention_count: int

    # Relations
    related_events: List[str]
    last_active: Optional[str]


class EntityListResponse(BaseModel):
    """Paginated entity list."""
    entities: List[EntitySummary]
    total: int
    offset: int
    limit: int


@router.get("", response_model=EntityListResponse)
async def list_entities(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    min_sources: int = Query(1, ge=0),
):
    """
    List canonical entities.

    Args:
        offset: Pagination offset
        limit: Page size (max 100)
        search: Search by name (substring match)
        min_sources: Minimum source count filter

    Returns:
        Paginated list of entity summaries
    """
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Build query
        where_clauses = ["e.source_count >= $min_sources"]
        params = {'offset': offset, 'limit': limit, 'min_sources': min_sources}

        if search:
            where_clauses.append("e.canonical_name CONTAINS $search")
            params['search'] = search

        where_str = " AND ".join(where_clauses)

        # Count total
        count_result = await neo4j._execute_read(f"""
            MATCH (e:Entity)
            WHERE {where_str}
            RETURN count(e) as total
        """, params)
        total = count_result[0]['total'] if count_result else 0

        # Get page
        results = await neo4j._execute_read(f"""
            MATCH (e:Entity)
            WHERE {where_str}
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.entity_type as entity_type,
                   e.narrative as narrative,
                   e.image_url as image_url,
                   e.source_count as source_count,
                   e.surface_count as surface_count
            ORDER BY e.source_count DESC
            SKIP $offset
            LIMIT $limit
        """, params)

        entities = [
            EntitySummary(
                id=r['id'],
                canonical_name=r['canonical_name'] or r['id'],
                entity_type=r['entity_type'],
                narrative=r['narrative'],
                image_url=r['image_url'],
                source_count=r['source_count'] or 0,
                surface_count=r['surface_count'] or 0,
            )
            for r in results
        ]

        return EntityListResponse(
            entities=entities,
            total=total,
            offset=offset,
            limit=limit,
        )

    finally:
        await neo4j.close()


@router.get("/{entity_id}", response_model=EntityDetail)
async def get_entity(entity_id: str):
    """
    Get single entity by ID.

    Args:
        entity_id: Entity ID (en_xxxxxxxx format)

    Returns:
        Full entity detail
    """
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Get entity with all NER and Wikidata data
        results = await neo4j._execute_read("""
            MATCH (e:Entity {id: $entity_id})
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.entity_type as entity_type,
                   e.aliases as aliases,
                   e.narrative as narrative,
                   e.profile_summary as profile_summary,
                   e.wikidata_qid as wikidata_qid,
                   e.wikidata_label as wikidata_label,
                   e.wikidata_description as wikidata_description,
                   e.image_url as image_url,
                   e.source_count as source_count,
                   e.surface_count as surface_count,
                   e.claim_count as claim_count,
                   e.mention_count as mention_count,
                   e.related_events as related_events,
                   e.last_active as last_active
        """, {'entity_id': entity_id})

        if not results:
            raise HTTPException(status_code=404, detail="Entity not found")

        r = results[0]

        return EntityDetail(
            id=r['id'],
            canonical_name=r['canonical_name'] or r['id'],
            entity_type=r['entity_type'],
            aliases=r['aliases'] or [],
            narrative=r['narrative'],
            profile_summary=r['profile_summary'],
            wikidata_qid=r['wikidata_qid'],
            wikidata_label=r['wikidata_label'],
            wikidata_description=r['wikidata_description'],
            image_url=r['image_url'],
            source_count=r['source_count'] or 0,
            surface_count=r['surface_count'] or 0,
            claim_count=r['claim_count'] or 0,
            mention_count=r['mention_count'] or 0,
            related_events=r['related_events'] or [],
            last_active=r['last_active'],
        )

    finally:
        await neo4j.close()


@router.get("/{entity_id}/events")
async def get_entity_events(
    entity_id: str,
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get events related to an entity.

    Args:
        entity_id: Entity ID
        limit: Max events to return

    Returns:
        List of related events
    """
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Get entity's related events
        entity_result = await neo4j._execute_read("""
            MATCH (e:Entity {id: $entity_id})
            RETURN e.related_events as event_ids
        """, {'entity_id': entity_id})

        if not entity_result:
            raise HTTPException(status_code=404, detail="Entity not found")

        event_ids = entity_result[0]['event_ids'] or []

        if not event_ids:
            return {"entity_id": entity_id, "events": []}

        # Get event details
        results = await neo4j._execute_read("""
            MATCH (e:Event)
            WHERE e.id IN $event_ids
            RETURN e.id as id,
                   e.title as title,
                   e.description as description,
                   e.source_count as source_count,
                   e.time_start as time_start
            ORDER BY e.source_count DESC
            LIMIT $limit
        """, {'event_ids': event_ids[:limit], 'limit': limit})

        return {
            "entity_id": entity_id,
            "events": [
                {
                    "id": r['id'],
                    "title": r['title'],
                    "description": r['description'],
                    "source_count": r['source_count'] or 0,
                    "time_start": r['time_start'],
                }
                for r in results
            ]
        }

    finally:
        await neo4j.close()
