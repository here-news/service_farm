"""
Canonical Events API
====================

REST endpoints for serving canonical events to downstream apps.
Events are L4 Cases built from L3 Incidents by the CanonicalWorker.

Endpoints:
- GET /api/events - List events (paginated, filterable)
- GET /api/events/{id} - Get single event with surfaces
- GET /api/events/{id}/surfaces - Get surfaces for event
- GET /api/events/{id}/claims - Get claims for event

Architecture:
- Uses CaseRepository for data access (proper data model layer)
- Cases require >= 2 incidents to be shown (validity filter in repository)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from services.neo4j_service import Neo4jService
from repositories.case_repository import CaseRepository


router = APIRouter(prefix="/api/events", tags=["Events"])


class EventSummary(BaseModel):
    """Event summary for list view. Supports both ev_ (legacy) and ca_ (Case) IDs."""
    id: str
    title: str
    description: str
    primary_entity: str
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    source_count: int = 0
    surface_count: int = 0
    case_type: Optional[str] = None  # For ca_ IDs: breaking, developing, etc.


class InquirySummary(BaseModel):
    """Inquiry summary for event aggregation."""
    id: str
    title: str
    status: str
    schema_type: str


class EventDetail(BaseModel):
    """Full event detail. Supports both ev_ (legacy) and ca_ (Case) IDs."""
    id: str
    title: str
    description: str
    primary_entity: str
    secondary_entities: List[str] = []
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    source_count: int = 0
    surface_count: int = 0
    claim_count: int = 0
    case_type: Optional[str] = None  # For ca_ IDs
    incident_count: int = 0  # For ca_ IDs: number of L3 incidents
    surfaces: Optional[List[dict]] = None
    inquiries: Optional[List[InquirySummary]] = None


class EventListResponse(BaseModel):
    """Paginated event list."""
    events: List[EventSummary]
    total: int
    offset: int
    limit: int


@router.get("", response_model=EventListResponse)
async def list_events(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    entity: Optional[str] = None,
    since: Optional[str] = None,
    type: Optional[str] = Query(None, description="Filter by case_type: developing, entity_storyline"),
):
    """
    List canonical events (L4 Cases).

    Uses CaseRepository to ensure proper data model access.
    Only cases with >= 2 incidents are returned.

    Args:
        offset: Pagination offset
        limit: Page size (max 100)
        entity: Filter by entity name (substring match)
        since: Filter by time_start >= this date (ISO format)
        type: Filter by case_type (developing, entity_storyline)

    Returns:
        Paginated list of event summaries
    """
    neo4j = Neo4jService()
    await neo4j.connect()
    repo = CaseRepository(neo4j)

    try:
        # Get count and cases from repository
        total = await repo.count_valid(entity_filter=entity, since=since, case_type=type)
        cases = await repo.list_valid(
            offset=offset,
            limit=limit,
            entity_filter=entity,
            since=since,
            case_type=type,
        )

        # Convert domain objects to API response
        events = [
            EventSummary(
                id=case.id,
                title=case.title or "Untitled",
                description=case.description or "",
                primary_entity=case.primary_entities[0] if case.primary_entities else "Unknown",
                time_start=case.time_start.isoformat() if case.time_start else None,
                time_end=case.time_end.isoformat() if case.time_end else None,
                source_count=case.source_count,
                surface_count=case.surface_count,
                case_type=case.case_type,
            )
            for case in cases
        ]

        return EventListResponse(
            events=events,
            total=total,
            offset=offset,
            limit=limit,
        )

    finally:
        await neo4j.close()


@router.get("/{event_id}", response_model=EventDetail)
async def get_event(
    event_id: str,
    include_surfaces: bool = Query(False),
    include_inquiries: bool = Query(True),
):
    """
    Get single event (L4 Case) by ID.

    Uses CaseRepository for proper data model access.

    Args:
        event_id: Case ID (case_xxxxxxxx format)
        include_surfaces: Include surface details
        include_inquiries: Include related inquiries (default: True)

    Returns:
        Full event detail with inquiries
    """
    import asyncpg
    import os

    neo4j = Neo4jService()
    await neo4j.connect()
    repo = CaseRepository(neo4j)

    try:
        # Get case from repository
        case = await repo.get_by_id(event_id)

        if not case:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get surfaces if requested
        surfaces = None
        if include_surfaces:
            surface_data = await repo.get_surfaces_for_case(event_id)
            surfaces = [
                {
                    "id": s['id'],
                    "source_count": len(s['sources']),
                    "time": s['time_start'],
                    "sample_claim": s['sample_claims'][0][:200] if s['sample_claims'] else None,
                }
                for s in surface_data
            ]

        # Get related inquiries from PostgreSQL
        inquiries = None
        if include_inquiries:
            conn = await asyncpg.connect(
                host=os.getenv('POSTGRES_HOST', 'db'),
                database=os.getenv('POSTGRES_DB', 'phi_here'),
                user=os.getenv('POSTGRES_USER', 'phi_user'),
                password=os.getenv('POSTGRES_PASSWORD'),
            )
            try:
                inquiry_rows = await conn.fetch("""
                    SELECT id, title, status, schema_type
                    FROM inquiries
                    WHERE source_event = $1
                    ORDER BY created_at DESC
                """, event_id)
                inquiries = [
                    InquirySummary(
                        id=row['id'],
                        title=row['title'],
                        status=row['status'],
                        schema_type=row['schema_type'],
                    )
                    for row in inquiry_rows
                ]
            finally:
                await conn.close()

        return EventDetail(
            id=case.id,
            title=case.title or "Untitled",
            description=case.description or "",
            primary_entity=case.primary_entities[0] if case.primary_entities else "Unknown",
            secondary_entities=case.primary_entities[1:5] if len(case.primary_entities) > 1 else [],
            time_start=case.time_start.isoformat() if case.time_start else None,
            time_end=case.time_end.isoformat() if case.time_end else None,
            source_count=case.source_count,
            surface_count=case.surface_count,
            claim_count=case.claim_count,
            case_type=case.case_type,
            incident_count=case.incident_count,
            surfaces=surfaces,
            inquiries=inquiries,
        )

    finally:
        await neo4j.close()


@router.get("/{event_id}/surfaces")
async def get_event_surfaces(
    event_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Get surfaces for an event.

    Args:
        event_id: Event ID
        offset: Pagination offset
        limit: Page size

    Returns:
        List of surfaces with claims
    """
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Verify event exists
        event_check = await neo4j._execute_read(
            "MATCH (e:Event {id: $id}) RETURN e.id",
            {'id': event_id}
        )
        if not event_check:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get surfaces - supports both legacy Event->Surface and Case->Incident->Surface
        results = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            OPTIONAL MATCH (e)-[:CONTAINS]->(i:Incident)-[:CONTAINS]->(s1:Surface)
            OPTIONAL MATCH (s2:Surface)-[:BELONGS_TO]->(e)
            WITH e, collect(DISTINCT s1) + collect(DISTINCT s2) as all_surfaces
            UNWIND all_surfaces as s
            WITH DISTINCT s WHERE s IS NOT NULL
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH s, collect({text: c.text, page_id: c.page_id})[0..5] as claims
            RETURN s.id as id,
                   s.sources as sources,
                   s.anchor_entities as anchors,
                   s.time_start as time_start,
                   claims
            ORDER BY s.time_start
            SKIP $offset
            LIMIT $limit
        """, {'event_id': event_id, 'offset': offset, 'limit': limit})

        return {
            "event_id": event_id,
            "surfaces": [
                {
                    "id": r['id'],
                    "sources": r['sources'] or [],
                    "anchors": r['anchors'] or [],
                    "time": r['time_start'],
                    "claims": r['claims'] or [],
                }
                for r in results
            ],
            "offset": offset,
            "limit": limit,
        }

    finally:
        await neo4j.close()


@router.get("/{event_id}/claims")
async def get_event_claims(
    event_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get all claims for an event (via surfaces).

    Args:
        event_id: Event ID
        offset: Pagination offset
        limit: Page size

    Returns:
        List of claims with provenance
    """
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Get claims - supports both legacy Event->Surface and Case->Incident->Surface
        results = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            OPTIONAL MATCH (e)-[:CONTAINS]->(i:Incident)-[:CONTAINS]->(s1:Surface)
            OPTIONAL MATCH (s2:Surface)-[:BELONGS_TO]->(e)
            WITH collect(DISTINCT s1) + collect(DISTINCT s2) as all_surfaces
            UNWIND all_surfaces as s
            WITH DISTINCT s WHERE s IS NOT NULL
            MATCH (s)-[:CONTAINS]->(c:Claim)
            OPTIONAL MATCH (p:Page {id: c.page_id})
            RETURN c.id as id,
                   c.text as text,
                   c.page_id as page_id,
                   p.title as page_title,
                   p.url as page_url,
                   c.reported_time as reported_time,
                   s.id as surface_id
            ORDER BY c.reported_time
            SKIP $offset
            LIMIT $limit
        """, {'event_id': event_id, 'offset': offset, 'limit': limit})

        return {
            "event_id": event_id,
            "claims": [
                {
                    "id": r['id'],
                    "text": r['text'],
                    "page_id": r['page_id'],
                    "page_title": r['page_title'],
                    "page_url": r['page_url'],
                    "reported_time": r['reported_time'],
                    "surface_id": r['surface_id'],
                }
                for r in results
            ],
            "offset": offset,
            "limit": limit,
        }

    finally:
        await neo4j.close()
