"""
Stories API
============

Unified REST endpoints for stories (L3 incidents + L4 cases).

One resource type with scale parameter:
- GET /api/stories?scale=incident  (L3)
- GET /api/stories?scale=case      (L4)
- GET /api/stories                 (both)

This replaces the confusing /api/events endpoint which mixed
IncidentEventView, CanonicalEvent, and EventBuilder concepts.

Response Models:
- StorySummary: list view (id, scale, title, stats)
- StoryDetail: single view (+ surfaces, incidents, inquiries)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Literal
from pydantic import BaseModel

from services.neo4j_service import Neo4jService
from repositories.story_repository import StoryRepository


router = APIRouter(prefix="/api/stories", tags=["Stories"])


# =============================================================================
# Response Models
# =============================================================================

class StorySummary(BaseModel):
    """Story summary for list views."""
    id: str
    scale: Literal["incident", "case"]
    scope_signature: str  # Stable ID for caching/dedup
    title: str
    description: str
    primary_entity: str
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    source_count: int = 0
    surface_count: int = 0
    incident_count: Optional[int] = None  # Only for scale="case"
    case_type: Optional[str] = None  # Only for scale="case": "developing" or "entity_storyline"


class SurfaceSummary(BaseModel):
    """Surface summary for story detail."""
    id: str
    source_count: int
    time: Optional[str] = None
    sample_claim: Optional[str] = None


class InquirySummary(BaseModel):
    """Inquiry summary for story aggregation."""
    id: str
    title: str
    status: str
    schema_type: str


class StoryDetail(BaseModel):
    """Full story detail for single-item views."""
    id: str
    scale: Literal["incident", "case"]
    title: str
    description: str
    primary_entity: str
    primary_entities: List[str] = []
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    source_count: int = 0
    surface_count: int = 0
    claim_count: int = 0
    incident_count: Optional[int] = None
    case_type: Optional[str] = None  # Only for scale="case"
    scope_signature: Optional[str] = None
    surfaces: Optional[List[SurfaceSummary]] = None
    incidents: Optional[List[StorySummary]] = None  # For case scale
    inquiries: Optional[List[InquirySummary]] = None


class StoryListResponse(BaseModel):
    """Paginated story list."""
    stories: List[StorySummary]
    total: int
    offset: int
    limit: int


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=StoryListResponse)
async def list_stories(
    scale: Optional[Literal["incident", "case"]] = Query(None, description="Filter by scale"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    entity: Optional[str] = Query(None, description="Filter by entity name"),
    since: Optional[str] = Query(None, description="Filter by time_start >= date (ISO)"),
    case_type: Optional[str] = Query(None, description="Filter by case_type: developing, entity_storyline"),
):
    """
    List stories with optional scale filter.

    - scale=incident: L3 single happenings
    - scale=case: L4 grouped happenings (requires ≥2 incidents)
    - scale=None: both types
    - case_type=developing: CaseCores formed by k>=2 motif recurrence
    - case_type=entity_storyline: EntityCases formed by focal entity

    Ordered by source_count descending.
    """
    neo4j = Neo4jService()
    await neo4j.connect()
    repo = StoryRepository(neo4j)

    try:
        total = await repo.count_stories(
            scale=scale,
            entity_filter=entity,
            since=since,
            case_type=case_type,
        )
        stories = await repo.list_stories(
            scale=scale,
            offset=offset,
            limit=limit,
            entity_filter=entity,
            since=since,
            case_type=case_type,
        )

        return StoryListResponse(
            stories=[
                StorySummary(
                    id=s.id,
                    scale=s.scale,
                    scope_signature=s.scope_signature or s.id,
                    title=s.title or "Untitled",
                    description=s.description or "",
                    primary_entity=s.primary_entity,
                    time_start=s.time_start.isoformat() if s.time_start else None,
                    time_end=s.time_end.isoformat() if s.time_end else None,
                    source_count=s.source_count,
                    surface_count=s.surface_count,
                    incident_count=s.incident_count if s.is_case else None,
                    case_type=s.case_type if s.is_case else None,
                )
                for s in stories
            ],
            total=total,
            offset=offset,
            limit=limit,
        )

    finally:
        await neo4j.close()


@router.get("/{story_id}", response_model=StoryDetail)
async def get_story(
    story_id: str,
    include_surfaces: bool = Query(False, description="Include surface details"),
    include_incidents: bool = Query(False, description="Include nested incidents (case only)"),
    include_inquiries: bool = Query(True, description="Include related inquiries"),
):
    """
    Get single story by ID.

    Works for both incident (L3) and case (L4) IDs.
    """
    import asyncpg
    import os

    neo4j = Neo4jService()
    await neo4j.connect()
    repo = StoryRepository(neo4j)

    try:
        story = await repo.get_by_id(story_id)

        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        # Get surfaces if requested
        surfaces = None
        if include_surfaces:
            surface_data = await repo.get_surfaces_for_story(story_id)
            surfaces = [
                SurfaceSummary(
                    id=s['id'],
                    source_count=len(s['sources']),
                    time=s['time_start'].isoformat() if hasattr(s['time_start'], 'isoformat') else s['time_start'],
                    sample_claim=s['sample_claims'][0][:200] if s['sample_claims'] else None,
                )
                for s in surface_data
            ]

        # Get nested incidents if requested (for cases)
        incidents = None
        if include_incidents and story.is_case:
            incident_stories = await repo.get_incidents_for_case(story_id)
            incidents = [
                StorySummary(
                    id=i.id,
                    scale=i.scale,
                    scope_signature=i.scope_signature or i.id,
                    title=i.title or "Untitled",
                    description=i.description or "",
                    primary_entity=i.primary_entity,
                    time_start=i.time_start.isoformat() if i.time_start else None,
                    time_end=i.time_end.isoformat() if i.time_end else None,
                    source_count=i.source_count,
                    surface_count=i.surface_count,
                    incident_count=None,
                )
                for i in incident_stories
            ]

        # Get related inquiries from PostgreSQL
        inquiries = None
        if include_inquiries:
            try:
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
                    """, story_id)
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
            except Exception:
                inquiries = []

        return StoryDetail(
            id=story.id,
            scale=story.scale,
            title=story.title or "Untitled",
            description=story.description or "",
            primary_entity=story.primary_entity,
            primary_entities=story.primary_entities,
            time_start=story.time_start.isoformat() if story.time_start else None,
            time_end=story.time_end.isoformat() if story.time_end else None,
            source_count=story.source_count,
            surface_count=story.surface_count,
            claim_count=story.claim_count,
            incident_count=story.incident_count if story.is_case else None,
            case_type=story.case_type if story.is_case else None,
            scope_signature=story.scope_signature,
            surfaces=surfaces,
            incidents=incidents,
            inquiries=inquiries,
        )

    finally:
        await neo4j.close()


@router.get("/{story_id}/surfaces")
async def get_story_surfaces(
    story_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Get surfaces for a story.

    Works for both incident and case IDs.
    For cases, returns surfaces from all contained incidents.
    """
    neo4j = Neo4jService()
    await neo4j.connect()
    repo = StoryRepository(neo4j)

    try:
        # Verify story exists
        story = await repo.get_by_id(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        surface_data = await repo.get_surfaces_for_story(story_id)

        # Apply pagination
        paginated = surface_data[offset:offset + limit]

        return {
            "story_id": story_id,
            "scale": story.scale,
            "surfaces": [
                {
                    "id": s['id'],
                    "sources": s['sources'],
                    "anchors": s['anchors'],
                    "time": s['time_start'],
                    "sample_claims": s['sample_claims'],
                }
                for s in paginated
            ],
            "total": len(surface_data),
            "offset": offset,
            "limit": limit,
        }

    finally:
        await neo4j.close()


@router.get("/{story_id}/incidents")
async def get_story_incidents(
    story_id: str,
):
    """
    Get incidents for a case story.

    Only applicable for scale="case" stories.
    Returns 404 if story is an incident.
    """
    neo4j = Neo4jService()
    await neo4j.connect()
    repo = StoryRepository(neo4j)

    try:
        story = await repo.get_by_id(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        if story.is_incident:
            raise HTTPException(
                status_code=400,
                detail="Incidents don't have nested incidents. Use /surfaces instead."
            )

        incident_stories = await repo.get_incidents_for_case(story_id)

        return {
            "story_id": story_id,
            "incidents": [
                {
                    "id": i.id,
                    "scale": i.scale,
                    "title": i.title or "Untitled",
                    "description": i.description or "",
                    "primary_entity": i.primary_entity,
                    "surface_count": i.surface_count,
                    "time_start": i.time_start.isoformat() if i.time_start else None,
                }
                for i in incident_stories
            ],
            "count": len(incident_stories),
        }

    finally:
        await neo4j.close()
