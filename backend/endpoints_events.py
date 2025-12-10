"""
Event Tree API Endpoints

GET /api/events - List all events
GET /api/events/{event_id} - Get event detail with tree structure
"""
import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from models.domain.event import Event
from models.domain.claim import Claim
from models.domain.entity import Entity
from utils.datetime_utils import neo4j_datetime_to_python

router = APIRouter()

# Globals
db_pool = None
neo4j_service = None
event_repo = None
claim_repo = None
entity_repo = None


async def init_services():
    """Initialize database pool and services"""
    global db_pool, neo4j_service, event_repo, claim_repo, entity_repo

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

    if neo4j_service is None:
        neo4j_service = Neo4jService(
            uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')
        )
        await neo4j_service.connect()

    if event_repo is None:
        event_repo = EventRepository(db_pool, neo4j_service)

    if claim_repo is None:
        claim_repo = ClaimRepository(db_pool, neo4j_service)

    if entity_repo is None:
        entity_repo = EntityRepository(db_pool, neo4j_service)

    return event_repo, claim_repo, entity_repo


@router.get("/events")
async def list_events(
    status: Optional[str] = None,
    scale: Optional[str] = None,
    limit: int = 50
):
    """
    List root events (events without parents) with filters

    Uses EventRepository to query Neo4j for root events
    """
    event_repo, _, _ = await init_services()

    # Use repository method instead of direct Neo4j query
    events_data = await event_repo.list_root_events(status=status, scale=scale, limit=limit)

    # Convert to API response format
    import json
    events = []
    for row in events_data:
        metadata = row.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        # Extract coherence and summary from metadata (they might be there)
        coherence = metadata.get('coherence', row.get('coherence'))
        summary = metadata.get('summary')

        # Convert datetimes to ISO format strings
        event_start = row.get('event_start')
        event_end = row.get('event_end')
        created_at = row.get('created_at')
        updated_at = row.get('updated_at')

        event_dict = {
            'id': row['id'],
            'title': row['canonical_name'],  # Frontend expects 'title'
            'canonical_name': row['canonical_name'],
            'event_type': row['event_type'],
            'event_scale': row.get('event_scale'),
            'status': row['status'],
            'confidence': row['confidence'],
            'event_start': event_start.isoformat() if event_start and hasattr(event_start, 'isoformat') else event_start,
            'event_end': event_end.isoformat() if event_end and hasattr(event_end, 'isoformat') else event_end,
            'created_at': created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else created_at,
            'updated_at': updated_at.isoformat() if updated_at and hasattr(updated_at, 'isoformat') else updated_at,
            'last_updated': updated_at.isoformat() if updated_at and hasattr(updated_at, 'isoformat') else updated_at,  # Frontend expects 'last_updated'
            'coherence': coherence,
            'child_count': row['child_count'],
            'summary': summary,
            # Note: claims_count removed - use graph relationships via API instead
        }
        events.append(event_dict)

    return {
        'events': events,
        'total': len(events)
    }


@router.get("/events/mine")
async def get_my_events():
    """
    Get user's submitted/followed events (placeholder)

    For now returns empty list - auth system not yet implemented
    """
    return {
        'events': [],
        'total': 0
    }


@router.get("/events/{event_id}")
async def get_event_tree(event_id: str):
    """
    Get event with full tree structure using domain models and repositories

    Returns:
    - Event details with narrative
    - Child events (sub-events via CONTAINS relationships)
    - Parent event (if this is a sub-event)
    - Entities
    - Claims
    """
    event_repo, claim_repo, entity_repo = await init_services()

    # Validate short ID format (ev_xxxxxxxx)
    if not event_id.startswith('ev_'):
        raise HTTPException(status_code=400, detail="Invalid event ID format. Expected format: ev_xxxxxxxx")

    # Get event using repository (accepts short ID string)
    event = await event_repo.get_by_id(event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Get sub-events
    sub_events = await event_repo.get_sub_events(event_id)

    # Get parent event (if any)
    parent_event = None
    if event.parent_event_id:
        parent_event = await event_repo.get_by_id(event.parent_event_id)

    # Get entities for this event using repository
    entities = await entity_repo.get_by_event_id(event_id)

    # Get claims linked to this event from Neo4j graph
    claims = await event_repo.get_event_claims(event_id)

    # Convert domain models to API response format
    import json

    def event_to_dict(e: Event) -> dict:
        """Convert Event domain model to API dict"""
        return {
            'id': str(e.id),
            'canonical_name': e.canonical_name,
            'event_type': e.event_type,
            'event_scale': e.event_scale,
            'status': e.status,
            'confidence': e.confidence,
            'coherence': e.coherence,
            'event_start': e.event_start.isoformat() if e.event_start else None,
            'event_end': e.event_end.isoformat() if e.event_end else None,
            'summary': e.summary,
            'location': e.location,
            'claims_count': e.claims_count,
            'created_at': e.created_at.isoformat() if e.created_at else None,
            'updated_at': e.updated_at.isoformat() if e.updated_at else None,
        }

    def claim_to_dict(c: Claim) -> dict:
        """Convert Claim domain model to API dict"""
        # Handle event_time - might be string or datetime
        event_time = c.event_time
        if event_time and hasattr(event_time, 'isoformat'):
            event_time = event_time.isoformat()

        return {
            'id': str(c.id),
            'text': c.text,
            'event_time': event_time,
            'confidence': c.confidence,
            'modality': c.modality,
        }

    def entity_to_dict(e: Entity) -> dict:
        """Convert Entity domain model to API dict"""
        return {
            'id': str(e.id),
            'canonical_name': e.canonical_name,
            'entity_type': e.entity_type,
            'confidence': e.confidence,
            'mention_count': e.mention_count,
            'wikidata_qid': e.wikidata_qid,
            'wikidata_label': e.wikidata_label,
            'wikidata_description': e.wikidata_description,
            'wikidata_image': e.wikidata_image,
            'wikidata_thumbnail': e.wikidata_image,  # Alias for frontend compatibility
        }

    # Build response
    return {
        'event': event_to_dict(event),
        'children': [event_to_dict(sub) for sub in sub_events],
        'parent': event_to_dict(parent_event) if parent_event else None,
        'entities': [entity_to_dict(e) for e in entities],
        'claims': [claim_to_dict(c) for c in claims],
    }
