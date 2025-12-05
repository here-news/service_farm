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
from models.event import Event
from models.claim import Claim
from models.entity import Entity
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


@router.get("/api/events")
async def list_events(
    status: Optional[str] = None,
    scale: Optional[str] = None,
    limit: int = 50
):
    """
    List root events (events without parents) with filters

    Uses Neo4j to find all root events (no incoming CONTAINS relationships)
    """
    event_repo, _, _ = await init_services()

    # Query Neo4j for root events
    query = """
        MATCH (e:Event)
        WHERE NOT exists((e)<-[:CONTAINS]-())
    """

    params = {}

    if status:
        query += " AND e.status = $status"
        params['status'] = status

    if scale:
        query += " AND e.event_scale = $scale"
        params['scale'] = scale

    query += """
        WITH e
        OPTIONAL MATCH (e)-[:CONTAINS]->(sub:Event)
        WITH e, count(sub) as child_count
        RETURN e.id as id,
               e.canonical_name as canonical_name,
               e.event_type as event_type,
               e.event_scale as event_scale,
               e.status as status,
               e.confidence as confidence,
               e.earliest_time as event_start,
               e.latest_time as event_end,
               e.created_at as created_at,
               e.updated_at as updated_at,
               e.metadata_json as metadata,
               e.coherence as coherence,
               child_count
        ORDER BY e.updated_at DESC
        LIMIT $limit
    """
    params['limit'] = limit

    results = await neo4j_service._execute_read(query, params)

    # Convert to API response format
    import json
    events = []
    for row in results:
        metadata = row.get('metadata')
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        # Convert Neo4j datetimes to ISO format strings
        event_start = neo4j_datetime_to_python(row.get('event_start'))
        event_end = neo4j_datetime_to_python(row.get('event_end'))
        created_at = neo4j_datetime_to_python(row.get('created_at'))
        updated_at = neo4j_datetime_to_python(row.get('updated_at'))

        event_dict = {
            'id': row['id'],
            'canonical_name': row['canonical_name'],
            'event_type': row['event_type'],
            'event_scale': row['event_scale'],
            'status': row['status'],
            'confidence': row['confidence'],
            'event_start': event_start.isoformat() if event_start else None,
            'event_end': event_end.isoformat() if event_end else None,
            'created_at': created_at.isoformat() if created_at else None,
            'updated_at': updated_at.isoformat() if updated_at else None,
            'coherence': row['coherence'],
            'child_count': row['child_count'],
            'summary': metadata.get('summary'),
            'claims_count': len(metadata.get('claim_ids', [])),
        }
        events.append(event_dict)

    return {
        'events': events,
        'total': len(events)
    }


@router.get("/api/events/{event_id}")
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

    try:
        event_uuid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event ID format")

    # Get event using repository
    event = await event_repo.get_by_id(event_uuid)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Get sub-events
    sub_events = await event_repo.get_sub_events(event_uuid)

    # Get parent event (if any)
    parent_event = None
    if event.parent_event_id:
        parent_event = await event_repo.get_by_id(event.parent_event_id)

    # Get entities for this event using repository
    entities = await entity_repo.get_by_event_id(event_uuid)

    # Get claims from claim_ids using repository
    claims = []
    if event.claim_ids:
        for claim_id in event.claim_ids:
            claim = await claim_repo.get_by_id(claim_id)
            if claim:
                claims.append(claim)

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
            'claims_count': len(e.claim_ids),
            'created_at': e.created_at.isoformat() if e.created_at else None,
            'updated_at': e.updated_at.isoformat() if e.updated_at else None,
        }

    def claim_to_dict(c: Claim) -> dict:
        """Convert Claim domain model to API dict"""
        return {
            'id': str(c.id),
            'text': c.text,
            'event_time': c.event_time.isoformat() if c.event_time else None,
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
