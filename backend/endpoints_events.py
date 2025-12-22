"""
Event Tree API Endpoints

GET /api/events - List all events
GET /api/event/{event_id} - Get single event with tree structure
GET /api/event/{event_id}/topology - Get epistemic topology for visualization
GET /api/entity/{entity_id} - Get single entity
GET /api/claim/{claim_id} - Get single claim
GET /api/page/{page_id} - Get page with claims and entities
GET /api/pages - List recent pages (for Archive page)
GET /api/pages/mine - Get user's submitted pages (placeholder)
"""
import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException
import asyncpg

from services.neo4j_service import Neo4jService
from services.topology_persistence import TopologyPersistence
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from repositories.page_repository import PageRepository
from models.domain.event import Event, StructuredNarrative
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
page_repo = None
topology_persistence = None


async def init_services():
    """Initialize database pool and services"""
    global db_pool, neo4j_service, event_repo, claim_repo, entity_repo, page_repo, topology_persistence

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

    if page_repo is None:
        page_repo = PageRepository(db_pool, neo4j_service)

    if topology_persistence is None:
        topology_persistence = TopologyPersistence(neo4j_service)

    return event_repo, claim_repo, entity_repo, page_repo, topology_persistence


@router.get("/events")
async def list_events(
    status: Optional[str] = None,
    scale: Optional[str] = None,
    limit: int = 50,
    min_coherence: float = 0.0
):
    """
    List root events (events without parents) with filters

    Uses EventRepository to query Neo4j for root events.
    Enriches with page thumbnails and latest thought for homepage display.
    """
    event_repo, _, _, _, _ = await init_services()

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

        # Extract coherence - prefer direct field, fallback to metadata
        coherence = row.get('coherence') or metadata.get('coherence')

        # Filter by minimum coherence
        if coherence is not None and coherence < min_coherence:
            continue

        # Extract summary - prefer direct field, fallback to metadata
        summary = row.get('summary') or metadata.get('summary')

        # Convert datetimes to ISO format strings
        event_start = row.get('event_start')
        event_end = row.get('event_end')
        created_at = row.get('created_at')
        updated_at = row.get('updated_at')

        event_id = row['id']

        # Fetch page thumbnails for this event (for fanning cards display)
        page_thumbnails = await event_repo.get_page_thumbnails_for_event(event_id, limit=5)

        # Fetch latest thought (for stimulating byline)
        latest_thought = await event_repo.get_latest_thought_for_event(event_id)

        # Fetch claim count and page count
        claim_count = await event_repo.get_event_claim_count(event_id)
        page_count = await event_repo.get_event_page_count(event_id)

        # Build version string from major.minor
        version_major = row.get('version_major', 0) or 0
        version_minor = row.get('version_minor', 1) or 1
        version = f"{version_major}.{version_minor}"

        event_dict = {
            'id': event_id,
            'title': row['canonical_name'],  # Frontend expects 'title'
            'canonical_name': row['canonical_name'],
            'event_type': row['event_type'],
            'event_scale': row.get('event_scale'),
            'status': row['status'],
            'confidence': row['confidence'],
            'coherence': coherence,
            'version': version,  # Semantic version (major.minor)
            'event_start': event_start.isoformat() if event_start and hasattr(event_start, 'isoformat') else event_start,
            'event_end': event_end.isoformat() if event_end and hasattr(event_end, 'isoformat') else event_end,
            'created_at': created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else created_at,
            'updated_at': updated_at.isoformat() if updated_at and hasattr(updated_at, 'isoformat') else updated_at,
            'last_updated': updated_at.isoformat() if updated_at and hasattr(updated_at, 'isoformat') else updated_at,  # Frontend expects 'last_updated'
            'child_count': row['child_count'],
            'summary': summary,
            'claim_count': claim_count,
            'page_count': page_count,
            # New fields for homepage display
            'page_thumbnails': page_thumbnails,
            'thought': latest_thought,
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


@router.get("/pages")
async def list_pages(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    List recent pages in the system.

    Returns pages ordered by creation date (newest first).
    Used by Archive page to display all pages.
    """
    _, _, _, page_repo, _ = await init_services()

    # Get pages using repository
    pages = await page_repo.list_recent(
        limit=limit,
        offset=offset,
        status_filter=status
    )

    # Get total count
    total = await page_repo.count(status_filter=status)

    # Convert to API response
    return {
        'pages': [
            {
                'id': page.id,
                'url': page.url,
                'title': page.title or 'Untitled',
                'domain': page.domain or (page.url.split('/')[2] if page.url else None),
                'thumbnail_url': page.thumbnail_url,
                'status': page.status,
                'created_at': page.created_at.isoformat() if page.created_at else None,
                'updated_at': page.updated_at.isoformat() if page.updated_at else None,
                'word_count': page.word_count,
            }
            for page in pages
        ],
        'total': total
    }


@router.get("/pages/mine")
async def get_my_pages():
    """
    Get user's submitted pages (placeholder)

    For now returns empty list - auth system not yet implemented
    """
    return {
        'pages': [],
        'total': 0
    }


@router.get("/event/{event_id}")
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
    event_repo, claim_repo, entity_repo, _, _ = await init_services()

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

    # Fetch page thumbnails for header background
    page_thumbnails = await event_repo.get_page_thumbnails_for_event(event_id, limit=5)

    # Fetch latest thought for header display
    latest_thought = await event_repo.get_latest_thought_for_event(event_id)

    # Convert domain models to API response format
    import json

    def event_to_dict(e: Event) -> dict:
        """Convert Event domain model to API dict"""
        result = {
            'id': str(e.id),
            'event_id': str(e.id),  # GraphPage expects event_id
            'title': e.canonical_name,  # GraphPage expects title
            'canonical_name': e.canonical_name,
            'event_type': e.event_type,
            'event_scale': e.event_scale,
            'status': e.status,
            'confidence': e.confidence,
            'coherence': e.coherence,
            'version': e.version,  # Semantic version (major.minor)
            'event_start': e.event_start.isoformat() if e.event_start else None,
            'event_end': e.event_end.isoformat() if e.event_end else None,
            'summary': e.summary,  # Keep flat text for backwards compat
            'location': e.location,
            'claims_count': e.claims_count,
            'created_at': e.created_at.isoformat() if e.created_at else None,
            'updated_at': e.updated_at.isoformat() if e.updated_at else None,
        }

        # Add structured narrative if available
        if e.narrative:
            result['narrative'] = e.narrative.to_dict()
        elif e.metadata and e.metadata.get('structured_narrative'):
            # Fallback: parse from metadata if not hydrated on model
            result['narrative'] = e.metadata['structured_narrative']

        return result

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
            'page_id': str(c.page_id) if c.page_id else None,
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
            'image_url': e.image_url,
            'latitude': e.latitude,
            'longitude': e.longitude
        }

    # Build response
    return {
        'event': event_to_dict(event),
        'children': [event_to_dict(sub) for sub in sub_events],
        'parent': event_to_dict(parent_event) if parent_event else None,
        'entities': [entity_to_dict(e) for e in entities],
        'claims': [claim_to_dict(c) for c in claims],
        'page_thumbnails': page_thumbnails,
        'thought': latest_thought,
    }


@router.get("/event/{event_id}/epistemic")
async def get_event_epistemic(event_id: str):
    """
    Get epistemic state for an event.

    Returns source diversity, coverage metrics, gaps, and perspective balance.
    Used by EpistemicStateCard component to show what we know and don't know.
    """
    from datetime import datetime, timedelta
    import re

    event_repo, claim_repo, _, page_repo, _ = await init_services()

    # Validate short ID format
    if not event_id.startswith('ev_'):
        raise HTTPException(status_code=400, detail="Invalid event ID format. Expected: ev_xxxxxxxx")

    # Get event
    event = await event_repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Get claims for this event
    claims = await event_repo.get_event_claims(event_id)

    # Classify sources by type
    SOURCE_TYPES = {
        'reuters.com': 'wire', 'apnews.com': 'wire', 'afp.com': 'wire',
        'bbc.com': 'international', 'bbc.co.uk': 'international',
        'theguardian.com': 'international', 'dw.com': 'international',
        'aljazeera.com': 'international', 'cnn.com': 'international',
        'nytimes.com': 'international', 'washingtonpost.com': 'international',
        'scmp.com': 'local', 'hk01.com': 'local', 'mingpao.com': 'local',
        'gov.hk': 'official', 'info.gov.hk': 'official',
        'amnesty.org': 'ngo', 'hrw.org': 'ngo', 'rsf.org': 'ngo',
    }

    source_diversity = {'wire': 0, 'international': 0, 'local': 0, 'official': 0, 'ngo': 0, 'other': 0}
    seen_domains = set()
    page_times = []

    for claim in claims:
        if claim.page_id:
            page = await page_repo.get_by_id(str(claim.page_id))
            if page and page.url:
                domain = page.url.lower().split('/')[2].replace('www.', '') if '/' in page.url else ''

                if domain not in seen_domains:
                    seen_domains.add(domain)
                    source_type = 'other'
                    for pattern, stype in SOURCE_TYPES.items():
                        if pattern in domain:
                            source_type = stype
                            break
                    source_diversity[source_type] += 1

                if page.pub_time:
                    page_times.append(page.pub_time)

    # Calculate heat (recency factor)
    heat = 0.0
    if page_times:
        now = datetime.utcnow()
        most_recent = max(page_times)
        if hasattr(most_recent, 'replace'):
            most_recent = most_recent.replace(tzinfo=None)
        hours_ago = (now - most_recent).total_seconds() / 3600
        # Decay: 100% if < 1hr, 50% at 24hr, ~10% at 72hr
        heat = max(0.0, min(1.0, 1.0 - (hours_ago / 168)))  # Week decay

    # Calculate coverage (based on source diversity)
    # Full coverage = at least one from each category except 'other'
    expected_categories = ['wire', 'international', 'local', 'official', 'ngo']
    covered = sum(1 for cat in expected_categories if source_diversity.get(cat, 0) > 0)
    coverage = covered / len(expected_categories)

    # Detect gaps
    gaps = []

    if source_diversity['official'] == 0:
        gaps.append({
            'type': 'missing_source',
            'description': 'No official government sources',
            'priority': 'high',
            'bounty': 15
        })

    if source_diversity['ngo'] == 0:
        gaps.append({
            'type': 'perspective_gap',
            'description': 'No NGO/human rights perspective',
            'priority': 'medium',
            'bounty': 20
        })

    if source_diversity['wire'] == 0:
        gaps.append({
            'type': 'missing_source',
            'description': 'No wire service (Reuters, AP, AFP) coverage',
            'priority': 'medium',
            'bounty': 10
        })

    if source_diversity['local'] == 0:
        gaps.append({
            'type': 'missing_source',
            'description': 'No local news sources',
            'priority': 'low',
            'bounty': 8
        })

    # Check for stale data
    if heat < 0.3 and len(claims) > 0:
        gaps.append({
            'type': 'stale',
            'description': 'Story may need fresh updates',
            'priority': 'low',
            'bounty': 5
        })

    # Detect contradictions (simplified - check topology if available)
    has_contradiction = False
    try:
        _, _, _, _, topo_persistence = await init_services()
        topology = await topo_persistence.get_topology(event_id)
        if topology and len(topology.contradictions) > 0:
            has_contradiction = True
    except:
        pass

    return {
        'event_id': event_id,
        'source_count': sum(source_diversity.values()),
        'source_diversity': source_diversity,
        'claim_count': len(claims),
        'coverage': round(coverage, 2),
        'heat': round(heat, 2),
        'has_contradiction': has_contradiction,
        'gaps': gaps,
        'last_updated': datetime.utcnow().isoformat()
    }


@router.get("/event/{event_id}/topology")
async def get_event_topology(event_id: str):
    """
    Get epistemic topology for event visualization.

    Returns pre-computed topology from Neo4j:
    - Claims with plausibility scores and metadata
    - Claim-to-claim relationships (CORROBORATES, CONTRADICTS, UPDATES)
    - Topology pattern (consensus, progressive, contradictory, mixed)
    - Consensus date
    - Update chains (metric progressions)
    - Contradictions
    - Source diversity
    - Organism state (coherence, temperature)

    No computation - reads persisted topology. If topology not computed,
    returns 404 with suggestion to trigger /retopologize command.
    """
    _, _, _, _, topo_persistence = await init_services()

    # Validate short ID format
    if not event_id.startswith('ev_'):
        raise HTTPException(status_code=400, detail="Invalid event ID format. Expected: ev_xxxxxxxx")

    # Fetch topology from Neo4j
    topology = await topo_persistence.get_topology(event_id)

    if not topology:
        raise HTTPException(
            status_code=404,
            detail=f"Topology not computed for event {event_id}. "
                   "Trigger /retopologize command via event worker to compute."
        )

    # Convert to API response format
    return {
        'event_id': topology.event_id,
        'pattern': topology.pattern,
        'consensus_date': topology.consensus_date,

        # Claims with plausibility
        'claims': [
            {
                'id': c.id,
                'text': c.text,
                'plausibility': c.plausibility,
                'prior': c.prior,
                'is_superseded': c.is_superseded,
                'event_time': c.event_time,
                'source_type': c.source_type,
                'corroboration_count': c.corroboration_count,
                'page_id': c.page_id,
            }
            for c in topology.claims
        ],

        # Relationships (edges)
        'relationships': [
            {
                'source': r.source_id,
                'target': r.target_id,
                'type': r.rel_type,
                'similarity': r.similarity,
            }
            for r in topology.relationships
        ],

        # Update chains (metric progressions)
        'update_chains': [
            {
                'metric': chain.metric,
                'chain': chain.chain,
                'current': chain.current_claim_id,
            }
            for chain in topology.update_chains
        ],

        # Active contradictions
        'contradictions': topology.contradictions,

        # Source diversity
        'source_diversity': topology.source_diversity,

        # Organism state
        'organism_state': {
            'coherence': topology.coherence,
            'temperature': topology.temperature,
            'active_tensions': len(topology.contradictions),
            'last_updated': topology.last_updated.isoformat() if topology.last_updated else None,
        }
    }


@router.get("/entity/{entity_id}")
async def get_entity(entity_id: str):
    """
    Get entity details by ID

    Returns entity with Wikidata enrichment, aliases, related events and claims
    """
    event_repo, claim_repo, entity_repo, _, _ = await init_services()

    # Validate ID format
    if not entity_id.startswith('en_'):
        raise HTTPException(status_code=400, detail="Invalid entity ID format. Expected: en_xxxxxxxx")

    entity = await entity_repo.get_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    # Get claims mentioning this entity
    claims = await claim_repo.get_claims_by_entity(entity_id)

    # Get related events (events that involve this entity)
    related_events = await event_repo.get_events_by_entity(entity_id)

    return {
        'entity': {
            'id': str(entity.id),
            'canonical_name': entity.canonical_name,
            'entity_type': entity.entity_type,
            'aliases': entity.aliases,
            'mention_count': entity.mention_count,
            'profile_summary': entity.profile_summary,
            'wikidata_qid': entity.wikidata_qid,
            'wikidata_label': entity.wikidata_label,
            'wikidata_description': entity.wikidata_description,
            'image_url': entity.image_url,
            'status': entity.status,
            'confidence': entity.confidence,
            'metadata': entity.metadata,
        },
        'claims': [
            {
                'id': str(c.id),
                'text': c.text,
                'event_time': c.event_time.isoformat() if c.event_time and hasattr(c.event_time, 'isoformat') else str(c.event_time) if c.event_time else None,
                'confidence': c.confidence,
                'page_id': str(c.page_id),
            }
            for c in claims
        ],
        'claims_count': len(claims),
        'related_events': related_events,
    }


@router.get("/claim/{claim_id}")
async def get_claim(claim_id: str):
    """
    Get claim details by ID

    Returns claim with source page info and mentioned entities
    """
    _, claim_repo, _, page_repo, _ = await init_services()

    # Validate ID format
    if not claim_id.startswith('cl_'):
        raise HTTPException(status_code=400, detail="Invalid claim ID format. Expected: cl_xxxxxxxx")

    claim = await claim_repo.get_by_id(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Get entities mentioned in this claim
    entities = await claim_repo.get_entities_for_claim(claim_id)

    # Get page info (page_repo from init_services)
    page = await page_repo.get_by_id(claim.page_id)

    return {
        'claim': {
            'id': str(claim.id),
            'text': claim.text,
            'event_time': claim.event_time.isoformat() if claim.event_time and hasattr(claim.event_time, 'isoformat') else str(claim.event_time) if claim.event_time else None,
            'confidence': claim.confidence,
            'modality': claim.modality,
            'topic_key': claim.topic_key,
            'page_id': str(claim.page_id),
        },
        'source': {
            'page_id': str(page.id) if page else None,
            'url': page.url if page else None,
            'title': page.title if page else None,
            'site_name': page.site_name if page else None,
            'domain': page.domain if page else None,
        } if page else None,
        'entities': [
            {
                'id': str(e.id),
                'canonical_name': e.canonical_name,
                'entity_type': e.entity_type,
                'wikidata_qid': e.wikidata_qid,
            }
            for e in entities
        ],
    }


@router.get("/page/{page_id}")
async def get_page(page_id: str):
    """
    Get page details with claims and entities

    Returns full page information including:
    - Page metadata (title, thumbnail, author, domain, etc.)
    - Processing status
    - All claims extracted from this page (with anchor IDs for deep linking)
    - All entities mentioned in this page's claims

    Frontend can use anchor links like /app/page/pg_xxx#cl_yyy to jump to specific claims.
    """
    _, claim_repo, entity_repo, page_repo, _ = await init_services()

    # Validate ID format
    if not page_id.startswith('pg_'):
        raise HTTPException(status_code=400, detail="Invalid page ID format. Expected: pg_xxxxxxxx")

    page = await page_repo.get_by_id(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Get claims using ClaimRepository (returns List[Claim] domain models)
    claims = await claim_repo.get_by_page(page_id)

    # Get entities for this page's claims
    entities = await entity_repo.get_by_page_id(page_id)

    # Calculate semantic confidence (same logic as in endpoints.py)
    word_count = page.word_count or 0
    claim_count = len(claims)
    entity_count = len(entities)
    status = page.status

    if status == 'semantic_complete':
        if word_count >= 300 and claim_count >= 3 and entity_count >= 3:
            semantic_confidence = 1.0
        elif word_count >= 150 and claim_count >= 1:
            semantic_confidence = 0.7
        else:
            semantic_confidence = 0.5
    elif status in ['extracted', 'preview'] and word_count >= 100:
        semantic_confidence = 0.3
    elif status == 'semantic_failed' and word_count < 100:
        semantic_confidence = 0.0
    elif status == 'semantic_failed':
        semantic_confidence = 0.1
    else:
        semantic_confidence = 0.0

    # Build claims with inline entities using domain models
    claims_with_entities = []
    for claim in claims:
        claim_entities = await claim_repo.get_entities_for_claim(claim.id)
        claims_with_entities.append({
            'id': str(claim.id),
            'text': claim.text,
            'confidence': claim.confidence,
            'event_time': claim.event_time.isoformat() if claim.event_time and hasattr(claim.event_time, 'isoformat') else str(claim.event_time) if claim.event_time else None,
            'created_at': claim.created_at.isoformat() if claim.created_at and hasattr(claim.created_at, 'isoformat') else None,
            'entities': [
                {
                    'id': str(e.id),
                    'canonical_name': e.canonical_name,
                    'entity_type': e.entity_type,
                    'wikidata_qid': e.wikidata_qid,
                }
                for e in claim_entities
            ]
        })

    return {
        'page': {
            'id': str(page.id),
            'url': page.url,
            'canonical_url': page.canonical_url,
            'title': page.title,
            'description': page.description,
            'author': page.author,
            'thumbnail_url': page.thumbnail_url,
            'site_name': page.site_name,
            'domain': page.domain,
            'language': page.language,
            'word_count': page.word_count,
            'status': page.status,
            'pub_time': page.pub_time.isoformat() if page.pub_time else None,
            'metadata_confidence': page.metadata_confidence or 0.0,
            'semantic_confidence': semantic_confidence,
            'created_at': page.created_at.isoformat() if page.created_at else None,
            'updated_at': page.updated_at.isoformat() if page.updated_at else None,
        },
        'claims': claims_with_entities,
        'claims_count': claim_count,
        'entities': [
            {
                'id': str(e.id),
                'canonical_name': e.canonical_name,
                'entity_type': e.entity_type,
                'wikidata_qid': e.wikidata_qid,
                'mention_count': e.mention_count,
                'confidence': e.confidence,
            }
            for e in entities
        ],
        'entities_count': entity_count,
    }
