"""
Surfaces API - L2 Epistemic Clusters

Serves surfaces (L2) computed from claims in Neo4j.
These are the building blocks for inquiry-oriented views.

Endpoints:
  GET /api/surfaces - List all surfaces
  GET /api/surfaces/{surface_id} - Get single surface with claims
  GET /api/surfaces/by-event/{event_id} - Get surfaces for an event
"""
import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.surface_repository import SurfaceRepository
from repositories.claim_repository import ClaimRepository
from utils.datetime_utils import neo4j_datetime_to_python

router = APIRouter()

# Globals
db_pool = None
neo4j_service = None
surface_repo = None
claim_repo = None


async def init_services():
    """Initialize database connections and repositories."""
    global db_pool, neo4j_service, surface_repo, claim_repo

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

    if surface_repo is None:
        surface_repo = SurfaceRepository(db_pool, neo4j_service)

    if claim_repo is None:
        claim_repo = ClaimRepository(db_pool, neo4j_service)

    return surface_repo, claim_repo


@router.get("/surfaces")
async def list_surfaces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_claims: int = Query(1, ge=1),
):
    """
    List all surfaces ordered by support (claim count * source diversity).

    Returns surfaces in frontend-compatible format with:
    - id, name (canonical title), claim_count, sources, entities
    - entropy (dispersion), in_scope (always true for now)
    """
    surface_repo, _ = await init_services()

    # Query Neo4j for surfaces
    results = await neo4j_service._execute_read("""
        MATCH (s:Surface)
        WHERE s.claim_count >= $min_claims
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
        WITH s, count(c) as actual_claim_count
        RETURN s.id as id,
               coalesce(s.canonical_title, s.id) as name,
               actual_claim_count as claim_count,
               s.sources as sources,
               s.entities as entities,
               s.anchor_entities as anchor_entities,
               s.support as support,
               s.time_start as time_start,
               s.time_end as time_end
        ORDER BY s.support DESC
        SKIP $offset
        LIMIT $limit
    """, {
        'min_claims': min_claims,
        'offset': offset,
        'limit': limit
    })

    surfaces = []
    for row in results:
        surfaces.append({
            'id': row['id'],
            'name': row['name'] or f"Surface {row['id'][-8:]}",
            'claim_count': row['claim_count'] or 0,
            'sources': row['sources'] or [],
            'entities': row['entities'] or [],
            'anchor_entities': row['anchor_entities'] or [],
            'in_scope': True,  # All surfaces in scope by default
            'entropy': 0.0,  # Will be computed if centroid available
            'support': row['support'] or 0.0,
            'time_start': row['time_start'],
            'time_end': row['time_end'],
            'relations': []  # Computed on detail view
        })

    # Get total count
    count_result = await neo4j_service._execute_read("""
        MATCH (s:Surface) WHERE s.claim_count >= $min_claims
        RETURN count(s) as total
    """, {'min_claims': min_claims})
    total = count_result[0]['total'] if count_result else 0

    return {
        'surfaces': surfaces,
        'total': total,
        'limit': limit,
        'offset': offset
    }


@router.get("/surfaces/{surface_id}")
async def get_surface(surface_id: str):
    """
    Get single surface with full details including claims.

    Returns surface with:
    - All claims contained in this surface
    - Inter-claim relations (CONFIRMS, CONFLICTS, etc.)
    - Computed canonical value if typed
    """
    surface_repo, claim_repo = await init_services()

    # Validate ID format
    if not surface_id.startswith('sf_'):
        raise HTTPException(status_code=400, detail="Invalid surface ID format. Expected: sf_xxxxxxxx")

    # Get surface from repository
    surface = await surface_repo.get_by_id(surface_id)
    if not surface:
        raise HTTPException(status_code=404, detail="Surface not found")

    # Get claims for this surface
    claims_result = await neo4j_service._execute_read("""
        MATCH (s:Surface {id: $surface_id})-[:CONTAINS]->(c:Claim)
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        RETURN c.id as id,
               c.text as text,
               c.event_time as event_time,
               c.confidence as confidence,
               p.url as source_url,
               p.domain as source_domain
        ORDER BY c.event_time
    """, {'surface_id': surface_id})

    claims = []
    for row in claims_result:
        source = row['source_domain'] or 'unknown'
        if row['source_url']:
            try:
                from urllib.parse import urlparse
                source = urlparse(row['source_url']).netloc.replace('www.', '')
            except:
                pass

        claims.append({
            'id': row['id'],
            'text': row['text'],
            'event_time': row['event_time'],
            'confidence': row['confidence'] or 0.8,
            'source': source,
            'source_url': row['source_url']
        })

    # Get internal relations (claim-to-claim within surface)
    relations_result = await neo4j_service._execute_read("""
        MATCH (s:Surface {id: $surface_id})-[:CONTAINS]->(c1:Claim)
        MATCH (s)-[:CONTAINS]->(c2:Claim)
        MATCH (c1)-[r]->(c2)
        WHERE type(r) IN ['CONFIRMS', 'REFINES', 'SUPERSEDES', 'CONFLICTS']
        RETURN c1.id as source, c2.id as target, type(r) as relation_type,
               r.confidence as confidence
    """, {'surface_id': surface_id})

    internal_relations = [
        {
            'source': row['source'],
            'target': row['target'],
            'type': row['relation_type'],
            'confidence': row['confidence'] or 0.5
        }
        for row in relations_result
    ]

    return {
        'surface': {
            'id': surface.id,
            'name': surface.canonical_title or f"Surface {surface.id[-8:]}",
            'claim_count': len(surface.claim_ids),
            'sources': list(surface.sources),
            'entities': list(surface.entities),
            'anchor_entities': list(surface.anchor_entities),
            'in_scope': True,
            'entropy': surface.entropy if hasattr(surface, 'entropy') else 0.0,
            'support': surface.support,
            'time_start': surface.time_start.isoformat() if surface.time_start else None,
            'time_end': surface.time_end.isoformat() if surface.time_end else None,
        },
        'claims': claims,
        'internal_relations': internal_relations
    }


@router.get("/surfaces/by-event/{event_id}")
async def get_surfaces_for_event(event_id: str):
    """
    Get all surfaces belonging to an event.

    Traverses Event-[CONTAINS]->Surface relationships in Neo4j.
    Returns surfaces with their membership level (CORE, PERIPHERY, QUARANTINE).
    """
    surface_repo, _ = await init_services()

    # Validate ID format
    if not event_id.startswith('ev_'):
        raise HTTPException(status_code=400, detail="Invalid event ID format. Expected: ev_xxxxxxxx")

    # Query surfaces for this event
    results = await neo4j_service._execute_read("""
        MATCH (e:Event {id: $event_id})-[m:CONTAINS]->(s:Surface)
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
        WITH s, m, count(c) as claim_count
        RETURN s.id as id,
               coalesce(s.canonical_title, s.id) as name,
               claim_count,
               s.sources as sources,
               s.entities as entities,
               s.anchor_entities as anchor_entities,
               s.support as support,
               m.level as membership_level,
               m.score as membership_score
        ORDER BY m.score DESC
    """, {'event_id': event_id})

    surfaces = []
    for row in results:
        surfaces.append({
            'id': row['id'],
            'name': row['name'] or f"Surface {row['id'][-8:]}",
            'claim_count': row['claim_count'] or 0,
            'sources': row['sources'] or [],
            'entities': row['entities'] or [],
            'anchor_entities': row['anchor_entities'] or [],
            'in_scope': True,
            'membership_level': row['membership_level'] or 'CORE',
            'membership_score': row['membership_score'] or 0.5,
            'support': row['support'] or 0.0
        })

    return {
        'event_id': event_id,
        'surfaces': surfaces,
        'total': len(surfaces)
    }


@router.get("/surfaces/stats")
async def get_surface_stats():
    """
    Get aggregate statistics about surfaces in the system.

    Returns:
    - Total surface count
    - Distribution by source count
    - Top anchor entities
    """
    await init_services()

    # Get basic stats
    stats_result = await neo4j_service._execute_read("""
        MATCH (s:Surface)
        WITH count(s) as total,
             avg(s.claim_count) as avg_claims,
             avg(size(s.sources)) as avg_sources
        RETURN total, avg_claims, avg_sources
    """)

    # Get top anchor entities
    anchors_result = await neo4j_service._execute_read("""
        MATCH (s:Surface)
        UNWIND s.anchor_entities as anchor
        WITH anchor, count(s) as surface_count
        ORDER BY surface_count DESC
        LIMIT 20
        RETURN anchor, surface_count
    """)

    stats = stats_result[0] if stats_result else {'total': 0, 'avg_claims': 0, 'avg_sources': 0}

    return {
        'total_surfaces': stats['total'],
        'avg_claims_per_surface': round(stats['avg_claims'] or 0, 2),
        'avg_sources_per_surface': round(stats['avg_sources'] or 0, 2),
        'top_anchors': [
            {'entity': row['anchor'], 'surface_count': row['surface_count']}
            for row in anchors_result
        ]
    }
