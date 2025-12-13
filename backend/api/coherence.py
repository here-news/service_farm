"""
Coherence feed API router
"""

from fastapi import APIRouter, HTTPException, Body
from services.neo4j_service import Neo4jService
from typing import List, Dict
import asyncio

router = APIRouter(tags=["coherence"])

# Create singleton Neo4j service instance
_neo4j_service = None

async def get_neo4j():
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
        await _neo4j_service.connect()
    return _neo4j_service


@router.get("/feed")
async def get_feed(limit: int = 20, offset: int = 0, min_coherence: float = 0.0):
    """
    Get coherence-ranked event feed

    Returns event organisms ranked by coherence (structural integrity):
    - Hub coverage (% claims touching hub entities)
    - Graph connectivity (claim-entity network density)

    Args:
        limit: Number of events to return (default 20, max 50)
        offset: Number of events to skip for pagination (default 0)
        min_coherence: Minimum coherence score filter 0.0-1.0 (default 0.0)
    """
    try:
        # Limit maximum page size to prevent overload
        limit = min(limit, 50)

        # Get Neo4j connection
        neo4j = await get_neo4j()

        # Get events from Neo4j ordered by coherence
        # Filter: coherence IS NOT NULL (ignore incomplete events)
        events = await neo4j._execute_read("""
            MATCH (e:Event)
            WHERE e.coherence IS NOT NULL
              AND e.coherence >= $min_coherence
              AND e.summary IS NOT NULL
            OPTIONAL MATCH (e)-[:INTAKES]->(c:Claim)
            WITH e, count(c) as claim_count
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.event_type as event_type,
                   e.coherence as coherence,
                   e.summary as summary,
                   e.status as status,
                   e.confidence as confidence,
                   e.created_at as created_at,
                   e.updated_at as updated_at,
                   claim_count
            ORDER BY e.coherence DESC, e.updated_at DESC
            SKIP $offset
            LIMIT $limit
        """, {
            'min_coherence': min_coherence,
            'offset': offset,
            'limit': limit
        })

        # Format events for frontend
        formatted_events = [{
            'event_id': e['id'],
            'id': e['id'],
            'title': e['canonical_name'],
            'summary': e['summary'],
            'coherence': e['coherence'],
            'status': e['status'],
            'confidence': e['confidence'],
            'claim_count': e['claim_count'],
            'event_type': e['event_type'],
            'created_at': e.get('created_at'),
            'last_updated': e.get('updated_at')
        } for e in events]

        return {
            "status": "success",
            "count": len(formatted_events),
            "offset": offset,
            "limit": limit,
            "algorithm": "Coherence-First (Hub Coverage + Graph Connectivity)",
            "weights": {"hub_coverage": 0.6, "graph_connectivity": 0.4},
            "events": formatted_events
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feed: {str(e)}")


@router.get("/stats")
async def get_stats():
    """Feed algorithm statistics and configuration"""
    return {
        "status": "success",
        "algorithm": "Coherence-First",
        "version": "1.0.0",
        "weights": {
            "hub_coverage": 0.6,
            "graph_connectivity": 0.4
        },
        "description": "Events are ranked by epistemic coherence: hub coverage (% claims touching hub entities) + graph connectivity (claim-entity network density)"
    }


@router.post("/search")
async def search_events(
    query: str = Body(..., embed=True),
    limit: int = Body(5, embed=True)
):
    """
    Search events by keyword

    Uses Neo4j substring search on event names and summaries
    """
    try:
        if not query or len(query.strip()) < 2:
            return {"matches": []}

        neo4j = await get_neo4j()

        # Search events by canonical_name or summary
        results = await neo4j._execute_read("""
            MATCH (e:Event)
            WHERE e.coherence IS NOT NULL
              AND (toLower(e.canonical_name) CONTAINS toLower($query)
                   OR toLower(e.summary) CONTAINS toLower($query))
            RETURN e.id as id,
                   e.canonical_name as title,
                   e.summary as summary,
                   e.coherence as coherence
            ORDER BY e.coherence DESC
            LIMIT $limit
        """, {'query': query.strip(), 'limit': limit})

        matches = [{
            'event_id': r['id'],
            'title': r['title'],
            'summary': r['summary'],
            'coherence': r['coherence']
        } for r in results]

        return {"matches": matches}

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
