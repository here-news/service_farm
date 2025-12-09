"""
Coherence feed API endpoint - events ranked by coherence
"""
from fastapi import APIRouter, Query
from typing import List, Optional
import os

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
import asyncpg

router = APIRouter()


@router.get("/coherence/feed")
async def get_coherence_feed(
    limit: int = Query(12, ge=1, le=100),
    min_coherence: float = Query(0.0, ge=0.0, le=1.0),
    offset: int = Query(0, ge=0)
):
    """
    Get events ordered by coherence (highest first).

    This shows the best-formed event organisms.
    """
    # Connect to services
    neo4j = Neo4jService()
    await neo4j.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    event_repo = EventRepository(db_pool, neo4j)

    # Get events ordered by coherence
    events = await event_repo.list_root_events(status=None)

    # Filter and sort
    filtered = [e for e in events if (e.get('coherence') or 0) >= min_coherence]
    sorted_events = sorted(filtered, key=lambda x: x.get('coherence') or 0, reverse=True)

    # Paginate
    paginated = sorted_events[offset:offset + limit]

    # Format response
    result = {
        'events': paginated,
        'total': len(filtered),
        'limit': limit,
        'offset': offset,
        'min_coherence': min_coherence
    }

    await neo4j.close()
    await db_pool.close()

    return result


@router.get("/events/mine")
async def get_my_events():
    """
    Placeholder for user's followed events.
    For now, returns all events.
    """
    # Connect to services
    neo4j = Neo4jService()
    await neo4j.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    event_repo = EventRepository(db_pool, neo4j)

    events = await event_repo.list_root_events(status=None)

    result = {
        'events': events,
        'total': len(events)
    }

    await neo4j.close()
    await db_pool.close()

    return result
