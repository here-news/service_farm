"""
Map API router - provides location data from events and entities
"""
from fastapi import APIRouter
from repositories.entity_repository import EntityRepository
from repositories.event_repository import EventRepository
from services.neo4j_service import Neo4jService
import asyncpg
import os

router = APIRouter()

# Database connection pool (will be initialized on first request)
_db_pool = None
_neo4j = None


async def get_services():
    """Get or create service instances"""
    global _db_pool, _neo4j

    if _neo4j is None:
        _neo4j = Neo4jService()
        await _neo4j.connect()

    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=1,
            max_size=5
        )

    return _db_pool, _neo4j


@router.get("/locations")
async def get_map_locations(limit: int = 100):
    """
    Get locations from events with their coordinates.

    Returns location entities (GPE/LOC/LOCATION) that have coordinates,
    along with the events they appear in.
    """
    try:
        db_pool, neo4j = await get_services()
        entity_repo = EntityRepository(db_pool, neo4j)
        event_repo = EventRepository(db_pool, neo4j)

        # Get recent events
        events_data = await event_repo.list_root_events(limit=limit)

        # Collect location entities from all events
        location_map = {}  # entity_id -> {entity, event_ids}

        for event_data in events_data:
            event_id = event_data['id']

            # Get entities for this event
            entities = await entity_repo.get_by_event_id(event_id)

            for entity in entities:
                # Filter to location entities with coordinates
                if entity.entity_type in ('GPE', 'LOC', 'LOCATION'):
                    if entity.latitude and entity.longitude:
                        entity_id = str(entity.id)

                        if entity_id not in location_map:
                            location_map[entity_id] = {
                                'entity': entity,
                                'event_ids': [],
                                'events': []
                            }

                        location_map[entity_id]['event_ids'].append(event_id)
                        location_map[entity_id]['events'].append({
                            'id': event_id,
                            'title': event_data.get('canonical_name', 'Untitled'),
                            'status': event_data.get('status')
                        })

        # Build response
        locations = []
        for entity_id, data in location_map.items():
            entity = data['entity']
            locations.append({
                'id': entity_id,
                'name': entity.canonical_name,
                'latitude': entity.latitude,
                'longitude': entity.longitude,
                'qid': entity.wikidata_qid,
                'thumbnail': entity.image_url,
                'description': entity.wikidata_description,
                'events': data['events'],
                'event_count': len(data['events'])
            })

        # Sort by event count (most mentioned locations first)
        locations.sort(key=lambda x: x['event_count'], reverse=True)

        # Build connections (locations that share events)
        connections = []
        location_list = list(location_map.items())
        for i, (loc1_id, loc1_data) in enumerate(location_list):
            for loc2_id, loc2_data in location_list[i+1:]:
                shared = set(loc1_data['event_ids']) & set(loc2_data['event_ids'])
                if shared:
                    connections.append({
                        'source': loc1_id,
                        'target': loc2_id,
                        'shared_event_count': len(shared),
                        'shared_event_ids': list(shared)
                    })

        return {
            "status": "success",
            "locations": locations,
            "connections": connections,
            "total_locations": len(locations),
            "total_connections": len(connections)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "locations": [],
            "connections": [],
            "total_locations": 0,
            "total_connections": 0,
            "error": str(e)
        }
