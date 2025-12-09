"""
Map API router - provides location data from top stories
"""

from fastapi import APIRouter
from services.neo4j_client import neo4j_client

router = APIRouter(prefix="/api/map", tags=["map"])


@router.get("/locations")
async def get_map_locations(limit: int = 100):
    """
    Get locations from top stories with their coordinates and connections

    Returns:
    - Locations with lat/lon coordinates
    - Stories associated with each location
    - Connections between locations via shared stories
    """
    try:
        with neo4j_client.driver.session(database=neo4j_client.database) as session:
            # Get top stories with their locations
            result = session.run('''
                // Get recent stories (fetch more to find diverse locations)
                MATCH (s:Story)
                WHERE s.created_at IS NOT NULL
                WITH s
                ORDER BY s.created_at DESC
                LIMIT $limit

                // Get locations mentioned in these stories
                MATCH (s)-[:MENTIONS_LOCATION]->(loc:Location)
                WHERE loc.latitude IS NOT NULL
                  AND loc.longitude IS NOT NULL
                  AND loc.latitude <> 0
                  AND loc.longitude <> 0

                // Collect stories for each location
                WITH loc, collect(DISTINCT {
                    id: s.id,
                    title: coalesce(s.title, s.topic),
                    coherence: s.health_indicator,
                    cover_image: s.cover_image
                }) as stories

                RETURN loc.canonical_id as id,
                       loc.canonical_name as name,
                       loc.latitude as latitude,
                       loc.longitude as longitude,
                       loc.wikidata_qid as qid,
                       loc.wikidata_thumbnail as thumbnail,
                       loc.description as description,
                       stories
                ORDER BY size(stories) DESC
            ''', limit=limit)

            locations = []
            location_map = {}

            for record in result:
                location_id = record['id']
                location_data = {
                    'id': location_id,
                    'name': record['name'],
                    'latitude': record['latitude'],
                    'longitude': record['longitude'],
                    'qid': record['qid'],
                    'thumbnail': record['thumbnail'],
                    'description': record['description'],
                    'stories': record['stories'],
                    'story_count': len(record['stories'])
                }
                locations.append(location_data)
                location_map[location_id] = location_data

            # Build connections between locations (via shared stories)
            connections = []
            for i, loc1 in enumerate(locations):
                for loc2 in locations[i+1:]:
                    # Find shared stories
                    stories1 = set(s['id'] for s in loc1['stories'])
                    stories2 = set(s['id'] for s in loc2['stories'])
                    shared_stories = stories1.intersection(stories2)

                    if shared_stories:
                        connections.append({
                            'source': loc1['id'],
                            'target': loc2['id'],
                            'shared_story_count': len(shared_stories),
                            'shared_story_ids': list(shared_stories)
                        })

            return {
                "status": "success",
                "locations": locations,
                "connections": connections,
                "total_locations": len(locations),
                "total_connections": len(connections)
            }

    except Exception as e:
        print(f"Error fetching map locations: {e}")
        return {
            "status": "error",
            "locations": [],
            "connections": [],
            "total_locations": 0,
            "total_connections": 0,
            "error": str(e)
        }
