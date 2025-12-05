#!/usr/bin/env python3
"""
Queue all entities for Wikidata enrichment
"""
import asyncio
import json
from services.neo4j_service import Neo4jService
from services.job_queue import JobQueue
import redis.asyncio as redis

async def queue_entities():
    """Queue all entities that need Wikidata enrichment"""

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis
    job_queue = JobQueue()
    await job_queue.connect()

    try:
        # Get all entities without wikidata_qid
        entities = await neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.wikidata_qid IS NULL
                AND e.entity_type IN ['PERSON', 'ORGANIZATION', 'LOCATION']
            RETURN e.id as id, e.canonical_name as name, e.entity_type as type,
                   e.profile_summary as profile
            LIMIT 50
        """)

        print(f"Found {len(entities)} entities to enrich")

        queued = 0
        for entity in entities:
            await job_queue.enqueue('wikidata_enrichment', {
                'entity_id': entity['id'],
                'canonical_name': entity['name'],
                'entity_type': entity['type'],
                'context': {
                    'description': entity.get('profile') or '',
                    'mentions': []
                }
            })
            queued += 1
            print(f"✅ Queued: {entity['name']} ({entity['type']})")

        print(f"\n✨ Queued {queued} entities for Wikidata enrichment")

    finally:
        await neo4j.close()
        await job_queue.close()

if __name__ == "__main__":
    asyncio.run(queue_entities())
