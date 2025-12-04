"""
Test entity hydration from Neo4j via ClaimRepository
"""
import asyncio
import asyncpg
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repositories.claim_repository import ClaimRepository
from services.neo4j_service import Neo4jService


async def test_hydration():
    """Test that we can hydrate entities from Neo4j"""

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Initialize ClaimRepository
    claim_repo = ClaimRepository(db_pool, neo4j)

    # Test claim ID with entities
    claim_id = uuid.UUID('e86f4697-263f-4f52-a9a7-9726dd2ca8e3')

    print(f"🔍 Testing claim: {claim_id}")

    # Fetch claim
    claim = await claim_repo.get_by_id(claim_id)

    if not claim:
        print("❌ Claim not found")
        return

    print(f"✅ Claim text: {claim.text[:80]}...")
    print(f"📊 Entity IDs in metadata: {claim.entity_ids}")
    print(f"📊 Entity names in metadata: {claim.entity_names}")

    # Hydrate entities
    print("\n🔄 Hydrating entities from Neo4j...")
    claim = await claim_repo.hydrate_entities(claim)

    if claim.entities:
        print(f"✅ Hydrated {len(claim.entities)} entities:")
        for entity in claim.entities:
            print(f"   - {entity.canonical_name} ({entity.entity_type})")
            print(f"     Wikidata: {entity.wikidata_qid or 'N/A'}")
            print(f"     Confidence: {entity.confidence}")
    else:
        print("❌ No entities hydrated")

    # Cleanup
    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(test_hydration())
