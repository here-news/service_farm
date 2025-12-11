"""
Test LiveEvent with Bayesian Topology Integration

Tests the integration of ClaimTopologyService into LiveEvent's
narrative regeneration, using stored publisher priors.
"""
import asyncio
import sys
import os

sys.path.insert(0, '/app')

from services.neo4j_service import Neo4jService
from services.claim_topology import ClaimTopologyService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from models.domain.live_event import LiveEvent
from openai import AsyncOpenAI
import asyncpg


async def main():
    print("=" * 80)
    print("🧬 TEST: LiveEvent with Bayesian Topology (Stored Priors)")
    print("=" * 80)

    # Connect to services
    neo4j = Neo4jService()
    await neo4j.connect()

    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews')
    )

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Create repositories
    event_repo = EventRepository(pg_pool, neo4j)
    claim_repo = ClaimRepository(pg_pool, neo4j)

    # Create topology service
    topology_service = ClaimTopologyService(openai_client)

    # Find an existing event with claims
    print("\n📄 Finding existing event...")
    events = await neo4j._execute_read("""
        MATCH (e:Event)-[:SUPPORTS]->(c:Claim)
        WITH e, count(c) as claim_count
        WHERE claim_count >= 5
        RETURN e.id as id, e.canonical_name as name, claim_count
        ORDER BY claim_count DESC
        LIMIT 1
    """)

    if not events:
        print("❌ No events with claims found")
        return

    event_data = events[0]
    print(f"   Found: {event_data['name']} ({event_data['claim_count']} claims)")

    # Load the event
    event = await event_repo.get_by_id(event_data['id'])
    if not event:
        print("❌ Failed to load event")
        return

    # Create a mock event_service (we only need event_repo for this test)
    class MockEventService:
        def __init__(self, event_repo):
            self.event_repo = event_repo

        async def _generate_event_narrative(self, event, claims):
            return "Fallback narrative"

    mock_service = MockEventService(event_repo)

    # Create LiveEvent with topology service
    print("\n🌱 Creating LiveEvent with topology service...")
    live_event = LiveEvent(event, mock_service, topology_service)

    # Hydrate claims - this now fetches publisher priors
    await live_event.hydrate(claim_repo)
    print(f"   Loaded {len(live_event.claims)} claims")

    # Show publisher prior coverage
    print(f"\n📊 Publisher Priors Coverage:")
    stored_count = sum(1 for p in live_event.publisher_priors.values() if p.get('base_prior'))
    print(f"   {stored_count}/{len(live_event.claims)} claims have stored priors")

    # Show sample priors
    if live_event.publisher_priors:
        print(f"\n   Sample publisher priors:")
        for claim_id, prior_info in list(live_event.publisher_priors.items())[:5]:
            publisher = prior_info.get('publisher_name', 'Unknown')
            source_type = prior_info.get('source_type', 'unknown')
            base_prior = prior_info.get('base_prior', 0.50)
            print(f"   [{base_prior:.2f}] {source_type:12s} - {publisher}")

    # Test: Regenerate narrative using Bayesian topology
    print(f"\n{'='*80}")
    print("📝 REGENERATING NARRATIVE WITH BAYESIAN TOPOLOGY")
    print(f"{'='*80}")

    await live_event.regenerate_narrative()

    print(f"\n{'='*80}")
    print("📖 FINAL NARRATIVE")
    print(f"{'='*80}")
    print(live_event.event.summary)

    # Check if plausibilities were stored
    print(f"\n{'='*80}")
    print("📊 CHECKING STORED PLAUSIBILITIES")
    print(f"{'='*80}")

    results = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[r:SUPPORTS]->(c:Claim)
        WHERE r.plausibility IS NOT NULL
        RETURN c.text as text, r.plausibility as plausibility
        ORDER BY r.plausibility DESC
        LIMIT 10
    """, {'event_id': event.id})

    if results:
        print(f"   Found {len(results)} claims with plausibility scores:")
        for r in results:
            print(f"   [{r['plausibility']:.2f}] {r['text'][:60]}...")
    else:
        print("   No plausibility scores stored yet")

    # Check publisher entities for source_type/base_prior
    print(f"\n{'='*80}")
    print("📰 CHECKING PUBLISHER ENTITIES")
    print(f"{'='*80}")

    publishers = await neo4j._execute_read("""
        MATCH (pub:Entity {is_publisher: true})
        WHERE pub.source_type IS NOT NULL
        RETURN pub.canonical_name as name, pub.domain as domain,
               pub.source_type as source_type, pub.base_prior as base_prior
        ORDER BY pub.base_prior DESC
        LIMIT 10
    """)

    if publishers:
        print(f"   Publishers with stored priors:")
        for p in publishers:
            print(f"   [{p['base_prior']:.2f}] {p['source_type']:12s} - {p['name']} ({p['domain']})")
    else:
        print("   No publishers with stored priors yet (run knowledge worker to populate)")

    # Cleanup
    await neo4j.close()
    await pg_pool.close()

    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
