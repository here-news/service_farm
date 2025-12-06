"""
Investigate why CNN timeline claims created separate event instead of matching Wang Fuk Court Fire
"""
import asyncio
import sys
import os
import asyncpg
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository


async def main():
    # Connect
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )
    neo4j = Neo4jService()
    await neo4j.connect()

    event_repo = EventRepository(pg_pool, neo4j)
    claim_repo = ClaimRepository(pg_pool, neo4j)

    print("="*80)
    print("MATCHING FAILURE INVESTIGATION")
    print("="*80)

    # Event IDs
    wang_fuk_id = UUID("2569aee8-017e-4202-a2b0-044060b9fa6d")
    wang_cheong_id = UUID("94197800-e1e7-45ea-b6e5-b1b9a96506a8")

    # Get both events
    wang_fuk = await event_repo.get_by_id(wang_fuk_id)
    wang_cheong = await event_repo.get_by_id(wang_cheong_id)

    print(f"\n📊 Event 1 (created first):")
    print(f"   Name: {wang_fuk.canonical_name}")
    print(f"   Time: {wang_fuk.event_start} → {wang_fuk.event_end}")
    print(f"   Created: {wang_fuk.created_at}")

    print(f"\n📊 Event 2 (created second - should have matched Event 1):")
    print(f"   Name: {wang_cheong.canonical_name}")
    print(f"   Time: {wang_cheong.event_start} → {wang_cheong.event_end}")
    print(f"   Created: {wang_cheong.created_at}")

    # Get entities for each event
    print("\n" + "="*80)
    print("ENTITY COMPARISON")
    print("="*80)

    entities1 = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:MENTIONS]->(ent:Entity)
        RETURN ent.id as id, ent.name as name, ent.type as type
    """, {'event_id': str(wang_fuk_id)})

    entities2 = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:MENTIONS]->(ent:Entity)
        RETURN ent.id as id, ent.name as name, ent.type as type
    """, {'event_id': str(wang_cheong_id)})

    print(f"\nEvent 1 entities ({len(entities1)}):")
    entity_ids_1 = set()
    for e in entities1:
        entity_ids_1.add(e['id'])
        print(f"   - {e['name']} ({e['type']})")

    print(f"\nEvent 2 entities ({len(entities2)}):")
    entity_ids_2 = set()
    for e in entities2:
        entity_ids_2.add(e['id'])
        print(f"   - {e['name']} ({e['type']})")

    # Calculate entity overlap
    overlap = entity_ids_1.intersection(entity_ids_2)
    if overlap:
        print(f"\n✅ Shared entities ({len(overlap)}):")
        for ent_id in overlap:
            ent = [e for e in entities1 if e['id'] == ent_id][0]
            print(f"   - {ent['name']} ({ent['type']})")
    else:
        print(f"\n❌ NO SHARED ENTITIES")

    # Check if Event 1 would have been found as candidate
    print("\n" + "="*80)
    print("CANDIDATE FINDING SIMULATION")
    print("="*80)

    # Get reference time from Event 2's first claim
    claims2 = await event_repo.get_event_claims_with_timeline_data(wang_cheong_id)
    if claims2:
        reference_time = claims2[0].event_time or claims2[0].created_at
        print(f"\nReference time (from Event 2's claims): {reference_time}")

        # Find candidates using Event 2's entities
        candidates = await event_repo.find_candidates(
            entity_ids=list(entity_ids_2),
            reference_time=reference_time,
            time_window_days=7
        )

        print(f"\nCandidates found: {len(candidates)}")
        for c in candidates:
            is_target = c['id'] == str(wang_fuk_id)
            marker = "🎯 TARGET" if is_target else ""
            print(f"   - {c['canonical_name']} {marker}")
            print(f"     ID: {c['id']}")
            print(f"     Time: {c['event_start']} → {c['event_end']}")

        # Check if Wang Fuk was in candidates
        if str(wang_fuk_id) in [c['id'] for c in candidates]:
            print(f"\n✅ Event 1 WAS found as candidate")
        else:
            print(f"\n❌ Event 1 was NOT found as candidate")
            print(f"\n   Possible reasons:")
            print(f"   1. No shared entities")
            print(f"   2. Time window (7 days) didn't include Event 1")
            print(f"   3. Entity matching query issue")

    # Show sample claims from each event
    print("\n" + "="*80)
    print("SAMPLE CLAIMS COMPARISON")
    print("="*80)

    claims1 = await event_repo.get_event_claims_with_timeline_data(wang_fuk_id)
    print(f"\nEvent 1 claims ({len(claims1)}):")
    for i, claim in enumerate(sorted(claims1, key=lambda c: c.event_time or c.created_at)[:3]):
        time_str = claim.event_time.strftime('%m-%d %H:%M') if claim.event_time else 'no-time'
        print(f"   {i+1}. [{time_str}] {claim.text[:80]}...")

    print(f"\nEvent 2 claims ({len(claims2)}):")
    for i, claim in enumerate(sorted(claims2, key=lambda c: c.event_time or c.created_at)[:3]):
        time_str = claim.event_time.strftime('%m-%d %H:%M') if claim.event_time else 'no-time'
        print(f"   {i+1}. [{time_str}] {claim.text[:80]}...")

    # Analysis
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    print("\nQuestions to answer:")
    print("1. Why didn't Event 2's claims match Event 1 during candidate finding?")
    print("2. What entities are extracted from CNN timeline claims?")
    print("3. Is the entity extraction missing key entities (Wang Fuk Court, etc)?")
    print("4. Or is the scoring threshold too high (> 0.3)?")

    await neo4j.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
