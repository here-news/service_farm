"""
Inspect event structure for semantic plausibility
"""
import asyncio
import sys
import os
import asyncpg
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository


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

    print("="*80)
    print("EVENT STRUCTURE PLAUSIBILITY CHECK")
    print("="*80)

    # Get root event
    root = await event_repo.get_by_id(UUID("7bf046d2-7d7c-49bb-95ac-06a676ec43a3"))

    print(f"\n🌳 ROOT: {root.canonical_name}")
    print(f"   Time: {root.event_start} → {root.event_end}")

    # Get direct claims (sampled)
    root_claims = await event_repo.get_event_claims_with_timeline_data(root.id)
    print(f"\n   Direct claims: {len(root_claims)}")
    print("   Sample (first 5):")
    for i, claim in enumerate(sorted(root_claims, key=lambda c: c.event_time or c.created_at)[:5]):
        time_str = claim.event_time.strftime('%m-%d %H:%M') if claim.event_time else 'no-time'
        print(f"     {i+1}. [{time_str}] {claim.text[:75]}...")

    # Get level 1 children
    level1_children = await event_repo.get_sub_events(root.id)
    print(f"\n   Level 1 children: {len(level1_children)}")

    for child1 in sorted(level1_children, key=lambda e: e.event_start or e.created_at):
        print(f"\n   ├─ {child1.canonical_name}")

        span_h = (child1.event_end - child1.event_start).total_seconds() / 3600 if child1.event_end and child1.event_start else 0
        print(f"   │  Time: {child1.event_start} → {child1.event_end} ({span_h:.1f}h)")

        # Get claims for this child
        child1_claims = await event_repo.get_event_claims_with_timeline_data(child1.id)
        print(f"   │  Claims: {len(child1_claims)}")

        # Show all claims (to check plausibility)
        print(f"   │  All claims:")
        for i, claim in enumerate(sorted(child1_claims, key=lambda c: c.event_time or c.created_at)):
            time_str = claim.event_time.strftime('%m-%d %H:%M') if claim.event_time else 'no-time'
            print(f"   │    {i+1}. [{time_str}] {claim.text[:70]}...")

        # Check for level 2 children
        level2_children = await event_repo.get_sub_events(child1.id)
        if level2_children:
            print(f"   │")
            print(f"   │  Level 2 children: {len(level2_children)}")

            for child2 in sorted(level2_children, key=lambda e: e.event_start or e.created_at):
                print(f"   │  ├─ {child2.canonical_name}")

                span_h = (child2.event_end - child2.event_start).total_seconds() / 3600 if child2.event_end and child2.event_start else 0
                print(f"   │  │  Time: {child2.event_start} → {child2.event_end} ({span_h:.1f}h)")

                # Get claims
                child2_claims = await event_repo.get_event_claims_with_timeline_data(child2.id)
                print(f"   │  │  Claims: {len(child2_claims)}")

                print(f"   │  │  All claims:")
                for i, claim in enumerate(sorted(child2_claims, key=lambda c: c.event_time or c.created_at)):
                    time_str = claim.event_time.strftime('%m-%d %H:%M') if claim.event_time else 'no-time'
                    print(f"   │  │    {i+1}. [{time_str}] {claim.text[:65]}...")

    # Analysis summary
    print("\n" + "="*80)
    print("PLAUSIBILITY ANALYSIS")
    print("="*80)

    print("\nQuestions to consider:")
    print("1. Do level 1 events represent coherent themes/phases?")
    print("2. Do level 2 events represent meaningful sub-themes?")
    print("3. Are temporal spans plausible for the event types?")
    print("4. Are claims properly grouped by topic?")
    print("5. Is there unnecessary fragmentation (should some be merged)?")
    print("6. Is there missing structure (should some be split)?")

    await neo4j.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
