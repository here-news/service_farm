"""
Test update detection on Wang Fuk Court Fire casualty data
"""
import asyncio
import sys
import os
import uuid
import asyncpg
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repositories.event_repository import EventRepository
from services.neo4j_service import Neo4jService
from services.update_detector import UpdateDetector


async def main():
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    event_repo = EventRepository(pg_pool, neo4j)
    update_detector = UpdateDetector()

    # Find Wang Fuk Court Fire event
    print("🔍 Finding Wang Fuk Court Fire event...")
    events = await event_repo.list_root_events(limit=100)

    root_event = None
    for event in events:
        event_name = event.get('canonical_name') if isinstance(event, dict) else event.canonical_name
        if "Wang Fuk" in event_name or "fire" in event_name.lower():
            if isinstance(event, dict):
                event_id = uuid.UUID(event['id'])
                root_event = await event_repo.get_by_id(event_id)
            else:
                root_event = event
            print(f"✅ Found: {root_event.canonical_name} ({root_event.id})")
            break

    if not root_event:
        print("❌ No event found!")
        await neo4j.close()
        await pg_pool.close()
        return

    # Get all claims WITH reported_time from page.pub_time
    print(f"\n📊 Getting claims with timeline data (event_time + reported_time)...")
    all_claims = await event_repo.get_event_claims_with_timeline_data(root_event.id)

    # Also get claims from sub-events
    sub_events = await event_repo.get_sub_events(root_event.id)
    for sub_event in sub_events:
        sub_claims = await event_repo.get_event_claims_with_timeline_data(sub_event.id)
        all_claims.extend(sub_claims)
        print(f"  Sub-event: {sub_event.canonical_name} ({len(sub_claims)} claims)")

    print(f"\n📊 Total claims: {len(all_claims)}")
    print(f"   With event_time: {len([c for c in all_claims if c.event_time])}")
    print(f"   With reported_time: {len([c for c in all_claims if c.reported_time])}")

    # Test 1: Auto-detect topics
    print("\n" + "="*80)
    print("TEST 1: AUTO-DETECT TOPICS")
    print("="*80)

    topic_counts = {}
    for claim in all_claims:
        topic = update_detector.detect_topic_key(claim)
        if topic:
            if topic not in topic_counts:
                topic_counts[topic] = []
            topic_counts[topic].append(claim)
            value = update_detector.extract_numeric_value(claim, topic)
            print(f"\n📌 {topic}: {value}")
            print(f"   Event time: {claim.event_time}")
            print(f"   Reported time: {claim.reported_time}")
            print(f"   Text: {claim.text[:100]}...")

    print(f"\n✨ Found {len(topic_counts)} topics:")
    for topic, claims in topic_counts.items():
        print(f"  - {topic}: {len(claims)} claims")

    # Test 2: Build update chains
    print("\n" + "="*80)
    print("TEST 2: BUILD UPDATE CHAINS")
    print("="*80)

    chains = update_detector.build_update_chains(all_claims)

    print(f"\n✨ Built {len(chains)} update chains:")
    for topic_key, chain in chains.items():
        print(f"\n📊 {topic_key.upper()} CHAIN ({len(chain)} claims):")
        for i, claim in enumerate(chain):
            value = update_detector.extract_numeric_value(claim, topic_key)
            current_marker = " [CURRENT]" if claim.is_current else " [SUPERSEDED]"
            print(f"  {i+1}. {value} - Reported: {claim.reported_time}{current_marker}")
            print(f"     Text: {claim.text[:80]}...")

    # Test 3: Detect update relationships
    print("\n" + "="*80)
    print("TEST 3: DETECT UPDATE RELATIONSHIPS")
    print("="*80)

    updates = update_detector.find_updates_in_event(all_claims)

    print(f"\n✨ Found {len(updates)} update relationships:")
    for new_claim, old_claim, topic_key in updates:
        new_val = update_detector.extract_numeric_value(new_claim, topic_key)
        old_val = update_detector.extract_numeric_value(old_claim, topic_key)
        print(f"\n📈 {topic_key}: {old_val} → {new_val}")
        print(f"   Old: {old_claim.text[:60]}...")
        print(f"   New: {new_claim.text[:60]}...")
        print(f"   Time gap: {(new_claim.reported_time - old_claim.reported_time).total_seconds() / 3600:.1f}h")

    # Test 4: Get current values
    print("\n" + "="*80)
    print("TEST 4: CURRENT STATE (SNAPSHOT)")
    print("="*80)

    print("\n✨ Current values for each topic:")
    for topic_key, chain in chains.items():
        current = update_detector.get_current_value(chain)
        if current:
            value = update_detector.extract_numeric_value(current, topic_key)
            print(f"\n  {topic_key}: {value}")
            print(f"    From: {current.text[:80]}...")
            print(f"    Reported: {current.reported_time}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total claims analyzed: {len(all_claims)}")
    print(f"Topics detected: {len(topic_counts)}")
    print(f"Update chains: {len(chains)}")
    print(f"Update relationships: {len(updates)}")
    print(f"\nUpdate chain topics: {', '.join(chains.keys())}")

    # Cleanup
    await neo4j.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
