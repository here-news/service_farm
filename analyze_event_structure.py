"""
Analyze current event structure to understand empirical patterns
"""
import asyncio
import sys
import os
import asyncpg
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository


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

    print("="*80)
    print("CURRENT EVENT STRUCTURE ANALYSIS")
    print("="*80)

    # 1. Get all root events
    print("\n📊 ROOT EVENTS:")
    root_dicts = await event_repo.list_root_events(limit=10)

    for root_dict in root_dicts:
        # Get full Event object
        from uuid import UUID
        root = await event_repo.get_by_id(UUID(root_dict['id']))

        print(f"\n  {root.canonical_name}")
        print(f"    id: {root.id}")
        print(f"    time: {root.event_start} → {root.event_end}")
        if root.event_end and root.event_start:
            span_hours = (root.event_end - root.event_start).total_seconds() / 3600
            print(f"    time span: {span_hours:.1f}h ({span_hours/24:.1f} days)")

        # Get sub-events
        subs = await event_repo.get_sub_events(root.id)

        if subs:
            print(f"    Sub-events: {len(subs)}")
            for sub in subs:
                print(f"      - {sub.canonical_name}")
                print(f"        time: {sub.event_start} → {sub.event_end}")

        # Get direct claims
        try:
            claims = await event_repo.get_event_claims_with_timeline_data(root.id)
            print(f"    Direct claims: {len(claims)}")
        except Exception as e:
            print(f"    Claims: (error)")

    # 2. Deep dive into Wang Fuk Fire
    print("\n" + "="*80)
    print("WANG FUK COURT FIRE - DETAILED STRUCTURE")
    print("="*80)

    # Find Wang Fuk event
    wang_fuk_event = None
    for root_dict in root_dicts:
        if 'Wang Fuk' in root_dict['canonical_name'] or 'wang' in root_dict['canonical_name'].lower():
            wang_fuk_event = await event_repo.get_by_id(UUID(root_dict['id']))
            break

    if wang_fuk_event:
        print(f"\nEvent: {wang_fuk_event.canonical_name}")
        print(f"ID: {wang_fuk_event.id}")
        print(f"Time: {wang_fuk_event.event_start} → {wang_fuk_event.event_end}")

        # Get sub-events
        subs = await event_repo.get_sub_events(wang_fuk_event.id)
        print(f"\nTotal sub-events: {len(subs)}")

        # Get all claims for the event hierarchy
        print(f"\n📁 Sub-events ({len(subs)}):")
        for sub in subs:
            print(f"\n  {sub.canonical_name}")
            print(f"    ID: {sub.id}")
            print(f"    Time: {sub.event_start} → {sub.event_end}")

            # Get claims for this sub-event
            try:
                sub_claims = await event_repo.get_event_claims_with_timeline_data(sub.id)
                print(f"    Claims: {len(sub_claims)}")

                if sub_claims:
                    times = [c.event_time for c in sub_claims if c.event_time]
                    if times:
                        print(f"    Claim times: {min(times)} → {max(times)}")
            except Exception as e:
                print(f"    Claims: (error loading)")

        # Analyze temporal patterns
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)

        # Get all claims from root + all sub-events
        all_claims = await event_repo.get_event_claims_with_timeline_data(wang_fuk_event.id)

        for sub in subs:
            try:
                sub_claims = await event_repo.get_event_claims_with_timeline_data(sub.id)
                all_claims.extend(sub_claims)
            except:
                pass

        claims_with_time = [c for c in all_claims if c.event_time]
        print(f"\nClaims with event_time: {len(claims_with_time)}")

        if claims_with_time:
            print("\nTemporal distribution:")

            # Group by day
            days = {}
            for claim in claims_with_time:
                t = claim.event_time
                day = f"{t.year}-{t.month:02d}-{t.day:02d}"
                if day not in days:
                    days[day] = []
                days[day].append(claim)

            for day, claims in sorted(days.items()):
                print(f"\n  {day}: {len(claims)} claims")
                for claim in sorted(claims, key=lambda c: c.event_time)[:3]:  # Show first 3
                    print(f"    [{claim.event_time.strftime('%H:%M')}] {claim.text[:60]}...")

        # Analyze thematic patterns
        print("\n" + "="*80)
        print("THEMATIC PATTERN ANALYSIS")
        print("="*80)

        # Group sub-events by name patterns
        theme_groups = {}
        for sub in subs:
            name = sub.canonical_name
            # Extract theme from name (e.g., "Fire Response and Alarm Levels")
            if ' - ' in name:
                theme = name.split(' - ')[-1]
            else:
                theme = name

            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(sub)

        print(f"\nIdentified themes: {len(theme_groups)}")
        for theme, events in theme_groups.items():
            print(f"\n  Theme: {theme}")
            print(f"    Events: {len(events)}")
            for event in events:
                try:
                    event_claims = await event_repo.get_event_claims_with_timeline_data(event.id)
                    claim_count = len(event_claims)
                except:
                    claim_count = 0
                print(f"      - {event.canonical_name} ({claim_count} claims)")

    await neo4j.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
