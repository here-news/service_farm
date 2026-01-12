#!/usr/bin/env python3
"""
Diagnose Entity ID Space for WFC Incidents
==========================================

Check 1: ID space alignment
- anchor_entities are name strings
- Need to verify Entity nodes exist for those names
- If not, fallback uses name as ID with type=UNKNOWN
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

import asyncio
import os

async def check_entity_ids():
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    # Get 3 WFC incidents with their anchor_entities
    wfc_query = '''
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id, i.anchor_entities as anchors, i.title as title
    ORDER BY i.id
    LIMIT 3
    '''
    wfc_results = await neo4j._execute_read(wfc_query)

    print("=" * 70)
    print("DIAGNOSTIC: Entity ID Space Analysis")
    print("=" * 70)

    # Collect all anchor names from these incidents
    all_anchor_names = set()
    for r in wfc_results:
        print(f"\nIncident: {r['id'][:40]}...")
        print(f"  anchor_entities: {r['anchors']}")
        all_anchor_names.update(r['anchors'] or [])

    print(f"\n\nAll unique anchor names from 3 WFC incidents:")
    for name in sorted(all_anchor_names):
        print(f"  - {name}")

    # Now try to look up Entity nodes for these names
    print(f"\n\nEntity node lookup (line 3142-3149 in principled_weaver.py):")
    entity_query = '''
    MATCH (e:Entity)
    WHERE e.canonical_name IN $names
    RETURN e.canonical_name as name,
           e.id as id,
           coalesce(e.entity_type, e.type, 'UNKNOWN') as type
    '''
    entity_results = await neo4j._execute_read(entity_query, {"names": list(all_anchor_names)})

    found_entities = {r["name"]: r for r in entity_results}

    print(f"\nFound {len(found_entities)} Entity nodes out of {len(all_anchor_names)} anchor names:")
    for name in sorted(all_anchor_names):
        if name in found_entities:
            e = found_entities[name]
            print(f"  FOUND: '{name}' -> id={e['id'][:20] if e['id'] else 'None'}..., type={e['type']}")
        else:
            print(f"  MISSING: '{name}' -> will use fallback id=name, type=UNKNOWN")

    # Check if WFC has any Entity nodes at all
    print(f"\n\nDirect search for 'Wang Fuk Court' Entity nodes:")
    wfc_entity_query = '''
    MATCH (e:Entity)
    WHERE e.canonical_name CONTAINS 'Wang' OR e.name CONTAINS 'Wang'
    RETURN e.id as id, e.canonical_name as canonical, e.name as name,
           coalesce(e.entity_type, e.type, 'UNKNOWN') as type
    LIMIT 10
    '''
    wfc_entity_results = await neo4j._execute_read(wfc_entity_query)

    if wfc_entity_results:
        for r in wfc_entity_results:
            print(f"  id={r['id'][:30] if r['id'] else 'None'}..., canonical={r['canonical']}, name={r['name']}, type={r['type']}")
    else:
        print("  NO Entity nodes found containing 'Wang'")

    # Check total Entity node count
    count_query = 'MATCH (e:Entity) RETURN count(e) as cnt'
    count_result = await neo4j._execute_read(count_query)
    print(f"\n\nTotal Entity nodes in Neo4j: {count_result[0]['cnt']}")

    await neo4j.close()

if __name__ == "__main__":
    asyncio.run(check_entity_ids())
