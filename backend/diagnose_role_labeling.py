#!/usr/bin/env python3
"""
Diagnose Role Labeling for WFC Incidents
=========================================

Check 2: Referent recall
- How many artifacts contain WFC as a referent?
- What role is assigned to WFC?

Check 3: Filtering correctness
- Is WFC being filtered out by DF filtering?
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

import asyncio
import os

async def check_role_labeling():
    from services.neo4j_service import Neo4jService
    from workers.principled_weaver import CaseBuilder
    from openai import AsyncOpenAI
    import asyncpg

    neo4j = Neo4jService()
    await neo4j.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD'),
    )

    llm_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Get 3 WFC incidents
    wfc_query = '''
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id, i.anchor_entities as anchors, i.title as title
    ORDER BY i.id
    LIMIT 3
    '''
    wfc_results = await neo4j._execute_read(wfc_query)
    wfc_ids = [r['id'] for r in wfc_results]

    print("=" * 70)
    print("DIAGNOSTIC: Role Labeling for WFC Incidents")
    print("=" * 70)

    # Initialize CaseBuilder
    cb = CaseBuilder(neo4j, db_pool=db_pool)
    await cb.load_incidents()

    # Filter to just these 3 WFC incidents
    cb.incidents = {iid: inc for iid, inc in cb.incidents.items() if iid in wfc_ids}
    print(f"\nLoaded {len(cb.incidents)} WFC incidents")
    print()

    # Show raw incident data
    print("RAW INCIDENT DATA:")
    print("-" * 70)
    for iid in wfc_ids:
        inc = cb.incidents.get(iid)
        if inc:
            print(f"Incident: {iid[:30]}...")
            print(f"  anchor_entities: {inc.anchor_entities}")
            print()

    # Build role artifacts (this calls LLM)
    print("Building role artifacts with LLM...")
    print("(This will take ~30 seconds for 3 incidents)")
    print()

    incidents_list = list(cb.incidents.values())
    artifacts = await cb.build_role_artifacts(incidents_list, llm_client)

    print("=" * 70)
    print("ROLE ARTIFACT RESULTS:")
    print("=" * 70)

    wfc_entity_id = None

    for r in wfc_results:
        iid = r['id']
        anchors = r['anchors']

        print(f"\nIncident: {iid[:30]}...")
        print(f"  anchor_entities (raw names): {anchors}")

        art = artifacts.get(iid)
        if art:
            print(f"\n  Role Artifact:")
            print(f"    referent_entity_ids: {set(art.referent_entity_ids)}")
            print(f"    role_map:")
            for eid, role in art.role_map.items():
                is_wfc = 'Wang Fuk Court' in eid or eid.startswith('en_')
                marker = " <-- WFC" if 'Wang Fuk Court' in eid else ""
                print(f"      '{eid[:40]}': {role.value}{marker}")
                if 'Wang Fuk Court' in eid or (wfc_entity_id is None and is_wfc):
                    wfc_entity_id = eid

            # Check if Wang Fuk Court is in referents
            wfc_in_referents = any('Wang Fuk Court' in str(eid) for eid in art.referent_entity_ids)
            print(f"\n    'Wang Fuk Court' in referents: {wfc_in_referents}")

            if not wfc_in_referents:
                print("    PROBLEM: WFC not in referent set!")
                # Check if it was in role_map but filtered
                for eid in art.role_map:
                    if 'Wang Fuk Court' in eid:
                        role = art.role_map[eid]
                        print(f"    WFC role: {role.value}")
                        if role.value == 'BROAD_LOCATION':
                            print("    CAUSE: WFC was demoted to BROAD_LOCATION (DF filtering)")
                        elif not role.is_referent:
                            print(f"    CAUSE: WFC role {role.value} is not a referent role")
        else:
            print(f"\n  NO ARTIFACT CREATED!")
        print()

    # Check DF statistics
    print("=" * 70)
    print("DF ANALYSIS:")
    print("=" * 70)

    # Query global DF for WFC
    df_query = '''
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN count(i) as wfc_count
    '''
    df_result = await neo4j._execute_read(df_query)
    wfc_df = df_result[0]['wfc_count']

    total_query = 'MATCH (i:Incident) RETURN count(i) as total'
    total_result = await neo4j._execute_read(total_query)
    total = total_result[0]['total']

    print(f"Wang Fuk Court global DF: {wfc_df}")
    print(f"Total incidents: {total}")
    print(f"DF fraction: {wfc_df/total:.1%}")
    print(f"DF threshold for demotion: 5%")
    print(f"WFC exceeds threshold: {wfc_df/total > 0.05}")

    if wfc_df/total > 0.05:
        print("\nWARNING: WFC exceeds DF threshold!")
        print("If WFC is typed as LOCATION, it will be demoted to BROAD_LOCATION")

    # Check WFC entity type in Neo4j
    print("\n" + "=" * 70)
    print("WFC ENTITY TYPE:")
    print("=" * 70)

    wfc_type_query = '''
    MATCH (e:Entity)
    WHERE e.canonical_name = 'Wang Fuk Court'
    RETURN e.id, coalesce(e.entity_type, e.type, 'UNKNOWN') as type
    '''
    wfc_type_result = await neo4j._execute_read(wfc_type_query)
    if wfc_type_result:
        print(f"WFC entity type in Neo4j: {wfc_type_result[0]['type']}")
        if wfc_type_result[0]['type'] == 'LOCATION':
            print("\nROOT CAUSE IDENTIFIED:")
            print("- WFC is typed as LOCATION in Neo4j")
            print("- LLM labels it as REFERENT_LOCATION")
            print("- DF filtering demotes REFERENT_LOCATION -> BROAD_LOCATION")
            print("- WFC removed from referent set -> no spine edges formed")
    else:
        print("WFC entity not found in Neo4j")

    await neo4j.close()
    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(check_role_labeling())
