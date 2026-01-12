#!/usr/bin/env python3
"""
Tier B Proof: Controlled Subset from Real Database
===================================================
"""

import os
import sys
import asyncio
import json
from datetime import datetime

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

import asyncpg
from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService
from workers.principled_weaver import CaseBuilder

PROOF_METADATA = {
    'tier': 'B',
    'timestamp': datetime.utcnow().isoformat(),
    'llm_model': 'gpt-4o-mini',
    'llm_temperature': 0.0,
}

async def run_tier_b():
    print('=' * 70)
    print('TIER B PROOF: Controlled Subset from Real Database')
    print('=' * 70)
    print(f'Timestamp: {PROOF_METADATA["timestamp"]}')
    print()

    # Connect
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

    # Load Tier B dataset: All WFC + 100 distractors
    print('Loading Tier B dataset...')

    wfc_query = '''
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id
    ORDER BY i.id
    '''
    wfc_results = await neo4j._execute_read(wfc_query)
    wfc_ids_list = [r['id'] for r in wfc_results]

    distractor_query = '''
    MATCH (i:Incident)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    WITH i ORDER BY i.id LIMIT 100
    RETURN i.id as id
    '''
    distractor_results = await neo4j._execute_read(distractor_query)
    distractor_ids = [r['id'] for r in distractor_results]

    all_ids = set(wfc_ids_list + distractor_ids)
    print(f'Dataset: {len(wfc_ids_list)} WFC + {len(distractor_ids)} distractors = {len(all_ids)} total')
    print()

    # Initialize CaseBuilder
    cb = CaseBuilder(neo4j, db_pool=db_pool)
    await cb.load_incidents()

    # Filter to Tier B subset
    cb.incidents = {iid: inc for iid, inc in cb.incidents.items() if iid in all_ids}
    print(f'Loaded {len(cb.incidents)} incidents from subset')
    print()

    # Ground truth WFC set
    wfc_ground_truth = {iid for iid in cb.incidents
                        if 'Wang Fuk Court' in (cb.incidents[iid].anchor_entities or set())}
    print(f'Ground truth WFC incidents: {len(wfc_ground_truth)}')
    print()

    # =========================================================================
    # ABLATION A1: Heuristic Overlap Baseline
    # =========================================================================
    print('=' * 70)
    print('ABLATION A1: Heuristic Overlap (Baseline)')
    print('=' * 70)

    edges_a1 = []
    incident_list = list(cb.incidents.keys())
    for i, id1 in enumerate(incident_list):
        for id2 in incident_list[i+1:]:
            inc1 = cb.incidents[id1]
            inc2 = cb.incidents[id2]
            ents1 = inc1.anchor_entities or set()
            ents2 = inc2.anchor_entities or set()
            overlap = ents1 & ents2
            if overlap:
                edges_a1.append((id1, id2, overlap))

    # Union-find for A1
    parent_a1 = {iid: iid for iid in cb.incidents}

    def find_a1(x):
        if parent_a1[x] != x:
            parent_a1[x] = find_a1(parent_a1[x])
        return parent_a1[x]

    def union_a1(x, y):
        px, py = find_a1(x), find_a1(y)
        if px != py:
            parent_a1[px] = py

    for id1, id2, _ in edges_a1:
        union_a1(id1, id2)

    comps_a1 = {}
    for iid in cb.incidents:
        root = find_a1(iid)
        if root not in comps_a1:
            comps_a1[root] = set()
        comps_a1[root].add(iid)

    cases_a1 = [c for c in comps_a1.values() if len(c) >= 2]

    wfc_case_a1 = None
    for c in sorted(cases_a1, key=len, reverse=True):
        if c & wfc_ground_truth:
            wfc_case_a1 = c
            break

    largest_a1 = max((len(c) for c in cases_a1), default=0)
    wfc_in_case_a1 = len(wfc_case_a1 & wfc_ground_truth) if wfc_case_a1 else 0
    coverage_a1 = wfc_in_case_a1 / len(wfc_ground_truth) if wfc_ground_truth else 0
    purity_a1 = wfc_in_case_a1 / len(wfc_case_a1) if wfc_case_a1 else 0
    contaminants_a1 = (wfc_case_a1 - wfc_ground_truth) if wfc_case_a1 else set()

    print(f'Total cases: {len(cases_a1)}')
    print(f'Largest case: {largest_a1}')
    print(f'Spine edges: {len(edges_a1)}')
    print(f'WFC coverage: {coverage_a1:.1%} ({wfc_in_case_a1}/{len(wfc_ground_truth)})')
    print(f'WFC purity: {purity_a1:.1%} ({wfc_in_case_a1}/{len(wfc_case_a1) if wfc_case_a1 else 0})')
    print(f'Contaminants: {len(contaminants_a1)}')

    if contaminants_a1:
        print('Top contaminants:')
        for iid in list(contaminants_a1)[:5]:
            inc = cb.incidents.get(iid)
            if inc:
                print(f'  {iid[:12]}: {list(inc.anchor_entities or set())[:4]}')
    print()

    # =========================================================================
    # ABLATION A3: Role-Based + DF Filtering (Full Implementation)
    # =========================================================================
    print('=' * 70)
    print('ABLATION A3: Role-Based Spine + DF Filtering')
    print('=' * 70)
    print('Running LLM role labeling (this takes ~15 minutes)...')
    print()

    cases_a3 = await cb.build_cases_relational(llm_client=llm_client)

    # Find WFC case in A3
    wfc_case_a3 = None
    for c in sorted(cases_a3, key=lambda x: len(x.incident_ids), reverse=True):
        case_anchors = set()
        for iid in c.incident_ids:
            inc = cb.incidents.get(iid)
            if inc:
                case_anchors.update(inc.anchor_entities or set())
        if 'Wang Fuk Court' in case_anchors:
            wfc_case_a3 = c
            break

    wfc_case_ids_a3 = wfc_case_a3.incident_ids if wfc_case_a3 else set()
    largest_a3 = max((len(c.incident_ids) for c in cases_a3), default=0)
    wfc_in_case_a3 = len(wfc_case_ids_a3 & wfc_ground_truth)
    coverage_a3 = wfc_in_case_a3 / len(wfc_ground_truth) if wfc_ground_truth else 0
    purity_a3 = wfc_in_case_a3 / len(wfc_case_ids_a3) if wfc_case_ids_a3 else 0
    contaminants_a3 = wfc_case_ids_a3 - wfc_ground_truth

    print(f'Total cases: {len(cases_a3)}')
    print(f'Largest case: {largest_a3}')
    print(f'WFC coverage: {coverage_a3:.1%} ({wfc_in_case_a3}/{len(wfc_ground_truth)})')
    print(f'WFC purity: {purity_a3:.1%} ({wfc_in_case_a3}/{len(wfc_case_ids_a3) if wfc_case_ids_a3 else 0})')
    print(f'Contaminants: {len(contaminants_a3)}')

    if contaminants_a3:
        print('Contaminants (bridge audit needed):')
        for iid in list(contaminants_a3)[:5]:
            inc = cb.incidents.get(iid)
            if inc:
                print(f'  {iid[:12]}: {list(inc.anchor_entities or set())[:4]}')
    else:
        print('No contaminants - clean membrane!')
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('=' * 70)
    print('TIER B RESULTS SUMMARY')
    print('=' * 70)
    print()
    print(f'Dataset: {len(wfc_ids_list)} WFC + {len(distractor_ids)} distractors')
    print()
    print('| Ablation | Cases | Largest | WFC Coverage | WFC Purity | Contaminants |')
    print('|----------|-------|---------|--------------|------------|--------------|')
    print(f'| A1 Heuristic | {len(cases_a1)} | {largest_a1} | {coverage_a1:.1%} | {purity_a1:.1%} | {len(contaminants_a1)} |')
    print(f'| A3 DF-Filter | {len(cases_a3)} | {largest_a3} | {coverage_a3:.1%} | {purity_a3:.1%} | {len(contaminants_a3)} |')
    print()

    if purity_a3 > purity_a1:
        print(f'IMPROVEMENT: Purity {purity_a1:.1%} -> {purity_a3:.1%} (+{(purity_a3-purity_a1)*100:.1f}pp)')
    elif purity_a3 < purity_a1:
        print(f'REGRESSION: Purity {purity_a1:.1%} -> {purity_a3:.1%} ({(purity_a3-purity_a1)*100:.1f}pp)')
    else:
        print(f'NO CHANGE: Purity remains {purity_a1:.1%}')

    await neo4j.close()
    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(run_tier_b())
