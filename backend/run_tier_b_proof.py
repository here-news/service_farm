#!/usr/bin/env python3
"""
Tier B Proof: Controlled Subset from Real Database
===================================================

Fixed inputs:
- 52 WFC incidents + 100 sampled distractors from Neo4j
- Git commit: [to be recorded]
- LLM: gpt-4o-mini, temp=0.0
- Artifacts cached: yes

Metrics (all approaches on same data):
- largest_case_size
- WFC_coverage, WFC_purity
- #cases, #spine_edges, #metabolic_edges
- Top contaminants with bridge-edge audit

Ablations:
- A1: Heuristic overlap (baseline)
- A2: Role-based only (no DF filter)
- A3: Role-based + DF filtering (full)
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Set, Dict, List

sys.path.insert(0, '/media/im3/plus/lab4/re_news/service_farm/backend')

import asyncpg
from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService
from workers.principled_weaver import CaseBuilder
from reee.contracts.case_formation import EntityRole

# Reproducibility metadata
PROOF_METADATA = {
    "tier": "B",
    "git_commit": "to_be_filled",
    "timestamp": datetime.utcnow().isoformat(),
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.0,
    "seed": 42,
}


async def load_tier_b_dataset(neo4j):
    """Load 52 WFC + 100 distractors with fixed query."""

    # Get all WFC incidents
    wfc_query = '''
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id
    ORDER BY i.id  // Deterministic ordering
    '''
    wfc_results = await neo4j._execute_read(wfc_query)
    wfc_ids = [r['id'] for r in wfc_results]

    # Get 100 non-WFC distractors (fixed seed for reproducibility)
    distractor_query = '''
    MATCH (i:Incident)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    WITH i
    ORDER BY i.id  // Deterministic
    LIMIT 100
    RETURN i.id as id
    '''
    distractor_results = await neo4j._execute_read(distractor_query)
    distractor_ids = [r['id'] for r in distractor_results]

    return wfc_ids, distractor_ids


async def run_ablation_a1_heuristic(cb, incident_ids: Set[str]):
    """A1: Heuristic overlap baseline."""

    print("\n" + "=" * 70)
    print("ABLATION A1: Heuristic Overlap (Baseline)")
    print("=" * 70)

    # Build adjacency from any shared entity
    edges = []
    for id1 in incident_ids:
        for id2 in incident_ids:
            if id1 >= id2:
                continue

            inc1 = cb.incidents.get(id1)
            inc2 = cb.incidents.get(id2)
            if not inc1 or not inc2:
                continue

            ents1 = inc1.anchor_entities or set()
            ents2 = inc2.anchor_entities or set()
            overlap = ents1 & ents2

            if overlap:
                edges.append((id1, id2, overlap))

    # Union-find
    parent = {iid: iid for iid in incident_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for id1, id2, _ in edges:
        union(id1, id2)

    # Collect components
    components = {}
    for iid in incident_ids:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    cases = [c for c in components.values() if len(c) >= 2]

    # Compute metrics
    wfc_ids = {iid for iid in incident_ids if 'Wang Fuk Court' in (cb.incidents.get(iid).anchor_entities or set())}

    wfc_case = None
    for c in cases:
        if c & wfc_ids:
            wfc_case = c
            break

    largest = max((len(c) for c in cases), default=0)
    wfc_in_case = len(wfc_case & wfc_ids) if wfc_case else 0
    coverage = wfc_in_case / len(wfc_ids) if wfc_ids else 0
    purity = wfc_in_case / len(wfc_case) if wfc_case else 0

    print(f"Total cases: {len(cases)}")
    print(f"Largest case: {largest}")
    print(f"Spine edges: {len(edges)}")
    print(f"WFC coverage: {coverage:.1%} ({wfc_in_case}/{len(wfc_ids)})")
    print(f"WFC purity: {purity:.1%} ({wfc_in_case}/{len(wfc_case) if wfc_case else 0})")

    # Show contaminants
    if wfc_case:
        contaminants = wfc_case - wfc_ids
        if contaminants:
            print(f"\nContaminants ({len(contaminants)}):")
            for iid in list(contaminants)[:5]:
                inc = cb.incidents.get(iid)
                if inc:
                    print(f"  {iid[:12]}: {list(inc.anchor_entities or set())[:3]}")

    return {
        "ablation": "A1_heuristic",
        "total_cases": len(cases),
        "largest_case": largest,
        "spine_edges": len(edges),
        "wfc_coverage": coverage,
        "wfc_purity": purity,
        "contaminant_count": len(wfc_case - wfc_ids) if wfc_case else 0,
    }


async def run_tier_b():
    """Run Tier B proof with all ablations."""

    # Connect to services
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

    # Load dataset
    print("Loading Tier B dataset...")
    wfc_ids, distractor_ids = await load_tier_b_dataset(neo4j)
    all_ids = set(wfc_ids + distractor_ids)

    print(f"Dataset: {len(wfc_ids)} WFC + {len(distractor_ids)} distractors = {len(all_ids)} total")
    print()

    # Initialize case builder
    cb = CaseBuilder(neo4j, db_pool=db_pool)
    await cb.load_incidents()

    # Filter to tier B subset
    cb.incidents = {iid: inc for iid, inc in cb.incidents.items() if iid in all_ids}
    print(f"Loaded {len(cb.incidents)} incidents from subset")
    print()

    # Run ablations
    results = {}

    # A1: Heuristic baseline
    results['A1'] = await run_ablation_a1_heuristic(cb, all_ids)

    # A2 and A3 would require LLM processing (17+ minutes)
    # For now, show that A1 baseline is captured

    print("\n" + "=" * 70)
    print("TIER B RESULTS SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print()
    print("NOTE: A2 (roles only) and A3 (roles + DF) require LLM processing")
    print("      Estimated time: ~17 minutes for 152 incidents")

    await neo4j.close()
    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(run_tier_b())
